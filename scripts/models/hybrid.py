"""Hybrid Tensor-Network + Variational Quantum Circuit classifier.

Architecture:
    ResNet backbone -> TN bridge (MPS / PEPS / TTN) -> PennyLane VQC -> BN -> linear head
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.classifiers import _RESNET_REGISTRY

# -- Maths helpers ---------------------------------------------------------


def _l2_normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
    return x / (x.norm(dim=dim, keepdim=True) + eps)


def _generalized_ZX(dim: int, device=None, dtype=None) -> Tuple[torch.Tensor, torch.Tensor]:
    """Clock-like (Z) and shift (X) operators.  For ``dim=2`` returns Pauli Z, X.

    For ``dim > 2`` this intentionally uses a real-valued surrogate for the
    canonical complex clock operator ``diag(exp(2*pi*i*k / dim))`` so the bridge
    stays in real arithmetic.
    """
    if dim == 2:
        Z = torch.tensor([[1., 0.], [0., -1.]], device=device, dtype=dtype)
        X = torch.tensor([[0., 1.], [1., 0.]], device=device, dtype=dtype)
        return Z, X
    ks = torch.arange(dim, device=device, dtype=dtype)
    Z = torch.diag(torch.cos(2 * math.pi * ks / dim))
    X = torch.zeros(dim, dim, device=device, dtype=dtype)
    for k in range(dim):
        X[k, (k + 1) % dim] = 1.0
    return Z, X


# -- MPS core --------------------------------------------------------------


class _SpectralCore(nn.Module):
    """Single MPS tensor with Frobenius-normalised base and learnable scale."""

    def __init__(self, phys_dim: int, left_dim: int, right_dim: int, eps: float = 1e-8) -> None:
        super().__init__()
        self.eps = eps
        self.base = nn.Parameter(torch.randn(phys_dim, left_dim, right_dim) * 0.02)
        self.log_scale = nn.Parameter(torch.zeros(()))

    def forward(self) -> torch.Tensor:
        norm = self.base.reshape(-1).norm(p=2) + self.eps
        return torch.exp(self.log_scale) * (self.base / norm)


# -- MPS bridge ------------------------------------------------------------


class MPSBridge(nn.Module):
    """Feature to qudit expectation values via a Matrix Product State.

    1. Split features into Q chunks; encode each into amplitudes + gate.
    2. Contract through MPS cores (bidirectional environments).
    3. Per-site Z/X expectations blended with learnable beta.
    4. Skip connection + learnable mixing + bias.
    """

    def __init__(
        self,
        feature_dim: int = 512,
        n_sites: int = 8,
        bond_dim: int = 64,
        phys_dim: int = 2,
        hidden_dim: int = 128,
        temperature: float = 1.0,
        renorm_stride: int = 2,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        if feature_dim % n_sites:
            raise ValueError("feature_dim must be divisible by n_sites.")

        self.Q = n_sites
        self.D = bond_dim
        self.P = phys_dim
        self.Fpc = feature_dim // n_sites
        self.temperature = temperature
        self.renorm_stride = max(1, int(renorm_stride))
        self.eps = eps

        self.ln = nn.ModuleList([nn.LayerNorm(self.Fpc) for _ in range(self.Q)])
        self.fc1 = nn.ModuleList([nn.Linear(self.Fpc, hidden_dim) for _ in range(self.Q)])
        self.fc_logits = nn.ModuleList([nn.Linear(hidden_dim, self.P) for _ in range(self.Q)])
        self.fc_gate = nn.ModuleList([nn.Linear(hidden_dim, 1) for _ in range(self.Q)])

        cores = []
        for i in range(self.Q):
            l_dim = 1 if i == 0 else self.D
            r_dim = 1 if i == self.Q - 1 else self.D
            cores.append(_SpectralCore(phys_dim, l_dim, r_dim, eps))
        self.cores = nn.ModuleList(cores)

        self.beta_logit = nn.Parameter(torch.zeros(self.Q))
        self.skip_proj = nn.Linear(feature_dim, self.Q)
        self.site_scale = nn.Parameter(torch.ones(self.Q))
        self.mix_logit = nn.Parameter(torch.zeros(()))
        self.bias = nn.Parameter(torch.zeros(self.Q))

        Z, X = _generalized_ZX(self.P)
        self.register_buffer("ObsZ", Z)
        self.register_buffer("ObsX", X)

    # -- encoding ---------------------------------------------------------

    def _encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B = x.shape[0]
        chunks = x.view(B, self.Q, self.Fpc)
        amps, gates = [], []
        for i in range(self.Q):
            h = F.gelu(self.fc1[i](self.ln[i](chunks[:, i])))
            logits = self.fc_logits[i](h) / self.temperature
            a = torch.sqrt(F.softmax(logits, dim=-1).clamp(min=self.eps))
            amps.append(_l2_normalize(a, eps=self.eps))
            gates.append(torch.sigmoid(self.fc_gate[i](h)).squeeze(-1))
        return torch.stack(amps, 1), torch.stack(gates, 1)

    # -- environment contraction ------------------------------------------

    def _left_envs(self, amps: torch.Tensor) -> List[torch.Tensor]:
        B = amps.shape[0]
        envs = [torch.ones(B, 1, device=amps.device, dtype=amps.dtype)]
        state = envs[0]
        for i in range(self.Q):
            c = torch.einsum("bp,plr->blr", amps[:, i], self.cores[i]())
            state = torch.einsum("bl,blr->br", state, c)
            if (i + 1) % self.renorm_stride == 0:
                state = _l2_normalize(state, eps=self.eps)
            envs.append(state)
        return envs

    def _right_envs(self, amps: torch.Tensor) -> List[torch.Tensor]:
        B = amps.shape[0]
        envs: list = [None] * (self.Q + 1)  # type: ignore[list-item]
        state = torch.ones(B, 1, device=amps.device, dtype=amps.dtype)
        envs[self.Q] = state
        for i in reversed(range(self.Q)):
            c = torch.einsum("plr,bp->blr", self.cores[i](), amps[:, i])
            state = torch.einsum("blr,br->bl", c, state)
            if (self.Q - i) % self.renorm_stride == 0:
                state = _l2_normalize(state, eps=self.eps)
            envs[i] = state
        return envs

    def _expectations(self, amps, left, right) -> torch.Tensor:
        z_list, x_list = [], []
        for i in range(self.Q):
            core = self.cores[i]()
            a = amps[:, i]
            cc = torch.einsum("plr,bp->blr", core, a)
            denom = torch.einsum("bl,blr,br->b", left[i], cc, right[i + 1]).abs().clamp_min(self.eps)
            for obs, out in ((self.ObsZ, z_list), (self.ObsX, x_list)):
                oc = torch.einsum("pq,qlr->plr", obs, core)
                fc = torch.einsum("plr,bp->blr", oc, a)
                out.append(torch.einsum("bl,blr,br->b", left[i], fc, right[i + 1]) / denom)
        beta = torch.sigmoid(self.beta_logit).unsqueeze(0)
        return beta * torch.stack(z_list, 1) + (1 - beta) * torch.stack(x_list, 1)

    # -- forward ----------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        amps, gates = self._encode(x)
        amps = amps * gates.unsqueeze(-1)
        z_mps = self._expectations(amps, self._left_envs(amps), self._right_envs(amps))
        z_skip = torch.tanh(self.skip_proj(x)) * self.site_scale
        alpha = torch.sigmoid(self.mix_logit)
        return alpha * z_mps + (1 - alpha) * z_skip + self.bias


# -- PEPS bridge -----------------------------------------------------------


class _PEPSSpectralCore(nn.Module):
    """Single PEPS tensor with Frobenius-normalised base and learnable scale."""

    def __init__(
        self,
        phys_dim: int,
        up_dim: int,
        down_dim: int,
        left_dim: int,
        right_dim: int,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.eps = eps
        self.base = nn.Parameter(torch.randn(phys_dim, up_dim, down_dim, left_dim, right_dim) * 0.02)
        self.log_scale = nn.Parameter(torch.zeros(()))

    def forward(self) -> torch.Tensor:
        norm = self.base.reshape(-1).norm(p=2) + self.eps
        return torch.exp(self.log_scale) * (self.base / norm)


class PEPSBridge(nn.Module):
    """Exact 4x4 PEPS bridge.

    The contraction follows the requested topology: first each of the four
    columns is contracted vertically, reducing the 4x4 lattice to four column
    tensors; then those four tensors are contracted horizontally. Local site
    values are computed by inserting a Z/X observable at one physical site at
    a time.
    """

    ROWS = 4
    COLS = 4
    N_SITES = ROWS * COLS
    MAX_BOND_DIM = 8

    def __init__(
        self,
        feature_dim: int = 512,
        n_sites: int = N_SITES,
        bond_dim: int = MAX_BOND_DIM,
        phys_dim: int = 2,
        hidden_dim: int = 128,
        temperature: float = 1.0,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        if n_sites != self.N_SITES:
            print(f"[PEPS] Exact topology is fixed to 4x4 ({self.N_SITES} sites); ignoring n_sites={n_sites}.")
        if bond_dim < 1:
            raise ValueError(f"bond_dim must be >= 1, got {bond_dim}.")
        if feature_dim % self.N_SITES:
            raise ValueError(f"feature_dim must be divisible by {self.N_SITES} PEPS sites.")
        if bond_dim > self.MAX_BOND_DIM:
            print(f"[PEPS] Exact contraction supports bond_dim <= {self.MAX_BOND_DIM}; using {self.MAX_BOND_DIM}.")
            bond_dim = self.MAX_BOND_DIM

        self.Q = self.N_SITES
        self.D = bond_dim
        self.P = phys_dim
        self.hidden_dim = hidden_dim
        self.Fps = feature_dim // self.Q
        self.temperature = temperature
        self.eps = eps

        self.ln = nn.ModuleList([nn.LayerNorm(self.Fps) for _ in range(self.Q)])
        self.fc1 = nn.ModuleList([nn.Linear(self.Fps, hidden_dim) for _ in range(self.Q)])
        self.fc_logits = nn.ModuleList([nn.Linear(hidden_dim, self.P) for _ in range(self.Q)])
        self.fc_gate = nn.ModuleList([nn.Linear(hidden_dim, 1) for _ in range(self.Q)])

        self.up_d, self.dn_d, self.lt_d, self.rt_d = [], [], [], []
        cores = []
        for i in range(self.Q):
            r, c = divmod(i, self.COLS)
            self.up_d.append(1 if r == 0 else self.D)
            self.dn_d.append(1 if r == self.ROWS - 1 else self.D)
            self.lt_d.append(1 if c == 0 else self.D)
            self.rt_d.append(1 if c == self.COLS - 1 else self.D)
            cores.append(
                _PEPSSpectralCore(
                    phys_dim,
                    self.up_d[i],
                    self.dn_d[i],
                    self.lt_d[i],
                    self.rt_d[i],
                    eps,
                )
            )
        self.cores = nn.ModuleList(cores)

        self.beta_logit = nn.Parameter(torch.zeros(self.Q))
        self.skip_proj = nn.Linear(feature_dim, self.Q)
        self.site_scale = nn.Parameter(torch.ones(self.Q))
        self.mix_logit = nn.Parameter(torch.zeros(()))
        self.bias = nn.Parameter(torch.zeros(self.Q))

        Z, X = _generalized_ZX(self.P)
        self.register_buffer("ObsZ", Z)
        self.register_buffer("ObsX", X)

    def _idx(self, r: int, c: int) -> int:
        return r * self.COLS + c

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        chunks = x.view(B, self.Q, self.Fps)
        amps = []
        for i in range(self.Q):
            h = F.gelu(self.fc1[i](self.ln[i](chunks[:, i])))
            logits = self.fc_logits[i](h) / self.temperature
            a = torch.sqrt(F.softmax(logits, -1).clamp(min=self.eps))
            gate = torch.sigmoid(self.fc_gate[i](h))
            uniform = torch.full_like(a, 1.0 / math.sqrt(self.P))
            amps.append(_l2_normalize(gate * a + (1.0 - gate) * uniform, eps=self.eps))
        return torch.stack(amps, 1)

    def _contract_physical(
        self,
        amps: torch.Tensor,
        cores: List[torch.Tensor],
    ) -> List[torch.Tensor]:
        return [torch.einsum("bp,pudlr->budlr", amps[:, i], cores[i]) for i in range(self.Q)]

    def _observable_amps(self, amps: torch.Tensor) -> List[torch.Tensor]:
        out = []
        for i in range(self.Q):
            beta = torch.sigmoid(self.beta_logit[i])
            obs = beta * self.ObsZ + (1.0 - beta) * self.ObsX
            out.append(torch.einsum("bp,pq->bq", amps[:, i], obs))
        return out

    def _contract_column_batch(self, col_tensors: List[torch.Tensor]) -> torch.Tensor:
        return torch.einsum(
            "zacde,zcfgh,zfijk,zilmn->zdgjmehkn",
            col_tensors[0],
            col_tensors[1],
            col_tensors[2],
            col_tensors[3],
        )

    def _contract_all_columns_batch(self, columns: List[torch.Tensor]) -> torch.Tensor:
        left = torch.einsum("zabcdefgh,zefghijkl->zijkl", columns[0], columns[1])
        left = torch.einsum("zijkl,zijklmnop->zmnop", left, columns[2])
        return torch.einsum("zmnop,zmnopqrst->z", left, columns[3])

    def _column_from_sites_batch(
        self,
        site_tensors: List[torch.Tensor],
        col: int,
        replacement: Optional[Tuple[int, torch.Tensor]] = None,
    ) -> torch.Tensor:
        tensors = [site_tensors[self._idx(row, col)] for row in range(self.ROWS)]
        if replacement is not None:
            row, tensor = replacement
            tensors[row] = tensor
        return self._contract_column_batch(tensors)

    def _exact_expectations(
        self,
        site_tensors: List[torch.Tensor],
        obs_site_tensors: List[torch.Tensor],
    ) -> torch.Tensor:
        # The lattice size is fixed at 4x4, so the site loop is cheap. Keep
        # the contraction batched across samples to avoid Python overhead on B.
        base_columns = [self._column_from_sites_batch(site_tensors, col) for col in range(self.COLS)]
        denom = self._contract_all_columns_batch(base_columns).abs().clamp_min(self.eps)

        outputs = []
        for site in range(self.Q):
            row, col = divmod(site, self.COLS)
            columns = list(base_columns)
            columns[col] = self._column_from_sites_batch(
                site_tensors,
                col,
                replacement=(row, obs_site_tensors[site]),
            )
            outputs.append(self._contract_all_columns_batch(columns) / denom)

        return torch.tanh(torch.stack(outputs, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        amps = self._encode(x)
        cores = [core() for core in self.cores]
        site_tensors = self._contract_physical(amps, cores)
        obs_site_tensors = self._contract_physical(torch.stack(self._observable_amps(amps), 1), cores)
        z = self._exact_expectations(site_tensors, obs_site_tensors)
        z_skip = torch.tanh(self.skip_proj(x)) * self.site_scale
        alpha = torch.sigmoid(self.mix_logit)
        return alpha * z + (1.0 - alpha) * z_skip + self.bias


class _TTNCore(nn.Module):
    """Single TTN contraction tensor with Frobenius-normalised base."""

    def __init__(self, left_dim: int, right_dim: int, parent_dim: int, eps: float = 1e-8) -> None:
        super().__init__()
        self.eps = eps
        self.base = nn.Parameter(torch.randn(left_dim, right_dim, parent_dim) * 0.02)
        self.log_scale = nn.Parameter(torch.zeros(()))

    def forward(self) -> torch.Tensor:
        norm = self.base.reshape(-1).norm(p=2) + self.eps
        return torch.exp(self.log_scale) * (self.base / norm)


class TTNBridge(nn.Module):
    """Hierarchical binary-tree TTN bridge with a fixed 16-leaf layout.

    The bridge mirrors the diagrammed TTN:
    1. split features into 16 leaves;
    2. encode each leaf into a local physical state;
    3. contract adjacent pairs bottom-up through TTN cores;
    4. stop when ``n_sites`` final states remain and measure each state.
    """

    N_LEAVES = 16
    VALID_OUTPUTS = (4, 8)

    def __init__(
        self,
        feature_dim: int = 512,
        n_sites: int = 8,
        bond_dim: int = 32,
        phys_dim: int = 2,
        hidden_dim: int = 128,
        temperature: float = 1.0,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        if feature_dim % self.N_LEAVES:
            raise ValueError(f"feature_dim must be divisible by {self.N_LEAVES} TTN leaves.")
        if bond_dim < 1:
            raise ValueError(f"bond_dim must be >= 1, got {bond_dim}.")
        if n_sites not in self.VALID_OUTPUTS:
            raise ValueError(
                f"TTN uses a fixed {self.N_LEAVES}-leaf tree; n_sites must be one of {self.VALID_OUTPUTS}, got {n_sites}."
            )

        self.Q = n_sites
        self.D = bond_dim
        self.P = phys_dim
        self.hidden_dim = hidden_dim
        self.temperature = temperature
        self.eps = eps
        self.Fpl = feature_dim // self.N_LEAVES

        self.ln = nn.ModuleList([nn.LayerNorm(self.Fpl) for _ in range(self.N_LEAVES)])
        self.fc1 = nn.ModuleList([nn.Linear(self.Fpl, hidden_dim) for _ in range(self.N_LEAVES)])
        self.fc_logits = nn.ModuleList([nn.Linear(hidden_dim, self.P) for _ in range(self.N_LEAVES)])
        self.fc_gate = nn.ModuleList([nn.Linear(hidden_dim, 1) for _ in range(self.N_LEAVES)])

        levels = []
        child_dim = self.P
        n_children = self.N_LEAVES
        while n_children > self.Q:
            n_parents = n_children // 2
            levels.append(nn.ModuleList([
                _TTNCore(child_dim, child_dim, self.D, eps) for _ in range(n_parents)
            ]))
            n_children = n_parents
            child_dim = self.D
        self.levels = nn.ModuleList(levels)

        self.beta_logit = nn.Parameter(torch.zeros(self.Q))
        self.skip_proj = nn.Linear(feature_dim, self.Q)
        self.site_scale = nn.Parameter(torch.ones(self.Q))
        self.mix_logit = nn.Parameter(torch.zeros(()))
        self.bias = nn.Parameter(torch.zeros(self.Q))

        Z, X = _generalized_ZX(self.D)
        self.register_buffer("ObsZ", Z)
        self.register_buffer("ObsX", X)

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        chunks = x.view(B, self.N_LEAVES, self.Fpl)
        leaves = []
        for i in range(self.N_LEAVES):
            h = F.gelu(self.fc1[i](self.ln[i](chunks[:, i])))
            logits = self.fc_logits[i](h) / self.temperature
            amp = torch.sqrt(F.softmax(logits, dim=-1).clamp(min=self.eps))
            gate = torch.sigmoid(self.fc_gate[i](h))
            uniform = torch.full_like(amp, 1.0 / math.sqrt(self.P))
            leaves.append(_l2_normalize(gate * amp + (1.0 - gate) * uniform, eps=self.eps))
        return torch.stack(leaves, 1)

    def _contract_level(self, states: torch.Tensor, cores: nn.ModuleList) -> torch.Tensor:
        left = states[:, 0::2]
        right = states[:, 1::2]
        core_stack = torch.stack([core() for core in cores], 0)
        parents = torch.einsum("bnl,bnr,nlrd->bnd", left, right, core_stack)
        return _l2_normalize(parents, eps=self.eps)

    def _bottom_up(self, leaves: torch.Tensor) -> torch.Tensor:
        states = leaves
        for level in self.levels:
            states = self._contract_level(states, level)
        return states

    def _measure(self, states: torch.Tensor) -> torch.Tensor:
        z = torch.einsum("bqd,df,bqf->bq", states, self.ObsZ, states)
        x = torch.einsum("bqd,df,bqf->bq", states, self.ObsX, states)
        beta = torch.sigmoid(self.beta_logit).unsqueeze(0)
        return beta * z + (1.0 - beta) * x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        leaves = self._encode(x)
        final_states = self._bottom_up(leaves)
        z_ttn = self._measure(final_states)
        z_skip = torch.tanh(self.skip_proj(x)) * self.site_scale
        alpha = torch.sigmoid(self.mix_logit)
        return alpha * z_ttn + (1.0 - alpha) * z_skip + self.bias


# -- PennyLane quantum layer ------------------------------------------------

class PennyLaneQuantumLayer(nn.Module):
    """Variational quantum circuit: RX(inputs) -> [RX/RY/RZ + ring CNOT]xL -> <Z>."""

    def __init__(self, n_qubits: int, n_layers: int) -> None:
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers

        try:
            import pennylane as qml
        except ImportError as exc:
            raise ImportError(
                "PennyLane is required for the quantum layer. Install it with `pip install pennylane`."
            ) from exc

        dev, diff = None, None
        for backend, dm in [("lightning.gpu", "adjoint"), ("lightning.qubit", "adjoint")]:
            try:
                dev = qml.device(backend, wires=n_qubits)
                diff = dm
                print(f"[Quantum] Using {backend} ({dm}).")
                break
            except Exception:
                continue
        if dev is None:
            dev = qml.device("default.qubit", wires=n_qubits)
            diff = "backprop"
            print("[Quantum] Falling back to default.qubit (backprop).")

        @qml.qnode(dev, interface="torch", diff_method=diff)
        def circuit(inputs, weights):
            for i in range(n_qubits):
                qml.RX(inputs[..., i], wires=i)
            for layer in range(n_layers):
                for i in range(n_qubits):
                    qml.RX(weights[layer, i, 0], wires=i)
                    qml.RY(weights[layer, i, 1], wires=i)
                    qml.RZ(weights[layer, i, 2], wires=i)
                if n_qubits > 1:
                    for i in range(n_qubits - 1):
                        qml.CNOT(wires=[i, i + 1])
                    if n_qubits > 2:
                        qml.CNOT(wires=[n_qubits - 1, 0])
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        self.qlayer = qml.qnn.TorchLayer(circuit, {"weights": (n_layers, n_qubits, 3)})

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.qlayer(x)


# -- ResNet feature extractor ---------------------------------------------

_FEAT_DIM = {"resnet18": 512, "resnet34": 512, "resnet50": 2048}


class _ResNetFeatureExtractor(nn.Module):
    """ResNet backbone with the final FC layer removed.

    Args:
        backbone: One of ``resnet18``/``resnet34``/``resnet50``.
        weights_path: Optional path to custom weights. If given,
            *pretrained* is ignored and the custom checkpoint is loaded with
            lenient key matching.
        pretrained: When ``True`` and *weights_path* is ``None``, load
            ImageNet-pretrained torchvision weights. When ``False``, the
            backbone is randomly initialised.
        freeze: When ``True``, freeze every parameter except ``layer4`` (the
            final residual stage), enabling efficient fine-tuning.
    """

    def __init__(
        self,
        backbone: str = "resnet18",
        weights_path: Optional[str] = None,
        pretrained: bool = True,
        freeze: bool = True,
    ) -> None:
        super().__init__()
        backbone = backbone.lower().strip()
        if backbone not in _FEAT_DIM:
            raise ValueError(f"Unknown backbone '{backbone}'. Choose from {list(_FEAT_DIM)}.")
        self.feature_dim = _FEAT_DIM[backbone]

        build_fn, default_w = _RESNET_REGISTRY[backbone]

        if weights_path:
            m = build_fn(weights=None)
            self._load_custom(m, weights_path)
            print(f"[Backbone] Loaded custom weights ({backbone}) from {weights_path}")
        elif pretrained:
            m = build_fn(weights=default_w)
            print(f"[Backbone] ImageNet-pretrained {backbone} ({self.feature_dim}-d)")
        else:
            m = build_fn(weights=None)
            print(f"[Backbone] Randomly-initialised {backbone} ({self.feature_dim}-d)")

        self.backbone = nn.Sequential(*list(m.children())[:-1])

        if freeze:
            for p in self.backbone.parameters():
                p.requires_grad = False
            for p in m.layer4.parameters():
                p.requires_grad = True

    @staticmethod
    def _load_custom(model: nn.Module, path: str) -> None:
        raw = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(raw, dict):
            for key in ("state_dict", "model_state_dict", "model", "teacher"):
                if key in raw:
                    raw = raw[key]
                    break
        if not isinstance(raw, dict):
            raise ValueError(f"Cannot interpret checkpoint at {path}.")

        skip_prefixes = ("fc.", "head.", "classifier.")
        strip_prefixes = ("module.", "backbone.", "encoder.", "resnet.", "model.")
        cleaned = {}
        for k, v in raw.items():
            for pfx in strip_prefixes:
                if k.startswith(pfx):
                    k = k[len(pfx):]
            if any(k.startswith(s) for s in skip_prefixes):
                continue
            cleaned[k] = v

        missing, unexpected = model.load_state_dict(cleaned, strict=False)
        missing_bb = [k for k in missing if not k.startswith("fc.")]
        if missing_bb:
            print(f"[Backbone] Warning missing keys: {missing_bb[:10]}")
        if unexpected:
            print(f"[Backbone] Warning unexpected keys: {unexpected[:10]}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x).flatten(1)


# -- Hybrid classifier ----------------------------------------------------


class HybridTNClassifier(nn.Module):
    """ResNet -> TN bridge (MPS/PEPS/TTN) -> VQC -> BN -> linear.

    Input:  ``(B, 3, H, W)``  Output: ``(B, n_classes)``
    """

    def __init__(
        self,
        n_classes: int,
        n_qubits: int = 8,
        tn_bond_dim: int = 64,
        tn_phys_dim: int = 2,
        bridge_hidden_dim: int = 128,
        vqc_layers: int = 2,
        freeze_resnet: bool = False,
        bridge_type: str = "mps",
        backbone: str = "resnet18",
        backbone_weights: Optional[str] = None,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        bridge_type = bridge_type.lower().strip()
        if bridge_type == "peps" and n_qubits != PEPSBridge.N_SITES:
            raise ValueError(
                f"PEPS uses a fixed 4x4 topology ({PEPSBridge.N_SITES} sites); "
                f"got n_qubits={n_qubits}."
            )
        if bridge_type == "ttn" and n_qubits not in TTNBridge.VALID_OUTPUTS:
            raise ValueError(
                f"TTN uses a fixed {TTNBridge.N_LEAVES}-leaf tree; "
                f"expected n_qubits in {TTNBridge.VALID_OUTPUTS}, got {n_qubits}."
            )

        self.n_classes = n_classes
        self.n_qubits = n_qubits

        self.resnet = _ResNetFeatureExtractor(
            backbone=backbone,
            weights_path=backbone_weights,
            pretrained=pretrained,
            freeze=freeze_resnet,
        )
        feat_dim = self.resnet.feature_dim

        if feat_dim % n_qubits:
            valid = [q for q in (8, 16, 32, 64) if feat_dim % q == 0]
            raise ValueError(
                f"feature_dim={feat_dim} not divisible by n_qubits={n_qubits}. "
                f"Try one of {valid}."
            )

        if bridge_type == "mps":
            self.tn_bridge = MPSBridge(feat_dim, 
                                       n_qubits, 
                                       tn_bond_dim, 
                                       tn_phys_dim, 
                                       bridge_hidden_dim
                                    )
        elif bridge_type == "peps":
            self.tn_bridge = PEPSBridge(feat_dim, 
                                        n_qubits, 
                                        tn_bond_dim, 
                                        tn_phys_dim, 
                                        bridge_hidden_dim
                                    )
        elif bridge_type == "ttn":
            self.tn_bridge = TTNBridge(feat_dim, 
                                       n_qubits, 
                                       tn_bond_dim, 
                                       tn_phys_dim, 
                                       bridge_hidden_dim
                                    )
        else:
            raise ValueError(f"Unknown bridge_type '{bridge_type}'.")

        self.quantum = PennyLaneQuantumLayer(n_qubits, vqc_layers)

        self.q_norm = nn.BatchNorm1d(n_qubits)
        self.q_proj = nn.Linear(n_qubits, n_classes)
        self.angle_scale = nn.Parameter(torch.tensor(0.75 * math.pi))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        feats = self.resnet(images)
        q_in = self.tn_bridge(feats) * self.angle_scale
        q_out = self.q_norm(self.quantum(q_in))
        return self.q_proj(q_out)


# -- Convenience factory --------------------------------------------------


def hybrid_tn_model(
    fc_out_features: int,
    n_qubits: int = 8,
    tn_bond_dim: int = 64,
    tn_phys_dim: int = 2,
    tn_hidden_dim: int = 128,
    vqc_layers: int = 2,
    freeze_resnet: bool = False,
    tn_bridge: str = "mps",
    backbone: str = "resnet18",
    backbone_weights: Optional[str] = None,
    pretrained: bool = True,
) -> HybridTNClassifier:
    """Factory for `HybridTNClassifier`."""
    return HybridTNClassifier(
        n_classes=fc_out_features,
        n_qubits=n_qubits,
        tn_bond_dim=tn_bond_dim,
        tn_phys_dim=tn_phys_dim,
        bridge_hidden_dim=tn_hidden_dim,
        vqc_layers=vqc_layers,
        freeze_resnet=freeze_resnet,
        bridge_type=tn_bridge,
        backbone=backbone,
        backbone_weights=backbone_weights,
        pretrained=pretrained,
    )
