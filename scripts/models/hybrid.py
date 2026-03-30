"""Hybrid Tensor-Network + Variational Quantum Circuit classifier.

Architecture:
    ResNet backbone → TN bridge (MPS *or* PEPS) → VQC (PennyLane *or* MLP) → BN → linear head
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Maths helpers ─────────────────────────────────────────────────────────


def _l2_normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
    return x / (x.norm(dim=dim, keepdim=True) + eps)


def _factor_grid(n: int, rows: Optional[int], cols: Optional[int]) -> Tuple[int, int]:
    """Resolve a 2-D grid whose area equals *n*."""
    if n < 1:
        raise ValueError(f"n must be ≥ 1, got {n}")
    if rows is not None and cols is not None:
        if int(rows) * int(cols) != n:
            raise ValueError(f"rows×cols={rows}×{cols} ≠ {n}")
        return int(rows), int(cols)
    if rows is not None:
        r = int(rows)
        if n % r:
            raise ValueError(f"n={n} not divisible by rows={r}")
        return r, n // r
    if cols is not None:
        c = int(cols)
        if n % c:
            raise ValueError(f"n={n} not divisible by cols={c}")
        return n // c, c
    r = int(math.sqrt(n))
    while r > 1 and n % r:
        r -= 1
    return r, n // r


def _generalized_ZX(dim: int, device=None, dtype=None) -> Tuple[torch.Tensor, torch.Tensor]:
    """Clock (Z) and shift (X) operators.  For ``dim=2`` returns Pauli Z, X."""
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


# ── MPS core ──────────────────────────────────────────────────────────────


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


# ── MPS bridge ────────────────────────────────────────────────────────────


class MPSBridge(nn.Module):
    """Feature → qudit expectation values via a Matrix Product State.

    1. Split features into Q chunks; encode each into amplitudes + gate.
    2. Contract through MPS cores (bidirectional environments).
    3. Per-site Z/X expectations blended with learnable β.
    4. Skip connection + learnable mixing α + bias.
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

    # ── encoding ─────────────────────────────────────────────────────────

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

    # ── environment contraction ──────────────────────────────────────────

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
            for obs, out in ((self.ObsZ, z_list), (self.ObsX, x_list)):
                oc = torch.einsum("pq,qlr->plr", obs, core)
                fc = torch.einsum("plr,bp->blr", oc, a)
                out.append(torch.einsum("bl,blr,br->b", left[i], fc, right[i + 1]))
        beta = torch.sigmoid(self.beta_logit).unsqueeze(0)
        return beta * torch.stack(z_list, 1) + (1 - beta) * torch.stack(x_list, 1)

    # ── forward ──────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        amps, gates = self._encode(x)
        amps = amps * gates.unsqueeze(-1)
        z_mps = self._expectations(amps, self._left_envs(amps), self._right_envs(amps))
        z_skip = torch.tanh(self.skip_proj(x)) * self.site_scale
        alpha = torch.sigmoid(self.mix_logit)
        return alpha * z_mps + (1 - alpha) * z_skip + self.bias


# ── PEPS bridge ───────────────────────────────────────────────────────────


class PEPSBridge(nn.Module):
    """2-D PEPS-inspired bridge with iterative message passing.

    Uses a factorised mean-field contraction with learnable directional
    projections and per-step message transforms.
    """

    def __init__(
        self,
        feature_dim: int = 512,
        n_sites: int = 8,
        bond_dim: int = 32,
        phys_dim: int = 2,
        hidden_dim: int = 128,
        temperature: float = 1.0,
        n_message_passing_steps: int = 3,
        grid_rows: Optional[int] = None,
        grid_cols: Optional[int] = None,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        if feature_dim % n_sites:
            raise ValueError("feature_dim must be divisible by n_sites.")

        self.Q = n_sites
        self.D = bond_dim
        self.P = phys_dim
        self.hidden_dim = hidden_dim
        self.Fps = feature_dim // n_sites
        self.temperature = temperature
        self.n_steps = max(1, int(n_message_passing_steps))
        self.eps = eps
        self.rows, self.cols = _factor_grid(n_sites, grid_rows, grid_cols)

        # Per-site encoders
        self.ln = nn.ModuleList([nn.LayerNorm(self.Fps) for _ in range(self.Q)])
        self.fc1 = nn.ModuleList([nn.Linear(self.Fps, hidden_dim) for _ in range(self.Q)])
        self.fc_logits = nn.ModuleList([nn.Linear(hidden_dim, self.P) for _ in range(self.Q)])
        self.fc_gate = nn.ModuleList([nn.Linear(hidden_dim, 1) for _ in range(self.Q)])

        # Directional bond dimensions (boundary = 1)
        self.up_d, self.dn_d, self.lt_d, self.rt_d = [], [], [], []
        for i in range(self.Q):
            r, c = divmod(i, self.cols)
            self.up_d.append(1 if r == 0 else self.D)
            self.dn_d.append(1 if r == self.rows - 1 else self.D)
            self.lt_d.append(1 if c == 0 else self.D)
            self.rt_d.append(1 if c == self.cols - 1 else self.D)

        # Directional projections from hidden_dim
        self.up_proj = nn.ModuleList([nn.Linear(hidden_dim, self.up_d[i], bias=False) for i in range(self.Q)])
        self.dn_proj = nn.ModuleList([nn.Linear(hidden_dim, self.dn_d[i], bias=False) for i in range(self.Q)])
        self.lt_proj = nn.ModuleList([nn.Linear(hidden_dim, self.lt_d[i], bias=False) for i in range(self.Q)])
        self.rt_proj = nn.ModuleList([nn.Linear(hidden_dim, self.rt_d[i], bias=False) for i in range(self.Q)])

        # Learnable message transforms per step
        self.msg_transforms = nn.ModuleList()
        for _ in range(self.n_steps):
            step_tf = nn.ModuleList([nn.Linear(self.D, self.D, bias=False) for _ in range(4)])
            for lt in step_tf:
                nn.init.eye_(lt.weight)
                lt.weight.data.mul_(0.9).add_(0.1 * torch.randn_like(lt.weight) * 0.02)
            self.msg_transforms.append(step_tf)

        self.env_weight_logits = nn.Parameter(torch.zeros(self.Q, 4))
        self.env_proj = nn.ModuleList([nn.Linear(4, self.P * self.P) for _ in range(self.Q)])

        self.beta_logit = nn.Parameter(torch.zeros(self.Q))
        self.skip_proj = nn.Linear(feature_dim, self.Q)
        self.site_scale = nn.Parameter(torch.ones(self.Q))
        self.mix_logit = nn.Parameter(torch.zeros(()))
        self.bias = nn.Parameter(torch.zeros(self.Q))

        Z, X = _generalized_ZX(self.P)
        self.register_buffer("ObsZ", Z)
        self.register_buffer("ObsX", X)

    # helpers
    def _idx(self, r: int, c: int) -> int:
        return r * self.cols + c

    def _unit_msg(self, B: int, dim: int, dev, dt) -> torch.Tensor:
        return _l2_normalize(torch.ones(B, dim, device=dev, dtype=dt), eps=self.eps)

    @staticmethod
    def _dot(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return (a * b).sum(-1, keepdim=True)

    def _apply_tf(self, msg: torch.Tensor, tf: nn.Linear) -> torch.Tensor:
        if msg.shape[-1] == 1:
            return msg
        return _l2_normalize(tf(msg), eps=self.eps)

    # encode
    def _encode(self, x: torch.Tensor):
        B = x.shape[0]
        chunks = x.view(B, self.Q, self.Fps)
        amps, gates, hiddens = [], [], []
        for i in range(self.Q):
            h = F.gelu(self.fc1[i](self.ln[i](chunks[:, i])))
            hiddens.append(h)
            logits = self.fc_logits[i](h) / self.temperature
            a = torch.sqrt(F.softmax(logits, -1).clamp(min=self.eps))
            amps.append(_l2_normalize(a, eps=self.eps))
            gates.append(torch.sigmoid(self.fc_gate[i](h)).squeeze(-1))
        return torch.stack(amps, 1), torch.stack(gates, 1), hiddens

    def _build_factors(self, amps, gates, hiddens):
        dirs = {"up": [], "dn": [], "lt": [], "rt": []}
        gated = []
        for i in range(self.Q):
            a = _l2_normalize(amps[:, i] * gates[:, i:i+1].clamp(min=self.eps), eps=self.eps)
            gated.append(a)
            h = hiddens[i]
            for key, proj in [("up", self.up_proj), ("dn", self.dn_proj),
                              ("lt", self.lt_proj), ("rt", self.rt_proj)]:
                dirs[key].append(_l2_normalize(torch.tanh(proj[i](h)), eps=self.eps))
        return dirs, gated

    def _message_pass(self, factors, B, dev, dt):
        in_u = [self._unit_msg(B, self.up_d[i], dev, dt) for i in range(self.Q)]
        in_d = [self._unit_msg(B, self.dn_d[i], dev, dt) for i in range(self.Q)]
        in_l = [self._unit_msg(B, self.lt_d[i], dev, dt) for i in range(self.Q)]
        in_r = [self._unit_msg(B, self.rt_d[i], dev, dt) for i in range(self.Q)]

        for step in range(self.n_steps):
            tf = self.msg_transforms[step]
            o_u, o_d, o_l, o_r = [], [], [], []
            for i in range(self.Q):
                u, d, l, r = factors["up"][i], factors["dn"][i], factors["lt"][i], factors["rt"][i]
                du, dd, dl, dr = self._dot(d, in_d[i]), self._dot(u, in_u[i]), self._dot(l, in_l[i]), self._dot(r, in_r[i])
                o_u.append(_l2_normalize(u * dd * dl * dr, eps=self.eps))
                o_d.append(_l2_normalize(d * du * dl * dr, eps=self.eps))
                o_l.append(_l2_normalize(l * du * dd * dr, eps=self.eps))
                o_r.append(_l2_normalize(r * du * dd * dl, eps=self.eps))

            nu = [None] * self.Q
            nd = [None] * self.Q
            nl = [None] * self.Q
            nr = [None] * self.Q
            for rr in range(self.rows):
                for cc in range(self.cols):
                    i = self._idx(rr, cc)
                    nu[i] = self._apply_tf(o_d[self._idx(rr-1, cc)], tf[0]) if rr > 0 else self._unit_msg(B, self.up_d[i], dev, dt)
                    nd[i] = self._apply_tf(o_u[self._idx(rr+1, cc)], tf[1]) if rr < self.rows - 1 else self._unit_msg(B, self.dn_d[i], dev, dt)
                    nl[i] = self._apply_tf(o_r[self._idx(rr, cc-1)], tf[2]) if cc > 0 else self._unit_msg(B, self.lt_d[i], dev, dt)
                    nr[i] = self._apply_tf(o_l[self._idx(rr, cc+1)], tf[3]) if cc < self.cols - 1 else self._unit_msg(B, self.rt_d[i], dev, dt)
            in_u, in_d, in_l, in_r = nu, nd, nl, nr

        return in_u, in_d, in_l, in_r

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        amps, gates, hiddens = self._encode(x)
        factors, gated = self._build_factors(amps, gates, hiddens)
        B = x.shape[0]
        in_u, in_d, in_l, in_r = self._message_pass(factors, B, x.device, x.dtype)

        outputs = []
        for i in range(self.Q):
            a = gated[i]
            scores = torch.stack([
                self._dot(factors["up"][i], in_u[i]).squeeze(-1),
                self._dot(factors["dn"][i], in_d[i]).squeeze(-1),
                self._dot(factors["lt"][i], in_l[i]).squeeze(-1),
                self._dot(factors["rt"][i], in_r[i]).squeeze(-1),
            ], -1)
            w = F.softmax(self.env_weight_logits[i].unsqueeze(0) + scores, dim=-1)
            env_vec = w * scores
            env_mat = self.env_proj[i](env_vec).view(B, self.P, self.P)
            env_mat = 0.5 * (env_mat + env_mat.transpose(-1, -2))
            env_mat = env_mat + torch.eye(self.P, device=env_mat.device).unsqueeze(0)

            beta = torch.sigmoid(self.beta_logit[i])
            O = beta * self.ObsZ + (1 - beta) * self.ObsX
            eff = torch.einsum("bpq,qr->bpr", env_mat, O)
            outputs.append(torch.einsum("bp,bpq,bq->b", a, eff, a))

        z = torch.stack(outputs, 1)
        z_skip = torch.tanh(self.skip_proj(x)) * self.site_scale
        alpha = torch.sigmoid(self.mix_logit)
        return alpha * z + (1 - alpha) * z_skip + self.bias


# ── PennyLane quantum layer ──────────────────────────────────────────────


class PennyLaneQuantumLayer(nn.Module):
    """Variational quantum circuit: RX(inputs) → [RX/RY/RZ + ring CNOT]×L → ⟨Z⟩."""

    def __init__(self, n_qubits: int, n_layers: int) -> None:
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers

        try:
            import pennylane as qml
        except ImportError as exc:
            raise ImportError(
                "PennyLane is required for the quantum layer. "
                "Install it or use --no_pennylane for the classical fallback."
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
                for i in range(n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
                qml.CNOT(wires=[n_qubits - 1, 0])
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        self.qlayer = qml.qnn.TorchLayer(circuit, {"weights": (n_layers, n_qubits, 3)})

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.qlayer(x)


# ── ResNet feature extractor ─────────────────────────────────────────────

_FEAT_DIM = {"resnet18": 512, "resnet34": 512, "resnet50": 2048}


class _ResNetFeatureExtractor(nn.Module):
    """Pretrained ResNet backbone (FC removed).

    Supports custom weight loading with lenient key matching.
    """

    def __init__(self, arch: str = "resnet18", weights_path: Optional[str] = None, freeze: bool = True) -> None:
        super().__init__()
        arch = arch.lower().strip()
        if arch not in _FEAT_DIM:
            raise ValueError(f"Unknown arch '{arch}'. Choose from {list(_FEAT_DIM)}.")
        self.feature_dim = _FEAT_DIM[arch]

        from torchvision import models as tvm
        factory = {
            "resnet18": (tvm.resnet18, tvm.ResNet18_Weights.DEFAULT),
            "resnet34": (tvm.resnet34, tvm.ResNet34_Weights.DEFAULT),
            "resnet50": (tvm.resnet50, tvm.ResNet50_Weights.DEFAULT),
        }
        build_fn, default_w = factory[arch]

        if weights_path:
            m = build_fn(weights=None)
            self._load_custom(m, weights_path)
            print(f"[Backbone] Loaded custom weights ({arch}) from {weights_path}")
        else:
            m = build_fn(weights=default_w)
            print(f"[Backbone] ImageNet-pretrained {arch} ({self.feature_dim}-d)")

        self.backbone = nn.Sequential(*list(m.children())[:-1])

        if freeze:
            for p in self.backbone.parameters():
                p.requires_grad = False
            for p in m.layer4.parameters():
                p.requires_grad = True
            trainable = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)
            total = sum(p.numel() for p in self.backbone.parameters())
            print(f"[Backbone] Frozen: {total - trainable:,} | Trainable (layer4): {trainable:,}")

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
            print(f"[Backbone] Warning – missing keys: {missing_bb[:10]}")
        if unexpected:
            print(f"[Backbone] Warning – unexpected keys: {unexpected[:10]}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x).flatten(1)


# ── Hybrid classifier ────────────────────────────────────────────────────


class HybridTNClassifier(nn.Module):
    """ResNet → TN bridge (MPS/PEPS) → VQC → BN → linear.

    Drop-in replacement for a plain ResNet classifier.
    Input:  ``(B, 3, H, W)``  Output: ``(B, n_classes)``
    """

    def __init__(
        self,
        n_classes: int,
        n_qubits: int = 8,
        mps_bond_dim: int = 64,
        mps_phys_dim: int = 2,
        bridge_hidden_dim: int = 128,
        vqc_layers: int = 2,
        freeze_resnet: bool = True,
        use_pennylane: bool = True,
        bridge_type: str = "mps",
        peps_rows: Optional[int] = None,
        peps_cols: Optional[int] = None,
        peps_message_passing_steps: int = 3,
        backbone: str = "resnet18",
        backbone_weights: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.n_classes = n_classes
        self.n_qubits = n_qubits

        self.resnet = _ResNetFeatureExtractor(arch=backbone, weights_path=backbone_weights, freeze=freeze_resnet)
        feat_dim = self.resnet.feature_dim

        if feat_dim % n_qubits:
            valid = [q for q in (8, 16, 32, 64) if feat_dim % q == 0]
            raise ValueError(
                f"feature_dim={feat_dim} not divisible by n_qubits={n_qubits}. "
                f"Try one of {valid}."
            )

        bridge_type = bridge_type.lower().strip()
        if bridge_type == "mps":
            self.tn_bridge = MPSBridge(feat_dim, n_qubits, mps_bond_dim, mps_phys_dim, bridge_hidden_dim)
        elif bridge_type == "peps":
            self.tn_bridge = PEPSBridge(
                feat_dim, n_qubits, mps_bond_dim, mps_phys_dim, bridge_hidden_dim,
                n_message_passing_steps=peps_message_passing_steps,
                grid_rows=peps_rows, grid_cols=peps_cols,
            )
        else:
            raise ValueError(f"Unknown bridge_type '{bridge_type}'.")

        if use_pennylane:
            self.quantum = PennyLaneQuantumLayer(n_qubits, vqc_layers)
        else:
            self.quantum = nn.Sequential(
                nn.Linear(n_qubits, 4 * n_qubits), nn.Tanh(),
                nn.Linear(4 * n_qubits, n_qubits),
            )

        self.q_norm = nn.BatchNorm1d(n_qubits)
        self.q_proj = nn.Linear(n_qubits, n_classes)
        self.angle_scale = nn.Parameter(torch.tensor(0.75 * math.pi))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        feats = self.resnet(images)
        q_in = self.tn_bridge(feats) * self.angle_scale
        q_out = self.q_norm(self.quantum(q_in))
        return self.q_proj(q_out)


# ── Convenience factory ──────────────────────────────────────────────────


def hybrid_tn_model(
    fc_out_features: int,
    n_qubits: int = 8,
    mps_bond_dim: int = 64,
    mps_phys_dim: int = 2,
    mps_hidden_dim: int = 128,
    vqc_layers: int = 2,
    freeze_resnet: bool = True,
    use_pennylane: bool = True,
    tn_bridge: str = "mps",
    peps_rows: Optional[int] = None,
    peps_cols: Optional[int] = None,
    peps_message_passing_steps: int = 3,
    backbone: str = "resnet18",
    backbone_weights: Optional[str] = None,
) -> HybridTNClassifier:
    """Factory for :class:`HybridTNClassifier`."""
    return HybridTNClassifier(
        n_classes=fc_out_features,
        n_qubits=n_qubits,
        mps_bond_dim=mps_bond_dim,
        mps_phys_dim=mps_phys_dim,
        bridge_hidden_dim=mps_hidden_dim,
        vqc_layers=vqc_layers,
        freeze_resnet=freeze_resnet,
        use_pennylane=use_pennylane,
        bridge_type=tn_bridge,
        peps_rows=peps_rows,
        peps_cols=peps_cols,
        peps_message_passing_steps=peps_message_passing_steps,
        backbone=backbone,
        backbone_weights=backbone_weights,
    )
