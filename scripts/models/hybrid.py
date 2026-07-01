"""Hybrid Quantum-Classical classifiers.

Architecture:
    ResNet backbone -> TN bridge (MPS / PEPS / TTN) -> PennyLane VQC -> BN -> linear head
    ResNet backbone -> linear angle map -> PennyLane VQC -> BN -> linear head
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .classifiers import _RESNET_REGISTRY, load_resnet_backbone_weights

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

    1. Split features into Q chunks; encode each into gated amplitudes.
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

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        chunks = x.view(B, self.Q, self.Fpc)
        amps = []
        for i in range(self.Q):
            h = F.gelu(self.fc1[i](self.ln[i](chunks[:, i])))
            logits = self.fc_logits[i](h) / self.temperature
            a = torch.sqrt(F.softmax(logits, dim=-1).clamp(min=self.eps))
            gate = torch.sigmoid(self.fc_gate[i](h))
            uniform = torch.full_like(a, 1.0 / math.sqrt(self.P))
            amps.append(_l2_normalize(gate * a + (1.0 - gate) * uniform, eps=self.eps))
        return torch.stack(amps, 1)

    # -- environment contraction ------------------------------------------

    def _left_envs(self, amps: torch.Tensor, cores: List[torch.Tensor]) -> List[torch.Tensor]:
        B = amps.shape[0]
        envs = [torch.ones(B, 1, device=amps.device, dtype=amps.dtype)]
        state = envs[0]
        for i in range(self.Q):
            c = torch.einsum("bp,plr->blr", amps[:, i], cores[i])
            state = torch.einsum("bl,blr->br", state, c)
            if (i + 1) % self.renorm_stride == 0:
                state = _l2_normalize(state, eps=self.eps)
            envs.append(state)
        return envs

    def _right_envs(self, amps: torch.Tensor, cores: List[torch.Tensor]) -> List[torch.Tensor]:
        B = amps.shape[0]
        envs: list = [None] * (self.Q + 1)
        state = torch.ones(B, 1, device=amps.device, dtype=amps.dtype)
        envs[self.Q] = state
        for i in reversed(range(self.Q)):
            c = torch.einsum("plr,bp->blr", cores[i], amps[:, i])
            state = torch.einsum("blr,br->bl", c, state)
            if (self.Q - i) % self.renorm_stride == 0:
                state = _l2_normalize(state, eps=self.eps)
            envs[i] = state
        return envs

    # -- observable amplitudes --------------------------------------------

    def _observable_amps(self, amps: torch.Tensor) -> List[torch.Tensor]:
        out = []
        for i in range(self.Q):
            beta = torch.sigmoid(self.beta_logit[i])
            obs = beta * self.ObsZ + (1.0 - beta) * self.ObsX
            out.append(torch.einsum("bp,pq->bq", amps[:, i], obs))
        return out

    def _expectations(
        self,
        amps: torch.Tensor,
        left: List[torch.Tensor],
        right: List[torch.Tensor],
        cores: List[torch.Tensor],
        obs_amps: List[torch.Tensor],
    ) -> torch.Tensor:
        out = []
        for i in range(self.Q):
            core = cores[i]
            a = amps[:, i]
            cc = torch.einsum("plr,bp->blr", core, a)
            denom = torch.einsum("bl,blr,br->b", left[i], cc, right[i + 1]).abs().clamp_min(self.eps)
            fc = torch.einsum("plr,bp->blr", core, obs_amps[i])
            out.append(torch.einsum("bl,blr,br->b", left[i], fc, right[i + 1]) / denom)
        return torch.stack(out, 1)

    # -- forward ----------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        amps = self._encode(x)
        cores = [core() for core in self.cores]
        left = self._left_envs(amps, cores)
        right = self._right_envs(amps, cores)
        obs_amps = self._observable_amps(amps)
        z_mps = torch.tanh(self._expectations(amps, left, right, cores, obs_amps))
        z_skip = torch.tanh(self.skip_proj(x)) * self.site_scale
        alpha = torch.sigmoid(self.mix_logit)
        return alpha * z_mps + (1 - alpha) * z_skip + self.bias


# -- PEPS core -------------------------------------------------------------


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


# -- PEPS bridge -----------------------------------------------------------


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
    N_SITES = ROWS * COLS   # 16 sites fixed by the exact contraction topology
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
            raise ValueError(
                f"PEPSBridge uses a fixed 4x4 exact-contraction topology "
                f"({self.N_SITES} sites); got n_sites={n_sites}."
            )
        if bond_dim < 1:
            raise ValueError(f"bond_dim must be >= 1, got {bond_dim}.")
        if feature_dim % self.N_SITES:
            raise ValueError(f"feature_dim must be divisible by {self.N_SITES} PEPS sites.")
        if bond_dim > self.MAX_BOND_DIM:
            raise ValueError(
                f"PEPSBridge exact contraction supports bond_dim <= {self.MAX_BOND_DIM}; "
                f"got bond_dim={bond_dim}. Increase MAX_BOND_DIM only after reviewing "
                "the contraction memory and compute cost."
            )

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

    # -- indexing ---------------------------------------------------------

    def _idx(self, r: int, c: int) -> int:
        return r * self.COLS + c

    # -- encoding ---------------------------------------------------------

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

    # -- physical contractions -------------------------------------------

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

    # -- column contraction -----------------------------------------------

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

    # -- expectation values -----------------------------------------------

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

    # -- forward ----------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        amps = self._encode(x)
        cores = [core() for core in self.cores]
        site_tensors = self._contract_physical(amps, cores)
        obs_site_tensors = self._contract_physical(torch.stack(self._observable_amps(amps), 1), cores)
        z = self._exact_expectations(site_tensors, obs_site_tensors)
        z_skip = torch.tanh(self.skip_proj(x)) * self.site_scale
        alpha = torch.sigmoid(self.mix_logit)
        return alpha * z + (1.0 - alpha) * z_skip + self.bias


# -- TTN core --------------------------------------------------------------


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


# -- TTN bridge ------------------------------------------------------------


class TTNBridge(nn.Module):
    """Hierarchical binary-tree TTN bridge with a configurable leaf layout.

    The bridge mirrors the diagrammed TTN:
    1. split features into ``n_leaves`` leaves;
    2. encode each leaf into a latent state of dimension ``bond_dim``;
    3. contract adjacent pairs bottom-up through TTN cores;
    4. stop when ``n_sites`` physical states remain and measure each state.
    """

    DEFAULT_LEAVES = 64
    VALID_LEAVES = (16, 32, 64, 128)
    VALID_OUTPUTS = (4, 8)

    def __init__(
        self,
        feature_dim: int = 512,
        n_sites: int = 8,
        bond_dim: int = 64,
        phys_dim: int = 2,
        hidden_dim: int = 128,
        n_leaves: int = DEFAULT_LEAVES,
        temperature: float = 1.0,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        if n_leaves not in self.VALID_LEAVES:
            raise ValueError(f"n_leaves must be one of {self.VALID_LEAVES}, got {n_leaves}.")
        if feature_dim % n_leaves:
            raise ValueError(f"feature_dim must be divisible by {n_leaves} TTN leaves.")
        if bond_dim < 1:
            raise ValueError(f"bond_dim must be >= 1, got {bond_dim}.")
        if n_sites not in self.VALID_OUTPUTS:
            raise ValueError(
                f"TTN final states must be one of {self.VALID_OUTPUTS}, got n_sites={n_sites}."
            )
        if n_leaves <= n_sites:
            raise ValueError(
                f"n_leaves must be greater than n_sites, got n_leaves={n_leaves}, n_sites={n_sites}."
            )

        self.Q = n_sites
        self.D = bond_dim
        self.P = phys_dim
        self.n_leaves = n_leaves
        self.hidden_dim = hidden_dim
        self.temperature = temperature
        self.eps = eps
        self.Fpl = feature_dim // self.n_leaves

        self.ln = nn.ModuleList([nn.LayerNorm(self.Fpl) for _ in range(self.n_leaves)])
        self.fc1 = nn.ModuleList([nn.Linear(self.Fpl, hidden_dim) for _ in range(self.n_leaves)])
        self.fc_logits = nn.ModuleList([nn.Linear(hidden_dim, self.D) for _ in range(self.n_leaves)])
        self.fc_gate = nn.ModuleList([nn.Linear(hidden_dim, 1) for _ in range(self.n_leaves)])

        levels = []
        child_dim = self.D
        n_children = self.n_leaves
        while n_children > self.Q:
            n_parents = n_children // 2
            parent_dim = self.P if n_parents == self.Q else self.D
            levels.append(nn.ModuleList([
                _TTNCore(child_dim, child_dim, parent_dim, eps) for _ in range(n_parents)
            ]))
            n_children = n_parents
            child_dim = parent_dim
        self.levels = nn.ModuleList(levels)

        self.beta_logit = nn.Parameter(torch.zeros(self.Q))
        self.skip_proj = nn.Linear(feature_dim, self.Q)
        self.site_scale = nn.Parameter(torch.ones(self.Q))
        self.mix_logit = nn.Parameter(torch.zeros(()))
        self.bias = nn.Parameter(torch.zeros(self.Q))

        Z, X = _generalized_ZX(self.P)
        self.register_buffer("ObsZ", Z)
        self.register_buffer("ObsX", X)

    # -- encoding ---------------------------------------------------------

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        chunks = x.view(B, self.n_leaves, self.Fpl)
        leaves = []
        for i in range(self.n_leaves):
            h = F.gelu(self.fc1[i](self.ln[i](chunks[:, i])))
            logits = self.fc_logits[i](h) / self.temperature
            amp = torch.sqrt(F.softmax(logits, dim=-1).clamp(min=self.eps))
            gate = torch.sigmoid(self.fc_gate[i](h))
            uniform = torch.full_like(amp, 1.0 / math.sqrt(self.D))
            leaves.append(_l2_normalize(gate * amp + (1.0 - gate) * uniform, eps=self.eps))
        return torch.stack(leaves, 1)

    # -- tree contraction -------------------------------------------------

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

    # -- measurement ------------------------------------------------------

    def _measure(self, states: torch.Tensor) -> torch.Tensor:
        z = torch.einsum("bqd,df,bqf->bq", states, self.ObsZ, states)
        x = torch.einsum("bqd,df,bqf->bq", states, self.ObsX, states)
        beta = torch.sigmoid(self.beta_logit).unsqueeze(0)
        return beta * z + (1.0 - beta) * x

    # -- forward ----------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        leaves = self._encode(x)
        final_states = self._bottom_up(leaves)
        z_ttn = torch.tanh(self._measure(final_states))
        z_skip = torch.tanh(self.skip_proj(x)) * self.site_scale
        alpha = torch.sigmoid(self.mix_logit)
        return alpha * z_ttn + (1.0 - alpha) * z_skip + self.bias


# -- Variational quantum layer ----------------------------------------------

class VariationalQuantumLayer(nn.Module):
    """Variational quantum circuit with data re-uploading and configurable read-out.

    Each of the ``n_layers`` blocks: 
    (1) re-uploads three features per qubit with trainable per-(layer, qubit, axis) 
    ``thetas`` and ``feature_weights`` as 
    ``RX(theta0 + w0*x0) RZ(theta1 + w1*x1) RY(theta2 + w2*x2)``,
    
    (2) applies a ring of CNOTs. 
    
    The preferred input shape is ``(..., 3 * n_qubits)``. 
    
    ``(..., n_qubits)`` is also accepted as a compatibility path and repeats each 
    scalar across the three rotation axes.

    The read-out is selected by ``readout``:

    * ``"z"``     single-qubit ``<Z_i>`` only (width ``n_qubits``).
    * ``"z+zz"``  ``<Z_i>`` plus nearest-neighbour ``<Z_i Z_j>`` correlators on
      the entangling ring (default; exposes entanglement to the head).
    * ``"z+zz_mean"`` ``<Z_i>`` plus one pooled mean of the nearest-neighbour
      ``<Z_i Z_j>`` ring correlators.
    * ``"bloch"`` the full local Bloch vector ``(<X_i>, <Y_i>, <Z_i>)``.
    * ``"full"``  Bloch vector plus ``{XX, YY, ZZ}`` ring correlators.

    Single-qubit X/Y are partly redundant with the trainable final rotations;
    the ring XX/YY correlators are not, so they probe independent entanglement
    channels -- at the cost of a wider (and, under adjoint, proportionally more
    expensive) read-out. The flat output width is ``readout_dim``.
    """

    VALID_READOUTS = ("z", "z+zz", "z+zz_mean", "bloch", "full")

    def __init__(self, n_qubits: int, n_layers: int, readout: str = "z+zz") -> None:
        super().__init__()
        readout = readout.lower().strip()
        if readout not in self.VALID_READOUTS:
            raise ValueError(f"readout must be one of {self.VALID_READOUTS}, got '{readout}'.")
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.readout = readout

        # Nearest-neighbour pairs mirror the entangling ring built below.
        pairs = []
        if n_qubits > 1:
            pairs = [(i, i + 1) for i in range(n_qubits - 1)]
            if n_qubits > 2:
                pairs.append((n_qubits - 1, 0))

        # Read-out observables: each term is a tuple of (Pauli, wire) factors.
        # Single-qubit terms first, then correlators, so ``"z+zz"`` keeps its
        # original [<Z_i>..., <Z_iZ_j>...] ordering.
        single_bases = {"z": "Z", "z+zz": "Z", "z+zz_mean": "Z", "bloch": "XYZ", "full": "XYZ"}[readout]
        corr_bases = {"z": "", "z+zz": "Z", "z+zz_mean": "", "bloch": "", "full": "XYZ"}[readout]
        spec = []
        for p in single_bases:
            spec += [((p, i),) for i in range(n_qubits)]
        for p in corr_bases:
            spec += [((p, a), (p, b)) for a, b in pairs]
        if readout == "z+zz_mean" and pairs:
            spec.append(("mean_corr", "Z", tuple(pairs)))

        self.corr_pairs = pairs if (corr_bases or readout == "z+zz_mean") else []
        self.readout_spec = spec
        self.readout_dim = len(spec)

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

        pauli = {"X": qml.PauliX, "Y": qml.PauliY, "Z": qml.PauliZ}

        def _pauli_product(term):
            (p0, w0), rest = term[0], term[1:]
            op = pauli[p0](w0)
            for p, w in rest:
                op = op @ pauli[p](w)
            return op

        def _observable(term):
            if term and term[0] == "mean_corr":
                _, basis, corr_pairs = term
                coeff = 1.0 / len(corr_pairs)
                ops = [_pauli_product(((basis, a), (basis, b))) for a, b in corr_pairs]
                return qml.Hamiltonian([coeff] * len(ops), ops)
            return _pauli_product(term)

        @qml.qnode(dev, interface="torch", diff_method=diff)
        def circuit(inputs, thetas, feature_weights):
            if inputs.shape[-1] == 3 * n_qubits:
                features = inputs.reshape(*inputs.shape[:-1], n_qubits, 3)
            elif inputs.shape[-1] == n_qubits:
                features = inputs.unsqueeze(-1).expand(*inputs.shape, 3)
            else:
                raise ValueError(
                    f"Expected last input dimension {3 * n_qubits} or "
                    f"{n_qubits}, got {inputs.shape[-1]}."
                )

            for layer in range(n_layers):
                # Data re-uploading with trainable per-layer encoding weights.
                for i in range(n_qubits):
                    qml.RX(
                        thetas[layer, i, 0] + feature_weights[layer, i, 0] * features[..., i, 0],
                        wires=i,
                    )
                    qml.RZ(
                        thetas[layer, i, 1] + feature_weights[layer, i, 1] * features[..., i, 1],
                        wires=i,
                    )
                    qml.RY(
                        thetas[layer, i, 2] + feature_weights[layer, i, 2] * features[..., i, 2],
                        wires=i,
                    )
                
                # Ring entanglement.
                if n_qubits > 1:
                    for i in range(n_qubits - 1):
                        qml.CNOT(wires=[i, i + 1])
                    if n_qubits > 2:
                        qml.CNOT(wires=[n_qubits - 1, 0])
            return [qml.expval(_observable(t)) for t in spec]

        weight_shapes = {
            "thetas": (n_layers, n_qubits, 3),
            "feature_weights": (n_layers, n_qubits, 3),
        }
        self.qlayer = qml.qnn.TorchLayer(circuit, weight_shapes)
        # Start re-uploading at unit feature weights so the initial encoding
        # scale matches the separately learnable ``angle_scale`` in the classifier.
        # Small-angle variational init keeps the circuit near the identity,
        # avoiding the flat (barren-plateau-prone) regions reached by the
        # TorchLayer default U(0, 2*pi) init.
        with torch.no_grad():
            self.qlayer.feature_weights.fill_(1.0)
            nn.init.normal_(self.qlayer.thetas, mean=0.0, std=0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] not in (self.n_qubits, 3 * self.n_qubits):
            raise ValueError(
                f"Expected last input dimension {self.n_qubits} or "
                f"{3 * self.n_qubits}, got {x.shape[-1]}."
            )
        return self.qlayer(x)


# -- ResNet feature extractor ---------------------------------------------

_FEAT_DIM = {"resnet18": 512, "resnet34": 512, "resnet50": 2048}


class _ResNetFeatureExtractor(nn.Module):
    """ResNet backbone with the final FC layer removed.

    Args:
        backbone: One of ``resnet18``/``resnet34``/``resnet50``.
        weights_path: Optional custom pretrained-weight source. It is valid
            only when *pretrained* is ``True``.
        pretrained: When ``True``, load either *weights_path* or ImageNet
            torchvision weights. When ``False``, the backbone is randomly
            initialised.
        freeze_mode: ``none`` trains the full ResNet extractor, ``partial``
            freezes everything except ``layer4``, ``layer34`` freezes everything
            except ``layer3`` + ``layer4``, and ``full`` freezes the complete
            ResNet extractor.
    """

    def __init__(
        self,
        backbone: str = "resnet18",
        weights_path: Optional[str] = None,
        pretrained: bool = True,
        freeze_mode: str = "none",
    ) -> None:
        super().__init__()
        backbone = backbone.lower().strip()
        if backbone not in _FEAT_DIM:
            raise ValueError(f"Unknown backbone '{backbone}'. Choose from {list(_FEAT_DIM)}.")
        if weights_path and not pretrained:
            raise ValueError("weights_path requires pretrained=True.")
        if freeze_mode != "none" and not pretrained:
            raise ValueError(
                "ResNet freeze modes partial/full require pretrained=True."
            )
        self.feature_dim = _FEAT_DIM[backbone]

        build_fn, default_w = _RESNET_REGISTRY[backbone]

        if pretrained and weights_path:
            m = build_fn(weights=None)
            info = load_resnet_backbone_weights(m, weights_path)
            print(
                f"[Backbone] Loaded custom weights ({backbone}) from {weights_path} "
                f"({info['loaded_keys']} tensors)"
            )
        elif pretrained:
            m = build_fn(weights=default_w)
            print(f"[Backbone] ImageNet-pretrained {backbone} ({self.feature_dim}-d)")
        else:
            m = build_fn(weights=None)
            print(f"[Backbone] Randomly-initialised {backbone} ({self.feature_dim}-d)")

        self.backbone = nn.Sequential(*list(m.children())[:-1])

        if freeze_mode in ("partial", "layer34", "full"):
            for p in self.backbone.parameters():
                p.requires_grad = False
            if freeze_mode in ("partial", "layer34"):
                for p in m.layer4.parameters():
                    p.requires_grad = True
            if freeze_mode == "layer34":
                for p in m.layer3.parameters():
                    p.requires_grad = True

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
        vqc_readout: str = "z+zz",
        vqc_dropout: float = 0.0,
        resnet_freeze_mode: str = "none",
        bridge_type: str = "mps",
        backbone: str = "resnet18",
        backbone_weights: Optional[str] = None,
        pretrained: bool = True,
        ttn_leaves: int = TTNBridge.DEFAULT_LEAVES,
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
                f"TTN final states must be one of {TTNBridge.VALID_OUTPUTS}; "
                f"got n_qubits={n_qubits}."
            )
        if bridge_type == "ttn" and ttn_leaves not in TTNBridge.VALID_LEAVES:
            raise ValueError(
                f"TTN leaves must be one of {TTNBridge.VALID_LEAVES}; "
                f"got ttn_leaves={ttn_leaves}."
            )
        if bridge_type == "ttn" and ttn_leaves <= n_qubits:
            raise ValueError(
                f"TTN leaves must be greater than final states; "
                f"got ttn_leaves={ttn_leaves}, n_qubits={n_qubits}."
            )
        if not 0.0 <= vqc_dropout < 1.0:
            raise ValueError(f"vqc_dropout must be in [0, 1), got {vqc_dropout}.")

        self.n_classes = n_classes
        self.n_qubits = n_qubits

        self.resnet = _ResNetFeatureExtractor(
            backbone=backbone,
            weights_path=backbone_weights,
            pretrained=pretrained,
            freeze_mode=resnet_freeze_mode,
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
            self.tn_bridge = TTNBridge(
                feat_dim,
                n_qubits,
                tn_bond_dim,
                tn_phys_dim,
                bridge_hidden_dim,
                n_leaves=ttn_leaves,
            )
        else:
            raise ValueError(f"Unknown bridge_type '{bridge_type}'.")

        self.quantum = VariationalQuantumLayer(n_qubits, vqc_layers, readout=vqc_readout)

        # The VQC read-out width depends on ``vqc_readout`` (single-qubit and/or
        # correlator observables); size the head to match.
        readout_dim = self.quantum.readout_dim
        self.q_norm = nn.BatchNorm1d(readout_dim)
        self.q_dropout = nn.Dropout(p=vqc_dropout) if vqc_dropout > 0.0 else nn.Identity()
        self.q_proj = nn.Linear(readout_dim, n_classes)
        self.angle_scale = nn.Parameter(torch.tensor(0.75 * math.pi))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        feats = self.resnet(images)
        q_in = self.tn_bridge(feats) * self.angle_scale
        q_raw = self.quantum(q_in)
        if q_raw.device != q_in.device:
            q_raw = q_raw.to(q_in.device)
        q_out = self.q_dropout(self.q_norm(q_raw))
        return self.q_proj(q_out)


_VQC_ANGLE_HIDDEN_DIM = 256
_VQC_ANGLE_DROPOUT = 0.2


class SimpleHybridVQCClassifier(nn.Module):
    """ResNet -> MLP angle map -> VQC -> BN -> linear.

    Minimal hybrid baseline without a tensor-network bridge. The
    ResNet feature vector is normalised, projected to three input features per
    qubit, bounded with ``tanh``, and passed to the same VQC used by
    ``HybridTNClassifier``. Each VQC layer then re-uploads the same per-qubit
    feature triplet with independent trainable encoding weights and offsets.
    The small adapter MLP before the quantum bottleneck is intentionally fixed
    while this variant is being evaluated.

    Input:  ``(B, 3, H, W)``  Output: ``(B, n_classes)``
    """

    def __init__(
        self,
        n_classes: int,
        n_qubits: int = 8,
        vqc_layers: int = 2,
        vqc_readout: str = "z+zz",
        vqc_dropout: float = 0.0,
        resnet_freeze_mode: str = "none",
        backbone: str = "resnet18",
        backbone_weights: Optional[str] = None,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        if not 0.0 <= vqc_dropout < 1.0:
            raise ValueError(f"vqc_dropout must be in [0, 1), got {vqc_dropout}.")

        self.n_classes = n_classes
        self.n_qubits = n_qubits

        self.resnet = _ResNetFeatureExtractor(
            backbone=backbone,
            weights_path=backbone_weights,
            pretrained=pretrained,
            freeze_mode=resnet_freeze_mode,
        )
        feat_dim = self.resnet.feature_dim

        self.feature_norm = nn.LayerNorm(feat_dim)
        self.angle_map = nn.Sequential(
            nn.Linear(feat_dim, _VQC_ANGLE_HIDDEN_DIM),
            nn.GELU(),
            nn.Dropout(p=_VQC_ANGLE_DROPOUT),
            nn.Linear(_VQC_ANGLE_HIDDEN_DIM, 3 * n_qubits),
        )
        self.quantum = VariationalQuantumLayer(n_qubits, vqc_layers, readout=vqc_readout)

        readout_dim = self.quantum.readout_dim
        self.q_norm = nn.BatchNorm1d(readout_dim)
        self.q_dropout = nn.Dropout(p=vqc_dropout) if vqc_dropout > 0.0 else nn.Identity()
        self.q_proj = nn.Linear(readout_dim, n_classes)
        self.angle_scale = nn.Parameter(torch.tensor(0.75 * math.pi))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        feats = self.feature_norm(self.resnet(images))
        q_in = torch.tanh(self.angle_map(feats)) * self.angle_scale
        q_raw = self.quantum(q_in)
        if q_raw.device != q_in.device:
            q_raw = q_raw.to(q_in.device)
        q_out = self.q_dropout(self.q_norm(q_raw))
        return self.q_proj(q_out)


# -- Convenience factory --------------------------------------------------


def hybrid_tn_model(
    fc_out_features: int,
    n_qubits: int = 8,
    tn_bond_dim: int = 64,
    tn_phys_dim: int = 2,
    tn_hidden_dim: int = 128,
    vqc_layers: int = 2,
    vqc_readout: str = "z+zz",
    vqc_dropout: float = 0.0,
    resnet_freeze_mode: str = "none",
    tn_bridge: str = "mps",
    backbone: str = "resnet18",
    backbone_weights: Optional[str] = None,
    pretrained: bool = True,
    ttn_leaves: int = TTNBridge.DEFAULT_LEAVES,
) -> HybridTNClassifier:
    """Factory for `HybridTNClassifier`."""
    return HybridTNClassifier(
        n_classes=fc_out_features,
        n_qubits=n_qubits,
        tn_bond_dim=tn_bond_dim,
        tn_phys_dim=tn_phys_dim,
        bridge_hidden_dim=tn_hidden_dim,
        vqc_layers=vqc_layers,
        vqc_readout=vqc_readout,
        vqc_dropout=vqc_dropout,
        resnet_freeze_mode=resnet_freeze_mode,
        bridge_type=tn_bridge,
        backbone=backbone,
        backbone_weights=backbone_weights,
        pretrained=pretrained,
        ttn_leaves=ttn_leaves,
    )


def hybrid_vqc_model(
    fc_out_features: int,
    n_qubits: int = 8,
    vqc_layers: int = 2,
    vqc_readout: str = "z+zz",
    vqc_dropout: float = 0.0,
    resnet_freeze_mode: str = "none",
    backbone: str = "resnet18",
    backbone_weights: Optional[str] = None,
    pretrained: bool = True,
) -> SimpleHybridVQCClassifier:
    """Factory for `SimpleHybridVQCClassifier`."""
    return SimpleHybridVQCClassifier(
        n_classes=fc_out_features,
        n_qubits=n_qubits,
        vqc_layers=vqc_layers,
        vqc_readout=vqc_readout,
        vqc_dropout=vqc_dropout,
        resnet_freeze_mode=resnet_freeze_mode,
        backbone=backbone,
        backbone_weights=backbone_weights,
        pretrained=pretrained,
    )
