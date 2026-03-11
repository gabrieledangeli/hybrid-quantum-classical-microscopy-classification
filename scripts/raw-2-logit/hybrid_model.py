# Hybrid MPS + Variational Quantum Circuit classifier for the raw-2-logit pipeline.

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _l2_normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
    return x / (x.norm(dim=dim, keepdim=True) + eps)


def _factor_grid_shape(n_sites: int, rows: Optional[int], cols: Optional[int]) -> Tuple[int, int]:
    """
    Resolve a 2D grid shape whose area equals n_sites.

    Rules:
      - If both rows/cols are provided, they must multiply to n_sites.
      - If only one is provided, infer the other and validate divisibility.
      - If none is provided, choose the most square factorization.
    """
    if n_sites < 1:
        raise ValueError(f"n_sites must be >= 1, got {n_sites}")

    if rows is not None:
        rows = int(rows)
        if rows < 1:
            raise ValueError(f"rows must be >= 1, got {rows}")
    if cols is not None:
        cols = int(cols)
        if cols < 1:
            raise ValueError(f"cols must be >= 1, got {cols}")

    if rows is not None and cols is not None:
        if rows * cols != n_sites:
            raise ValueError(
                f"Invalid PEPS grid: rows*cols={rows*cols} but n_sites={n_sites}."
            )
        return rows, cols

    if rows is not None:
        if n_sites % rows != 0:
            raise ValueError(
                f"Cannot infer cols: n_sites={n_sites} is not divisible by rows={rows}."
            )
        return rows, n_sites // rows

    if cols is not None:
        if n_sites % cols != 0:
            raise ValueError(
                f"Cannot infer rows: n_sites={n_sites} is not divisible by cols={cols}."
            )
        return n_sites // cols, cols

    # Auto: most square factorization.
    rows_auto = int(math.sqrt(n_sites))
    while rows_auto > 1 and n_sites % rows_auto != 0:
        rows_auto -= 1
    cols_auto = n_sites // rows_auto
    return rows_auto, cols_auto


def generalized_ZX(phys_dim: int, device=None, dtype=None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generalized Z (clock) and X (shift) operators for qudits.
    For qubits (phys_dim=2) returns standard Pauli Z and X.
    For higher dimensions uses real approximation (cosine phases).
    """
    if phys_dim == 2:
        Z = torch.tensor([[1.0, 0.0],
                          [0.0, -1.0]], device=device, dtype=dtype)
        X = torch.tensor([[0.0, 1.0],
                          [1.0, 0.0]], device=device, dtype=dtype)
        return Z, X

    ks = torch.arange(phys_dim, device=device, dtype=dtype)
    phases = torch.cos(2 * math.pi * ks / phys_dim)
    Z = torch.diag(phases)

    X = torch.zeros((phys_dim, phys_dim), device=device, dtype=dtype)
    for k in range(phys_dim):
        X[k, (k + 1) % phys_dim] = 1.0
    return Z, X


# ---------------------------------------------------------------------------
# MPS Core
# ---------------------------------------------------------------------------

class SpectralCore(nn.Module):
    """Single MPS tensor core with Frobenius-normalized base and learnable scale."""

    def __init__(self, phys_dim: int, left_dim: int, right_dim: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.base = nn.Parameter(torch.randn(phys_dim, left_dim, right_dim) * 0.02)
        self.log_scale = nn.Parameter(torch.zeros(()))

    def forward(self) -> torch.Tensor:
        norm = self.base.reshape(-1).norm(p=2) + self.eps
        return torch.exp(self.log_scale) * (self.base / norm)


# ---------------------------------------------------------------------------
# MPS Bridge
# ---------------------------------------------------------------------------

class CompetitiveMPSBridgeV2(nn.Module):
    """
    Maps a classical feature vector to Q expectation values via a
    Matrix Product State tensor network.

    Steps:
      1. Split features into Q chunks, encode each into qudit amplitudes + gate
      2. Contract through MPS cores (left-to-right and right-to-left environments)
      3. Compute per-site Z/X expectations, blend with learnable beta
      4. Skip connection + learnable mixing alpha + bias
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
    ):
        super().__init__()
        assert feature_dim % n_sites == 0, "feature_dim must be divisible by n_sites."
        self.feature_dim = feature_dim
        self.Q = n_sites
        self.D = bond_dim
        self.P = phys_dim
        self.hidden_dim = hidden_dim
        self.Fpc = feature_dim // n_sites
        self.temperature = temperature
        self.renorm_stride = max(1, int(renorm_stride))
        self.eps = eps

        # Per-site encoders
        self.ln = nn.ModuleList([nn.LayerNorm(self.Fpc) for _ in range(self.Q)])
        self.fc1 = nn.ModuleList([nn.Linear(self.Fpc, hidden_dim) for _ in range(self.Q)])
        self.fc_logits = nn.ModuleList([nn.Linear(hidden_dim, self.P) for _ in range(self.Q)])
        self.fc_gate = nn.ModuleList([nn.Linear(hidden_dim, 1) for _ in range(self.Q)])

        # MPS cores: boundary dims 1-D-...-D-1
        cores: List[SpectralCore] = []
        for i in range(self.Q):
            left_dim = 1 if i == 0 else self.D
            right_dim = 1 if i == self.Q - 1 else self.D
            cores.append(SpectralCore(phys_dim=self.P, left_dim=left_dim, right_dim=right_dim, eps=eps))
        self.cores = nn.ModuleList(cores)

        # Learnable beta per site (blend Z/X)
        self.beta_logit = nn.Parameter(torch.zeros(self.Q))

        # Skip + mixing
        self.skip_proj = nn.Linear(feature_dim, self.Q)
        self.site_scale = nn.Parameter(torch.ones(self.Q))
        self.mix_logit = nn.Parameter(torch.zeros(()))
        self.bias = nn.Parameter(torch.zeros(self.Q))

        # Observables
        Z, X = generalized_ZX(self.P, device=None, dtype=None)
        self.register_buffer("ObsZ", Z)
        self.register_buffer("ObsX", X)

    # ----- internal helpers -----

    def _amps(self, input_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B = input_features.shape[0]
        chunks = input_features.view(B, self.Q, self.Fpc)

        amps = []
        gates = []
        for i in range(self.Q):
            x = chunks[:, i, :]
            x = self.ln[i](x)
            h = F.gelu(self.fc1[i](x))
            logits = self.fc_logits[i](h) / self.temperature
            probs = F.softmax(logits, dim=-1)
            a = torch.sqrt(torch.clamp(probs, min=self.eps))
            a = _l2_normalize(a, dim=-1, eps=self.eps)

            g = torch.sigmoid(self.fc_gate[i](h)).squeeze(-1)
            amps.append(a)
            gates.append(g)

        return torch.stack(amps, dim=1), torch.stack(gates, dim=1)

    def _left_envs(self, amplitudes: torch.Tensor) -> List[torch.Tensor]:
        B = amplitudes.shape[0]
        left_envs = []
        state = torch.ones((B, 1), device=amplitudes.device, dtype=amplitudes.dtype)
        left_envs.append(state)

        for i in range(self.Q):
            core_i = self.cores[i]()
            amp_i = amplitudes[:, i, :]
            contr = torch.einsum("bp,plr->blr", amp_i, core_i)
            state = torch.einsum("bl,blr->br", state, contr)
            if (i + 1) % self.renorm_stride == 0:
                state = _l2_normalize(state, dim=-1, eps=self.eps)
            left_envs.append(state)

        return left_envs

    def _right_envs(self, amplitudes: torch.Tensor) -> List[torch.Tensor]:
        B = amplitudes.shape[0]
        right_envs = [None] * (self.Q + 1)
        state = torch.ones((B, 1), device=amplitudes.device, dtype=amplitudes.dtype)
        right_envs[self.Q] = state

        for i in reversed(range(self.Q)):
            core_i = self.cores[i]()
            amp_i = amplitudes[:, i, :]
            contr = torch.einsum("plr,bp->blr", core_i, amp_i)
            state = torch.einsum("blr,br->bl", contr, state)
            if (self.Q - i) % self.renorm_stride == 0:
                state = _l2_normalize(state, dim=-1, eps=self.eps)
            right_envs[i] = state

        return right_envs

    def _site_expectations(
        self,
        amplitudes: torch.Tensor,
        left_envs: List[torch.Tensor],
        right_envs: List[torch.Tensor],
    ) -> torch.Tensor:
        z_list = []
        x_list = []
        for i in range(self.Q):
            core_i = self.cores[i]()

            core_Z = torch.einsum("pq,qlr->plr", self.ObsZ, core_i)
            core_X = torch.einsum("pq,qlr->plr", self.ObsX, core_i)

            amp_i = amplitudes[:, i, :]

            full_Z = torch.einsum("plr,bp->blr", core_Z, amp_i)
            full_X = torch.einsum("plr,bp->blr", core_X, amp_i)

            z = torch.einsum("bl,blr,br->b", left_envs[i], full_Z, right_envs[i + 1])
            x = torch.einsum("bl,blr,br->b", left_envs[i], full_X, right_envs[i + 1])

            z_list.append(z)
            x_list.append(x)

        Zexp = torch.stack(z_list, dim=1)
        Xexp = torch.stack(x_list, dim=1)

        beta = torch.sigmoid(self.beta_logit).unsqueeze(0)
        return beta * Zexp + (1.0 - beta) * Xexp

    # ----- forward -----

    def forward(self, input_features: torch.Tensor) -> torch.Tensor:
        amplitudes, gates = self._amps(input_features)
        amplitudes = amplitudes * gates.unsqueeze(-1)

        left_envs = self._left_envs(amplitudes)
        right_envs = self._right_envs(amplitudes)

        z_mps = self._site_expectations(amplitudes, left_envs, right_envs)

        z_skip = torch.tanh(self.skip_proj(input_features)) * self.site_scale

        alpha = torch.sigmoid(self.mix_logit)
        return alpha * z_mps + (1.0 - alpha) * z_skip + self.bias


# ---------------------------------------------------------------------------
# PEPS Bridge (factorized mean-field contraction)
# ---------------------------------------------------------------------------

class CompetitivePEPSBridgeV2(nn.Module):
    """
    Maps a classical feature vector to Q outputs through a 2D PEPS-like lattice.

    Practical note:
      Exact PEPS contraction scales poorly in 2D. This bridge uses a factorized
      site parameterization + iterative learnable message passing to keep training
      feasible while preserving 2D neighborhood structure (up/down/left/right).

    Architecture (vs original):
      - Directional projections operate on hidden_dim (not phys_dim P).
      - Message passing uses learnable linear transforms per step.
      - Environment modulates a per-site density matrix (not just scalar scaling).
      - Border sites use adaptive learned weighting instead of uniform 0.25.
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
    ):
        super().__init__()
        assert feature_dim % n_sites == 0, "feature_dim must be divisible by n_sites."

        self.feature_dim = feature_dim
        self.Q = n_sites
        self.D = bond_dim
        self.P = phys_dim
        self.hidden_dim = hidden_dim
        self.Fps = feature_dim // n_sites
        self.temperature = temperature
        self.n_message_passing_steps = max(1, int(n_message_passing_steps))
        self.eps = eps

        self.rows, self.cols = _factor_grid_shape(n_sites, grid_rows, grid_cols)

        # Per-site encoders: chunk -> hidden (H) -> amplitudes (P) + gate.
        self.ln = nn.ModuleList([nn.LayerNorm(self.Fps) for _ in range(self.Q)])
        self.fc1 = nn.ModuleList([nn.Linear(self.Fps, hidden_dim) for _ in range(self.Q)])
        self.fc_logits = nn.ModuleList([nn.Linear(hidden_dim, self.P) for _ in range(self.Q)])
        self.fc_gate = nn.ModuleList([nn.Linear(hidden_dim, 1) for _ in range(self.Q)])

        # Directional bond dimensions (OBC: boundary bonds have dim 1).
        self.up_dims: List[int] = []
        self.down_dims: List[int] = []
        self.left_dims: List[int] = []
        self.right_dims: List[int] = []
        # Number of real neighbors per site (for adaptive env weighting).
        self._n_neighbors: List[int] = []
        for i in range(self.Q):
            r, c = divmod(i, self.cols)
            self.up_dims.append(1 if r == 0 else self.D)
            self.down_dims.append(1 if r == self.rows - 1 else self.D)
            self.left_dims.append(1 if c == 0 else self.D)
            self.right_dims.append(1 if c == self.cols - 1 else self.D)
            n_nbr = 4
            if r == 0:
                n_nbr -= 1
            if r == self.rows - 1:
                n_nbr -= 1
            if c == 0:
                n_nbr -= 1
            if c == self.cols - 1:
                n_nbr -= 1
            self._n_neighbors.append(n_nbr)

        # [FIX 1] Directional projections from hidden_dim (not P).
        # This gives the message passing access to rich feature information.
        self.up_proj = nn.ModuleList([nn.Linear(hidden_dim, self.up_dims[i], bias=False) for i in range(self.Q)])
        self.down_proj = nn.ModuleList([nn.Linear(hidden_dim, self.down_dims[i], bias=False) for i in range(self.Q)])
        self.left_proj = nn.ModuleList([nn.Linear(hidden_dim, self.left_dims[i], bias=False) for i in range(self.Q)])
        self.right_proj = nn.ModuleList([nn.Linear(hidden_dim, self.right_dims[i], bias=False) for i in range(self.Q)])

        # [FIX 5] Learnable message transforms per step.
        # Each step applies a linear transform to messages before passing them.
        self.msg_transforms = nn.ModuleList()
        for _ in range(self.n_message_passing_steps):
            step_transforms = nn.ModuleList([
                nn.Linear(self.D, self.D, bias=False) for _ in range(4)  # up, down, left, right
            ])
            # Initialize close to identity for stable start.
            for lt in step_transforms:
                nn.init.eye_(lt.weight)
                lt.weight.data.mul_(0.9)
                lt.weight.data.add_(0.1 * torch.randn_like(lt.weight.data) * 0.02)
            self.msg_transforms.append(step_transforms)

        # [FIX 4] Learnable per-site direction weights (4 logits per site).
        # Replaces uniform 0.25 average; softmax normalizes adaptively.
        self.env_weight_logits = nn.Parameter(torch.zeros(self.Q, 4))

        # [FIX 3] Environment modulates density matrix via per-site rotation.
        # env_proj maps aggregated env scalar (4-d vector) to a P×P mixing matrix.
        self.env_proj = nn.ModuleList([nn.Linear(4, self.P * self.P) for _ in range(self.Q)])

        # Learnable beta per site (blend Z/X), skip + mixing.
        self.beta_logit = nn.Parameter(torch.zeros(self.Q))
        self.skip_proj = nn.Linear(feature_dim, self.Q)
        self.site_scale = nn.Parameter(torch.ones(self.Q))
        self.mix_logit = nn.Parameter(torch.zeros(()))
        self.bias = nn.Parameter(torch.zeros(self.Q))

        Z, X = generalized_ZX(self.P, device=None, dtype=None)
        self.register_buffer("ObsZ", Z)
        self.register_buffer("ObsX", X)

    def _site_index(self, row: int, col: int) -> int:
        return row * self.cols + col

    def _unit_message(self,
                      batch_size: int,
                      dim: int,
                      device: torch.device,
                      dtype: torch.dtype) -> torch.Tensor:
        msg = torch.ones((batch_size, dim), device=device, dtype=dtype)
        return _l2_normalize(msg, dim=-1, eps=self.eps)

    @staticmethod
    def _dot(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return (a * b).sum(dim=-1, keepdim=True)

    def _amps_and_hiddens(self, input_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        """Encode features into amplitudes, gates, AND per-site hidden vectors."""
        B = input_features.shape[0]
        chunks = input_features.view(B, self.Q, self.Fps)

        amps = []
        gates = []
        hiddens: List[torch.Tensor] = []
        for i in range(self.Q):
            x = chunks[:, i, :]
            x = self.ln[i](x)
            h = F.gelu(self.fc1[i](x))                     # [B, hidden_dim]
            hiddens.append(h)

            logits = self.fc_logits[i](h) / self.temperature
            probs = F.softmax(logits, dim=-1)
            a = torch.sqrt(torch.clamp(probs, min=self.eps))
            a = _l2_normalize(a, dim=-1, eps=self.eps)

            g = torch.sigmoid(self.fc_gate[i](h)).squeeze(-1)
            amps.append(a)
            gates.append(g)

        return torch.stack(amps, dim=1), torch.stack(gates, dim=1), hiddens

    def _build_site_factors(
        self,
        amplitudes: torch.Tensor,
        gates: torch.Tensor,
        hiddens: List[torch.Tensor],
    ) -> Tuple[dict, List[torch.Tensor]]:
        """Build directional bond factors from hidden vectors (not amplitudes)."""
        factors = {"up": [], "down": [], "left": [], "right": []}
        amps_gated: List[torch.Tensor] = []

        for i in range(self.Q):
            a = amplitudes[:, i, :]
            g = torch.clamp(gates[:, i:i + 1], min=self.eps)
            a = _l2_normalize(a * g, dim=-1, eps=self.eps)
            amps_gated.append(a)

            # [FIX 1] Project from hidden_dim, not from P-dim amplitudes.
            h = hiddens[i]  # [B, hidden_dim]
            up = torch.tanh(self.up_proj[i](h))
            down = torch.tanh(self.down_proj[i](h))
            left = torch.tanh(self.left_proj[i](h))
            right = torch.tanh(self.right_proj[i](h))

            factors["up"].append(_l2_normalize(up, dim=-1, eps=self.eps))
            factors["down"].append(_l2_normalize(down, dim=-1, eps=self.eps))
            factors["left"].append(_l2_normalize(left, dim=-1, eps=self.eps))
            factors["right"].append(_l2_normalize(right, dim=-1, eps=self.eps))

        return factors, amps_gated

    def _apply_msg_transform(self, msg: torch.Tensor, transform: nn.Linear) -> torch.Tensor:
        """Apply learnable transform to a message, handling boundary dim=1 gracefully."""
        if msg.shape[-1] == 1:
            # Boundary messages are scalar — no transform needed.
            return msg
        return _l2_normalize(transform(msg), dim=-1, eps=self.eps)

    def _run_message_passing(
        self,
        factors: dict,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        incoming_up = [self._unit_message(batch_size, self.up_dims[i], device, dtype) for i in range(self.Q)]
        incoming_down = [self._unit_message(batch_size, self.down_dims[i], device, dtype) for i in range(self.Q)]
        incoming_left = [self._unit_message(batch_size, self.left_dims[i], device, dtype) for i in range(self.Q)]
        incoming_right = [self._unit_message(batch_size, self.right_dims[i], device, dtype) for i in range(self.Q)]

        for step in range(self.n_message_passing_steps):
            # [FIX 5] Retrieve learnable transforms for this step.
            step_tf = self.msg_transforms[step]  # [up_tf, down_tf, left_tf, right_tf]

            out_up = []
            out_down = []
            out_left = []
            out_right = []

            for i in range(self.Q):
                u = factors["up"][i]
                d = factors["down"][i]
                l = factors["left"][i]
                r = factors["right"][i]

                in_u = incoming_up[i]
                in_d = incoming_down[i]
                in_l = incoming_left[i]
                in_r = incoming_right[i]

                up_scale = self._dot(d, in_d) * self._dot(l, in_l) * self._dot(r, in_r)
                down_scale = self._dot(u, in_u) * self._dot(l, in_l) * self._dot(r, in_r)
                left_scale = self._dot(u, in_u) * self._dot(d, in_d) * self._dot(r, in_r)
                right_scale = self._dot(u, in_u) * self._dot(d, in_d) * self._dot(l, in_l)

                out_up.append(_l2_normalize(u * up_scale, dim=-1, eps=self.eps))
                out_down.append(_l2_normalize(d * down_scale, dim=-1, eps=self.eps))
                out_left.append(_l2_normalize(l * left_scale, dim=-1, eps=self.eps))
                out_right.append(_l2_normalize(r * right_scale, dim=-1, eps=self.eps))

            next_up: List[torch.Tensor] = [torch.empty(0)] * self.Q
            next_down: List[torch.Tensor] = [torch.empty(0)] * self.Q
            next_left: List[torch.Tensor] = [torch.empty(0)] * self.Q
            next_right: List[torch.Tensor] = [torch.empty(0)] * self.Q

            for rr in range(self.rows):
                for cc in range(self.cols):
                    i = self._site_index(rr, cc)

                    if rr > 0:
                        # [FIX 2] Assert dimension compatibility between neighbors.
                        j = self._site_index(rr - 1, cc)
                        assert self.down_dims[j] == self.up_dims[i], (
                            f"Bond mismatch: site {j} down_dim={self.down_dims[j]} "
                            f"!= site {i} up_dim={self.up_dims[i]}"
                        )
                        next_up[i] = self._apply_msg_transform(out_down[j], step_tf[0])
                    else:
                        next_up[i] = self._unit_message(batch_size, self.up_dims[i], device, dtype)

                    if rr < self.rows - 1:
                        j = self._site_index(rr + 1, cc)
                        assert self.up_dims[j] == self.down_dims[i], (
                            f"Bond mismatch: site {j} up_dim={self.up_dims[j]} "
                            f"!= site {i} down_dim={self.down_dims[i]}"
                        )
                        next_down[i] = self._apply_msg_transform(out_up[j], step_tf[1])
                    else:
                        next_down[i] = self._unit_message(batch_size, self.down_dims[i], device, dtype)

                    if cc > 0:
                        j = self._site_index(rr, cc - 1)
                        assert self.right_dims[j] == self.left_dims[i], (
                            f"Bond mismatch: site {j} right_dim={self.right_dims[j]} "
                            f"!= site {i} left_dim={self.left_dims[i]}"
                        )
                        next_left[i] = self._apply_msg_transform(out_right[j], step_tf[2])
                    else:
                        next_left[i] = self._unit_message(batch_size, self.left_dims[i], device, dtype)

                    if cc < self.cols - 1:
                        j = self._site_index(rr, cc + 1)
                        assert self.left_dims[j] == self.right_dims[i], (
                            f"Bond mismatch: site {j} left_dim={self.left_dims[j]} "
                            f"!= site {i} right_dim={self.right_dims[i]}"
                        )
                        next_right[i] = self._apply_msg_transform(out_left[j], step_tf[3])
                    else:
                        next_right[i] = self._unit_message(batch_size, self.right_dims[i], device, dtype)

            incoming_up, incoming_down, incoming_left, incoming_right = (
                next_up, next_down, next_left, next_right
            )

        return incoming_up, incoming_down, incoming_left, incoming_right

    def forward(self, input_features: torch.Tensor) -> torch.Tensor:
        amplitudes, gates, hiddens = self._amps_and_hiddens(input_features)
        factors, amps_gated = self._build_site_factors(amplitudes, gates, hiddens)

        B = input_features.shape[0]
        incoming_up, incoming_down, incoming_left, incoming_right = self._run_message_passing(
            factors=factors,
            batch_size=B,
            device=input_features.device,
            dtype=input_features.dtype,
        )

        peps_outputs = []
        for i in range(self.Q):
            a = amps_gated[i]  # [B, P]

            # [FIX 4] Adaptive per-direction weighting (learned, not uniform 0.25).
            dir_scores = torch.stack([
                self._dot(factors["up"][i], incoming_up[i]).squeeze(-1),
                self._dot(factors["down"][i], incoming_down[i]).squeeze(-1),
                self._dot(factors["left"][i], incoming_left[i]).squeeze(-1),
                self._dot(factors["right"][i], incoming_right[i]).squeeze(-1),
            ], dim=-1)  # [B, 4]

            # Learnable per-site logits bias the softmax toward real neighbors.
            dir_weights = F.softmax(self.env_weight_logits[i].unsqueeze(0) + dir_scores, dim=-1)  # [B, 4]
            env_vec = (dir_weights * dir_scores)  # [B, 4]

            # [FIX 3] Environment modulates density matrix, not just a scalar.
            # env_proj maps the 4-d aggregated env signal into a P×P matrix that
            # rotates/mixes the observable, giving the 2D structure real influence
            # on *which* observable combination is measured.
            env_mat = self.env_proj[i](env_vec).view(B, self.P, self.P)  # [B, P, P]
            # Symmetrize to keep it Hermitian-like (real symmetric).
            env_mat = 0.5 * (env_mat + env_mat.transpose(-1, -2))
            # Soft identity residual: start close to identity for stable training.
            env_mat = env_mat + torch.eye(self.P, device=env_mat.device, dtype=env_mat.dtype).unsqueeze(0)

            # Compute modulated expectation values: <a| env_mat @ O |a>
            beta = torch.sigmoid(self.beta_logit[i])
            O = beta * self.ObsZ + (1.0 - beta) * self.ObsX  # [P, P]
            effective_O = torch.einsum("bpq,qr->bpr", env_mat, O)  # [B, P, P]
            site_val = torch.einsum("bp,bpq,bq->b", a, effective_O, a)

            peps_outputs.append(site_val)

        z_peps = torch.stack(peps_outputs, dim=1)

        z_skip = torch.tanh(self.skip_proj(input_features)) * self.site_scale
        alpha = torch.sigmoid(self.mix_logit)
        return alpha * z_peps + (1.0 - alpha) * z_skip + self.bias


# ---------------------------------------------------------------------------
# PennyLane Quantum Layer
# ---------------------------------------------------------------------------

class PennyLaneQuantumLayer(nn.Module):
    """
    Variational quantum circuit implemented via PennyLane TorchLayer.
    Circuit: RX(inputs) -> [RX/RY/RZ trainable + ring CNOT] x n_layers -> <Z> per qubit.
    """

    def __init__(self, n_qubits: int, n_layers: int):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers

        try:
            import pennylane as qml
        except ImportError:
            raise ImportError(
                "PennyLane is required for the quantum layer but is not installed. "
                "Or use --no_pennylane to fall back to a classical MLP."
            )

        # Try fast backends in order: lightning.gpu > lightning.qubit > default.qubit
        dev = None
        diff_method = None
        for backend, dm in [
            ("lightning.gpu", "adjoint"),
            ("lightning.qubit", "adjoint"),
        ]:
            try:
                dev = qml.device(backend, wires=n_qubits)
                diff_method = dm
                print(f"[Quantum] Using {backend} with {dm} differentiation.")
                break
            except Exception:
                continue

        if dev is None:
            dev = qml.device("default.qubit", wires=n_qubits)
            diff_method = "backprop"
            print(f"[Quantum] Falling back to default.qubit with backprop (slow). "
                  f"Install pennylane-lightning for 5-20x speedup: pip install pennylane-lightning")

        @qml.qnode(dev, interface="torch", diff_method=diff_method)
        def circuit(inputs, weights):
            for i in range(n_qubits):
                qml.RX(inputs[..., i], wires=i)

            for l in range(n_layers):
                for i in range(n_qubits):
                    qml.RX(weights[l, i, 0], wires=i)
                    qml.RY(weights[l, i, 1], wires=i)
                    qml.RZ(weights[l, i, 2], wires=i)

                for i in range(n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
                qml.CNOT(wires=[n_qubits - 1, 0])

            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        weight_shapes = {"weights": (n_layers, n_qubits, 3)}
        self.qlayer = qml.qnn.TorchLayer(circuit, weight_shapes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.qlayer(x)


# ---------------------------------------------------------------------------
# ResNet18 Feature Extractor
# ---------------------------------------------------------------------------

# Output feature dimensions per ResNet variant.
_RESNET_FEAT_DIM = {"resnet18": 512, "resnet34": 512, "resnet50": 2048}


class ResNetFeatureExtractor(nn.Module):
    """Pretrained ResNet feature extractor (FC removed).

    Supports resnet18 / resnet34 / resnet50 with either ImageNet weights
    or custom weights from a .pth file (e.g. pathology-pretrained RetCCL).

    Output dim: 512 (resnet18/34) or 2048 (resnet50).

    Args:
        arch: One of 'resnet18', 'resnet34', 'resnet50'.
        weights_path: Path to a .pth / .pt state-dict file, or None for
                      default ImageNet weights.  When loading custom weights
                      the loader performs lenient key matching (strips common
                      prefixes like 'module.', 'backbone.', 'encoder.') so
                      most public checkpoints work out of the box.
        freeze: If True, freeze all layers except layer4.
    """

    def __init__(
        self,
        arch: str = "resnet18",
        weights_path: Optional[str] = None,
        freeze: bool = True,
    ):
        super().__init__()
        arch = arch.lower().strip()
        if arch not in _RESNET_FEAT_DIM:
            raise ValueError(f"Unknown ResNet arch '{arch}'. Choose from {list(_RESNET_FEAT_DIM)}.")
        self.feature_dim = _RESNET_FEAT_DIM[arch]

        # --- build torchvision ResNet ---
        from torchvision import models as tvm
        _factory = {
            "resnet18": (tvm.resnet18, tvm.ResNet18_Weights.DEFAULT),
            "resnet34": (tvm.resnet34, tvm.ResNet34_Weights.DEFAULT),
            "resnet50": (tvm.resnet50, tvm.ResNet50_Weights.DEFAULT),
        }
        build_fn, default_weights = _factory[arch]

        if weights_path is not None:
            # Build without pretrained weights; we'll load custom ones.
            m = build_fn(weights=None)
            self._load_custom_weights(m, weights_path)
            print(f"[Backbone] Loaded custom weights for {arch} from {weights_path}")
        else:
            m = build_fn(weights=default_weights)
            print(f"[Backbone] Using ImageNet-pretrained {arch} ({self.feature_dim}-d)")

        # Remove final FC → backbone outputs (B, feat_dim, 1, 1) after avgpool.
        self.backbone = nn.Sequential(*list(m.children())[:-1])

        if freeze:
            for p in self.backbone.parameters():
                p.requires_grad = False
            # Keep only layer4 trainable for domain adaptation.
            for p in m.layer4.parameters():
                p.requires_grad = True
            trainable = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)
            total = sum(p.numel() for p in self.backbone.parameters())
            print(f"[Backbone] Frozen: {total - trainable:,} params | Trainable (layer4): {trainable:,} params")

    @staticmethod
    def _load_custom_weights(model: nn.Module, path: str) -> None:
        """Load a state-dict .pth file with lenient key matching."""
        raw = torch.load(path, map_location="cpu", weights_only=False)
        # Handle checkpoints that wrap state_dict under a key.
        if isinstance(raw, dict):
            for key in ("state_dict", "model_state_dict", "model", "teacher"):
                if key in raw:
                    raw = raw[key]
                    break
        if not isinstance(raw, dict):
            raise ValueError(f"Cannot interpret checkpoint at {path} as a state dict.")

        # Strip common prefixes from checkpoint keys.
        prefixes_to_strip = ["module.", "backbone.", "encoder.", "resnet.", "model."]
        cleaned: dict = {}
        for k, v in raw.items():
            new_k = k
            for pfx in prefixes_to_strip:
                if new_k.startswith(pfx):
                    new_k = new_k[len(pfx):]
            # Skip keys that belong to FC / classification head.
            if new_k.startswith("fc.") or new_k.startswith("head.") or new_k.startswith("classifier."):
                continue
            cleaned[new_k] = v

        missing, unexpected = model.load_state_dict(cleaned, strict=False)
        # Only warn for genuinely missing backbone keys (ignore fc.*).
        missing_backbone = [k for k in missing if not k.startswith("fc.")]
        if missing_backbone:
            print(f"[Backbone] Warning: missing keys after loading: {missing_backbone[:10]}")
        if unexpected:
            print(f"[Backbone] Warning: unexpected keys (ignored): {unexpected[:10]}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x).flatten(1)


# ---------------------------------------------------------------------------
# Hybrid Classifier (drop-in replacement for ResNet in EndToEndModel)
# ---------------------------------------------------------------------------

class HybridTNClassifier(nn.Module):
    """
    End-to-end hybrid classifier: ResNet -> TN Bridge (MPS/PEPS) -> VQC -> BN -> Linear.

    Interface matches raw-2-logit's classifier contract:
      Input:  (B, 3, H, W)  — RGB images (from processor + augmentation)
      Output: (B, n_classes) — logits for cross-entropy loss

    The backbone feature dimension is set automatically based on the
    chosen ResNet architecture (512 for resnet18/34, 2048 for resnet50).
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
    ):
        super().__init__()
        self.n_classes = n_classes
        self.n_qubits = n_qubits

        self.resnet = ResNetFeatureExtractor(
            arch=backbone,
            weights_path=backbone_weights,
            freeze=freeze_resnet,
        )
        feature_dim = self.resnet.feature_dim  # 512 or 2048

        # Ensure feature_dim is divisible by n_qubits.
        if feature_dim % n_qubits != 0:
            raise ValueError(
                f"feature_dim={feature_dim} (from {backbone}) is not divisible "
                f"by n_qubits={n_qubits}. Choose n_qubits that divides {feature_dim} "
                f"(e.g. {[q for q in (8, 16, 32, 64) if feature_dim % q == 0]})."
            )

        bridge_type = bridge_type.lower().strip()
        self.bridge_type = bridge_type
        if bridge_type == "mps":
            self.tn_bridge = CompetitiveMPSBridgeV2(
                feature_dim=feature_dim,
                n_sites=n_qubits,
                bond_dim=mps_bond_dim,
                phys_dim=mps_phys_dim,
                hidden_dim=bridge_hidden_dim,
                temperature=1.0,
                renorm_stride=2,
            )
        elif bridge_type == "peps":
            self.tn_bridge = CompetitivePEPSBridgeV2(
                feature_dim=feature_dim,
                n_sites=n_qubits,
                bond_dim=mps_bond_dim,
                phys_dim=mps_phys_dim,
                hidden_dim=bridge_hidden_dim,
                temperature=1.0,
                n_message_passing_steps=peps_message_passing_steps,
                grid_rows=peps_rows,
                grid_cols=peps_cols,
            )
        else:
            raise ValueError(f"Unknown bridge_type '{bridge_type}'. Expected one of: mps, peps.")

        if use_pennylane:
            self.quantum = PennyLaneQuantumLayer(n_qubits=n_qubits, n_layers=vqc_layers)
        else:
            self.quantum = nn.Sequential(
                nn.Linear(n_qubits, 4 * n_qubits),
                nn.Tanh(),
                nn.Linear(4 * n_qubits, n_qubits),
            )

        self.q_norm = nn.BatchNorm1d(n_qubits)
        self.q_proj = nn.Linear(n_qubits, n_classes)

        # Learnable angle scaling — initialized to 0.75π but adapts during training.
        self.angle_scale = nn.Parameter(torch.tensor(0.75 * math.pi))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        classical_features = self.resnet(images)                    # [B, feat_dim]
        quantum_inputs = self.tn_bridge(classical_features)         # [B, Q]
        quantum_inputs = quantum_inputs * self.angle_scale           # learnable scaling
        quantum_outputs = self.quantum(quantum_inputs)              # [B, Q]
        quantum_outputs = self.q_norm(quantum_outputs)
        logits = self.q_proj(quantum_outputs)                       # [B, n_classes]
        return logits
