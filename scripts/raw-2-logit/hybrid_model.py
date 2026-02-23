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
        renorm_after_gating: bool = True,
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
        self.renorm_after_gating = renorm_after_gating
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
        right_envs: List[Optional[torch.Tensor]] = [None] * (self.Q + 1)
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

        return [t for t in right_envs]  # type: ignore

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
        if self.renorm_after_gating:
            amplitudes = _l2_normalize(amplitudes, dim=-1, eps=self.eps)

        left_envs = self._left_envs(amplitudes)
        right_envs = self._right_envs(amplitudes)

        z_mps = self._site_expectations(amplitudes, left_envs, right_envs)

        z_skip = torch.tanh(self.skip_proj(input_features)) * self.site_scale

        alpha = torch.sigmoid(self.mix_logit)
        return alpha * z_mps + (1.0 - alpha) * z_skip + self.bias


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
                "Install it with: pip install pennylane\n"
                "Or use --no_pennylane to fall back to a classical MLP substitute."
            )

        dev = qml.device("default.qubit", wires=n_qubits)

        @qml.qnode(dev, interface="torch", diff_method="backprop")
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

class ResNet18FeatureExtractor(nn.Module):
    """Pretrained ResNet18 with final FC removed, outputs 512-d feature vector."""

    def __init__(self, pretrained: bool = True, freeze: bool = True):
        super().__init__()
        from torchvision.models import resnet18, ResNet18_Weights
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        m = resnet18(weights=weights)
        self.backbone = nn.Sequential(*list(m.children())[:-1])
        if freeze:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x).flatten(1)


# ---------------------------------------------------------------------------
# Hybrid Classifier (drop-in replacement for ResNet in EndToEndModel)
# ---------------------------------------------------------------------------

class HybridMPSClassifier(nn.Module):
    """
    End-to-end hybrid classifier: ResNet18 -> MPS Bridge -> VQC -> BN -> Linear.

    Interface matches raw-2-logit's classifier contract:
      Input:  (B, 3, H, W)  — RGB images (from processor + augmentation)
      Output: (B, n_classes) — logits for cross-entropy loss
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
    ):
        super().__init__()
        self.n_classes = n_classes
        self.n_qubits = n_qubits

        self.resnet = ResNet18FeatureExtractor(pretrained=True, freeze=freeze_resnet)

        self.mps_bridge = CompetitiveMPSBridgeV2(
            feature_dim=512,
            n_sites=n_qubits,
            bond_dim=mps_bond_dim,
            phys_dim=mps_phys_dim,
            hidden_dim=bridge_hidden_dim,
            temperature=1.0,
            renorm_stride=2,
            renorm_after_gating=True,
        )

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

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        classical_features = self.resnet(images)                    # [B, 512]
        quantum_inputs = self.mps_bridge(classical_features)        # [B, Q]
        quantum_inputs = quantum_inputs * (0.75 * math.pi)          # scale to angles
        quantum_outputs = self.quantum(quantum_inputs)              # [B, Q]
        quantum_outputs = self.q_norm(quantum_outputs)
        logits = self.q_proj(quantum_outputs)                       # [B, n_classes]
        return logits
