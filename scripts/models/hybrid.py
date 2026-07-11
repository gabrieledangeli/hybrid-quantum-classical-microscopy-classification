"""Hybrid Quantum-Classical classifiers.

Architecture:
    ResNet backbone -> angle map (MLP / linear) -> PennyLane VQC -> BN -> linear head
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn

from .classifiers import _RESNET_REGISTRY, load_resnet_backbone_weights


# -- Variational quantum layer ----------------------------------------------

class VariationalQuantumLayer(nn.Module):
    """Variational quantum circuit with data re-uploading and configurable read-out.

    Each of the ``n_layers`` blocks:
    (1) re-uploads three features per qubit with trainable per-(layer, qubit, axis)
    ``thetas`` and ``feature_weights`` as
    ``RX(theta0 + w0*x0) RZ(theta1 + w1*x1) RY(theta2 + w2*x2)``,

    (2) applies an entangling ring selected by ``entangler``:

    * ``"cnot"``     fixed CNOT ring (default; no extra parameters).
    * ``"crz"``      trainable ``CRZ(phi)`` ring, one angle per (layer, pair).
    * ``"ising_zz"`` trainable ``IsingZZ(phi)`` ring, one angle per (layer, pair).

    The trainable rings add ``n_layers * n_pairs`` parameters and start near the
    identity (no entanglement), letting the model learn how much interaction to
    use. Both are diagonal in the Z basis, so in the *final* layer they commute
    with Z-only observables: with ``readout`` ``"z"``/``"z+zz"``/``"z+zz_mean"``
    combine them with ``final_rotation=True`` (or a Bloch read-out), otherwise
    the last entangling ring is invisible to the measurement.

    When ``final_rotation`` is ``True``, a trainable ``Rot(a, b, g)`` is applied
    per qubit after the last entangling ring, making the measurement basis
    trainable (+``3 * n_qubits`` parameters, initialised near the identity).

    When ``incremental_reuploading`` is ``False`` (default) every layer
    re-uploads the *same* ``3 * n_qubits`` features (standard data
    re-uploading). When ``True``, layer ``l`` encodes its own fresh feature
    slice ``inputs[..., l*3*n_qubits:(l+1)*3*n_qubits]``, so each layer adds
    new information instead of re-encoding the old: the circuit consumes
    ``n_layers * 3 * n_qubits`` features with no extra quantum parameters.

    The preferred input shape is ``(..., 3 * n_qubits)``, or
    ``(..., n_layers * 3 * n_qubits)`` with incremental re-uploading
    (layer-major slices). The expected width is exposed as ``input_dim``.

    ``(..., n_qubits)`` is also accepted as a compatibility path and repeats
    each scalar across the three rotation axes (incremental re-uploading has
    no such path).

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
    VALID_ENTANGLERS = ("cnot", "crz", "ising_zz")

    def __init__(
        self,
        n_qubits: int,
        n_layers: int,
        readout: str = "z+zz",
        entangler: str = "cnot",
        final_rotation: bool = False,
        incremental_reuploading: bool = False,
    ) -> None:
        super().__init__()
        readout = readout.lower().strip()
        if readout not in self.VALID_READOUTS:
            raise ValueError(f"readout must be one of {self.VALID_READOUTS}, got '{readout}'.")
        entangler = entangler.lower().strip()
        if entangler not in self.VALID_ENTANGLERS:
            raise ValueError(
                f"entangler must be one of {self.VALID_ENTANGLERS}, got '{entangler}'."
            )
        _validate_vqc_dimensions(n_qubits, n_layers)
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.readout = readout
        self.entangler = entangler
        self.final_rotation = final_rotation
        self.incremental_reuploading = incremental_reuploading
        if incremental_reuploading:
            self.input_dim = n_layers * 3 * n_qubits
            self._valid_widths = (self.input_dim,)
        else:
            self.input_dim = 3 * n_qubits
            self._valid_widths = (n_qubits, self.input_dim)

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

        # Trainable entangler angles only exist for parametrised rings with at
        # least one pair; the fixed-CNOT ring has no parameters.
        use_ent_weights = entangler != "cnot" and bool(pairs)

        def _circuit_body(inputs, thetas, feature_weights, entangle_weights=None, final_rot=None):
            if incremental_reuploading:
                if inputs.shape[-1] != n_layers * 3 * n_qubits:
                    raise ValueError(
                        f"Expected last input dimension {n_layers * 3 * n_qubits} "
                        f"with incremental re-uploading, got {inputs.shape[-1]}."
                    )
                features = inputs.reshape(*inputs.shape[:-1], n_layers, n_qubits, 3)
            elif inputs.shape[-1] == 3 * n_qubits:
                features = inputs.reshape(*inputs.shape[:-1], n_qubits, 3)
            elif inputs.shape[-1] == n_qubits:
                features = inputs.unsqueeze(-1).expand(*inputs.shape, 3)
            else:
                raise ValueError(
                    f"Expected last input dimension {3 * n_qubits} or "
                    f"{n_qubits}, got {inputs.shape[-1]}."
                )

            for layer in range(n_layers):
                # Data (re-)uploading with trainable per-layer encoding weights;
                # incremental mode encodes a fresh feature slice per layer.
                f = features[..., layer, :, :] if incremental_reuploading else features
                for i in range(n_qubits):
                    qml.RX(
                        thetas[layer, i, 0] + feature_weights[layer, i, 0] * f[..., i, 0],
                        wires=i,
                    )
                    qml.RZ(
                        thetas[layer, i, 1] + feature_weights[layer, i, 1] * f[..., i, 1],
                        wires=i,
                    )
                    qml.RY(
                        thetas[layer, i, 2] + feature_weights[layer, i, 2] * f[..., i, 2],
                        wires=i,
                    )

                # Ring entanglement (fixed CNOTs or trainable diagonal gates).
                for k, (a, b) in enumerate(pairs):
                    if entangler == "cnot":
                        qml.CNOT(wires=[a, b])
                    elif entangler == "crz":
                        qml.CRZ(entangle_weights[layer, k], wires=[a, b])
                    else:  # ising_zz
                        qml.IsingZZ(entangle_weights[layer, k], wires=[a, b])

            # Trainable measurement basis.
            if final_rot is not None:
                for i in range(n_qubits):
                    qml.Rot(final_rot[i, 0], final_rot[i, 1], final_rot[i, 2], wires=i)
            return [qml.expval(_observable(t)) for t in spec]

        # TorchLayer requires the QNode signature to list exactly the weights
        # in ``weight_shapes``, so expose only the parameters actually in use.
        if use_ent_weights and final_rotation:
            def circuit(inputs, thetas, feature_weights, entangle_weights, final_rot):
                return _circuit_body(inputs, thetas, feature_weights, entangle_weights, final_rot)
        elif use_ent_weights:
            def circuit(inputs, thetas, feature_weights, entangle_weights):
                return _circuit_body(inputs, thetas, feature_weights, entangle_weights)
        elif final_rotation:
            def circuit(inputs, thetas, feature_weights, final_rot):
                return _circuit_body(inputs, thetas, feature_weights, final_rot=final_rot)
        else:
            def circuit(inputs, thetas, feature_weights):
                return _circuit_body(inputs, thetas, feature_weights)

        circuit = qml.qnode(dev, interface="torch", diff_method=diff)(circuit)

        weight_shapes = {
            "thetas": (n_layers, n_qubits, 3),
            "feature_weights": (n_layers, n_qubits, 3),
        }
        if use_ent_weights:
            weight_shapes["entangle_weights"] = (n_layers, len(pairs))
        if final_rotation:
            weight_shapes["final_rot"] = (n_qubits, 3)
        self.qlayer = qml.qnn.TorchLayer(circuit, weight_shapes)
        # Start re-uploading at unit feature weights so the initial encoding
        # scale matches the angle_scale applied in the classifier.
        # Small-angle variational init keeps the circuit near the identity,
        # avoiding the flat (barren-plateau-prone) regions reached by the
        # TorchLayer default U(0, 2*pi) init. The same near-identity init is
        # used for the entangler angles (CRZ(0) = IsingZZ(0) = I) and the
        # final rotations (Rot(0, 0, 0) = I).
        with torch.no_grad():
            self.qlayer.feature_weights.fill_(1.0)
            nn.init.normal_(self.qlayer.thetas, mean=0.0, std=0.1)
            if use_ent_weights:
                nn.init.normal_(self.qlayer.entangle_weights, mean=0.0, std=0.1)
            if final_rotation:
                nn.init.normal_(self.qlayer.final_rot, mean=0.0, std=0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] not in self._valid_widths:
            raise ValueError(
                f"Expected last input dimension in {self._valid_widths}, "
                f"got {x.shape[-1]}."
            )
        return self.qlayer(x)


def _ring_pair_count(n_qubits: int) -> int:
    """Number of nearest-neighbour pairs on the entangling ring."""
    if n_qubits < 2:
        return 0
    return (n_qubits - 1) + (1 if n_qubits > 2 else 0)


def _validate_vqc_dimensions(n_qubits: int, n_layers: int, n_blocks: int = 1) -> int:
    """Validate VQC dimensions and return the qubits per block."""
    if n_qubits < 2:
        raise ValueError(f"n_qubits must be >= 2, got {n_qubits}.")
    if n_layers < 1:
        raise ValueError(f"n_layers must be >= 1, got {n_layers}.")
    if n_blocks < 1:
        raise ValueError(f"n_blocks must be >= 1, got {n_blocks}.")
    if n_qubits % n_blocks != 0:
        raise ValueError(
            f"n_blocks must divide n_qubits, got n_blocks={n_blocks} "
            f"with n_qubits={n_qubits}."
        )
    block_qubits = n_qubits // n_blocks
    if block_qubits < 2:
        raise ValueError(
            f"Each VQC block must have >= 2 qubits to keep ring read-outs "
            f"and entanglers well-defined; got n_qubits={n_qubits}, "
            f"n_blocks={n_blocks}."
        )
    return block_qubits


def _vqc_param_count(
    n_qubits: int,
    n_layers: int,
    entangler: str = "cnot",
    final_rotation: bool = False,
    n_blocks: int = 1,
) -> int:
    """
    Trainable-parameter count of the (possibly parallel) VQC head.

    Mirrors the weight shapes registered in VariationalQuantumLayer so the
    classical controls can match the quantum parameter budget without
    instantiating a quantum device.
    """
    entangler = entangler.lower().strip()
    if entangler not in VariationalQuantumLayer.VALID_ENTANGLERS:
        raise ValueError(
            f"entangler must be one of {VariationalQuantumLayer.VALID_ENTANGLERS}, "
            f"got '{entangler}'."
        )
    block_qubits = _validate_vqc_dimensions(n_qubits, n_layers, n_blocks)
    per_block = 6 * block_qubits * n_layers
    if entangler != "cnot":
        per_block += n_layers * _ring_pair_count(block_qubits)
    if final_rotation:
        per_block += 3 * block_qubits
    return n_blocks * per_block


def _vqc_readout_dim(n_qubits: int, readout: str) -> int:
    """
    Read-out width of VariationalQuantumLayer for a given qubit count.

    Mirrors the observable spec built in VariationalQuantumLayer.__init__ so
    the classical control head can expose an identical ``readout_dim`` without
    instantiating a quantum device.
    """
    readout = readout.lower().strip()
    if n_qubits < 2:
        raise ValueError(f"n_qubits must be >= 2, got {n_qubits}.")

    if readout not in VariationalQuantumLayer.VALID_READOUTS:
        raise ValueError(
            f"readout must be one of {VariationalQuantumLayer.VALID_READOUTS}, got '{readout}'."
        )

    n_pairs = _ring_pair_count(n_qubits)

    single = {
        "z": 1, 
        "z+zz": 1, 
        "z+zz_mean": 1, 
        "bloch": 3, 
        "full": 3
    }[readout]
    
    corr = {
        "z": 0, 
        "z+zz": 1, 
        "z+zz_mean": 0, 
        "bloch": 0, 
        "full": 3
    }[readout]
    
    dim = single * n_qubits + corr * n_pairs
    
    if readout == "z+zz_mean" and n_pairs:
        dim += 1

    return dim


class ParallelVQCLayer(nn.Module):
    """``n_blocks`` independent VQC sub-circuits over disjoint qubit groups.

    The ``n_qubits`` total budget is split into ``n_blocks`` blocks of
    ``n_qubits / n_blocks`` qubits, each an independent
    :class:`VariationalQuantumLayer` (own device, own trainable weights, own
    entangling ring). Block ``j`` consumes the contiguous input slice
    ``[j * block_input_dim, (j + 1) * block_input_dim)`` of the preferred
    ``(..., n_blocks * block_input_dim)`` input (or the matching
    ``(..., n_qubits)`` compatibility slice), and the block read-outs are
    concatenated block-major, so ``readout_dim = n_blocks * block_readout_dim``.
    With ``incremental_reuploading`` the layout is block-major then
    layer-major: each block's slice stays contiguous and is split internally
    into one fresh sub-slice per layer.

    Rationale: keeps the circuit shallow while widening the feature space, and
    the statevector simulation cost drops from ``2^n_qubits`` to
    ``n_blocks * 2^(n_qubits / n_blocks)``. Each block must contain at least
    two qubits so ring read-outs and entanglers do not silently disappear.
    """

    def __init__(
        self,
        n_qubits: int,
        n_layers: int,
        readout: str = "z+zz",
        n_blocks: int = 2,
        entangler: str = "cnot",
        final_rotation: bool = False,
        incremental_reuploading: bool = False,
    ) -> None:
        super().__init__()
        block_qubits = _validate_vqc_dimensions(n_qubits, n_layers, n_blocks)
        self.n_qubits = n_qubits
        self.n_blocks = n_blocks
        self.block_qubits = block_qubits
        self.incremental_reuploading = incremental_reuploading
        self.blocks = nn.ModuleList(
            VariationalQuantumLayer(
                self.block_qubits,
                n_layers,
                readout=readout,
                entangler=entangler,
                final_rotation=final_rotation,
                incremental_reuploading=incremental_reuploading,
            )
            for _ in range(n_blocks)
        )
        self.readout = self.blocks[0].readout
        self.readout_dim = sum(block.readout_dim for block in self.blocks)
        self.input_dim = sum(block.input_dim for block in self.blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] == self.input_dim:
            chunk = self.blocks[0].input_dim
        elif x.shape[-1] == self.n_qubits and not self.incremental_reuploading:
            chunk = self.block_qubits
        else:
            raise ValueError(
                f"Expected last input dimension {self.input_dim}"
                + ("" if self.incremental_reuploading else f" or {self.n_qubits}")
                + f", got {x.shape[-1]}."
            )
        outs = [
            block(x[..., j * chunk:(j + 1) * chunk])
            for j, block in enumerate(self.blocks)
        ]
        return torch.cat(outs, dim=-1)


class ClassicalReadoutBlock(nn.Module):
    """
    Classical drop-in replacement for VariationalQuantumLayer.

    Modes:
      * ``classical_affine`` Exactly matches the VQC
        trainable-parameter budget when 3*n_qubits == readout_dim (true for bloch). 
      * ``classical_lowrank`` rank-r linear map U V.T with r chosen so
        the parameter count matches the VQC budget as closely as possible.
      * ``classical_mlp`` in -> hidden -> out MLP (GELU). 
        Deliberately over-capacity control.
    """
    def __init__(
        self,
        n_qubits: int,
        n_layers: int,
        readout: str = "z+zz",
        mode: str = "classical_lowrank",
        mlp_hidden: int = 64,
        n_blocks: int = 1,
        entangler: str = "cnot",
        final_rotation: bool = False,
        incremental_reuploading: bool = False,
    ) -> None:
        super().__init__()
        if mode not in ("classical_affine", "classical_lowrank", "classical_mlp"):
            raise ValueError(f"mode must be one of "
                             f"('classical_affine', 'classical_lowrank', 'classical_mlp'),"
                             f" got '{{mode}}'.")
        block_qubits = _validate_vqc_dimensions(n_qubits, n_layers, n_blocks)

        self.n_qubits = n_qubits
        self.readout = readout.lower().strip()
        self.incremental_reuploading = incremental_reuploading
        # Mirror the (possibly parallel) VQC read-out width and param budget.
        self.readout_dim = n_blocks * _vqc_readout_dim(block_qubits, self.readout)

        self.mode = mode
        in_dim = (n_layers if incremental_reuploading else 1) * 3 * n_qubits
        self.input_dim = in_dim
        out_dim = self.readout_dim
        self.vqc_budget = _vqc_param_count(
            n_qubits, n_layers,
            entangler=entangler, final_rotation=final_rotation, n_blocks=n_blocks,
        )

        if mode == "classical_affine":
            if in_dim != out_dim:
                raise ValueError(
                    f"classical_affine requires input width == readout_dim "
                    f"({in_dim} vs {out_dim}); use classical_lowrank for readout='{self.readout}'."
                )
            self.scale = nn.Parameter(torch.ones(out_dim))
            self.shift = nn.Parameter(torch.zeros(out_dim))
        elif mode == "classical_lowrank":
            self.rank = max(1, round(self.vqc_budget / (in_dim + out_dim)))
            self.U = nn.Parameter(torch.randn(out_dim, self.rank) * 0.02)
            self.V = nn.Parameter(torch.randn(in_dim, self.rank) * 0.02)
        else:
            # MLP
            self.net = nn.Sequential(
                nn.Linear(in_dim, mlp_hidden), 
                nn.GELU(), 
                nn.Linear(mlp_hidden, out_dim)
            )

        self.n_params = sum(p.numel() for p in self.parameters())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] == self.n_qubits and not self.incremental_reuploading:
            x = x.unsqueeze(-1).expand(*x.shape, 3).reshape(*x.shape[:-1], 3 * self.n_qubits)
        elif x.shape[-1] != self.input_dim:
            raise ValueError(
                f"Expected last input dimension {self.input_dim}"
                + ("" if self.incremental_reuploading else f" or {self.n_qubits}")
                + f", got {x.shape[-1]}."
            )
        if self.mode == "classical_affine":
            return x * self.scale + self.shift
        if self.mode == "classical_lowrank":
            return (x @ self.V) @ self.U.t()
        return self.net(x)


# -- ResNet feature extractor ---------------------------------------------
_FEAT_DIM = {
    "resnet18": 512, 
    "resnet34": 512, 
    "resnet50": 2048
}


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


class SimpleHybridVQCClassifier(nn.Module):
    """ResNet -> angle map (linear or MLP) -> VQC -> BN -> linear.

    The ResNet feature vector is normalised, projected to three input features
    per qubit, bounded with ``tanh``, and passed to the VQC. Each VQC layer then
    re-uploads the same per-qubit feature triplet with independent trainable
    encoding weights and offsets.

    With ``incremental_reuploading`` the angle map instead emits
    ``vqc_layers * 3 * n_qubits`` angles and each VQC layer encodes its own
    fresh feature slice (incremental data re-uploading): depth then widens the
    classical-to-quantum bottleneck rather than re-encoding the same features.
    The quantum parameter count is unchanged; only the angle map grows.

    ``angle_map_type`` selects the adapter in front of the quantum bottleneck:

    * ``"linear"`` a single feat -> 3*n_qubits map (default).
    * ``"mlp"``    feat -> angle_map_hidden_dim -> 3*n_qubits with GELU and
      ``angle_map_dropout``.

    The circuit itself is configured by ``vqc_entangler`` (fixed CNOT ring or
    trainable CRZ / IsingZZ ring), ``vqc_final_rotation`` (trainable
    measurement basis after the last entangling ring) and ``vqc_blocks``
    (split the qubit budget into independent parallel sub-circuits with
    concatenated read-outs; must divide ``n_qubits`` and leave at least two
    qubits per block). See
    :class:`VariationalQuantumLayer` and :class:`ParallelVQCLayer`.

    Input:  ``(B, 3, H, W)``  Output: ``(B, n_classes)``
    """

    def __init__(
        self,
        n_classes: int,
        n_qubits: int = 8,
        vqc_layers: int = 2,
        vqc_readout: str = "z+zz",
        resnet_freeze_mode: str = "none",
        backbone: str = "resnet18",
        backbone_weights: Optional[str] = None,
        pretrained: bool = True,
        head_type: str = "vqc",
        angle_map_type: str = "linear",
        angle_map_hidden_dim: int = 64,
        angle_map_dropout: float = 0.2,
        vqc_entangler: str = "cnot",
        vqc_final_rotation: bool = False,
        vqc_blocks: int = 1,
        incremental_reuploading: bool = False,
    ) -> None:
        super().__init__()
        angle_map_type = angle_map_type.lower().strip()
        if angle_map_type not in ('mlp', 'linear'):
            raise ValueError(
                f"angle_map_type must be one of ('mlp', 'linear'), got '{angle_map_type}'."
            )
        if angle_map_type == "mlp" and angle_map_hidden_dim < 1:
            raise ValueError(
                f"angle_map_hidden_dim must be >= 1, got {angle_map_hidden_dim}."
            )
        if angle_map_type == "mlp" and not 0.0 <= angle_map_dropout < 1.0:
            raise ValueError(
                f"angle_map_dropout must be in [0, 1), got {angle_map_dropout}."
            )
        self.head_type = head_type
        self.angle_map_type = angle_map_type

        self.n_classes = n_classes
        self.n_qubits = n_qubits
        self.incremental_reuploading = incremental_reuploading

        self.resnet = _ResNetFeatureExtractor(
            backbone=backbone,
            weights_path=backbone_weights,
            pretrained=pretrained,
            freeze_mode=resnet_freeze_mode,
        )
        feat_dim = self.resnet.feature_dim

        self.feature_norm = nn.LayerNorm(feat_dim)
        angle_dim = (vqc_layers if incremental_reuploading else 1) * 3 * n_qubits
        if angle_map_type == "mlp":
            self.angle_map = nn.Sequential(
                nn.Linear(feat_dim, angle_map_hidden_dim),
                nn.GELU(),
                nn.Dropout(p=angle_map_dropout),
                nn.Linear(angle_map_hidden_dim, angle_dim),
            )
        else:
            self.angle_map = nn.Linear(feat_dim, angle_dim)
        _validate_vqc_dimensions(n_qubits, vqc_layers, vqc_blocks)
        if head_type == "vqc":
            if vqc_blocks > 1:
                self.quantum = ParallelVQCLayer(
                    n_qubits, vqc_layers, readout=vqc_readout, n_blocks=vqc_blocks,
                    entangler=vqc_entangler, final_rotation=vqc_final_rotation,
                    incremental_reuploading=incremental_reuploading,
                )
            else:
                self.quantum = VariationalQuantumLayer(
                    n_qubits, vqc_layers, readout=vqc_readout,
                    entangler=vqc_entangler, final_rotation=vqc_final_rotation,
                    incremental_reuploading=incremental_reuploading,
                )
        else:
            self.quantum = ClassicalReadoutBlock(
                n_qubits, vqc_layers, readout=vqc_readout, mode=head_type,
                n_blocks=vqc_blocks, entangler=vqc_entangler,
                final_rotation=vqc_final_rotation,
                incremental_reuploading=incremental_reuploading,
            )

        readout_dim = self.quantum.readout_dim
        self.q_norm = nn.BatchNorm1d(readout_dim)
        self.q_proj = nn.Linear(readout_dim, n_classes)
        self.register_buffer("angle_scale", torch.tensor(0.75 * math.pi))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        feats = self.feature_norm(self.resnet(images))
        q_in = torch.tanh(self.angle_map(feats)) * self.angle_scale
        q_raw = self.quantum(q_in)
        if q_raw.device != q_in.device:
            q_raw = q_raw.to(q_in.device)
        return self.q_proj(self.q_norm(q_raw))


# -- Convenience factory --------------------------------------------------


def hybrid_vqc_model(
    fc_out_features: int,
    n_qubits: int = 8,
    vqc_layers: int = 2,
    vqc_readout: str = "z+zz",
    resnet_freeze_mode: str = "none",
    backbone: str = "resnet18",
    backbone_weights: Optional[str] = None,
    pretrained: bool = True,
    head_type: str = "vqc",
    angle_map_type: str = "linear",
    angle_map_hidden_dim: int = 64,
    angle_map_dropout: float = 0.2,
    vqc_entangler: str = "cnot",
    vqc_final_rotation: bool = False,
    vqc_blocks: int = 1,
    incremental_reuploading: bool = False,
) -> SimpleHybridVQCClassifier:
    """Factory for `SimpleHybridVQCClassifier`."""
    return SimpleHybridVQCClassifier(
        n_classes=fc_out_features,
        n_qubits=n_qubits,
        vqc_layers=vqc_layers,
        vqc_readout=vqc_readout,
        resnet_freeze_mode=resnet_freeze_mode,
        backbone=backbone,
        backbone_weights=backbone_weights,
        pretrained=pretrained,
        head_type=head_type,
        angle_map_type=angle_map_type,
        angle_map_hidden_dim=angle_map_hidden_dim,
        angle_map_dropout=angle_map_dropout,
        vqc_entangler=vqc_entangler,
        vqc_final_rotation=vqc_final_rotation,
        vqc_blocks=vqc_blocks,
        incremental_reuploading=incremental_reuploading,
    )
