import torch
import torch.nn as nn
import pennylane as qml
from pennylane.qnn import TorchLayer
import sys
import os
import numpy as np

from ResNet18 import ResNet18


class QuantumHead(nn.Module):
    """
    A PyTorch module containing a PennyLane Quantum Neural Network.
    Replaces the final fully connected layer of the classical network.
    """
    def __init__(self, n_qubits, n_layers, num_classes, backend="default.qubit", shots=None, ansatz="strong"):
        super(QuantumHead, self).__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.num_classes = num_classes
        self.ansatz = ansatz
        
        # Define the quantum device
        # 'default.qubit' is a pure state simulator (perfect for backprop).
        self.dev = qml.device(backend, wires=n_qubits, shots=shots)

        # Define the QNode
        # interface="torch" allows PyTorch to differentiate through the quantum circuit.
        @qml.qnode(self.dev, interface="torch")
        def circuit(inputs, weights):
            # 1. Encoding: Amplitude Embedding
            qml.AmplitudeEmbedding(features=inputs, wires=range(n_qubits), pad_with=0.0, normalize=True)

            # 2. Ansatz
            if self.ansatz == "strong":
                qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
                
            elif self.ansatz == "hardware":
                # Hardware Efficient Ansatz
                # Layer of Ry rotations followed by a chain of CNOTs.
                for l in range(n_layers):
                    for q in range(n_qubits):
                        qml.RY(weights[l, q], wires=q)
                    
                    # Entangling layer
                    for q in range(n_qubits - 1):
                        qml.CNOT(wires=[q, q+1])
            else:
                raise ValueError(f"Unknown ansatz type: {self.ansatz}")

            # 3. Measurement
            # We measure the expectation value of PauliZ on each qubit.
            return [qml.expval(qml.PauliZ(w)) for w in range(n_qubits)]

        # Define weight shapes based on ansatz structure
        if self.ansatz == "strong":
            weight_shapes = {"weights": (n_layers, n_qubits, 3)}
        elif self.ansatz == "hardware":
            weight_shapes = {"weights": (n_layers, n_qubits)}

        # Create the TorchLayer (wraps QNode as nn.Module)
        self.qnn = TorchLayer(circuit, weight_shapes) # type: ignore
        
        # We use a classical Linear layer to map the quantum measurements to class logits.
        self.post_process = nn.Linear(n_qubits, num_classes)

    def forward(self, x):
        q_out = self.qnn(x)
        logits = self.post_process(q_out)
        return logits


class HybridResNet18(nn.Module):
    """
    ResNet18 backbone with a pluggable Quantum Head.
    Compatible with the existing MedicalDataset training pipeline.
    """
    def __init__(self, num_classes=10, n_qubits=9, n_layers=2, 
                 freeze_backbone=False, 
                 ansatz="strong", backend="default.qubit", shots=None, in_channels=3):
        super(HybridResNet18, self).__init__()
                
        # 1. Load the backbone
        self.backbone = ResNet18(num_classes=num_classes, in_channels=in_channels)
        
        # 2. Extract Feature Dimensions
        # In ResNet18, the feature vector before the final FC is 512.
        self.feature_dim = 512
        
        # Remove the original classification head to save memory/parameters
        del self.backbone.fc

        # 3. Freeze backbone if requested (Transfer Learning)
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # 4. Define the QuantumHead
        # If n_qubits != 9, we project the features using a classical Linear layer.
        target_dim = 2**n_qubits
        
        if self.feature_dim == target_dim:
            self.pre_quantum_layer = nn.Identity()
        else:
            print(f"Projecting {self.feature_dim} features to {target_dim} for quantum layer.")
            self.pre_quantum_layer = nn.Linear(self.feature_dim, target_dim)
        
        self.head = QuantumHead(n_qubits=n_qubits, 
                                n_layers=n_layers, 
                                num_classes=num_classes, 
                                backend=backend,
                                shots=shots,
                                ansatz=ansatz)

    def forward(self, x):
        # --- Backbone Feed Forward ---
        # Replicating ResNet18 forward logic manually to bypass 'fc'
        x = self.backbone.relu(self.backbone.bn1(self.backbone.conv1(x)))
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1) # Shape: (Batch, 512)
        
        # Project to 2^n if necessary
        x = self.pre_quantum_layer(x)
            
        # --- Head Feed Forward ---
        x = self.head(x)
        
        return x
