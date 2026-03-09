import os
from collections import defaultdict
from typing import Dict, Any, Optional, List, Tuple, Set

import torch
import torch.nn as nn
import torch.optim
from torchvision.models import resnet18, resnet34, resnet50, ResNet, ResNet18_Weights, ResNet34_Weights, ResNet50_Weights
import torchvision.utils as vutils

from hybrid_model import HybridMPSClassifier


def resnet_model(model_name: str = 'resnet18',
                 pretrained: bool = True,
                 fc_out_features: int = 2,
                 freeze_backbone: bool = False) -> ResNet:
    """
    Factory function to create a ResNet model with a custom fully connected layer.
    """
    model_name_lower = model_name.lower()
    
    if model_name_lower == 'resnet18':
        resnet = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
    elif model_name_lower == 'resnet34':
        resnet = resnet34(weights=ResNet34_Weights.DEFAULT if pretrained else None)
    elif model_name_lower == 'resnet50':
        resnet = resnet50(weights=ResNet50_Weights.DEFAULT if pretrained else None)
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    # Modify the final fully connected layer
    resnet.fc = nn.Linear(
        in_features=resnet.fc.in_features, 
        out_features=fc_out_features, 
        bias=True
    )

    if freeze_backbone:
        # Keep only layer4 + classification head trainable.
        for param in resnet.parameters():
            param.requires_grad = False
        for param in resnet.layer4.parameters():
            param.requires_grad = True
        for param in resnet.fc.parameters():
            param.requires_grad = True
    
    return resnet


def hybrid_mps_vqc_model(
    fc_out_features: int,
    n_qubits: int = 8,
    mps_bond_dim: int = 64,
    mps_phys_dim: int = 2,
    mps_hidden_dim: int = 128,
    vqc_layers: int = 2,
    freeze_resnet: bool = True,
    use_pennylane: bool = True,
) -> HybridMPSClassifier:
    """
    Factory function to create a Hybrid MPS + VQC classifier.
    ResNet18 -> MPS Bridge -> VQC -> BN -> Linear.
    """
    return HybridMPSClassifier(
        n_classes=fc_out_features,
        n_qubits=n_qubits,
        mps_bond_dim=mps_bond_dim,
        mps_phys_dim=mps_phys_dim,
        bridge_hidden_dim=mps_hidden_dim,
        vqc_layers=vqc_layers,
        freeze_resnet=freeze_resnet,
        use_pennylane=use_pennylane,
    )


class EndToEndModel(nn.Module):
    """
    End-to-End model that combines a raw-processor, augmentation, and a classifier.
    """

    def __init__(self,
                 classifier: nn.Module,
                 processor: Optional[nn.Module] = None,
                 normalization: Optional[nn.Module] = None,
                 augmentation: Optional[nn.Module] = None,
                 augmentation_target_classes: Optional[List[int]] = None,
                 non_target_augmentation_prob: float = 1.0,
                 augmentation_on_eval: bool = False,
                 freeze_processor: bool = False):
        super().__init__()

        self.classifier = classifier
        self.processor = processor
        self.normalization = normalization
        self.augmentation = augmentation
        self.augmentation_target_classes: Optional[Set[int]] = (
            set(augmentation_target_classes) if augmentation_target_classes is not None else None
        )
        self.non_target_augmentation_prob = float(non_target_augmentation_prob)

        self.augmentation_on_eval = augmentation_on_eval
        self.freeze_processor = freeze_processor
        
        if freeze_processor and self.processor is not None:
             self._freeze_module(self.processor)

    def _freeze_module(self, module: nn.Module) -> None:
        for param in module.parameters():
            param.requires_grad = False

    def _apply_conditional_augmentation(self, x: torch.Tensor, labels: Optional[torch.Tensor]) -> torch.Tensor:
        if self.augmentation is None:
            return x
        if self.augmentation_target_classes is None or labels is None:
            return self.augmentation(x)
        if x.ndim != 4:
            return self.augmentation(x)

        label_list = labels.detach().cpu().tolist()
        augmented = []
        for i, sample in enumerate(x):
            label = int(label_list[i])
            should_augment = label in self.augmentation_target_classes
            if not should_augment and self.non_target_augmentation_prob > 0.0:
                should_augment = bool(torch.rand(1, device=sample.device).item() < self.non_target_augmentation_prob)

            if should_augment:
                sample = self.augmentation(sample)
            augmented.append(sample)

        return torch.stack(augmented, dim=0)

    def forward(self, x: torch.Tensor, labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        # 1. Image Processing (Raw -> RGB)
        if self.processor is not None:
            x = self.processor(x)

        # 2. Augmentation (Training only, unless specified)
        apply_augmentation = self.training or self.augmentation_on_eval
        if self.augmentation is not None and apply_augmentation:
            x = self._apply_conditional_augmentation(x, labels)

        # 3. Normalization (after augmentation to keep transform domain consistent)
        if self.normalization is not None:
            x = self.normalization(x)

        # 4. Classification
        x = self.classifier(x)
        return x


class TensorNormalize(nn.Module):
    """Channel-wise normalization module for tensors of shape (B, C, H, W) or (C, H, W)."""

    def __init__(self, mean: List[float], std: List[float], eps: float = 1e-6):
        super().__init__()
        mean_tensor = torch.as_tensor(mean, dtype=torch.float32).view(1, -1, 1, 1)
        std_tensor = torch.as_tensor(std, dtype=torch.float32).view(1, -1, 1, 1)
        self.register_buffer("mean", mean_tensor)
        self.register_buffer("std", torch.clamp(std_tensor, min=eps))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3:
            return (x - self.mean.squeeze(0)) / self.std.squeeze(0)
        if x.ndim == 4:
            return (x - self.mean) / self.std
        raise ValueError(f"TensorNormalize expects 3D or 4D tensor, got shape {tuple(x.shape)}")


def log_tensor(batch: torch.Tensor, path: str, save_tensors: bool = True, nrow: int = 8) -> None:
    """Helper to save tensors as images and/or .pt files."""
    if save_tensors:
        torch.save(batch, path)

    # Only save as image if it looks like an image batch (B, C, H, W) or single image (C, H, W)
    if batch.ndim == 4 or (batch.ndim == 3 and batch.shape[0] in [1, 3, 4]):
        img_path = path.replace('.pt', '.png')
        
        # Simple path manipulation to insert prefix
        directory, filename = os.path.split(img_path)
        img_path = os.path.join(directory, f"img_{filename}")

        try:
            # Create grid and save
            # Normalize to [0, 1] for visualization if needed, but make_grid handles some
            grid = vutils.make_grid(batch, nrow=nrow, normalize=True).squeeze()
            vutils.save_image(grid, img_path)
        except Exception as e:
            print(f"Warning: Could not save image for {path}: {e}")
    else:
        # For logits/labels (2D or 1D), we skip image generation
        pass


def track_images(model: EndToEndModel, 
                 data_loader: torch.utils.data.DataLoader, 
                 loss_fn: Optional[nn.Module] = None, 
                 device: torch.device = torch.device('cpu'), 
                 reference_processor: Optional[nn.Module] = None, 
                 track_processing: bool = True, 
                 track_gradients: bool = True, 
                 track_predictions: bool = True, 
                 save_tensors: bool = True, 
                 save_loc: str = 'results') -> None:
    """
    Debug function to track intermediate outputs, gradients, and predictions.
    """
    processor = model.processor
    classifier = model.classifier

    # Valid only if processor exposes intermediate stages
    if not hasattr(processor, 'stages'):
        return

    os.makedirs(save_loc, exist_ok=True)

    labels_full = []
    logits_full = []
    stages_full = defaultdict(list)
    grads_full = defaultdict(list)
    diffs_full = defaultdict(list)

    track_differences = reference_processor is not None

    print("Tracking images...")
    
    for inputs, labels in data_loader:
        inputs = inputs.to(device) 
        labels = labels.to(device)
        inputs.requires_grad = True

        processed_rgb = processor(inputs)

        if track_differences and reference_processor is not None:
            # reference_processor(inputs) # Run ref processor if needed
             pass

        if track_gradients or track_predictions:
            logits = classifier(processed_rgb)

            if track_gradients and loss_fn is not None:
                model.zero_grad()
                loss = loss_fn(logits, labels)
                loss.backward()

            if track_predictions:
                labels_full.append(labels.detach().cpu())
                logits_full.append(logits.detach().cpu())

        # Collect stages
        if hasattr(processor, 'stages'):
             for stage, batch in processor.stages.items():
                stages_full[stage].append(batch.detach().cpu())
                
                if track_differences and reference_processor:
                     # Placeholder for diff logic if implemented
                     pass

                if track_gradients:
                    batch_grad = batch.grad
                    if batch_grad is not None:
                         grads_full[stage].append(batch_grad.detach().cpu())

    # Concatenate and Save
    with torch.no_grad():
        if track_processing:
            for stage_name, batches in stages_full.items():
                full_batch = torch.cat(batches)
                log_tensor(full_batch, os.path.join(save_loc, f'processing_{stage_name}.pt'), save_tensors)

        if track_gradients:
            for stage_name, batches in grads_full.items():
                full_batch = torch.cat(batches)
                full_batch = full_batch.abs()
                # Min-max normalization for visualization
                if full_batch.max() > full_batch.min():
                    full_batch = (full_batch - full_batch.min()) / (full_batch.max() - full_batch.min())
                
                log_tensor(full_batch, os.path.join(save_loc, f'gradients_{stage_name}.pt'), save_tensors)

        if track_predictions:
            labels_cat = torch.cat(labels_full).unsqueeze(1) # shape (N, 1)
            logits_cat = torch.cat(logits_full)
            
            # Since this is classification, logits are (N, C). Visualization depends on task.
            # Just saving raw tensors for now as per original code
            log_tensor(labels_cat.float(), os.path.join(save_loc, 'targets.pt'), save_tensors)
            log_tensor(logits_cat, os.path.join(save_loc, 'preds.pt'), save_tensors)
