import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from ResNet18 import ResNet18
from HQCNN import HybridResNet18
from hybrid_mps_qnn import MicroscopyHybridModelMPS
from dataset import MedicalDataset, class_to_idx
from args import get_args

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

if __name__ == '__main__':
    # Get arguments
    args = get_args()

    # Load dataset paths first to determine channels
    script_dir = os.path.dirname(os.path.abspath(__file__))
    workspace_dir = os.path.dirname(script_dir)
    
    # Select dataset based on args
    if args.mode == 'rgb':
        #data_path = os.path.join(workspace_dir, 'data', 'microscopy', 'images', 'rgb_scale100')
        data_path = os.path.join('C:\\Users\\gabriele.dangeli\\Downloads', 'data', 'microscopy', 'images', 'rgb_scale075')
    else:
        data_path = os.path.join(workspace_dir, 'data', 'microscopy', 'images', 'raw_scale100')

    annotation_path = os.path.join(workspace_dir, 'data', 'microscopy', 'labels', 'Ma190c_annotations.dat')
    
    # Determine mode based on folder name/args
    if args.mode == 'rgb':
        input_mode = 'RGB'
        in_channels = 3
        norm_mean = (0.5, 0.5, 0.5)
        norm_std = (0.5, 0.5, 0.5)
    else:
        input_mode = 'L' # Grayscale
        in_channels = 1
        norm_mean = (0.5,)
        norm_std = (0.5,)
    
    print(f"Detected mode: {input_mode} ({in_channels} channels)")

    # Define transforms for 256x256 images
    transform_train = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(90),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std),
    ])

    transform_test = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std),
    ])

    full_dataset = MedicalDataset(root_dir=data_path, annotation_file=annotation_path, transform=transform_train, preload=not args.no_preload, mode=input_mode)
    num_classes = len(class_to_idx)
    
    print(f"Dataset loaded from {data_path} with {len(full_dataset)} images.")

    # Split dataset into train and test
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    trainset, testset = torch.utils.data.random_split(full_dataset, [train_size, test_size])
    
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True, persistent_workers=True, prefetch_factor=4)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True, persistent_workers=True, prefetch_factor=4)

    # Instantiate the model, define the loss function and the optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if args.use_mps:
        print(f"Initializing Hybrid MPS Model (Qubits={args.n_qubits})")
        model = MicroscopyHybridModelMPS(
            n_classes=num_classes,
            n_qubits=args.n_qubits,
            mps_bond_dim=64, # Default bond dim
            mps_phys_dim=2,
            bridge_hidden_dim=128,
            vqc_layers=args.n_layers,
            freeze_resnet=args.freeze_backbone,
            use_pennylane=True
        ).to(device)
    elif args.use_quantum:
        print(f"Initializing Hybrid Model (Quantum={args.use_quantum})")
        model = HybridResNet18(
            num_classes=num_classes, 
            n_qubits=args.n_qubits, 
            n_layers=args.n_layers, 
            freeze_backbone=args.freeze_backbone, 
            ansatz=args.ansatz, 
            backend=args.backend, 
            shots=args.shots, 
            in_channels=in_channels
        ).to(device)
    else:
        print("Initializing Classical ResNet18 Model")
        model = ResNet18(num_classes=num_classes, in_channels=in_channels).to(device)

    criterion = nn.CrossEntropyLoss(weight=None).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
        
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    # Training loop
    num_epochs = args.epochs
    train_losses, train_acc_list, test_acc_list = [], [], []

    print("\n" + "="*50)
    print(f"{'TRAINING CONFIGURATION':^50}")
    print("="*50)
    print(f"{'Device':<20}: {device}")
    
    if args.use_mps:
        model_type = "Hybrid MPS Quantum Model"
    elif args.use_quantum:
        model_type = "Hybrid Quantum ResNet18"
    elif args.freeze_backbone:
        model_type = "Frozen Classical ResNet18"
    else:
        model_type = "Classical ResNet18"
        
    print(f"{'Model Architecture':<20}: {model_type}")
    print(f"{'Output Classes':<20}: {num_classes}")

    # Calculate model parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{'Total Parameters':<20}: {total_params:,}")
    print(f"{'Trainable Params':<20}: {trainable_params:,}")
    
    if args.use_quantum or args.use_mps:
        print("-" * 50)
        print(f"{'>>> QUANTUM CONFIG':<20}")
        print(f"{'Qubits':<20}: {args.n_qubits}")
        print(f"{'Layers':<20}: {args.n_layers}")
        print(f"{'Ansatz':<20}: {args.ansatz}")
        print(f"{'Backend':<20}: {args.backend}")
        print(f"{'Shots':<20}: {args.shots if args.shots else 'Analytic (None)'}")
        print(f"{'Backbone Frozen':<20}: {args.freeze_backbone}")

    print("-" * 50)
    print(f"{'Mode':<20}: {args.mode}")
    print(f"{'Epochs':<20}: {num_epochs}")
    print(f"{'Batch Size':<20}: {args.batch_size}")
    print(f"{'Learning Rate':<20}: {args.lr}")
    print(f"{'Optimizer':<20}: Adam (wd={args.wd})")
    print(f"{'Scheduler':<20}: StepLR (step={args.step_size}, gamma={args.gamma})")
    print("-" * 50)
    print(f"{'Total Dataset':<20}: {len(full_dataset)}")
    print(f"{'Training Set':<20}: {train_size} ({train_size/len(full_dataset):.0%})")
    print(f"{'Test Set':<20}: {test_size} ({test_size/len(full_dataset):.0%})")
    print(f"{'Preloading':<20}: {full_dataset.preload}")
    print("="*50 + "\n")

    print("="*20)
    print("Starting Training...")
    print("="*20)

    # Epoch loop with progress bar
    epoch_pbar = tqdm(range(num_epochs), desc="Training", unit="epoch")
    
    for epoch in epoch_pbar:
        model.train()
        running_loss = 0.0
        correct, total = 0, 0
        
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_loss = running_loss / total
        train_acc = 100. * correct / total
        train_losses.append(train_loss)
        train_acc_list.append(train_acc)

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for inputs, labels in testloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        test_acc = 100. * correct / total
        test_acc_list.append(test_acc)

        scheduler.step()
        
        # Update progress bar with metrics
        epoch_pbar.set_postfix({
            'Loss': f'{train_loss:.4f}', 
            'Train Acc': f'{train_acc:.2f}%', 
            'Test Acc': f'{test_acc:.2f}%'
        })


    # Plot training loss and accuracy
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_acc_list, label='Train Accuracy')
    plt.plot(test_acc_list, label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Training and Test Accuracy')
    plt.legend()

    plt.tight_layout()
    
    # Save the plot
    plots_dir = os.path.join(workspace_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # Create a unique filename based on config
    filename = f"training_plot_{model_type.replace(' ', '_')}_{args.mode}_e{args.epochs}"
    if args.use_quantum or args.use_mps:
        filename += f"_q{args.n_qubits}_l{args.n_layers}"
    filename += ".png"
    
    save_path = os.path.join(plots_dir, filename)
    plt.savefig(save_path)
    print(f"Plot saved to: {save_path}")
    
    # --- Detailed Metrics & Visualization ---
    print("\n" + "="*20)
    print("Calculating Detailed Metrics...")
    
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(testloader, desc="Final Evaluation"):
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    # Sort class names by index to ensure correct mapping
    class_names = [k for k, v in sorted(class_to_idx.items(), key=lambda item: item[1])]
    
    # Print Classification Report (Precision, Recall, F1)
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names, digits=4))
    
    # Generate and Save Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    plt.figure(figsize=(12, 10))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues, xticks_rotation=45, ax=plt.gca(), values_format='d')
    plt.title(f'Confusion Matrix - {model_type}')
    plt.tight_layout()
    
    cm_filename = f"confusion_matrix_{model_type.replace(' ', '_')}_{args.mode}.png"
    cm_save_path = os.path.join(plots_dir, cm_filename)
    plt.savefig(cm_save_path)
    print(f"Confusion Matrix saved to: {cm_save_path}")
    # ----------------------------------------

    plt.show()