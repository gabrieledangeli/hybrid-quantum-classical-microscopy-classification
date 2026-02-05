import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm
from ResNet18 import ResNet18
from dataset import MedicalDataset, class_to_idx

if __name__ == '__main__':
    # Define transforms for 256x256 images
    transform_train = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    transform_test = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    # Load dataset from data/microscopy/images/rgb_scale100
    # Use absolute paths or paths relative to the script location to avoid CWD issues
    script_dir = os.path.dirname(os.path.abspath(__file__))
    workspace_dir = os.path.dirname(script_dir)
    
    data_path = os.path.join(workspace_dir, 'data', 'microscopy', 'images', 'rgb_scale100')
    annotation_path = os.path.join(workspace_dir, 'data', 'microscopy', 'labels', 'Ma190c_annotations.dat')
    
    full_dataset = MedicalDataset(root_dir=data_path, annotation_file=annotation_path, transform=transform_train, preload=True)
    num_classes = len(class_to_idx)
    
    print(f"Dataset loaded from {data_path} with {len(full_dataset)} images.")

    # Split dataset into train and test
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    trainset, testset = torch.utils.data.random_split(full_dataset, [train_size, test_size])
    
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False, num_workers=2)

    # Instantiate the model, define the loss function and the optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNet18(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    # Training loop
    num_epochs = 100
    train_losses, train_acc_list, test_acc_list = [], [], []

    print("\n" + "="*50)
    print(f"{'TRAINING CONFIGURATION':^50}")
    print("="*50)
    print(f"{'Device':<20}: {device}")
    print(f"{'Model Architecture':<20}: ResNet18")
    print(f"{'Combined Classes':<20}: {num_classes}")
    print("-" * 50)
    print(f"{'Epochs':<20}: {num_epochs}")
    print(f"{'Batch Size':<20}: 32")
    print(f"{'Learning Rate':<20}: 0.001")
    print(f"{'Optimizer':<20}: Adam (wd=1e-4)")
    print(f"{'Scheduler':<20}: StepLR (step=30, gamma=0.1)")
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
    plt.show()