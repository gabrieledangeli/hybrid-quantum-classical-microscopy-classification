import argparse

def get_args():
    parser = argparse.ArgumentParser(description='Train ResNet18 on Medical Dataset')
    parser.add_argument('--mode', type=str, default='rgb', choices=['rgb', 'raw'], help='Dataset mode: rgb or raw (grayscale)')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--wd', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--step_size', type=int, default=30, help='Step size for LR scheduler')
    parser.add_argument('--gamma', type=float, default=0.1, help='Gamma for LR scheduler')
    parser.add_argument('--no_preload', action='store_true', help='Disable dataset preloading')
    
    return parser.parse_args()
