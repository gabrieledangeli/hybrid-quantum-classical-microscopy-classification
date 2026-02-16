import argparse

def get_args():
    parser = argparse.ArgumentParser(description='Train ResNet18 on Medical Dataset')
    parser.add_argument('--mode', type=str, default='rgb', choices=['rgb', 'raw'], help='Dataset mode: rgb or raw (grayscale)')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--wd', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--step_size', type=int, default=30, help='Step size for LR scheduler')
    parser.add_argument('--gamma', type=float, default=0.1, help='Gamma for LR scheduler')
    parser.add_argument('--no_preload', action='store_true', help='Disable dataset preloading')

    # Quantum / Hybrid Parameters
    parser.add_argument('--use_quantum', action='store_true', help='Use Hybrid Quantum-Classical Model')
    parser.add_argument('--use_mps', action='store_true', help='Use Hybrid MPS-Quantum Model')
    parser.add_argument('--n_qubits', type=int, default=9, help='Number of qubits for quantum layer')
    parser.add_argument('--n_layers', type=int, default=2, help='Number of layers in quantum ansatz')
    parser.add_argument('--ansatz', type=str, default='strong', choices=['strong', 'hardware'], help='Quantum ansatz type')
    parser.add_argument('--backend', type=str, default='default.qubit', help='PennyLane backend device')
    parser.add_argument('--shots', type=int, default=None, help='Number of shots for quantum measurement (None = analytic)')
    parser.add_argument('--freeze_backbone', action='store_true', help='Freeze the Classical ResNet backbone')
    
    return parser.parse_args()
