"""
Level 3: GPU-Accelerated Neural Network with PyTorch
=====================================================

Same concepts as 1_improved_network.py but using GPU for 10-100x speedup!

Topics covered:
1. Cross-Entropy Cost Function
2. L2 Regularization (weight_decay)
3. Better Weight Initialization (Xavier/He)
4. Full training with ALL improvements

Run from src/ directory:
    python level3/2_gpu_accelerated.py

Requirements:
    - PyTorch with CUDA support
    - GPU with CUDA capability
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
import gzip
import pickle
from utils import level3_picture, reset_run_timestamp


# =============================================================================
# SETUP: Device Selection
# =============================================================================

def setup_device():
    """Select GPU if available, otherwise CPU."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"🚀 Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"   CUDA version: {torch.version.cuda}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        device = torch.device("cpu")
        print("⚠️  GPU not available, using CPU")
    return device


# =============================================================================
# DATA LOADING
# =============================================================================

def load_mnist_pytorch(device):
    """Load MNIST and convert to PyTorch tensors on device."""
    data_path = Path(__file__).parent.parent.parent / "data" / "mnist.pkl.gz"
    
    with gzip.open(data_path, 'rb') as f:
        training_data, validation_data, test_data = pickle.load(f, encoding='latin1')
    
    # Convert to PyTorch tensors
    X_train = torch.FloatTensor(training_data[0]).to(device)
    y_train = torch.LongTensor(training_data[1]).to(device)
    
    X_val = torch.FloatTensor(validation_data[0]).to(device)
    y_val = torch.LongTensor(validation_data[1]).to(device)
    
    X_test = torch.FloatTensor(test_data[0]).to(device)
    y_test = torch.LongTensor(test_data[1]).to(device)
    
    print(f"\nData loaded to {device}:")
    print(f"  Training:   {X_train.shape[0]:,} samples")
    print(f"  Validation: {X_val.shape[0]:,} samples")
    print(f"  Test:       {X_test.shape[0]:,} samples")
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


# =============================================================================
# NEURAL NETWORK MODEL
# =============================================================================

class ImprovedNetwork(nn.Module):
    """
    Neural network with all Level 3 improvements built-in:
    - Cross-Entropy loss (via nn.CrossEntropyLoss)
    - Xavier/He initialization
    - Regularization via optimizer weight_decay
    """
    
    def __init__(self, sizes, init_method='xavier'):
        super().__init__()
        
        self.sizes = sizes
        layers = []
        
        for i in range(len(sizes) - 1):
            layer = nn.Linear(sizes[i], sizes[i+1])
            
            # Better weight initialization
            if init_method == 'xavier':
                # Xavier/Glorot initialization (good for sigmoid/tanh)
                nn.init.xavier_uniform_(layer.weight)
            elif init_method == 'he':
                # He initialization (good for ReLU)
                nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
            elif init_method == 'large':
                # Old method (for comparison) - N(0, 1)
                nn.init.normal_(layer.weight, mean=0, std=1)
            else:
                # Default scaled initialization like network2.py
                nn.init.normal_(layer.weight, mean=0, std=1/np.sqrt(sizes[i]))
            
            nn.init.zeros_(layer.bias)
            layers.append(layer)
            
            # Add activation (sigmoid like original, but ReLU is faster)
            if i < len(sizes) - 2:  # Not on last layer
                layers.append(nn.Sigmoid())
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


# =============================================================================
# TRAINING FUNCTIONS
# =============================================================================

def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * X_batch.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(y_batch).sum().item()
        total += y_batch.size(0)
    
    return total_loss / total, 100 * correct / total


def evaluate(model, X, y, criterion):
    """Evaluate model accuracy."""
    model.eval()
    with torch.no_grad():
        outputs = model(X)
        loss = criterion(outputs, y).item()
        _, predicted = outputs.max(1)
        accuracy = 100 * predicted.eq(y).sum().item() / y.size(0)
    return loss, accuracy


def train_model(model, train_data, val_data, epochs, batch_size, lr, weight_decay, device, verbose=True):
    """Full training loop with monitoring."""
    X_train, y_train = train_data
    X_val, y_val = val_data
    
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Cross-entropy loss (better than quadratic!)
    criterion = nn.CrossEntropyLoss()
    
    # SGD with weight decay (L2 regularization!)
    optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    history = {'train_acc': [], 'val_acc': [], 'train_loss': [], 'val_loss': []}
    
    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, X_val, y_val, criterion)
        
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        if verbose:
            print(f"Epoch {epoch:3d}: Train {train_acc:.2f}% | Val {val_acc:.2f}%")
    
    return history


# =============================================================================
# EXPERIMENTS
# =============================================================================

def section_header(title: str) -> None:
    """Print a formatted section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70 + "\n")


def experiment_cost_functions(train_data, val_data, device):
    """Compare different loss functions (GPU accelerated)."""
    section_header("EXPERIMENT 1: Cross-Entropy vs MSE (GPU)")
    
    X_train, y_train = train_data
    X_val, y_val = val_data
    
    # Use subset for fair comparison
    X_sub = X_train[:10000]
    y_sub = y_train[:10000]
    
    epochs = 30
    results = {}
    
    # MSE Loss (similar to quadratic cost)
    print("--- Training with MSE Loss (Quadratic) ---")
    model_mse = ImprovedNetwork([784, 30, 10], init_method='scaled').to(device)
    
    train_dataset = TensorDataset(X_sub, y_sub)
    train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
    
    criterion_mse = nn.MSELoss()
    optimizer = optim.SGD(model_mse.parameters(), lr=0.5)
    
    mse_acc = []
    for epoch in range(epochs):
        model_mse.train()
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model_mse(X_batch)
            # Convert labels to one-hot for MSE
            y_onehot = torch.zeros(y_batch.size(0), 10, device=device)
            y_onehot.scatter_(1, y_batch.unsqueeze(1), 1)
            loss = criterion_mse(outputs, y_onehot)
            loss.backward()
            optimizer.step()
        
        # Evaluate
        model_mse.eval()
        with torch.no_grad():
            outputs = model_mse(X_val)
            _, predicted = outputs.max(1)
            acc = 100 * predicted.eq(y_val).sum().item() / y_val.size(0)
        mse_acc.append(acc)
        print(f"Epoch {epoch}: {acc:.2f}%")
    
    results['mse'] = mse_acc
    
    # Cross-Entropy Loss
    print("\n--- Training with Cross-Entropy Loss ---")
    model_ce = ImprovedNetwork([784, 30, 10], init_method='scaled').to(device)
    
    criterion_ce = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model_ce.parameters(), lr=0.5)
    
    ce_acc = []
    for epoch in range(epochs):
        model_ce.train()
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model_ce(X_batch)
            loss = criterion_ce(outputs, y_batch)
            loss.backward()
            optimizer.step()
        
        # Evaluate
        model_ce.eval()
        with torch.no_grad():
            outputs = model_ce(X_val)
            _, predicted = outputs.max(1)
            acc = 100 * predicted.eq(y_val).sum().item() / y_val.size(0)
        ce_acc.append(acc)
        print(f"Epoch {epoch}: {acc:.2f}%")
    
    results['ce'] = ce_acc
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(range(1, epochs+1), mse_acc, 'b-o', label='MSE (Quadratic)', markersize=4)
    ax.plot(range(1, epochs+1), ce_acc, 'g-s', label='Cross-Entropy', markersize=4)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Validation Accuracy (%)', fontsize=12)
    ax.set_title('Cost Function Comparison (GPU Accelerated)', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    fig_path = level3_picture("gpu_cost_comparison")
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"\n✓ Saved: {fig_path}")
    
    return results


def experiment_regularization(train_data, val_data, device):
    """Demonstrate overfitting and regularization (GPU accelerated)."""
    section_header("EXPERIMENT 2: Regularization (GPU)")
    
    X_train, y_train = train_data
    X_val, y_val = val_data
    
    # Small dataset to induce overfitting
    X_small = X_train[:1000]
    y_small = y_train[:1000]
    
    epochs = 30  # Reduced for faster demo
    
    # Without regularization
    print("--- Training WITHOUT regularization ---")
    model_no_reg = ImprovedNetwork([784, 30, 10]).to(device)
    hist_no_reg = train_model(
        model_no_reg, (X_small, y_small), (X_val, y_val),
        epochs=epochs, batch_size=10, lr=0.5, weight_decay=0.0, device=device
    )
    
    # With regularization
    print("\n--- Training WITH regularization (weight_decay=0.0001) ---")
    model_reg = ImprovedNetwork([784, 30, 10]).to(device)
    hist_reg = train_model(
        model_reg, (X_small, y_small), (X_val, y_val),
        epochs=epochs, batch_size=10, lr=0.5, weight_decay=0.0001, device=device
    )
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1 = axes[0]
    ax1.plot(hist_no_reg['train_acc'], 'b-', label='Training', alpha=0.7)
    ax1.plot(hist_no_reg['val_acc'], 'r-', label='Validation')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_title('WITHOUT Regularization\n(Overfitting!)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2 = axes[1]
    ax2.plot(hist_reg['train_acc'], 'b-', label='Training', alpha=0.7)
    ax2.plot(hist_reg['val_acc'], 'g-', label='Validation')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('WITH Regularization\n(Better generalization)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig_path = level3_picture("gpu_regularization")
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"\n✓ Saved: {fig_path}")
    
    return hist_no_reg, hist_reg


def experiment_initialization(train_data, val_data, device):
    """Compare weight initialization methods (GPU accelerated)."""
    section_header("EXPERIMENT 3: Weight Initialization (GPU)")
    
    epochs = 30
    results = {}
    
    for init_name in ['large', 'scaled', 'xavier']:
        print(f"\n--- Training with {init_name.upper()} initialization ---")
        model = ImprovedNetwork([784, 30, 10], init_method=init_name).to(device)
        hist = train_model(
            model, train_data, val_data,
            epochs=epochs, batch_size=10, lr=0.5, weight_decay=0.0001, device=device
        )
        results[init_name] = hist['val_acc']
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {'large': 'red', 'scaled': 'blue', 'xavier': 'green'}
    labels = {'large': 'Large N(0,1)', 'scaled': 'Scaled 1/√n', 'xavier': 'Xavier'}
    
    for name, accs in results.items():
        ax.plot(range(1, epochs+1), accs, color=colors[name], label=labels[name], linewidth=2)
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Validation Accuracy (%)', fontsize=12)
    ax.set_title('Weight Initialization Comparison (GPU)', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    fig_path = level3_picture("gpu_initialization")
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"\n✓ Saved: {fig_path}")
    
    return results


def full_training(train_data, val_data, test_data, device):
    """Train the best network with all improvements."""
    section_header("FULL TRAINING: All Improvements (GPU)")
    
    X_train, y_train = train_data
    X_test, y_test = test_data
    
    print("""
    Training with ALL improvements:
    ✓ Cross-Entropy Loss
    ✓ L2 Regularization (weight_decay)
    ✓ Xavier Initialization
    ✓ Larger hidden layer [784, 100, 10]
    ✓ GPU Acceleration! 🚀
    """)
    
    # Create best model
    model = ImprovedNetwork([784, 100, 10], init_method='xavier').to(device)
    
    # Train
    history = train_model(
        model, train_data, val_data,
        epochs=30, batch_size=32, lr=0.5, weight_decay=0.0001, device=device
    )
    
    # Final test evaluation
    criterion = nn.CrossEntropyLoss()
    test_loss, test_acc = evaluate(model, X_test, y_test, criterion)
    
    print(f"\n🎯 FINAL TEST ACCURACY: {test_acc:.2f}%")
    
    # Plot training progress
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1 = axes[0]
    ax1.plot(history['train_acc'], 'b-', label='Training', alpha=0.7)
    ax1.plot(history['val_acc'], 'g-', label='Validation')
    ax1.axhline(y=test_acc, color='r', linestyle='--', label=f'Test: {test_acc:.1f}%')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_title('Training Progress - Accuracy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2 = axes[1]
    ax2.plot(history['train_loss'], 'b-', label='Training', alpha=0.7)
    ax2.plot(history['val_loss'], 'g-', label='Validation')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.set_title('Training Progress - Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig_path = level3_picture("gpu_full_training")
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"✓ Saved: {fig_path}")
    
    return model, test_acc


def main():
    reset_run_timestamp()
    
    print("\n" + "╔" + "═" * 68 + "╗")
    print("║" + " " * 8 + "LEVEL 3: GPU-ACCELERATED NEURAL NETWORK" + " " * 19 + "║")
    print("╚" + "═" * 68 + "╝")
    
    # Setup
    device = setup_device()
    train_data, val_data, test_data = load_mnist_pytorch(device)
    
    # Run experiments
    import time
    start = time.time()
    
    experiment_cost_functions(train_data, val_data, device)
    experiment_regularization(train_data, val_data, device)
    experiment_initialization(train_data, val_data, device)
    model, accuracy = full_training(train_data, val_data, test_data, device)
    
    elapsed = time.time() - start
    
    print("\n" + "=" * 70)
    print("  LEVEL 3 (GPU) COMPLETE!")
    print("=" * 70)
    print(f"""
    ⏱️  Total time: {elapsed:.1f} seconds
    
    You've learned (with GPU speedup!):
    
    ✅ Cross-Entropy Loss - faster learning
    ✅ L2 Regularization - prevents overfitting  
    ✅ Xavier Initialization - avoids saturation
    
    🎯 Final Accuracy: {accuracy:.2f}%
    
    → Ready for Level 4: CNNs with PyTorch! 🚀
    """)


if __name__ == "__main__":
    main()

