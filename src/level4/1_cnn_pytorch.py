#!/usr/bin/env python3
"""
Level 4: Convolutional Neural Networks (CNNs) with PyTorch
==========================================================

This script demonstrates CNNs for MNIST digit recognition using modern PyTorch.
We progress from simple architectures to more complex ones, showing the 
improvement in accuracy.

Run from project root:
    cd /home/alcidesticlla/Documents/MOOC/mniels/neural-networks-and-deep-learning/src
    python level4/1_cnn_pytorch.py

Key Concepts Demonstrated:
- Convolutional layers (local receptive fields, weight sharing)
- Max pooling (downsampling, translation invariance)
- ReLU activation (avoiding vanishing gradients)
- Dropout regularization (preventing overfitting)
- Softmax output layer (probability distribution)
- Batch normalization (faster training, better generalization)

Expected Results:
- Simple CNN: ~98.5% accuracy
- Two-layer CNN: ~99.0% accuracy  
- CNN + ReLU + Dropout: ~99.3%+ accuracy
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import time

# Import our utilities
import mnist_loader
from utils import level4_picture, reset_run_timestamp

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🔧 Using device: {device}")
if torch.cuda.is_available():
    print(f"   GPU: {torch.cuda.get_device_name(0)}")


# =============================================================================
# CNN ARCHITECTURES
# =============================================================================

class SimpleCNN(nn.Module):
    """
    Simple CNN: One convolutional layer + one FC layer
    Architecture: Conv(20) → Pool → FC(100) → Output(10)
    Expected accuracy: ~98.5%
    """
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # Convolutional layer: 1 input channel, 20 output channels, 5x5 kernel
        self.conv1 = nn.Conv2d(1, 20, kernel_size=5)  # 28→24
        self.pool = nn.MaxPool2d(2, 2)                 # 24→12
        
        # Fully connected layers
        self.fc1 = nn.Linear(20 * 12 * 12, 100)
        self.fc2 = nn.Linear(100, 10)
    
    def forward(self, x):
        # Reshape flat input to 2D image: (batch, 784) → (batch, 1, 28, 28)
        if x.dim() == 2:
            x = x.view(-1, 1, 28, 28)
        
        # Conv block: Conv → ReLU → Pool
        x = self.pool(F.relu(self.conv1(x)))  # (batch, 20, 12, 12)
        
        # Flatten and FC
        x = x.view(-1, 20 * 12 * 12)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return F.log_softmax(x, dim=1)


class TwoLayerCNN(nn.Module):
    """
    Two-layer CNN: Two convolutional layers + FC
    Architecture: Conv(20) → Pool → Conv(40) → Pool → FC(100) → Output(10)
    Expected accuracy: ~99.0%
    """
    def __init__(self, activation=F.relu):
        super(TwoLayerCNN, self).__init__()
        self.activation = activation
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 20, kernel_size=5)   # 28→24
        self.conv2 = nn.Conv2d(20, 40, kernel_size=5)  # 12→8
        self.pool = nn.MaxPool2d(2, 2)                  # Halves dimensions
        
        # Fully connected
        self.fc1 = nn.Linear(40 * 4 * 4, 100)
        self.fc2 = nn.Linear(100, 10)
    
    def forward(self, x):
        if x.dim() == 2:
            x = x.view(-1, 1, 28, 28)
        
        # Conv block 1: 28→24→12
        x = self.pool(self.activation(self.conv1(x)))
        
        # Conv block 2: 12→8→4
        x = self.pool(self.activation(self.conv2(x)))
        
        # Flatten and FC
        x = x.view(-1, 40 * 4 * 4)
        x = self.activation(self.fc1(x))
        x = self.fc2(x)
        
        return F.log_softmax(x, dim=1)


class AdvancedCNN(nn.Module):
    """
    Advanced CNN: Two conv layers + Dropout + Batch Normalization
    Architecture: Conv(20) → BN → Pool → Conv(40) → BN → Pool → FC(100) → Dropout → Output(10)
    Expected accuracy: ~99.3%+
    """
    def __init__(self, dropout_rate=0.5):
        super(AdvancedCNN, self).__init__()
        
        # Convolutional layers with batch normalization
        self.conv1 = nn.Conv2d(1, 20, kernel_size=5)
        self.bn1 = nn.BatchNorm2d(20)
        
        self.conv2 = nn.Conv2d(20, 40, kernel_size=5)
        self.bn2 = nn.BatchNorm2d(40)
        
        self.pool = nn.MaxPool2d(2, 2)
        
        # Fully connected layers with dropout
        self.fc1 = nn.Linear(40 * 4 * 4, 100)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(100, 10)
    
    def forward(self, x):
        if x.dim() == 2:
            x = x.view(-1, 1, 28, 28)
        
        # Conv block 1 with batch norm
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        
        # Conv block 2 with batch norm
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        
        # FC with dropout
        x = x.view(-1, 40 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return F.log_softmax(x, dim=1)


# =============================================================================
# DATA LOADING
# =============================================================================

def load_mnist_pytorch():
    """Load MNIST data and convert to PyTorch format."""
    print("📊 Loading MNIST data...")
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    
    # Convert to numpy arrays
    train_x = np.array([x.flatten() for x, _ in training_data]).astype(np.float32)
    train_y = np.array([np.argmax(y) for _, y in training_data]).astype(np.int64)
    
    test_x = np.array([x.flatten() for x, _ in test_data]).astype(np.float32)
    test_y = np.array([y for _, y in test_data]).astype(np.int64)
    
    # Convert to tensors
    train_x = torch.from_numpy(train_x)
    train_y = torch.from_numpy(train_y)
    test_x = torch.from_numpy(test_x)
    test_y = torch.from_numpy(test_y)
    
    # Create data loaders
    train_dataset = TensorDataset(train_x, train_y)
    test_dataset = TensorDataset(test_x, test_y)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
    
    print(f"   Training samples: {len(train_x)}")
    print(f"   Test samples: {len(test_x)}")
    
    return train_loader, test_loader


# =============================================================================
# TRAINING AND EVALUATION
# =============================================================================

def train_epoch(model, device, train_loader, optimizer):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)
    
    return total_loss / len(train_loader), 100. * correct / total


def evaluate(model, device, test_loader):
    """Evaluate model on test set."""
    model.eval()
    test_loss = 0
    correct = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    
    return test_loss, accuracy


def train_model(model, train_loader, test_loader, epochs=10, lr=0.01, name="CNN"):
    """Train a model and track progress."""
    print(f"\n{'='*60}")
    print(f"Training: {name}")
    print(f"{'='*60}")
    
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    train_losses = []
    test_accuracies = []
    
    start_time = time.time()
    
    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(model, device, train_loader, optimizer)
        test_loss, test_acc = evaluate(model, device, test_loader)
        
        train_losses.append(train_loss)
        test_accuracies.append(test_acc)
        
        print(f"Epoch {epoch+1:2d}/{epochs}: Train Loss={train_loss:.4f}, "
              f"Train Acc={train_acc:.2f}%, Test Acc={test_acc:.2f}%")
    
    elapsed = time.time() - start_time
    print(f"\n⏱️  Total training time: {elapsed:.1f}s")
    print(f"🎯 Final test accuracy: {test_accuracies[-1]:.2f}%")
    
    return train_losses, test_accuracies


# =============================================================================
# VISUALIZATION
# =============================================================================

def visualize_filters(model, save_path):
    """Visualize the learned convolutional filters."""
    # Get first conv layer weights
    conv1_weights = model.conv1.weight.data.cpu().numpy()
    
    n_filters = conv1_weights.shape[0]
    n_cols = 5
    n_rows = (n_filters + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 10))
    fig.suptitle('Learned Convolutional Filters (Layer 1)\n'
                 'Each filter detects a specific feature pattern', 
                 fontsize=14, fontweight='bold')
    
    for i in range(n_rows * n_cols):
        ax = axes.flat[i]
        if i < n_filters:
            # Show filter
            ax.imshow(conv1_weights[i, 0], cmap='RdBu', vmin=-0.5, vmax=0.5)
            ax.set_title(f'Filter {i+1}', fontsize=9)
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"📸 Saved filters visualization to {save_path}")


def visualize_feature_maps(model, sample_image, save_path):
    """Visualize feature maps for a sample image."""
    model.eval()
    
    # Prepare input
    x = torch.from_numpy(sample_image.astype(np.float32)).view(1, 1, 28, 28).to(device)
    
    # Get activations after each conv layer
    with torch.no_grad():
        # After conv1
        conv1_out = F.relu(model.conv1(x))
        pool1_out = model.pool(conv1_out)
        
        # After conv2 (if exists)
        if hasattr(model, 'conv2'):
            conv2_out = F.relu(model.conv2(pool1_out))
            pool2_out = model.pool(conv2_out)
    
    # Plot
    fig = plt.figure(figsize=(16, 10))
    
    # Original image
    ax = fig.add_subplot(3, 1, 1)
    ax.imshow(sample_image.reshape(28, 28), cmap='gray')
    ax.set_title('Original Input Image (28×28)', fontsize=12, fontweight='bold')
    ax.axis('off')
    
    # Conv1 feature maps (show first 10)
    for i in range(min(10, conv1_out.shape[1])):
        ax = fig.add_subplot(3, 10, 11 + i)
        ax.imshow(pool1_out[0, i].cpu().numpy(), cmap='viridis')
        ax.set_title(f'{i+1}', fontsize=8)
        ax.axis('off')
    fig.text(0.5, 0.62, 'After Conv1 + Pool (12×12 × 20 feature maps, showing first 10)', 
             ha='center', fontsize=11, fontweight='bold')
    
    # Conv2 feature maps (if exists)
    if hasattr(model, 'conv2'):
        for i in range(min(10, pool2_out.shape[1])):
            ax = fig.add_subplot(3, 10, 21 + i)
            ax.imshow(pool2_out[0, i].cpu().numpy(), cmap='viridis')
            ax.set_title(f'{i+1}', fontsize=8)
            ax.axis('off')
        fig.text(0.5, 0.30, 'After Conv2 + Pool (4×4 × 40 feature maps, showing first 10)', 
                 ha='center', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"📸 Saved feature maps to {save_path}")


def plot_comparison(results, save_path):
    """Plot accuracy comparison between different architectures."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6']
    
    # Training curves
    ax1.set_title('Test Accuracy During Training', fontsize=12, fontweight='bold')
    for i, (name, (_, accs)) in enumerate(results.items()):
        ax1.plot(range(1, len(accs)+1), accs, 
                marker='o', markersize=4, 
                color=colors[i % len(colors)],
                label=f'{name}', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Test Accuracy (%)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(95, 100)
    
    # Final accuracy bar chart
    ax2.set_title('Final Test Accuracy Comparison', fontsize=12, fontweight='bold')
    names = list(results.keys())
    final_accs = [results[name][1][-1] for name in names]
    bars = ax2.bar(range(len(names)), final_accs, color=colors[:len(names)])
    ax2.set_xticks(range(len(names)))
    ax2.set_xticklabels(names, rotation=15, ha='right')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_ylim(97, 100)
    
    # Add value labels on bars
    for bar, acc in zip(bars, final_accs):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{acc:.2f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"📸 Saved comparison plot to {save_path}")


def save_model(model, path):
    """Save trained model."""
    torch.save(model.state_dict(), path)
    print(f"💾 Model saved to {path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main execution."""
    print("=" * 70)
    print("  LEVEL 4: CONVOLUTIONAL NEURAL NETWORKS (CNNs)")
    print("  Modern PyTorch Implementation")
    print("=" * 70)
    
    reset_run_timestamp()
    
    # Load data
    train_loader, test_loader = load_mnist_pytorch()
    
    # Store results for comparison
    results = {}
    
    # ==========================================================================
    # EXPERIMENT 1: Simple CNN (One conv layer)
    # ==========================================================================
    print("\n" + "═" * 70)
    print("EXPERIMENT 1: Simple CNN (One Convolutional Layer)")
    print("Architecture: Input → Conv(20,5×5) → Pool → FC(100) → Output(10)")
    print("═" * 70)
    
    simple_cnn = SimpleCNN()
    losses, accs = train_model(simple_cnn, train_loader, test_loader, 
                               epochs=10, lr=0.001, name="Simple CNN")
    results["Simple CNN\n(1 conv)"] = (losses, accs)
    
    # Visualize filters
    visualize_filters(simple_cnn, level4_picture("simple_cnn_filters"))
    
    # ==========================================================================
    # EXPERIMENT 2: Two-layer CNN
    # ==========================================================================
    print("\n" + "═" * 70)
    print("EXPERIMENT 2: Two-Layer CNN")
    print("Architecture: Input → Conv(20) → Pool → Conv(40) → Pool → FC(100) → Output")
    print("═" * 70)
    
    two_layer_cnn = TwoLayerCNN()
    losses, accs = train_model(two_layer_cnn, train_loader, test_loader,
                               epochs=10, lr=0.001, name="Two-Layer CNN")
    results["Two-Layer CNN\n(2 conv)"] = (losses, accs)
    
    # Visualize feature maps
    sample_x, _ = next(iter(test_loader))
    visualize_feature_maps(two_layer_cnn, sample_x[0].numpy(), 
                          level4_picture("feature_maps"))
    
    # ==========================================================================
    # EXPERIMENT 3: Advanced CNN with Dropout + BatchNorm
    # ==========================================================================
    print("\n" + "═" * 70)
    print("EXPERIMENT 3: Advanced CNN (BatchNorm + Dropout)")
    print("Architecture: Conv → BN → Pool → Conv → BN → Pool → FC → Dropout → Out")
    print("═" * 70)
    
    advanced_cnn = AdvancedCNN(dropout_rate=0.5)
    losses, accs = train_model(advanced_cnn, train_loader, test_loader,
                               epochs=15, lr=0.001, name="Advanced CNN")
    results["Advanced CNN\n(BN+Dropout)"] = (losses, accs)
    
    # Visualize filters
    visualize_filters(advanced_cnn, level4_picture("advanced_cnn_filters"))
    
    # ==========================================================================
    # COMPARISON
    # ==========================================================================
    print("\n" + "═" * 70)
    print("RESULTS COMPARISON")
    print("═" * 70)
    
    plot_comparison(results, level4_picture("cnn_comparison"))
    
    print("\n📊 Summary:")
    print("-" * 50)
    for name, (_, accs) in results.items():
        clean_name = name.replace('\n', ' ')
        print(f"  {clean_name:30s}: {accs[-1]:.2f}%")
    
    # Save best model
    models_dir = Path(__file__).parent.parent.parent / "models"
    models_dir.mkdir(exist_ok=True)
    save_model(advanced_cnn, models_dir / "level4_cnn.pt")
    
    print("\n" + "═" * 70)
    print("✅ Level 4 CNN training complete!")
    print("   Key improvements over Level 3:")
    print("   • Convolutional layers exploit spatial structure")
    print("   • Max pooling adds translation invariance")
    print("   • BatchNorm accelerates training")
    print("   • Dropout prevents overfitting")
    print("═" * 70)


if __name__ == "__main__":
    main()

