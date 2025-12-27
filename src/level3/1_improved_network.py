"""
Level 3: Improved Neural Network Techniques
============================================

This script explores the improvements in network2.py:
1. Cross-Entropy Cost Function (faster learning)
2. L2 Regularization (prevent overfitting)
3. Better Weight Initialization (avoid vanishing gradients)

Run from src/ directory:
    python level3/1_improved_network.py

Expected improvement: 94.5% → ~97-98% accuracy
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt
import mnist_loader
import network   # Basic network (Level 2)
import network2  # Improved network (Level 3)
from utils import level3_picture, reset_run_timestamp


def section_header(title: str) -> None:
    """Print a formatted section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70 + "\n")


# =============================================================================
# SECTION 1: Cross-Entropy Cost Function
# =============================================================================

def explain_cross_entropy():
    """
    Explain why cross-entropy is better than quadratic cost.
    """
    section_header("SECTION 1: Cross-Entropy Cost Function")
    
    print("""
    ╔════════════════════════════════════════════════════════════════════╗
    ║                 THE LEARNING SLOWDOWN PROBLEM                       ║
    ╠════════════════════════════════════════════════════════════════════╣
    ║                                                                     ║
    ║   With QUADRATIC COST, the gradient contains σ'(z):                ║
    ║                                                                     ║
    ║   ∂C/∂w = (a - y) × σ'(z) × input                                  ║
    ║                      ↑                                              ║
    ║                  PROBLEM!                                           ║
    ║                                                                     ║
    ║   When σ(z) ≈ 0 or σ(z) ≈ 1:                                       ║
    ║   • σ'(z) becomes very small (near 0)                              ║
    ║   • Gradient becomes tiny                                          ║
    ║   • Learning SLOWS DOWN when predictions are very wrong!           ║
    ║                                                                     ║
    ╚════════════════════════════════════════════════════════════════════╝
    
    
    ╔════════════════════════════════════════════════════════════════════╗
    ║                    CROSS-ENTROPY SOLUTION                           ║
    ╠════════════════════════════════════════════════════════════════════╣
    ║                                                                     ║
    ║   Cross-Entropy Cost:                                               ║
    ║   C = -1/n × Σ [y·ln(a) + (1-y)·ln(1-a)]                           ║
    ║                                                                     ║
    ║   Gradient (magic happens!):                                        ║
    ║   ∂C/∂w = (a - y) × input                                          ║
    ║           ↑                                                         ║
    ║        NO σ'(z)!                                                    ║
    ║                                                                     ║
    ║   Benefits:                                                         ║
    ║   • Larger error → Larger gradient → Faster learning               ║
    ║   • No slowdown when predictions are very wrong                    ║
    ║   • Network learns faster from mistakes                            ║
    ║                                                                     ║
    ╚════════════════════════════════════════════════════════════════════╝
    """)
    
    # Visualize sigmoid derivative problem
    z = np.linspace(-6, 6, 100)
    sigmoid = 1 / (1 + np.exp(-z))
    sigmoid_prime = sigmoid * (1 - sigmoid)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Sigmoid and its derivative
    ax1 = axes[0]
    ax1.plot(z, sigmoid, 'b-', linewidth=2, label='σ(z)')
    ax1.plot(z, sigmoid_prime, 'r-', linewidth=2, label="σ'(z)")
    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax1.axhline(y=0.25, color='r', linestyle=':', alpha=0.5, label="max σ'(z) = 0.25")
    ax1.fill_between(z, 0, sigmoid_prime, alpha=0.2, color='red')
    ax1.set_xlabel('z (weighted input)', fontsize=12)
    ax1.set_ylabel('Value', fontsize=12)
    ax1.set_title("The Vanishing Gradient Problem\nσ'(z) is very small at extremes", fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Annotate problem areas
    ax1.annotate('σ\'(z) ≈ 0\nLearning slow!', xy=(-4, 0.02), fontsize=10,
                ha='center', color='red')
    ax1.annotate('σ\'(z) ≈ 0\nLearning slow!', xy=(4, 0.02), fontsize=10,
                ha='center', color='red')
    
    # Plot 2: Compare gradients
    ax2 = axes[1]
    
    # For a wrong prediction where target y=1 but output a is varying
    a_values = np.linspace(0.01, 0.99, 100)
    y = 1  # Target is 1
    
    # Quadratic cost gradient (contains σ' which we approximate)
    # The gradient magnitude is proportional to |a - y| * σ'
    z_approx = np.log(a_values / (1 - a_values))  # Inverse sigmoid
    sigma_prime = a_values * (1 - a_values)
    quadratic_grad = np.abs(a_values - y) * sigma_prime
    
    # Cross-entropy gradient is just |a - y|
    cross_entropy_grad = np.abs(a_values - y)
    
    ax2.plot(a_values, quadratic_grad, 'b-', linewidth=2, label='Quadratic: |a-y| × σ\'(z)')
    ax2.plot(a_values, cross_entropy_grad, 'g-', linewidth=2, label='Cross-Entropy: |a-y|')
    ax2.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Network output a (target y=1)', fontsize=12)
    ax2.set_ylabel('Gradient magnitude', fontsize=12)
    ax2.set_title("Gradient Comparison (target y=1)\nCross-entropy learns faster when wrong!", fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Highlight the key difference
    ax2.annotate('Wrong prediction\n(a ≈ 0)\nQuadratic: slow\nCE: fast!', 
                xy=(0.1, 0.7), fontsize=10, ha='center',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
    
    plt.tight_layout()
    fig_path = level3_picture("cross_entropy_advantage")
    plt.savefig(fig_path, dpi=150)
    plt.close()
    
    print(f"✓ Saved: {fig_path}")


def compare_cost_functions():
    """
    Train networks with quadratic vs cross-entropy cost and compare.
    """
    section_header("EXPERIMENT: Quadratic vs Cross-Entropy Cost")
    
    print("Loading MNIST data...")
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    
    # Reduce training data for faster comparison
    training_subset = training_data[:10000]
    
    print(f"\nUsing {len(training_subset)} training examples for comparison")
    print("Training two networks for 10 epochs each...\n")
    
    epochs = 10
    
    # Network with Quadratic Cost (old way)
    print("--- Training with QUADRATIC COST ---")
    net_quadratic = network2.Network([784, 30, 10], cost=network2.QuadraticCost)
    _, quad_acc, _, _ = net_quadratic.SGD(
        training_subset, epochs, 10, 0.5,
        evaluation_data=validation_data,
        monitor_evaluation_accuracy=True
    )
    
    print("\n--- Training with CROSS-ENTROPY COST ---")
    net_cross_entropy = network2.Network([784, 30, 10], cost=network2.CrossEntropyCost)
    _, ce_acc, _, _ = net_cross_entropy.SGD(
        training_subset, epochs, 10, 0.5,
        evaluation_data=validation_data,
        monitor_evaluation_accuracy=True
    )
    
    # Plot comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    
    epochs_range = range(1, epochs + 1)
    ax.plot(epochs_range, [a/10000*100 for a in quad_acc], 'b-o', 
            linewidth=2, markersize=8, label='Quadratic Cost')
    ax.plot(epochs_range, [a/10000*100 for a in ce_acc], 'g-s', 
            linewidth=2, markersize=8, label='Cross-Entropy Cost')
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Validation Accuracy (%)', fontsize=12)
    ax.set_title('Cost Function Comparison\n(10,000 training examples, same architecture)', fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(epochs_range)
    
    # Add final accuracy annotations
    ax.annotate(f'{quad_acc[-1]/100:.1f}%', xy=(epochs, quad_acc[-1]/100), 
               xytext=(5, 0), textcoords='offset points', fontsize=10, color='blue')
    ax.annotate(f'{ce_acc[-1]/100:.1f}%', xy=(epochs, ce_acc[-1]/100),
               xytext=(5, 0), textcoords='offset points', fontsize=10, color='green')
    
    plt.tight_layout()
    fig_path = level3_picture("cost_comparison")
    plt.savefig(fig_path, dpi=150)
    plt.close()
    
    print(f"\n✓ Saved: {fig_path}")
    print(f"\nResults after {epochs} epochs:")
    print(f"  Quadratic Cost:     {quad_acc[-1]/100:.2f}% accuracy")
    print(f"  Cross-Entropy Cost: {ce_acc[-1]/100:.2f}% accuracy")
    
    return quad_acc, ce_acc


# =============================================================================
# SECTION 2: L2 Regularization
# =============================================================================

def explain_regularization():
    """
    Explain L2 regularization and overfitting.
    """
    section_header("SECTION 2: L2 Regularization (Weight Decay)")
    
    print("""
    ╔════════════════════════════════════════════════════════════════════╗
    ║                     THE OVERFITTING PROBLEM                         ║
    ╠════════════════════════════════════════════════════════════════════╣
    ║                                                                     ║
    ║   Overfitting = Network memorizes training data instead of         ║
    ║                 learning general patterns                           ║
    ║                                                                     ║
    ║   Symptoms:                                                         ║
    ║   • Training accuracy: 99%+ (too good!)                            ║
    ║   • Test accuracy: 95% (much worse)                                ║
    ║   • Network doesn't generalize to new data                         ║
    ║                                                                     ║
    ║   ┌──────────────────────────────────────────────────────────────┐ ║
    ║   │                                                              │ ║
    ║   │  Training    ●●●●●●●●●●   (memorized)                       │ ║
    ║   │  Test        ●●●●●○○○○○   (can't generalize)                │ ║
    ║   │                                                              │ ║
    ║   └──────────────────────────────────────────────────────────────┘ ║
    ║                                                                     ║
    ╚════════════════════════════════════════════════════════════════════╝
    
    
    ╔════════════════════════════════════════════════════════════════════╗
    ║                    L2 REGULARIZATION SOLUTION                       ║
    ╠════════════════════════════════════════════════════════════════════╣
    ║                                                                     ║
    ║   Modified cost function:                                           ║
    ║                                                                     ║
    ║   C = C₀ + (λ/2n) × Σ w²                                           ║
    ║        ↑       ↑                                                    ║
    ║   original  penalty for                                             ║
    ║    cost     large weights                                           ║
    ║                                                                     ║
    ║   Effect:                                                           ║
    ║   • Penalizes large weights                                        ║
    ║   • Forces network to use smaller, distributed weights             ║
    ║   • Prevents any single weight from dominating                     ║
    ║   • Results in smoother, more general solutions                    ║
    ║                                                                     ║
    ║   Weight update becomes:                                            ║
    ║   w → (1 - η×λ/n) × w - η × ∇C₀                                    ║
    ║       ↑                                                             ║
    ║   "weight decay" - weights shrink toward 0                          ║
    ║                                                                     ║
    ║   λ (lambda) controls regularization strength:                      ║
    ║   • λ = 0: no regularization                                       ║
    ║   • λ = 0.1-1: mild regularization                                 ║
    ║   • λ = 5-10: strong regularization                                ║
    ║                                                                     ║
    ╚════════════════════════════════════════════════════════════════════╝
    """)


def demonstrate_overfitting():
    """
    Show overfitting by training on small dataset, then fix with regularization.
    """
    section_header("EXPERIMENT: Overfitting and Regularization")
    
    print("Loading MNIST data...")
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    
    # Use small training set to induce overfitting
    small_training = training_data[:1000]
    
    print(f"\nUsing only {len(small_training)} training examples (to induce overfitting)")
    print("Training for 50 epochs...\n")
    
    epochs = 50  # Reduced for faster demo
    
    # Without regularization (will overfit)
    print("--- Training WITHOUT regularization (λ=0) ---")
    net_no_reg = network2.Network([784, 30, 10], cost=network2.CrossEntropyCost)
    _, val_acc_no_reg, _, train_acc_no_reg = net_no_reg.SGD(
        small_training, epochs, 10, 0.5,
        lmbda=0.0,  # No regularization
        evaluation_data=validation_data,
        monitor_evaluation_accuracy=True,
        monitor_training_accuracy=True
    )
    
    print("\n--- Training WITH regularization (λ=5.0) ---")
    net_reg = network2.Network([784, 30, 10], cost=network2.CrossEntropyCost)
    _, val_acc_reg, _, train_acc_reg = net_reg.SGD(
        small_training, epochs, 10, 0.5,
        lmbda=5.0,  # With regularization
        evaluation_data=validation_data,
        monitor_evaluation_accuracy=True,
        monitor_training_accuracy=True
    )
    
    # Plot comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs_range = range(1, epochs + 1)
    
    # Left plot: Without regularization (overfitting)
    ax1 = axes[0]
    ax1.plot(epochs_range, [a/1000*100 for a in train_acc_no_reg], 'b-', 
            linewidth=2, label='Training (1000 samples)', alpha=0.7)
    ax1.plot(epochs_range, [a/10000*100 for a in val_acc_no_reg], 'r-', 
            linewidth=2, label='Validation (10000 samples)')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Accuracy (%)', fontsize=12)
    ax1.set_title('WITHOUT Regularization (λ=0)\nOverfitting: Training >> Validation', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Highlight the gap
    gap_no_reg = train_acc_no_reg[-1]/10 - val_acc_no_reg[-1]/100
    ax1.annotate(f'Gap: {gap_no_reg:.1f}%', xy=(epochs*0.7, 85), fontsize=12,
                bbox=dict(boxstyle='round', facecolor='red', alpha=0.3))
    
    # Right plot: With regularization
    ax2 = axes[1]
    ax2.plot(epochs_range, [a/1000*100 for a in train_acc_reg], 'b-', 
            linewidth=2, label='Training (1000 samples)', alpha=0.7)
    ax2.plot(epochs_range, [a/10000*100 for a in val_acc_reg], 'g-', 
            linewidth=2, label='Validation (10000 samples)')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('WITH Regularization (λ=5.0)\nSmaller gap = Better generalization', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    gap_reg = train_acc_reg[-1]/10 - val_acc_reg[-1]/100
    ax2.annotate(f'Gap: {gap_reg:.1f}%', xy=(epochs*0.7, 85), fontsize=12,
                bbox=dict(boxstyle='round', facecolor='green', alpha=0.3))
    
    plt.tight_layout()
    fig_path = level3_picture("regularization_effect")
    plt.savefig(fig_path, dpi=150)
    plt.close()
    
    print(f"\n✓ Saved: {fig_path}")
    print(f"\nResults after {epochs} epochs:")
    print(f"  WITHOUT regularization:")
    print(f"    Training:   {train_acc_no_reg[-1]/10:.1f}%")
    print(f"    Validation: {val_acc_no_reg[-1]/100:.1f}%")
    print(f"    Gap: {gap_no_reg:.1f}% (OVERFITTING)")
    print(f"  WITH regularization (λ=5.0):")
    print(f"    Training:   {train_acc_reg[-1]/10:.1f}%")
    print(f"    Validation: {val_acc_reg[-1]/100:.1f}%")
    print(f"    Gap: {gap_reg:.1f}% (better generalization)")


# =============================================================================
# SECTION 3: Better Weight Initialization
# =============================================================================

def explain_weight_initialization():
    """
    Explain why weight initialization matters.
    """
    section_header("SECTION 3: Better Weight Initialization")
    
    print("""
    ╔════════════════════════════════════════════════════════════════════╗
    ║                  THE SATURATION PROBLEM                             ║
    ╠════════════════════════════════════════════════════════════════════╣
    ║                                                                     ║
    ║   With 784 inputs and standard normal weights:                      ║
    ║                                                                     ║
    ║   z = Σ(w_i × x_i) + b                                             ║
    ║                                                                     ║
    ║   If weights ~ N(0, 1), then z has std ≈ √784 ≈ 28                 ║
    ║                                                                     ║
    ║   Result: |z| is often very large (10-30)                          ║
    ║           σ(z) ≈ 0 or 1 (saturated)                                ║
    ║           σ'(z) ≈ 0 (learning frozen!)                             ║
    ║                                                                     ║
    ╚════════════════════════════════════════════════════════════════════╝
    
    
    ╔════════════════════════════════════════════════════════════════════╗
    ║                    THE SOLUTION: 1/√n SCALING                       ║
    ╠════════════════════════════════════════════════════════════════════╣
    ║                                                                     ║
    ║   Initialize weights with:                                          ║
    ║                                                                     ║
    ║   w ~ N(0, 1/√n_in)  where n_in = number of inputs                 ║
    ║                                                                     ║
    ║   For 784 inputs: w ~ N(0, 1/√784) = N(0, 1/28)                    ║
    ║                                                                     ║
    ║   Now z has std ≈ 1, and neurons stay in the                       ║
    ║   "active" region of the sigmoid where σ'(z) is larger.            ║
    ║                                                                     ║
    ║   Code comparison:                                                  ║
    ║   ─────────────────                                                 ║
    ║   # Old (network.py):                                               ║
    ║   weights = np.random.randn(y, x)                                  ║
    ║                                                                     ║
    ║   # New (network2.py):                                              ║
    ║   weights = np.random.randn(y, x) / np.sqrt(x)                     ║
    ║                                                                     ║
    ╚════════════════════════════════════════════════════════════════════╝
    """)


def compare_initializations():
    """
    Compare old vs new weight initialization.
    """
    section_header("EXPERIMENT: Weight Initialization Comparison")
    
    print("Loading MNIST data...")
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    
    print("\nComparing initialization methods on full training set...")
    print("Training for 10 epochs each...\n")
    
    epochs = 10  # Reduced for faster demo
    
    # Old initialization (large weights)
    print("--- Training with LARGE weights (old method) ---")
    net_large = network2.Network([784, 30, 10], cost=network2.CrossEntropyCost)
    net_large.large_weight_initializer()  # Use old method
    _, large_acc, _, _ = net_large.SGD(
        training_data, epochs, 10, 0.5,
        lmbda=5.0,
        evaluation_data=validation_data,
        monitor_evaluation_accuracy=True
    )
    
    print("\n--- Training with 1/√n weights (new method) ---")
    net_small = network2.Network([784, 30, 10], cost=network2.CrossEntropyCost)
    # Uses default_weight_initializer automatically
    _, small_acc, _, _ = net_small.SGD(
        training_data, epochs, 10, 0.5,
        lmbda=5.0,
        evaluation_data=validation_data,
        monitor_evaluation_accuracy=True
    )
    
    # Plot comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    
    epochs_range = range(1, epochs + 1)
    ax.plot(epochs_range, [a/10000*100 for a in large_acc], 'r-o', 
            linewidth=2, markersize=6, label='Large weights (std=1)', alpha=0.8)
    ax.plot(epochs_range, [a/10000*100 for a in small_acc], 'g-s', 
            linewidth=2, markersize=6, label='Small weights (std=1/√n)', alpha=0.8)
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Validation Accuracy (%)', fontsize=12)
    ax.set_title('Weight Initialization Comparison\n(50,000 training examples)', fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig_path = level3_picture("initialization_comparison")
    plt.savefig(fig_path, dpi=150)
    plt.close()
    
    print(f"\n✓ Saved: {fig_path}")
    print(f"\nResults after {epochs} epochs:")
    print(f"  Large weights (std=1):   {large_acc[-1]/100:.2f}%")
    print(f"  Small weights (std=1/√n): {small_acc[-1]/100:.2f}%")
    
    return large_acc, small_acc


# =============================================================================
# SECTION 4: Full Training with All Improvements
# =============================================================================

def train_improved_network():
    """
    Train the best network with all improvements.
    """
    section_header("SECTION 4: Full Training with All Improvements")
    
    print("Loading MNIST data...")
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    
    print("""
    Training with ALL improvements:
    ✓ Cross-Entropy Cost
    ✓ L2 Regularization (λ=5.0)
    ✓ Better weight initialization (1/√n)
    
    Network architecture: [784, 100, 10] (larger hidden layer)
    """)
    
    epochs = 15  # Reduced for faster demo
    
    net = network2.Network([784, 100, 10], cost=network2.CrossEntropyCost)
    eval_cost, eval_acc, train_cost, train_acc = net.SGD(
        training_data, epochs, 10, 0.5,
        lmbda=5.0,
        evaluation_data=test_data,
        monitor_evaluation_accuracy=True,
        monitor_evaluation_cost=True,
        monitor_training_accuracy=True,
        monitor_training_cost=True
    )
    
    # Plot training progress
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs_range = range(1, epochs + 1)
    
    # Accuracy plot
    ax1 = axes[0]
    ax1.plot(epochs_range, [a/50000*100 for a in train_acc], 'b-', 
            linewidth=2, label='Training', alpha=0.7)
    ax1.plot(epochs_range, [a/10000*100 for a in eval_acc], 'g-', 
            linewidth=2, label='Test')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Accuracy (%)', fontsize=12)
    ax1.set_title('Training Progress - Accuracy', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Cost plot
    ax2 = axes[1]
    ax2.plot(epochs_range, train_cost, 'b-', linewidth=2, label='Training', alpha=0.7)
    ax2.plot(epochs_range, eval_cost, 'g-', linewidth=2, label='Test')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Cost', fontsize=12)
    ax2.set_title('Training Progress - Cost', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig_path = level3_picture("full_training")
    plt.savefig(fig_path, dpi=150)
    plt.close()
    
    print(f"\n✓ Saved: {fig_path}")
    
    final_accuracy = eval_acc[-1] / 10000 * 100
    print(f"\n🎯 FINAL TEST ACCURACY: {final_accuracy:.2f}%")
    print(f"   (vs ~94.5% from basic network in Level 2)")
    
    return net, eval_acc


def main():
    # Start a fresh run
    reset_run_timestamp()
    
    print("\n" + "╔" + "═" * 68 + "╗")
    print("║" + " " * 10 + "LEVEL 3: IMPROVED NEURAL NETWORK" + " " * 24 + "║")
    print("╚" + "═" * 68 + "╝")
    
    # Section 1: Cross-Entropy
    explain_cross_entropy()
    compare_cost_functions()
    
    # Section 2: Regularization
    explain_regularization()
    demonstrate_overfitting()
    
    # Section 3: Weight Initialization
    explain_weight_initialization()
    compare_initializations()
    
    # Section 4: Full Training
    net, accuracy = train_improved_network()
    
    print("\n" + "=" * 70)
    print("  LEVEL 3 COMPLETE!")
    print("=" * 70)
    print("""
    You've learned:
    
    ✅ Cross-Entropy Cost
       - No σ'(z) in gradient → faster learning when wrong
    
    ✅ L2 Regularization  
       - Penalizes large weights → prevents overfitting
    
    ✅ Better Initialization
       - Weights ~ N(0, 1/√n) → avoids saturation
    
    📈 Result: ~97-98% accuracy (vs 94.5% from Level 2)
    
    → Ready for Level 4: Convolutional Neural Networks! 🚀
       (Get to 99%+ accuracy)
    """)


if __name__ == "__main__":
    main()

