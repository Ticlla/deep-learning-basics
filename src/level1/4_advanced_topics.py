"""
Level 1: Advanced Topics - Why Column Vectors & Data Augmentation
==================================================================

This script covers:
1. Why we use (784, 1) column vectors instead of (784,) flat arrays
2. Data augmentation - how to expand the training dataset
3. Preview of baseline classifiers (SVM, darkness-based)

Run from project root:
    python src/level1/4_advanced_topics.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt
import mnist_loader
from utils import level1_picture, reset_run_timestamp


def section_header(title: str) -> None:
    """Print a formatted section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70 + "\n")


# =============================================================================
# TOPIC 1: Why Column Vectors (784, 1) instead of (784,)?
# =============================================================================

def explain_column_vectors():
    """
    Demonstrate why neural networks use column vectors (n, 1)
    instead of flat arrays (n,).
    """
    section_header("TOPIC 1: Why Column Vectors?")
    
    print("""
    In neural networks, we perform matrix multiplication:
    
        z = W · a + b
    
    Where:
        W = weights matrix (neurons_out × neurons_in)
        a = activations (column vector)
        b = biases (column vector)
        z = weighted input (column vector)
    
    Let's see why shape matters...
    """)
    
    # Example: 3 input neurons → 2 output neurons
    print("\n--- Example: 3 inputs → 2 outputs ---\n")
    
    # Weights: 2 output neurons, each connected to 3 inputs
    W = np.array([
        [0.1, 0.2, 0.3],  # weights to neuron 0
        [0.4, 0.5, 0.6]   # weights to neuron 1
    ])
    print(f"Weights W shape: {W.shape}  (2 outputs × 3 inputs)")
    print(f"W = \n{W}\n")
    
    # Biases: one per output neuron
    b = np.array([[0.1], [0.2]])  # Column vector!
    print(f"Biases b shape: {b.shape}  (2 outputs × 1)")
    print(f"b = \n{b}\n")
    
    # Input as FLAT array (784,) - PROBLEM!
    a_flat = np.array([1.0, 2.0, 3.0])
    print(f"Input a_flat shape: {a_flat.shape}  ← FLAT array")
    
    try:
        z_flat = np.dot(W, a_flat) + b
        print(f"Result z shape: {z_flat.shape}")
        print(f"z = \n{z_flat}")
        print("\n⚠️  This works but produces shape (2, 2) instead of (2, 1)!")
        print("    The bias addition broadcasts incorrectly.\n")
    except Exception as e:
        print(f"ERROR: {e}\n")
    
    # Input as COLUMN vector (784, 1) - CORRECT!
    a_col = np.array([[1.0], [2.0], [3.0]])  # Column vector!
    print(f"Input a_col shape: {a_col.shape}  ← COLUMN vector")
    
    z_col = np.dot(W, a_col) + b
    print(f"Result z shape: {z_col.shape}  ← Correct!")
    print(f"z = \n{z_col}\n")
    
    print("""
    ✅ KEY INSIGHT:
    ───────────────
    Using (n, 1) column vectors ensures:
    
    1. Matrix dimensions align correctly: (m×n) · (n×1) = (m×1)
    2. Bias addition is unambiguous: (m×1) + (m×1) = (m×1)
    3. Consistent shapes throughout the network
    4. Batch processing: (m×n) · (n×batch) = (m×batch)
    """)
    
    # Show real MNIST example
    print("\n--- Real MNIST Example ---\n")
    
    training_data, _, _ = mnist_loader.load_data_wrapper()
    x, y = training_data[0]
    
    print(f"Image x shape: {x.shape}  ← Column vector (784 × 1)")
    print(f"Label y shape: {y.shape}  ← Column vector (10 × 1)")
    print(f"\nThis allows: W · x + b where W is (hidden_neurons × 784)")
    
    # Simulate first layer computation
    hidden_neurons = 30
    W1 = np.random.randn(hidden_neurons, 784) * 0.1
    b1 = np.random.randn(hidden_neurons, 1) * 0.1
    
    z1 = np.dot(W1, x) + b1
    print(f"\nFirst layer: W1·x + b1")
    print(f"  W1 shape: {W1.shape}")
    print(f"  x shape:  {x.shape}")
    print(f"  b1 shape: {b1.shape}")
    print(f"  z1 shape: {z1.shape}  ← Perfect (30 × 1)!")


# =============================================================================
# TOPIC 2: Data Augmentation
# =============================================================================

def explain_data_augmentation():
    """
    Demonstrate data augmentation techniques to expand training data.
    """
    section_header("TOPIC 2: Data Augmentation")
    
    print("""
    Data augmentation creates MORE training examples from existing ones.
    This helps the network generalize better and reduces overfitting.
    
    Common techniques for MNIST:
    1. Translation (shift image up/down/left/right)
    2. Rotation (small angles)
    3. Scaling (zoom in/out)
    4. Elastic deformation
    5. Adding noise
    """)
    
    # Load one image
    training_data, _, _ = mnist_loader.load_data()
    original = training_data[0][0].reshape(28, 28)
    label = training_data[1][0]
    
    print(f"\nOriginal image: digit {label}")
    print(f"Shape: {original.shape}")
    
    # Create augmented versions
    augmented = []
    titles = []
    
    # 1. Original
    augmented.append(original)
    titles.append(f"Original\n(digit {label})")
    
    # 2. Shift up
    shifted_up = np.roll(original, -2, axis=0)
    shifted_up[-2:, :] = 0  # Clear bottom rows
    augmented.append(shifted_up)
    titles.append("Shift Up\n(2 pixels)")
    
    # 3. Shift down
    shifted_down = np.roll(original, 2, axis=0)
    shifted_down[:2, :] = 0  # Clear top rows
    augmented.append(shifted_down)
    titles.append("Shift Down\n(2 pixels)")
    
    # 4. Shift left
    shifted_left = np.roll(original, -2, axis=1)
    shifted_left[:, -2:] = 0
    augmented.append(shifted_left)
    titles.append("Shift Left\n(2 pixels)")
    
    # 5. Shift right
    shifted_right = np.roll(original, 2, axis=1)
    shifted_right[:, :2] = 0
    augmented.append(shifted_right)
    titles.append("Shift Right\n(2 pixels)")
    
    # 6. Add noise
    noisy = original + np.random.normal(0, 0.1, original.shape)
    noisy = np.clip(noisy, 0, 1)
    augmented.append(noisy)
    titles.append("Add Noise\n(Gaussian)")
    
    # 7. Rotate slightly (simple approximation)
    # Using scipy would be better, but let's keep dependencies minimal
    rotated = np.rot90(original)  # 90 degrees for demonstration
    augmented.append(rotated)
    titles.append("Rotate 90°\n(demo only)")
    
    # 8. Invert intensity slightly
    inverted = 1.0 - original
    inverted = np.where(original > 0.1, original * 0.8, original)
    augmented.append(inverted)
    titles.append("Darken\n(80%)")
    
    # Save visualization
    fig, axes = plt.subplots(2, 4, figsize=(12, 7))
    fig.suptitle("Data Augmentation: 1 Image → 8 Training Examples", fontsize=14)
    
    for ax, img, title in zip(axes.flatten(), augmented, titles):
        ax.imshow(img, cmap='gray')
        ax.set_title(title, fontsize=10)
        ax.axis('off')
    
    plt.tight_layout()
    fig_path = level1_picture("data_augmentation")
    plt.savefig(fig_path, dpi=150)
    plt.close()
    
    print(f"\n✓ Saved augmentation examples to: {fig_path}")
    
    print("""
    
    📈 IMPACT OF DATA AUGMENTATION:
    ──────────────────────────────
    
    Original dataset:  50,000 images
    With 5x augment:  250,000 images (shifts only)
    
    This is used in `expand_mnist.py` which creates 250,000 
    training images by shifting each original 1 pixel in 
    each direction (up, down, left, right).
    
    Result: Better generalization, ~0.5% accuracy improvement!
    """)
    
    return augmented


def show_expand_mnist_preview():
    """
    Preview what expand_mnist.py does.
    """
    print("\n--- expand_mnist.py Preview ---\n")
    
    print("""
    The file `src/expand_mnist.py` expands 50,000 → 250,000 images:
    
    For each original image, create 4 shifted versions:
    
        Original    Shift ↑    Shift ↓    Shift ←    Shift →
           ○          ○          ○          ○          ○
         ╱│╲         ╱│╲         ╱│╲         ╱│╲         ╱│╲
        ○ ○ ○       ○ ○ ○       ○ ○ ○       ○ ○ ○       ○ ○ ○
    
    Total: 1 + 4 = 5 images per original = 250,000 images
    
    To generate expanded data:
        python expand_mnist.py
    
    This creates: data/mnist_expanded.pkl.gz
    """)


# =============================================================================
# TOPIC 3: Baseline Classifiers Preview
# =============================================================================

def preview_baseline_classifiers():
    """
    Preview non-neural network approaches for comparison.
    """
    section_header("TOPIC 3: Baseline Classifiers (Preview)")
    
    print("""
    Before neural networks, how would you classify digits?
    
    This project includes two baseline approaches:
    
    ┌─────────────────────────────────────────────────────────────────┐
    │  BASELINE 1: Average Darkness (mnist_average_darkness.py)       │
    ├─────────────────────────────────────────────────────────────────┤
    │                                                                 │
    │  Idea: Classify by how "dark" (ink-filled) the image is        │
    │                                                                 │
    │  digit 0: 0.173 avg brightness (lots of ink - circular)        │
    │  digit 1: 0.076 avg brightness (little ink - just a line)      │
    │                                                                 │
    │  Accuracy: ~22% (barely better than random 10%)                │
    │                                                                 │
    └─────────────────────────────────────────────────────────────────┘
    
    ┌─────────────────────────────────────────────────────────────────┐
    │  BASELINE 2: Support Vector Machine (mnist_svm.py)              │
    ├─────────────────────────────────────────────────────────────────┤
    │                                                                 │
    │  Idea: Find hyperplanes that separate classes in 784D space    │
    │                                                                 │
    │  Uses scikit-learn's SVC with RBF kernel                       │
    │                                                                 │
    │  Accuracy: ~98.5% (very good! but slower to train)             │
    │                                                                 │
    └─────────────────────────────────────────────────────────────────┘
    
    ┌─────────────────────────────────────────────────────────────────┐
    │  OUR GOAL: Neural Network (network.py)                          │
    ├─────────────────────────────────────────────────────────────────┤
    │                                                                 │
    │  Simple network [784, 30, 10]:     ~95% accuracy               │
    │  Better network [784, 100, 10]:    ~97% accuracy               │
    │  With improvements (network2.py):   ~98% accuracy               │
    │  CNN (network3.py):                 ~99%+ accuracy              │
    │                                                                 │
    └─────────────────────────────────────────────────────────────────┘
    """)
    
    # Quick demonstration of darkness-based classification idea
    print("\n--- Quick Darkness Analysis ---\n")
    
    training_data, _, _ = mnist_loader.load_data()
    images = training_data[0]
    labels = training_data[1]
    
    print("Average pixel brightness by digit:")
    print("-" * 40)
    
    brightness_by_digit = {}
    for digit in range(10):
        mask = labels == digit
        avg_brightness = images[mask].mean()
        brightness_by_digit[digit] = avg_brightness
        bar = "█" * int(avg_brightness * 100)
        print(f"  Digit {digit}: {avg_brightness:.4f} {bar}")
    
    print("\n" + "-" * 40)
    darkest = max(brightness_by_digit, key=brightness_by_digit.get)
    lightest = min(brightness_by_digit, key=brightness_by_digit.get)
    print(f"  Darkest digit:  {darkest} (most ink)")
    print(f"  Lightest digit: {lightest} (least ink)")
    
    print("""
    
    🤔 Can we classify just by brightness? Not very well!
    
    Many digits have similar brightness (2,3,5,6,8,9 are all close).
    This is why we need neural networks - they learn FEATURES,
    not just simple statistics.
    """)


def main():
    # Start a fresh run (creates new folder with timestamp)
    reset_run_timestamp()
    
    print("\n" + "╔" + "═" * 68 + "╗")
    print("║" + " " * 12 + "LEVEL 1: ADVANCED TOPICS" + " " * 31 + "║")
    print("╚" + "═" * 68 + "╝")
    
    # Topic 1: Column Vectors
    explain_column_vectors()
    
    # Topic 2: Data Augmentation
    explain_data_augmentation()
    show_expand_mnist_preview()
    
    # Topic 3: Baseline Classifiers
    preview_baseline_classifiers()
    
    print("\n" + "=" * 70)
    print("  LEVEL 1 ADVANCED TOPICS COMPLETE!")
    print("=" * 70)
    print("""
    Key Takeaways:
    
    1. COLUMN VECTORS (784,1):
       - Required for correct matrix multiplication
       - Ensures W·a + b works with proper dimensions
       - Consistent shapes through all layers
    
    2. DATA AUGMENTATION:
       - Creates more training data from existing images
       - Simple shifts can 5x your dataset
       - Improves generalization significantly
    
    3. BASELINES:
       - Darkness alone: ~22% accuracy
       - SVM: ~98.5% accuracy
       - Neural networks: 95-99%+ accuracy
    
    → Ready for Level 2: Building the Neural Network! 🧠
    """)


if __name__ == "__main__":
    main()

