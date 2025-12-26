"""
Level 1: Foundations - Exploring the MNIST Dataset
===================================================

This script helps you understand the MNIST dataset structure,
how images are represented as vectors, and one-hot encoding.

Run from project root:
    python src/level1/1_explore_mnist.py

Learning Objectives:
    1. Understand the MNIST dataset structure
    2. Learn how images are represented as vectors
    3. Understand one-hot encoding for labels
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
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60 + "\n")


def explore_raw_data() -> tuple:
    """
    Section 1.1: Understanding the raw MNIST data structure.

    The load_data() function returns data in its original format:
    - training_data: tuple of (images, labels)
    - validation_data: tuple of (images, labels)
    - test_data: tuple of (images, labels)
    """
    section_header("1.1 Raw Data Structure (load_data)")

    training_data, validation_data, test_data = mnist_loader.load_data()

    print("MNIST Dataset Overview:")
    print("-" * 40)
    print(f"Training set:   {len(training_data[0]):,} images")
    print(f"Validation set: {len(validation_data[0]):,} images")
    print(f"Test set:       {len(test_data[0]):,} images")
    print(f"Total:          {len(training_data[0]) + len(validation_data[0]) + len(test_data[0]):,} images")

    print("\nData shapes (raw format):")
    print("-" * 40)
    print(f"training_data[0] (images): {training_data[0].shape}")
    print(f"training_data[1] (labels): {training_data[1].shape}")

    print("\nSingle image properties:")
    print("-" * 40)
    single_image = training_data[0][0]
    print(f"Shape: {single_image.shape} (flattened 28×28 = 784 pixels)")
    print(f"Dtype: {single_image.dtype}")
    print(f"Min value: {single_image.min():.4f} (white)")
    print(f"Max value: {single_image.max():.4f} (black)")

    print("\nLabel format (raw):")
    print("-" * 40)
    print(f"First 10 labels: {training_data[1][:10]}")
    print("Labels are simple integers 0-9")

    return training_data, validation_data, test_data


def explore_wrapper_data() -> tuple:
    """
    Section 1.2: Understanding the wrapper data format.

    The load_data_wrapper() function returns data formatted for neural networks:
    - Images reshaped to (784, 1) column vectors
    - Training labels converted to one-hot vectors (10, 1)
    - Validation/test labels remain as integers (for easier accuracy calculation)
    """
    section_header("1.2 Wrapper Data Format (load_data_wrapper)")

    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

    print("Wrapper format (ready for neural network):")
    print("-" * 40)
    print(f"Training examples: {len(training_data):,}")
    print(f"Validation examples: {len(validation_data):,}")
    print(f"Test examples: {len(test_data):,}")

    print("\nEach example is a tuple (x, y):")
    print("-" * 40)
    x, y = training_data[0]
    print(f"x (image) shape: {x.shape}  ← Column vector!")
    print(f"y (label) shape: {y.shape}  ← One-hot encoded!")

    print("\nWhy (784, 1) instead of (784,)?")
    print("-" * 40)
    print("Column vector shape (784, 1) allows matrix multiplication:")
    print("  W @ x + b  where W is (n_hidden, 784)")
    print("  Result: (n_hidden, 1) - output for next layer")

    return training_data, validation_data, test_data


def explain_one_hot_encoding() -> None:
    """
    Section 1.3: Understanding one-hot encoding.

    One-hot encoding converts a digit (0-9) to a vector where:
    - The vector has 10 elements (one for each possible digit)
    - Only the position corresponding to the digit is 1.0
    - All other positions are 0.0
    """
    section_header("1.3 One-Hot Encoding")

    print("What is one-hot encoding?")
    print("-" * 40)
    print("Converts a digit to a 10-element vector")
    print("Only the position matching the digit is 1.0\n")

    print("Examples:")
    print("-" * 40)
    for digit in range(10):
        one_hot = mnist_loader.vectorized_result(digit)
        vector_str = [int(v[0]) for v in one_hot]
        print(f"Digit {digit} → {vector_str}")

    print("\nWhy use one-hot encoding?")
    print("-" * 40)
    print("1. Neural network outputs 10 neurons (one per digit)")
    print("2. Each output neuron represents P(image is digit i)")
    print("3. We can directly compare output with target")
    print("4. Enables cross-entropy loss calculation")

    print("\nCode implementation:")
    print("-" * 40)
    print("""
def vectorized_result(j):
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e
""")


def visualize_digits(training_data: list) -> None:
    """
    Section 1.4: Visualizing MNIST digits.

    Display sample digits from the training set to understand
    what the neural network will learn to recognize.
    """
    section_header("1.4 Visualizing Digits")

    print("Displaying 20 random digits from training set...")
    print("(Check the matplotlib window)\n")

    fig, axes = plt.subplots(2, 10, figsize=(15, 4))
    fig.suptitle("MNIST Handwritten Digits", fontsize=14, fontweight="bold")

    # Get random indices
    indices = np.random.choice(len(training_data), 20, replace=False)

    for i, idx in enumerate(indices):
        x, y = training_data[idx]
        # Reshape from (784, 1) to (28, 28)
        image = x.reshape(28, 28)
        # Get the digit (argmax of one-hot vector)
        digit = np.argmax(y)

        ax = axes[i // 10, i % 10]
        ax.imshow(image, cmap="gray_r")  # gray_r: black digits on white
        ax.set_title(f"{digit}", fontsize=12)
        ax.axis("off")

    plt.tight_layout()
    save_path = level1_picture("sample_digits")
    plt.savefig(save_path, dpi=150)
    print(f"Saved to: {save_path}")
    plt.show()


def visualize_single_digit_detail(training_data: list) -> None:
    """
    Section 1.5: Deep dive into a single digit.

    Look at one digit in detail to understand the pixel values
    and how an image becomes a vector.
    """
    section_header("1.5 Single Digit Deep Dive")

    x, y = training_data[0]
    digit = np.argmax(y)
    image = x.reshape(28, 28)

    print(f"Examining first training example (digit: {digit})")
    print("-" * 40)
    print(f"Vector shape: {x.shape}")
    print(f"Image shape (reshaped): {image.shape}")
    print(f"Total pixels: {x.size}")

    print("\nPixel value statistics:")
    print("-" * 40)
    print(f"Min: {x.min():.4f}")
    print(f"Max: {x.max():.4f}")
    print(f"Mean: {x.mean():.4f}")
    print(f"Non-zero pixels: {np.count_nonzero(x)} / {x.size}")

    # Visualize
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # Original image
    axes[0].imshow(image, cmap="gray_r")
    axes[0].set_title(f"Digit: {digit}", fontsize=12)
    axes[0].axis("off")

    # Pixel values as heatmap with values
    im = axes[1].imshow(image, cmap="Blues")
    axes[1].set_title("Pixel Intensities", fontsize=12)
    plt.colorbar(im, ax=axes[1], fraction=0.046)

    # Histogram of pixel values
    axes[2].hist(x.flatten(), bins=50, color="steelblue", edgecolor="black")
    axes[2].set_xlabel("Pixel Value")
    axes[2].set_ylabel("Count")
    axes[2].set_title("Pixel Value Distribution", fontsize=12)
    axes[2].axvline(x=0, color="red", linestyle="--", label="Background")

    plt.tight_layout()
    save_path = level1_picture("single_digit_analysis")
    plt.savefig(save_path, dpi=150)
    print(f"\nSaved to: {save_path}")
    plt.show()


def visualize_digit_variations(training_data: list) -> None:
    """
    Section 1.6: Variations of the same digit.

    Show multiple examples of the same digit to understand
    the challenge: same digit can look very different!
    """
    section_header("1.6 Digit Variations")

    print("Showing 10 examples of each digit (0-9)")
    print("Notice how much variation exists within each class!\n")

    fig, axes = plt.subplots(10, 10, figsize=(12, 12))
    fig.suptitle("Variations Within Each Digit Class", fontsize=14, fontweight="bold")

    # For each digit 0-9
    for digit in range(10):
        # Find examples of this digit
        examples = [(x, y) for x, y in training_data if np.argmax(y) == digit]
        # Take first 10
        for i in range(10):
            x, y = examples[i]
            image = x.reshape(28, 28)
            ax = axes[digit, i]
            ax.imshow(image, cmap="gray_r")
            ax.axis("off")
            if i == 0:
                ax.set_ylabel(f"{digit}", fontsize=14, rotation=0, labelpad=20)

    plt.tight_layout()
    save_path = level1_picture("digit_variations")
    plt.savefig(save_path, dpi=150)
    print(f"Saved to: {save_path}")
    plt.show()


def summary() -> None:
    """Print a summary of what we learned."""
    section_header("Summary: Level 1 Complete!")

    print("""
What you learned:
-----------------
✓ MNIST has 70,000 images (50k train, 10k validation, 10k test)
✓ Each image is 28×28 = 784 pixels
✓ Pixel values range from 0.0 (white) to 1.0 (black)
✓ Images are stored as (784, 1) column vectors
✓ Labels are one-hot encoded as (10, 1) vectors

Key functions:
--------------
• load_data() → Raw numpy arrays
• load_data_wrapper() → Formatted for neural networks
• vectorized_result(j) → One-hot encoding

Next step:
----------
→ Level 2: Build your first neural network (network.py)
   Run: python src/level2/1_neural_network.py
""")


def main() -> None:
    """Run all Level 1 explorations."""
    # Start a fresh run (creates new folder with timestamp)
    reset_run_timestamp()
    
    print("\n" + "╔" + "═" * 58 + "╗")
    print("║" + " " * 10 + "LEVEL 1: MNIST DATA EXPLORATION" + " " * 16 + "║")
    print("╚" + "═" * 58 + "╝")

    # 1.1 Raw data structure
    raw_train, raw_val, raw_test = explore_raw_data()

    # 1.2 Wrapper format
    training_data, validation_data, test_data = explore_wrapper_data()

    # 1.3 One-hot encoding
    explain_one_hot_encoding()

    # 1.4 Visualize sample digits
    visualize_digits(training_data)

    # 1.5 Single digit deep dive
    visualize_single_digit_detail(training_data)

    # 1.6 Digit variations
    visualize_digit_variations(training_data)

    # Summary
    summary()


if __name__ == "__main__":
    main()

