"""
Level 1: Hands-On Exercises
============================

Complete these exercises to reinforce your understanding of MNIST data.
Each exercise has:
- A description of what to do
- Hints if you get stuck
- A solution (try not to peek!)

Run from project root:
    python src/level1/5_exercises.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt
import mnist_loader
from utils import level1_picture, reset_run_timestamp


def exercise_header(num: int, title: str) -> None:
    """Print exercise header."""
    print("\n" + "─" * 70)
    print(f"  EXERCISE {num}: {title}")
    print("─" * 70 + "\n")


# =============================================================================
# EXERCISE 1: Count digits
# =============================================================================

def exercise_1_count_digits():
    """
    Exercise 1: Count how many images of each digit are in the TRAINING set.
    
    Goal: Use numpy to count occurrences of each digit (0-9)
    Expected: A dictionary or array with counts for digits 0-9
    """
    exercise_header(1, "Count Digits in Training Set")
    
    print("TASK: Count how many images of each digit (0-9) are in training data.")
    print("HINT: Use mnist_loader.load_data() to get labels, then np.bincount()")
    print("\n" + "-" * 50)
    
    # YOUR CODE HERE (uncomment and modify):
    # training_data, _, _ = mnist_loader.load_data()
    # labels = training_data[1]
    # counts = ???
    # print(counts)
    
    # SOLUTION:
    print("\n📖 SOLUTION:")
    print("-" * 50)
    
    training_data, _, _ = mnist_loader.load_data()
    labels = training_data[1]
    counts = np.bincount(labels)
    
    print("counts = np.bincount(labels)")
    print(f"\nResult: {counts}")
    print(f"\nTotal: {counts.sum()} images")
    
    print("\nDetailed breakdown:")
    for digit, count in enumerate(counts):
        print(f"  Digit {digit}: {count:,} images")
    
    return counts


# =============================================================================
# EXERCISE 2: Find specific digits
# =============================================================================

def exercise_2_find_digit():
    """
    Exercise 2: Find the FIRST occurrence of digit 7 in the training set.
    
    Goal: Find the index of the first "7" in the labels array
    """
    exercise_header(2, "Find First Occurrence of Digit 7")
    
    print("TASK: Find the INDEX of the first image labeled '7' in training data.")
    print("HINT: Use np.where() or np.argmax() with a boolean mask")
    print("\n" + "-" * 50)
    
    # YOUR CODE HERE:
    # training_data, _, _ = mnist_loader.load_data()
    # labels = training_data[1]
    # first_seven_idx = ???
    # print(f"First 7 is at index: {first_seven_idx}")
    
    # SOLUTION:
    print("\n📖 SOLUTION:")
    print("-" * 50)
    
    training_data, _, _ = mnist_loader.load_data()
    labels = training_data[1]
    
    # Method 1: np.where
    indices = np.where(labels == 7)[0]
    first_seven_idx = indices[0]
    
    # Method 2: np.argmax (finds first True)
    first_seven_idx_alt = np.argmax(labels == 7)
    
    print(f"Method 1 (np.where):  first_seven_idx = np.where(labels == 7)[0][0]")
    print(f"Method 2 (np.argmax): first_seven_idx = np.argmax(labels == 7)")
    print(f"\nFirst '7' is at index: {first_seven_idx}")
    print(f"Verification: labels[{first_seven_idx}] = {labels[first_seven_idx]}")
    
    # Show the image
    images = training_data[0]
    plt.figure(figsize=(4, 4))
    plt.imshow(images[first_seven_idx].reshape(28, 28), cmap='gray')
    plt.title(f"First '7' in training set (index {first_seven_idx})")
    plt.axis('off')
    
    fig_path = level1_picture("exercise2_first_seven")
    plt.savefig(fig_path, dpi=100)
    plt.close()
    print(f"\n✓ Saved image to: {fig_path}")
    
    return first_seven_idx


# =============================================================================
# EXERCISE 3: Compute mean image
# =============================================================================

def exercise_3_mean_image():
    """
    Exercise 3: Compute the MEAN IMAGE of all training images.
    
    Goal: Average all 50,000 images into one "mean" image
    """
    exercise_header(3, "Compute Mean Image of All Training Data")
    
    print("TASK: Compute the average of all 50,000 training images.")
    print("HINT: Use images.mean(axis=0) to average across first axis")
    print("\n" + "-" * 50)
    
    # YOUR CODE HERE:
    # training_data, _, _ = mnist_loader.load_data()
    # images = training_data[0]  # Shape: (50000, 784)
    # mean_image = ???  # Should be shape (784,)
    # mean_image_2d = mean_image.reshape(28, 28)
    
    # SOLUTION:
    print("\n📖 SOLUTION:")
    print("-" * 50)
    
    training_data, _, _ = mnist_loader.load_data()
    images = training_data[0]
    
    mean_image = images.mean(axis=0)  # Average across all 50,000 images
    mean_image_2d = mean_image.reshape(28, 28)
    
    print(f"mean_image = images.mean(axis=0)")
    print(f"mean_image shape: {mean_image.shape}")
    print(f"mean_image_2d shape: {mean_image_2d.shape}")
    
    # Visualize
    plt.figure(figsize=(5, 5))
    plt.imshow(mean_image_2d, cmap='hot')
    plt.title("Mean Image of All 50,000 Training Images")
    plt.colorbar(label='Pixel intensity')
    plt.axis('off')
    
    fig_path = level1_picture("exercise3_mean_image")
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"\n✓ Saved image to: {fig_path}")
    
    print("\nThis 'ghost' image shows where digits typically have ink!")
    
    return mean_image


# =============================================================================
# EXERCISE 4: One-hot encoding
# =============================================================================

def exercise_4_one_hot():
    """
    Exercise 4: Implement your own one-hot encoding function.
    
    Goal: Convert a digit (0-9) to a one-hot vector of shape (10, 1)
    """
    exercise_header(4, "Implement One-Hot Encoding")
    
    print("TASK: Write a function that converts a digit to a one-hot vector.")
    print("      Example: 3 → [[0], [0], [0], [1], [0], [0], [0], [0], [0], [0]]")
    print("HINT: Create a zeros array, then set the appropriate index to 1")
    print("\n" + "-" * 50)
    
    # YOUR CODE HERE:
    # def my_one_hot(digit):
    #     """Convert digit (0-9) to one-hot vector of shape (10, 1)."""
    #     ???
    #     return result
    
    # SOLUTION:
    print("\n📖 SOLUTION:")
    print("-" * 50)
    
    def my_one_hot(digit):
        """Convert digit (0-9) to one-hot vector of shape (10, 1)."""
        result = np.zeros((10, 1))
        result[digit] = 1.0
        return result
    
    print("""
    def my_one_hot(digit):
        result = np.zeros((10, 1))
        result[digit] = 1.0
        return result
    """)
    
    # Test it
    print("Testing:")
    for d in [0, 3, 7, 9]:
        one_hot = my_one_hot(d)
        print(f"  my_one_hot({d}) = {one_hot.flatten()}")
    
    # Compare with mnist_loader's version
    print("\nCompare with mnist_loader.vectorized_result:")
    for d in [0, 3, 7, 9]:
        original = mnist_loader.vectorized_result(d).flatten()
        ours = my_one_hot(d).flatten()
        match = "✓" if np.array_equal(original, ours) else "✗"
        print(f"  digit {d}: {match} Match!")
    
    return my_one_hot


# =============================================================================
# EXERCISE 5: Find most similar images
# =============================================================================

def exercise_5_similar_images():
    """
    Exercise 5: Find images most similar to a target image.
    
    Goal: Use Euclidean distance to find the 5 most similar images
    """
    exercise_header(5, "Find Most Similar Images")
    
    print("TASK: Find the 5 images most similar to the first '3' in training data.")
    print("HINT: Use np.linalg.norm() or np.sum((a-b)**2) for distance")
    print("\n" + "-" * 50)
    
    # YOUR CODE HERE:
    # training_data, _, _ = mnist_loader.load_data()
    # images = training_data[0]
    # labels = training_data[1]
    # 
    # # Find first 3
    # target_idx = np.argmax(labels == 3)
    # target_image = images[target_idx]
    # 
    # # Compute distances to all other images
    # distances = ???
    # 
    # # Find 5 closest (excluding target itself)
    # closest_indices = ???
    
    # SOLUTION:
    print("\n📖 SOLUTION:")
    print("-" * 50)
    
    training_data, _, _ = mnist_loader.load_data()
    images = training_data[0]
    labels = training_data[1]
    
    # Find first 3
    target_idx = np.argmax(labels == 3)
    target_image = images[target_idx]
    
    print(f"Target: First '3' at index {target_idx}")
    
    # Compute Euclidean distance to all images
    # Using broadcasting: (50000, 784) - (784,) = (50000, 784)
    distances = np.linalg.norm(images - target_image, axis=1)
    
    # Sort by distance and get indices (skip index 0 which is the target itself)
    sorted_indices = np.argsort(distances)
    closest_indices = sorted_indices[1:6]  # Skip first (distance=0 is target)
    
    print(f"\nCode:")
    print("  distances = np.linalg.norm(images - target_image, axis=1)")
    print("  sorted_indices = np.argsort(distances)")
    print("  closest_indices = sorted_indices[1:6]")
    
    print(f"\n5 most similar images:")
    for i, idx in enumerate(closest_indices):
        print(f"  #{i+1}: index {idx}, label={labels[idx]}, distance={distances[idx]:.2f}")
    
    # Visualize
    fig, axes = plt.subplots(1, 6, figsize=(15, 3))
    
    axes[0].imshow(target_image.reshape(28, 28), cmap='gray')
    axes[0].set_title(f"TARGET\n(label={labels[target_idx]})")
    axes[0].axis('off')
    
    for i, idx in enumerate(closest_indices):
        axes[i+1].imshow(images[idx].reshape(28, 28), cmap='gray')
        axes[i+1].set_title(f"#{i+1}\n(label={labels[idx]})\ndist={distances[idx]:.1f}")
        axes[i+1].axis('off')
    
    plt.suptitle("Target Image and 5 Most Similar Images", fontsize=14)
    plt.tight_layout()
    
    fig_path = level1_picture("exercise5_similar_images")
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"\n✓ Saved image to: {fig_path}")
    
    return closest_indices


# =============================================================================
# BONUS EXERCISE: Pixel importance
# =============================================================================

def bonus_exercise_pixel_importance():
    """
    Bonus: Which pixels are most important for classification?
    
    Goal: Find pixels with highest variance across the dataset
    """
    exercise_header("BONUS", "Pixel Importance Analysis")
    
    print("TASK: Find which pixels vary most across all images.")
    print("      High variance = important for distinguishing digits")
    print("HINT: Use images.var(axis=0) to get variance per pixel")
    print("\n" + "-" * 50)
    
    # SOLUTION:
    print("\n📖 SOLUTION:")
    print("-" * 50)
    
    training_data, _, _ = mnist_loader.load_data()
    images = training_data[0]
    
    # Variance per pixel
    pixel_variance = images.var(axis=0)
    
    print(f"pixel_variance = images.var(axis=0)")
    print(f"Shape: {pixel_variance.shape}")
    print(f"Max variance: {pixel_variance.max():.4f}")
    print(f"Min variance: {pixel_variance.min():.4f}")
    
    # Find top 10 most important pixel positions
    top_10_indices = np.argsort(pixel_variance)[-10:][::-1]
    print(f"\nTop 10 most important pixel indices: {top_10_indices}")
    
    # Convert to 2D positions
    print("\nAs (row, col) positions:")
    for idx in top_10_indices:
        row, col = idx // 28, idx % 28
        print(f"  Pixel {idx}: ({row}, {col})")
    
    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    # Variance heatmap
    variance_map = pixel_variance.reshape(28, 28)
    im1 = axes[0].imshow(variance_map, cmap='hot')
    axes[0].set_title("Pixel Variance Map\n(brighter = more important)")
    plt.colorbar(im1, ax=axes[0])
    
    # Mark top pixels
    axes[1].imshow(variance_map, cmap='gray')
    for idx in top_10_indices:
        row, col = idx // 28, idx % 28
        axes[1].plot(col, row, 'r+', markersize=15, markeredgewidth=2)
    axes[1].set_title("Top 10 Most Important Pixels\n(marked with +)")
    
    plt.tight_layout()
    
    fig_path = level1_picture("bonus_pixel_importance")
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"\n✓ Saved image to: {fig_path}")
    
    print("""
    
    📊 INSIGHT:
    ──────────
    High variance pixels are in the CENTER of the image.
    Edge pixels have low variance (usually black).
    
    This tells us: Most digit information is in the center!
    Neural networks will learn to focus on these pixels.
    """)
    
    return pixel_variance


def main():
    # Start a fresh run (creates new folder with timestamp)
    reset_run_timestamp()
    
    print("\n" + "╔" + "═" * 68 + "╗")
    print("║" + " " * 15 + "LEVEL 1: HANDS-ON EXERCISES" + " " * 25 + "║")
    print("╚" + "═" * 68 + "╝")
    
    print("""
    Complete these exercises to master MNIST data handling.
    Each exercise builds on the previous concepts.
    """)
    
    exercises = [
        ("Exercise 1", exercise_1_count_digits),
        ("Exercise 2", exercise_2_find_digit),
        ("Exercise 3", exercise_3_mean_image),
        ("Exercise 4", exercise_4_one_hot),
        ("Exercise 5", exercise_5_similar_images),
        ("Bonus", bonus_exercise_pixel_importance),
    ]
    
    print("\nAvailable exercises:")
    for i, (name, _) in enumerate(exercises):
        print(f"  {i+1}. {name}")
    
    print("\nRunning all exercises...\n")
    
    for name, func in exercises:
        func()
        print("\n" + "=" * 70)
        print(f"✓ {name} complete!")
    
    print("\n" + "╔" + "═" * 68 + "╗")
    print("║" + " " * 12 + "🎉 ALL EXERCISES COMPLETE! 🎉" + " " * 25 + "║")
    print("╚" + "═" * 68 + "╝")
    
    print("""
    You've mastered:
    
    ✅ Counting and analyzing labels
    ✅ Finding specific digits
    ✅ Computing statistical aggregates (mean, variance)
    ✅ Implementing one-hot encoding
    ✅ Measuring image similarity
    ✅ Analyzing pixel importance
    
    → Ready for Level 2: Building Neural Networks! 🧠
    """)


if __name__ == "__main__":
    main()

