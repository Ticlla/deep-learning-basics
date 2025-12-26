"""
Level 1: Reading All 50,000 MNIST Images
=========================================

This script demonstrates how to access and iterate through
all images in the MNIST dataset.

Run from project root:
    python src/level1/2_read_all_images.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import mnist_loader


def demo_raw_data_access():
    """
    Method 1: Using load_data() - Raw numpy arrays
    
    This is the FASTEST way to access all images because
    they're stored as a single numpy array (matrix).
    """
    print("\n" + "=" * 60)
    print("  METHOD 1: Raw Data Access (load_data)")
    print("=" * 60)
    
    # Load raw data
    training_data, validation_data, test_data = mnist_loader.load_data()
    
    # training_data is a tuple: (images_array, labels_array)
    images = training_data[0]  # Shape: (50000, 784)
    labels = training_data[1]  # Shape: (50000,)
    
    print(f"\nImages array shape: {images.shape}")
    print(f"Labels array shape: {labels.shape}")
    print(f"Memory size: {images.nbytes / 1024 / 1024:.2f} MB")
    
    # Access specific images by index
    print("\n--- Accessing specific images ---")
    print(f"Image 0 shape: {images[0].shape}")
    print(f"Image 0 label: {labels[0]}")
    print(f"Image 100 label: {labels[100]}")
    print(f"Last image (49999) label: {labels[-1]}")
    
    # Iterate through ALL images (fast with numpy)
    print("\n--- Iterating through all 50,000 images ---")
    
    # Example: Count how many of each digit
    digit_counts = np.bincount(labels)
    print("\nDigit distribution in training set:")
    for digit, count in enumerate(digit_counts):
        bar = "█" * (count // 200)
        print(f"  Digit {digit}: {count:5d} {bar}")
    
    # Example: Calculate statistics across all images
    print("\n--- Statistics across ALL images ---")
    print(f"Mean pixel value: {images.mean():.4f}")
    print(f"Std pixel value:  {images.std():.4f}")
    print(f"Min pixel value:  {images.min():.4f}")
    print(f"Max pixel value:  {images.max():.4f}")
    
    # Example: Find the "brightest" and "darkest" images
    image_brightness = images.mean(axis=1)  # Mean of each row (image)
    brightest_idx = image_brightness.argmax()
    darkest_idx = image_brightness.argmin()
    print(f"\nBrightest image: index {brightest_idx} (digit {labels[brightest_idx]}), mean={image_brightness[brightest_idx]:.4f}")
    print(f"Darkest image:   index {darkest_idx} (digit {labels[darkest_idx]}), mean={image_brightness[darkest_idx]:.4f}")
    
    return images, labels


def demo_wrapper_data_access():
    """
    Method 2: Using load_data_wrapper() - List of tuples
    
    This format is ready for neural network training.
    Access is slightly slower (list iteration vs numpy indexing).
    """
    print("\n" + "=" * 60)
    print("  METHOD 2: Wrapper Data Access (load_data_wrapper)")
    print("=" * 60)
    
    # Load wrapped data
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    
    print(f"\nTraining data type: {type(training_data)}")
    print(f"Number of examples: {len(training_data)}")
    
    # Each element is a tuple (x, y)
    x, y = training_data[0]
    print(f"\nFirst example:")
    print(f"  x (image) shape: {x.shape}")
    print(f"  y (label) shape: {y.shape}")
    print(f"  y (one-hot): {y.flatten()}")
    print(f"  Actual digit: {np.argmax(y)}")
    
    # Iterate through all examples
    print("\n--- Iterating through all 50,000 examples ---")
    
    # Example: Count digits (slower than numpy but demonstrates iteration)
    digit_counts = [0] * 10
    total_pixels = 0
    
    for i, (x, y) in enumerate(training_data):
        digit = np.argmax(y)
        digit_counts[digit] += 1
        total_pixels += x.sum()
        
        # Progress indicator
        if (i + 1) % 10000 == 0:
            print(f"  Processed {i + 1:,} images...")
    
    print(f"\nTotal pixels summed: {total_pixels:,.0f}")
    print(f"Average pixels per image: {total_pixels / len(training_data):.2f}")
    
    return training_data


def demo_batch_processing():
    """
    Method 3: Processing images in batches
    
    This is how neural networks actually process data during training.
    """
    print("\n" + "=" * 60)
    print("  METHOD 3: Batch Processing")
    print("=" * 60)
    
    training_data, _, _ = mnist_loader.load_data()
    images = training_data[0]
    labels = training_data[1]
    
    batch_size = 32
    num_batches = len(images) // batch_size
    
    print(f"\nTotal images: {len(images)}")
    print(f"Batch size: {batch_size}")
    print(f"Number of batches: {num_batches}")
    print(f"Images per epoch: {num_batches * batch_size}")
    
    # Process in batches
    print("\n--- Processing in batches ---")
    
    batch_means = []
    for batch_idx in range(num_batches):
        start = batch_idx * batch_size
        end = start + batch_size
        
        batch_images = images[start:end]  # Shape: (32, 784)
        batch_labels = labels[start:end]  # Shape: (32,)
        
        # Compute something for each batch
        batch_mean = batch_images.mean()
        batch_means.append(batch_mean)
        
        if batch_idx < 3:
            print(f"  Batch {batch_idx}: images[{start}:{end}], mean={batch_mean:.4f}")
    
    print(f"  ...")
    print(f"  Batch {num_batches-1}: images[{(num_batches-1)*batch_size}:{num_batches*batch_size}]")
    
    print(f"\nOverall mean from batches: {np.mean(batch_means):.4f}")


def demo_specific_digit_access():
    """
    Method 4: Access all images of a specific digit
    """
    print("\n" + "=" * 60)
    print("  METHOD 4: Access All Images of Specific Digit")
    print("=" * 60)
    
    training_data, _, _ = mnist_loader.load_data()
    images = training_data[0]
    labels = training_data[1]
    
    # Get all images of digit 7
    target_digit = 7
    mask = labels == target_digit  # Boolean array
    digit_7_images = images[mask]
    
    print(f"\nAll images of digit {target_digit}:")
    print(f"  Count: {len(digit_7_images)}")
    print(f"  Shape: {digit_7_images.shape}")
    
    # Compute the "average 7"
    average_7 = digit_7_images.mean(axis=0)
    print(f"  Average image shape: {average_7.shape}")
    
    # Do this for all digits
    print("\n--- Average image for each digit ---")
    for digit in range(10):
        mask = labels == digit
        digit_images = images[mask]
        avg = digit_images.mean(axis=0)
        print(f"  Digit {digit}: {len(digit_images)} images, avg brightness={avg.mean():.4f}")


def demo_memory_efficient():
    """
    Method 5: Memory-efficient iteration (generator)
    
    For very large datasets that don't fit in memory.
    (MNIST easily fits, but this pattern is useful for larger datasets)
    """
    print("\n" + "=" * 60)
    print("  METHOD 5: Memory-Efficient Generator Pattern")
    print("=" * 60)
    
    def image_generator(images, labels, batch_size=100):
        """Yield batches of images one at a time."""
        num_samples = len(images)
        for start in range(0, num_samples, batch_size):
            end = min(start + batch_size, num_samples)
            yield images[start:end], labels[start:end]
    
    training_data, _, _ = mnist_loader.load_data()
    images = training_data[0]
    labels = training_data[1]
    
    print("\nUsing generator to process in chunks:")
    
    total_sum = 0
    chunk_count = 0
    
    for batch_images, batch_labels in image_generator(images, labels, batch_size=5000):
        total_sum += batch_images.sum()
        chunk_count += 1
        print(f"  Chunk {chunk_count}: processed {len(batch_images)} images")
    
    print(f"\nTotal sum from generator: {total_sum:,.0f}")


def main():
    print("\n" + "╔" + "═" * 58 + "╗")
    print("║" + " " * 8 + "READING ALL 50,000 MNIST IMAGES" + " " * 18 + "║")
    print("╚" + "═" * 58 + "╝")
    
    # Method 1: Raw numpy access (fastest)
    images, labels = demo_raw_data_access()
    
    # Method 2: Wrapper format (list of tuples)
    demo_wrapper_data_access()
    
    # Method 3: Batch processing
    demo_batch_processing()
    
    # Method 4: Access specific digits
    demo_specific_digit_access()
    
    # Method 5: Generator pattern
    demo_memory_efficient()
    
    print("\n" + "=" * 60)
    print("  SUMMARY: How to Read All Images")
    print("=" * 60)
    print("""
    FASTEST (numpy array indexing):
    ────────────────────────────────
    training_data, _, _ = mnist_loader.load_data()
    images = training_data[0]  # Shape: (50000, 784)
    labels = training_data[1]  # Shape: (50000,)
    
    # Access any image:
    image_42 = images[42]      # Get image at index 42
    label_42 = labels[42]      # Get its label
    
    # Access multiple images:
    first_100 = images[:100]   # First 100 images
    last_50 = images[-50:]     # Last 50 images
    
    # Access by condition:
    sevens = images[labels == 7]  # All images of digit 7
    
    
    FOR NEURAL NETWORK TRAINING:
    ────────────────────────────────
    training_data, _, _ = mnist_loader.load_data_wrapper()
    
    for x, y in training_data:
        # x.shape = (784, 1) - column vector
        # y.shape = (10, 1)  - one-hot encoded
        pass
    """)


if __name__ == "__main__":
    main()

