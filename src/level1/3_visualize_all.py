"""
Level 1: Visualize All 50,000 MNIST Images
==========================================

Multiple ways to visualize the entire dataset:
1. Mega-grid (250x200 = 50,000 images in one picture)
2. Average digit for each class
3. Random sampling grid
4. Digit-by-digit grids

Run from project root:
    python src/level1/3_visualize_all.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt
import mnist_loader
from utils import level1_picture, reset_run_timestamp, get_run_folder


def create_mega_grid(images: np.ndarray, save_path: str) -> None:
    """
    Create a single HUGE image containing ALL 50,000 digits.
    
    Layout: 250 columns × 200 rows = 50,000 images
    Each digit is 28×28, so final size: 7000 × 5600 pixels
    """
    print("\n[1/4] Creating mega-grid of ALL 50,000 images...")
    
    n_images = len(images)
    n_cols = 250
    n_rows = n_images // n_cols  # 200
    
    # Create the mega image
    # Each image is 28x28, so total size is (200*28, 250*28) = (5600, 7000)
    mega_height = n_rows * 28
    mega_width = n_cols * 28
    
    mega_image = np.zeros((mega_height, mega_width))
    
    for idx in range(n_images):
        row = idx // n_cols
        col = idx % n_cols
        
        # Get the image and reshape to 28x28
        img = images[idx].reshape(28, 28)
        
        # Place in the mega image
        y_start = row * 28
        x_start = col * 28
        mega_image[y_start:y_start+28, x_start:x_start+28] = img
        
        if (idx + 1) % 10000 == 0:
            print(f"    Processed {idx + 1:,} / {n_images:,} images...")
    
    # Save as high-resolution image
    fig, ax = plt.subplots(figsize=(70, 56), dpi=100)  # 7000x5600 pixels
    ax.imshow(mega_image, cmap='gray')
    ax.axis('off')
    ax.set_title(f'ALL {n_images:,} MNIST Training Images (250×200 grid)', fontsize=40, pad=20)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches='tight', facecolor='black')
    plt.close()
    
    print(f"    ✓ Saved: {save_path}")
    print(f"    Image size: {mega_width} × {mega_height} pixels")


def create_average_digits(images: np.ndarray, labels: np.ndarray, save_path: str) -> None:
    """
    Create the "average" image for each digit (0-9).
    This shows what each digit looks like on average.
    """
    print("\n[2/4] Creating average digit for each class...")
    
    fig, axes = plt.subplots(2, 5, figsize=(15, 7))
    fig.suptitle('Average Image for Each Digit (from 50,000 samples)', fontsize=16, y=1.02)
    
    for digit in range(10):
        ax = axes[digit // 5, digit % 5]
        
        # Get all images of this digit
        mask = labels == digit
        digit_images = images[mask]
        
        # Compute average
        avg_image = digit_images.mean(axis=0).reshape(28, 28)
        
        ax.imshow(avg_image, cmap='hot')
        ax.set_title(f'Digit {digit}\n({len(digit_images):,} samples)', fontsize=12)
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"    ✓ Saved: {save_path}")


def create_digit_grids(images: np.ndarray, labels: np.ndarray, save_dir: str) -> None:
    """
    Create a separate grid for each digit (0-9).
    Shows first 1000 examples of each digit in a 40x25 grid.
    """
    print("\n[3/4] Creating individual grids for each digit...")
    
    for digit in range(10):
        # Get all images of this digit
        mask = labels == digit
        digit_images = images[mask]
        
        # Take first 1000 (or all if less)
        n_show = min(1000, len(digit_images))
        n_cols = 40
        n_rows = n_show // n_cols
        
        # Create grid
        grid_height = n_rows * 28
        grid_width = n_cols * 28
        grid_image = np.zeros((grid_height, grid_width))
        
        for idx in range(n_rows * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            img = digit_images[idx].reshape(28, 28)
            y_start = row * 28
            x_start = col * 28
            grid_image[y_start:y_start+28, x_start:x_start+28] = img
        
        # Save
        fig, ax = plt.subplots(figsize=(20, 12.5))
        ax.imshow(grid_image, cmap='gray')
        ax.axis('off')
        ax.set_title(f'Digit {digit}: {n_rows * n_cols} examples (of {len(digit_images):,} total)', fontsize=14)
        
        save_path = level1_picture(f"all_digit_{digit}")
        plt.tight_layout()
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close()
        
        print(f"    ✓ Digit {digit}: saved {n_rows * n_cols} images to {save_path}")


def create_random_mosaic(images: np.ndarray, labels: np.ndarray, save_path: str) -> None:
    """
    Create a colorful mosaic with random samples, colored by digit.
    """
    print("\n[4/4] Creating colorful random mosaic...")
    
    n_show = 2500  # 50x50 grid
    n_cols = 50
    n_rows = 50
    
    # Random indices
    indices = np.random.choice(len(images), n_show, replace=False)
    
    # Create RGB image (colored by digit)
    mosaic = np.zeros((n_rows * 28, n_cols * 28, 3))
    
    # Colors for each digit (rainbow)
    colors = plt.cm.tab10(np.linspace(0, 1, 10))[:, :3]
    
    for i, idx in enumerate(indices):
        row = i // n_cols
        col = i % n_cols
        
        img = images[idx].reshape(28, 28)
        label = labels[idx]
        color = colors[label]
        
        y_start = row * 28
        x_start = col * 28
        
        # Apply color tint to grayscale image
        for c in range(3):
            mosaic[y_start:y_start+28, x_start:x_start+28, c] = img * color[c]
    
    fig, ax = plt.subplots(figsize=(20, 20))
    ax.imshow(mosaic)
    ax.axis('off')
    ax.set_title('Random Mosaic (2,500 samples, colored by digit)', fontsize=16)
    
    # Add legend
    for digit in range(10):
        ax.plot([], [], 's', color=colors[digit], markersize=15, label=f'Digit {digit}')
    ax.legend(loc='upper right', fontsize=10, ncol=2)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"    ✓ Saved: {save_path}")


def create_statistics_visualization(images: np.ndarray, labels: np.ndarray, save_path: str) -> None:
    """
    Create statistical visualizations of the entire dataset.
    """
    print("\n[BONUS] Creating statistical visualization...")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('MNIST Dataset Statistics (50,000 images)', fontsize=18, y=1.02)
    
    # 1. Digit distribution
    ax = axes[0, 0]
    counts = np.bincount(labels)
    bars = ax.bar(range(10), counts, color=plt.cm.tab10(np.linspace(0, 1, 10)))
    ax.set_xlabel('Digit')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of Digits')
    ax.set_xticks(range(10))
    for i, v in enumerate(counts):
        ax.text(i, v + 50, str(v), ha='center', fontsize=9)
    
    # 2. Pixel intensity histogram
    ax = axes[0, 1]
    ax.hist(images.flatten(), bins=50, color='steelblue', edgecolor='black', alpha=0.7)
    ax.set_xlabel('Pixel Intensity')
    ax.set_ylabel('Frequency')
    ax.set_title('Pixel Intensity Distribution')
    ax.axvline(images.mean(), color='red', linestyle='--', label=f'Mean: {images.mean():.3f}')
    ax.legend()
    
    # 3. Average brightness per digit
    ax = axes[0, 2]
    avg_brightness = [images[labels == d].mean() for d in range(10)]
    bars = ax.bar(range(10), avg_brightness, color=plt.cm.tab10(np.linspace(0, 1, 10)))
    ax.set_xlabel('Digit')
    ax.set_ylabel('Average Brightness')
    ax.set_title('Average Brightness by Digit')
    ax.set_xticks(range(10))
    
    # 4. Variance heatmap (which pixels vary most)
    ax = axes[1, 0]
    variance_map = images.var(axis=0).reshape(28, 28)
    im = ax.imshow(variance_map, cmap='hot')
    ax.set_title('Pixel Variance Map\n(Where digits differ most)')
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046)
    
    # 5. Mean image (all digits combined)
    ax = axes[1, 1]
    mean_image = images.mean(axis=0).reshape(28, 28)
    ax.imshow(mean_image, cmap='gray')
    ax.set_title('Mean Image\n(Average of all 50,000)')
    ax.axis('off')
    
    # 6. Std image (where variation happens)
    ax = axes[1, 2]
    std_image = images.std(axis=0).reshape(28, 28)
    ax.imshow(std_image, cmap='hot')
    ax.set_title('Std Dev Image\n(High variation areas)')
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"    ✓ Saved: {save_path}")


def main():
    # Start a fresh run (creates new folder with timestamp)
    reset_run_timestamp()
    run_folder = get_run_folder(1)
    
    print("\n" + "╔" + "═" * 58 + "╗")
    print("║" + " " * 8 + "VISUALIZING ALL 50,000 MNIST IMAGES" + " " * 13 + "║")
    print("╚" + "═" * 58 + "╝")
    
    # Load data
    print("\nLoading MNIST data...")
    training_data, _, _ = mnist_loader.load_data()
    images = training_data[0]  # (50000, 784)
    labels = training_data[1]  # (50000,)
    print(f"Loaded {len(images):,} images")
    
    # Create visualizations with sequential numbering
    create_mega_grid(images, str(level1_picture("all_50000_images")))
    create_average_digits(images, labels, str(level1_picture("average_digits")))
    create_digit_grids(images, labels, str(run_folder))
    create_random_mosaic(images, labels, str(level1_picture("color_mosaic")))
    create_statistics_visualization(images, labels, str(level1_picture("dataset_stats")))
    
    print("\n" + "=" * 60)
    print("  VISUALIZATION COMPLETE!")
    print("=" * 60)
    print(f"""
    Generated images in {run_folder}:
    
    1_all_50000_images.png   - MEGA grid (ALL 50,000 in one image!)
    2_average_digits.png     - Average image for each digit
    3-12_all_digit_X.png     - Grid for each digit (0-9)
    13_color_mosaic.png      - Random samples colored by digit
    14_dataset_stats.png     - Statistical visualizations
    
    Total: 15 visualization files
    """)


if __name__ == "__main__":
    main()

