#!/usr/bin/env python3
"""
Reorganize project files into a cleaner structure.

New structure:
    pictures/
        level1/
            YYYYMMDD_HHMMSS-imagename.png
        level2/
            YYYYMMDD_HHMMSS-imagename.png
        level3/
        level4/
    
    scripts/
        level1/
            explore_mnist.py
            read_all_images.py
            ...
        level2/
            neural_network.py
            ...
    
    docs/
        level1/
            data_understanding.md
        level2/
            neural_network.md

Run from project root:
    python reorganize_project.py
"""

import os
import shutil
from pathlib import Path
from datetime import datetime


def get_timestamp() -> str:
    """Generate timestamp for file naming."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def create_directory_structure(root: Path) -> None:
    """Create the new directory structure."""
    dirs = [
        "pictures/level1",
        "pictures/level2",
        "pictures/level3",
        "pictures/level4",
        "scripts/level1",
        "scripts/level2",
        "scripts/level3",
        "scripts/level4",
        "docs/level1",
        "docs/level2",
        "docs/level3",
        "docs/level4",
    ]
    
    for d in dirs:
        (root / d).mkdir(parents=True, exist_ok=True)
        print(f"  ✓ Created: {d}/")


def move_pictures(root: Path) -> None:
    """Move pictures from fig/ to pictures/levelX/."""
    fig_dir = root / "fig"
    
    if not fig_dir.exists():
        print("  No fig/ directory found, skipping pictures migration.")
        return
    
    # Define mappings: pattern -> level
    level_patterns = {
        1: ["level1_", "mnist_"],
        2: ["level2_", "sigmoid", "backprop", "sgd", "predictions", "weights"],
        3: ["level3_", "cross_entropy", "regularization"],
        4: ["level4_", "conv", "cnn"],
    }
    
    timestamp = get_timestamp()
    moved_count = 0
    
    for file in fig_dir.glob("*"):
        if not file.is_file():
            continue
        
        # Determine level based on filename
        target_level = None
        for level, patterns in level_patterns.items():
            for pattern in patterns:
                if pattern in file.name.lower():
                    target_level = level
                    break
            if target_level:
                break
        
        if target_level is None:
            # Keep in a general folder or level1 by default
            target_level = 1
        
        # Create new filename with timestamp
        # Remove old "level1_" or "level2_" prefix if present
        base_name = file.name
        for prefix in ["level1_", "level2_", "level3_", "level4_"]:
            if base_name.startswith(prefix):
                base_name = base_name[len(prefix):]
                break
        
        new_name = f"{timestamp}-{base_name}"
        target_dir = root / "pictures" / f"level{target_level}"
        target_path = target_dir / new_name
        
        # Copy file (don't move to preserve original during transition)
        shutil.copy2(file, target_path)
        print(f"  ✓ Copied: {file.name} → pictures/level{target_level}/{new_name}")
        moved_count += 1
    
    print(f"\n  Total pictures organized: {moved_count}")


def move_scripts(root: Path) -> None:
    """Move level scripts from src/ to scripts/levelX/."""
    src_dir = root / "src"
    
    # Define mappings: pattern -> level
    level_patterns = {
        1: ["level1_"],
        2: ["level2_"],
        3: ["level3_"],
        4: ["level4_"],
    }
    
    moved_count = 0
    
    for file in src_dir.glob("level*_*.py"):
        # Determine level based on filename
        target_level = None
        for level, patterns in level_patterns.items():
            for pattern in patterns:
                if file.name.startswith(pattern):
                    target_level = level
                    break
            if target_level:
                break
        
        if target_level is None:
            continue
        
        # Create cleaner name (remove "level1_" prefix)
        base_name = file.name
        for prefix in ["level1_", "level2_", "level3_", "level4_"]:
            if base_name.startswith(prefix):
                base_name = base_name[len(prefix):]
                break
        
        target_dir = root / "scripts" / f"level{target_level}"
        target_path = target_dir / base_name
        
        # Copy file
        shutil.copy2(file, target_path)
        print(f"  ✓ Copied: src/{file.name} → scripts/level{target_level}/{base_name}")
        moved_count += 1
    
    print(f"\n  Total scripts organized: {moved_count}")


def move_docs(root: Path) -> None:
    """Move documentation to docs/levelX/."""
    docs_dir = root / "docs"
    
    # Define mappings
    doc_mappings = {
        "level1_data_understanding.md": ("level1", "data_understanding.md"),
        "level2_neural_network.md": ("level2", "neural_network.md"),
    }
    
    moved_count = 0
    
    for old_name, (level, new_name) in doc_mappings.items():
        old_path = docs_dir / old_name
        if old_path.exists():
            target_dir = docs_dir / level
            target_path = target_dir / new_name
            
            # Copy file
            shutil.copy2(old_path, target_path)
            print(f"  ✓ Copied: docs/{old_name} → docs/{level}/{new_name}")
            moved_count += 1
    
    print(f"\n  Total docs organized: {moved_count}")


def create_readme_for_structure(root: Path) -> None:
    """Create README files explaining the new structure."""
    
    # Main pictures README
    pictures_readme = root / "pictures" / "README.md"
    pictures_readme.write_text("""# Pictures Directory

Organized by learning level with timestamped filenames.

## Structure

```
pictures/
├── level1/          # MNIST data exploration visualizations
├── level2/          # Basic neural network visualizations
├── level3/          # Improved techniques visualizations
└── level4/          # CNN and deep learning visualizations
```

## Naming Convention

Files are named with timestamp prefix for chronological ordering:

```
YYYYMMDD_HHMMSS-descriptive_name.png
```

Example: `20241226_143052-sigmoid.png`

## Generating New Pictures

Use the utility functions from `src/utils/file_utils.py`:

```python
from utils import level2_picture
import matplotlib.pyplot as plt

# Get path with automatic timestamp
path = level2_picture("my_visualization")
plt.savefig(path)
print(f"Saved to: {path}")
```
""")
    
    # Scripts README
    scripts_readme = root / "scripts" / "README.md"
    scripts_readme.write_text("""# Scripts Directory

Exploration and demonstration scripts organized by learning level.

## Structure

```
scripts/
├── level1/          # Data loading and exploration
│   ├── explore_mnist.py
│   ├── read_all_images.py
│   ├── visualize_all.py
│   ├── advanced_topics.py
│   └── exercises.py
│
├── level2/          # Basic neural network
│   └── neural_network.py
│
├── level3/          # Improved techniques
│   └── (coming soon)
│
└── level4/          # CNNs and deep learning
    └── (coming soon)
```

## Running Scripts

```bash
# From project root
cd scripts/level1
python explore_mnist.py

# Or add to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
python scripts/level2/neural_network.py
```
""")
    
    print("  ✓ Created README files for new structure")


def main():
    """Run the reorganization."""
    root = Path(__file__).parent
    
    print("\n" + "=" * 60)
    print("  PROJECT REORGANIZATION")
    print("=" * 60)
    
    print("\n1. Creating new directory structure...")
    create_directory_structure(root)
    
    print("\n2. Organizing pictures...")
    move_pictures(root)
    
    print("\n3. Organizing scripts...")
    move_scripts(root)
    
    print("\n4. Organizing documentation...")
    move_docs(root)
    
    print("\n5. Creating README files...")
    create_readme_for_structure(root)
    
    print("\n" + "=" * 60)
    print("  REORGANIZATION COMPLETE!")
    print("=" * 60)
    
    print("""
New structure created:

    pictures/
    ├── level1/    (MNIST exploration images)
    ├── level2/    (Neural network images)
    ├── level3/    (Improved techniques)
    └── level4/    (CNNs)
    
    scripts/
    ├── level1/    (Data exploration scripts)
    ├── level2/    (Basic NN scripts)
    ├── level3/
    └── level4/
    
    docs/
    ├── level1/    (Level 1 documentation)
    ├── level2/    (Level 2 documentation)
    ├── level3/
    └── level4/

Note: Original files in src/ and fig/ are preserved.
      You can delete them after verifying the new structure.

To use the new picture paths in scripts:

    from utils import level2_picture
    
    path = level2_picture("sigmoid")  # pictures/level2/TIMESTAMP-sigmoid.png
    plt.savefig(path)
""")


if __name__ == "__main__":
    main()

