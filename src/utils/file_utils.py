"""
Utility functions for file management across all levels.

Provides consistent file naming and organization:
- pictures/level1/YYYYMMDD_HHMMSS/imagename.png   (each run in its own folder)
- pictures/level2/YYYYMMDD_HHMMSS/imagename.png
- scripts/level1/script.py
- scripts/level2/script.py
- docs/level1/documentation.md
- docs/level2/documentation.md
"""

import os
import time
from pathlib import Path
from datetime import datetime

# Global run timestamp - set once per execution
_RUN_TIMESTAMP: str | None = None

# Global file counter - tracks order of file creation within a run
_FILE_COUNTER: int = 0


def get_project_root() -> Path:
    """Get the project root directory."""
    # Navigate up from src/utils to project root
    return Path(__file__).parent.parent.parent


def get_timestamp() -> str:
    """Generate a timestamp string for file naming."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def get_run_timestamp() -> str:
    """
    Get the timestamp for the current run.
    This is set once per execution and reused for all files in the same run.
    """
    global _RUN_TIMESTAMP
    if _RUN_TIMESTAMP is None:
        _RUN_TIMESTAMP = get_timestamp()
    return _RUN_TIMESTAMP


def reset_run_timestamp() -> str:
    """
    Reset the run timestamp to start a new run.
    Also resets the file counter.
    Returns the new timestamp.
    """
    global _RUN_TIMESTAMP, _FILE_COUNTER
    _RUN_TIMESTAMP = get_timestamp()
    _FILE_COUNTER = 0
    return _RUN_TIMESTAMP


def get_next_file_number() -> int:
    """
    Get the next file number in the sequence.
    Increments the counter each time it's called.
    """
    global _FILE_COUNTER
    _FILE_COUNTER += 1
    return _FILE_COUNTER


def get_run_folder(level: int) -> Path:
    """
    Get the folder for the current run.
    
    Returns:
        Path like: pictures/level2/20241226_143052/
    """
    root = get_project_root()
    run_timestamp = get_run_timestamp()
    run_dir = root / "pictures" / f"level{level}" / run_timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def get_picture_path(level: int, name: str, ext: str = "png") -> Path:
    """
    Get the path for saving a picture in the current run's folder.
    
    Files are numbered sequentially (1_, 2_, 3_, ...) to show execution order.
    
    Args:
        level: The level number (1, 2, 3, or 4)
        name: The base name of the image (e.g., "sigmoid", "weights")
        ext: File extension (default: "png")
    
    Returns:
        Path object like: pictures/level2/20241226_143052/1_sigmoid.png
    
    Example:
        >>> path = get_picture_path(2, "sigmoid")   # 1_sigmoid.png
        >>> path = get_picture_path(2, "backprop")  # 2_backprop.png
        >>> plt.savefig(path)
    """
    run_dir = get_run_folder(level)
    file_num = get_next_file_number()
    filename = f"{file_num}_{name}.{ext}"
    return run_dir / filename


def get_latest_run(level: int) -> Path | None:
    """
    Get the most recent run folder for a level.
    
    Args:
        level: The level number
    
    Returns:
        Path to the most recent run folder, or None if not found
    """
    root = get_project_root()
    level_dir = root / "pictures" / f"level{level}"
    
    if not level_dir.exists():
        return None
    
    # Get all run folders (they are named with timestamps)
    run_folders = [d for d in level_dir.iterdir() if d.is_dir() and d.name[0].isdigit()]
    if not run_folders:
        return None
    
    # Sort by name (timestamp) and return newest
    return max(run_folders, key=lambda p: p.name)


def get_latest_picture(level: int, name_pattern: str) -> Path | None:
    """
    Get the most recent picture matching a pattern from the latest run.
    
    Args:
        level: The level number
        name_pattern: Part of the filename to match (e.g., "sigmoid")
    
    Returns:
        Path to the most recent matching file, or None if not found
    """
    latest_run = get_latest_run(level)
    if latest_run is None:
        return None
    
    matching_files = list(latest_run.glob(f"*{name_pattern}*"))
    if not matching_files:
        return None
    
    return matching_files[0]


def ensure_level_structure(level: int) -> dict:
    """
    Ensure all directories for a level exist.
    
    Returns:
        Dictionary with paths to pictures, scripts, and docs directories
    """
    root = get_project_root()
    
    dirs = {
        "pictures": root / "pictures" / f"level{level}",
        "scripts": root / "scripts" / f"level{level}",
        "docs": root / "docs" / f"level{level}",
    }
    
    for dir_path in dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)
    
    return dirs


def list_runs(level: int) -> list[Path]:
    """List all run folders for a given level, sorted by timestamp."""
    root = get_project_root()
    level_dir = root / "pictures" / f"level{level}"
    
    if not level_dir.exists():
        return []
    
    run_folders = [d for d in level_dir.iterdir() if d.is_dir() and d.name[0].isdigit()]
    return sorted(run_folders, key=lambda p: p.name, reverse=True)  # newest first


def list_level_pictures(level: int, run: str | None = None) -> list[Path]:
    """
    List all pictures for a given level.
    
    Args:
        level: The level number
        run: Optional specific run timestamp. If None, uses latest run.
    """
    root = get_project_root()
    
    if run:
        run_dir = root / "pictures" / f"level{level}" / run
    else:
        run_dir = get_latest_run(level)
    
    if run_dir is None or not run_dir.exists():
        return []
    
    pictures = list(run_dir.glob("*.png")) + list(run_dir.glob("*.jpg"))
    return sorted(pictures, key=lambda p: p.name)


# Convenience functions for each level
def level1_picture(name: str, ext: str = "png") -> Path:
    """Get picture path for Level 1."""
    return get_picture_path(1, name, ext)


def level2_picture(name: str, ext: str = "png") -> Path:
    """Get picture path for Level 2."""
    return get_picture_path(2, name, ext)


def level3_picture(name: str, ext: str = "png") -> Path:
    """Get picture path for Level 3."""
    return get_picture_path(3, name, ext)


def level4_picture(name: str, ext: str = "png") -> Path:
    """Get picture path for Level 4."""
    return get_picture_path(4, name, ext)


if __name__ == "__main__":
    # Demo the utilities
    print("Project root:", get_project_root())
    print(f"\nCurrent run timestamp: {get_run_timestamp()}")
    
    print("\nExample paths (numbered sequentially in same run folder):")
    print(f"  1st save: {get_picture_path(2, 'initial_weights')}")
    print(f"  2nd save: {get_picture_path(2, 'sigmoid')}")
    print(f"  3rd save: {get_picture_path(2, 'backprop_flow')}")
    print(f"  4th save: {get_picture_path(2, 'predictions')}")
    
    print("\nRun folder structure:")
    ts = get_run_timestamp()
    print(f"  pictures/level2/{ts}/")
    print(f"    ├── 1_initial_weights.png")
    print(f"    ├── 2_sigmoid.png")
    print(f"    ├── 3_backprop_flow.png")
    print(f"    └── 4_predictions.png")
    
    # List existing runs
    print("\nExisting runs for Level 2:")
    for run in list_runs(2)[:5]:  # Show last 5
        print(f"  📁 {run.name}/")
    
    # Create structure for all levels
    print("\nCreating level structures...")
    for level in range(1, 5):
        dirs = ensure_level_structure(level)
        print(f"  Level {level}: {dirs['pictures']}")

