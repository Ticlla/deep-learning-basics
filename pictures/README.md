# Pictures Directory

Organized by learning level, with each execution run in its own subfolder.  
Files are numbered sequentially to show execution order.

## Structure

```
pictures/
├── level1/
│   └── 20251226_153346/           ← Run timestamp
│       ├── 1_sample_digits.png    ← 1st file saved
│       ├── 2_average_digits.png   ← 2nd file saved
│       ├── 3_digit_variations.png ← 3rd file saved
│       └── ...
│
├── level2/
│   └── 20251226_154154/
│       ├── 1_initial_weights.png
│       ├── 2_sigmoid.png
│       ├── 3_backprop_flow.png
│       ├── 4_sgd.png
│       ├── 5_predictions.png
│       └── 6_learned_weights.png
│
├── level3/
└── level4/
```

## Naming Convention

- **Folder:** `YYYYMMDD_HHMMSS` (timestamp when run started)
- **Files:** `N_descriptive_name.png` (N = order of creation)

## Using the Utilities

```python
from utils import level2_picture, reset_run_timestamp

# Files are numbered automatically in order
path1 = level2_picture("initial_weights")  # 1_initial_weights.png
path2 = level2_picture("sigmoid")          # 2_sigmoid.png
path3 = level2_picture("backprop")         # 3_backprop.png

# Start a NEW run (resets counter to 1)
reset_run_timestamp()
path4 = level2_picture("sigmoid")          # NEW FOLDER: 1_sigmoid.png
```

## Example Script

```python
from utils import level2_picture, reset_run_timestamp
import matplotlib.pyplot as plt

# Start fresh run
reset_run_timestamp()

# Create visualizations in order
fig1, ax1 = plt.subplots()
# ... create initial weights plot ...
plt.savefig(level2_picture("initial_weights"))  # → 1_initial_weights.png

fig2, ax2 = plt.subplots()
# ... create sigmoid plot ...
plt.savefig(level2_picture("sigmoid"))          # → 2_sigmoid.png

fig3, ax3 = plt.subplots()
# ... create training results ...
plt.savefig(level2_picture("predictions"))      # → 3_predictions.png
```

## Listing Runs

```python
from utils import list_runs, list_level_pictures

# List all runs for a level (newest first)
runs = list_runs(2)
for run in runs:
    print(run.name)  # 20251226_154154, 20251226_153346, ...

# List pictures in the latest run (sorted by number)
pictures = list_level_pictures(2)
for p in pictures:
    print(p.name)  # 1_initial_weights.png, 2_sigmoid.png, ...
```

## Benefits

✓ **Order preserved:** Numbers show execution sequence  
✓ **Easy sorting:** Files sort correctly by creation order  
✓ **Run isolation:** Each execution in its own folder  
✓ **Clean comparison:** Compare same step across runs  
