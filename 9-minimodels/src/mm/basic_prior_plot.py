# basic_prior_plot.py

import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

PKL_PATH = Path("1-mpt/src/mm/results.pkl")
FIG_PATH = Path("1-mpt/src/mm/")

# --- Load Data ---
try:
    with open(PKL_PATH, "rb") as f:
        results = pkl.load(f)
except FileNotFoundError:
    print("Error: results.pkl not found. Run hierarchical.mm.py first.")
    exit()
except Exception as e:
    print(f"Error loading results.pkl: {e}")
    exit()

# --- Extract Prior Data for Shared Model ---
prior_shared = results.get("prior", {}).get("shared_model")

if prior_shared is None:
    print("Error: Prior data for shared model not found in results.pkl.")
    exit()

# --- Check for delta1 and delta2 in the prior group ---
if "delta1" not in prior_shared.prior or "delta2" not in prior_shared.prior:
    print("Error: 'delta1' or 'delta2' not found in the prior group of the shared model.")
    exit()

# --- Extract delta samples as NumPy arrays ---
# Flatten the arrays to remove chain/draw dimensions
prior_delta1 = prior_shared.prior["delta1"].values.flatten()
prior_delta2 = prior_shared.prior["delta2"].values.flatten()

print(f"Extracted {len(prior_delta1)} prior samples for delta1 and delta2.")

# --- Create Basic 2D Histogram Plot ---
fig, ax = plt.subplots(figsize=(7, 7))

# Create the 2D histogram
# 'bins' controls the number of bins in each dimension
# 'cmap' sets the color map (e.g., 'viridis', 'Blues', 'Greys')
counts, xedges, yedges, im = ax.hist2d(
    prior_delta1,
    prior_delta2,
    bins=50,          # Adjust number of bins as needed
    cmap='viridis',   # Choose a colormap
    cmin=1            # Optionally hide bins with 0 or 1 count
)

# Add a color bar to show the density scale
fig.colorbar(im, ax=ax, label='Sample Count')

# --- Add Labels and Title ---
ax.set_title("Shared Model: Basic 2D Histogram of Prior Deltas")
ax.set_xlabel("delta1 (p_no_handler - p_original)")
ax.set_ylabel("delta2 (p_no_q - p_no_handler)")

# Optional: Add reference lines at 0
ax.axvline(0, color='k', linestyle=':', alpha=0.5)
ax.axhline(0, color='k', linestyle=':', alpha=0.5)

# Optional: Set axis limits if needed (deltas are between -1 and 1)
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)

# --- Save or Show Plot ---
plt.tight_layout()
plt.savefig(FIG_PATH / "basic_shared_prior_delta_hist2d.png")
print("Saved basic prior plot to basic_shared_prior_delta_hist2d.png")
# plt.show() # Uncomment to display interactively if backend supports it
plt.close(fig) # Close the figure

print("Plotting complete.")
