import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import os
import matplotlib.cm as cm # Import colormap

# --- Configuration ---
SEED = 42
NUM_POINTS = 30
DATA_MEAN = 2.0 # Used for partial pooling figure, represents one group's mean
DATA_STD = 0.25 # Used for partial pooling figure, represents one group's std dev

PRIOR_MEAN = 0.0 # Population mean for prior figure
PRIOR_STD = 1.0  # Population std dev for prior figure

# Shrinkage Figure Config
SHRINK_POP_MEAN = 1.0
SHRINK_POP_STD = 3.0  # Between-group variability (sigma_alpha)
SHRINK_WITHIN_STD = 10.0 # Within-group variability (sigma_y), assumed constant
SHRINK_NUM_GROUPS = 5
# Simulate different group sizes/precisions
SHRINK_GROUP_NS = np.array([5, 10, 20, 50, 100])
# Simulate true group means around the population mean
np.random.seed(SEED + 1) # Use different seed for this part
SHRINK_TRUE_GROUP_MEANS = np.random.normal(loc=SHRINK_POP_MEAN, scale=SHRINK_POP_STD, size=SHRINK_NUM_GROUPS)
# Simulate observed group means (no-pooling estimates) based on true means and sample size
# More variance in estimate for smaller N
SHRINK_OBS_GROUP_MEANS = np.random.normal(loc=SHRINK_TRUE_GROUP_MEANS, scale=SHRINK_WITHIN_STD / np.sqrt(SHRINK_GROUP_NS))

# Multilevel Structure Figure Config
ML_POP_MEAN = 0.0
ML_POP_STD = 2.0  # Between-group SD (sigma_alpha)
ML_WITHIN_STD = 0.75 # Within-group SD (sigma_y)
ML_NUM_GROUPS = 5
ML_X_RANGE_MIN = -6
ML_X_RANGE_MAX = 6

# Aesthetics
POINT_COLOR = 'blue'
POINT_SIZE = 50
POINT_ALPHA = 0.6
JITTER_AMOUNT = 0.01 # Small vertical jitter for visibility
PDF_COLOR_POOLED = 'red'
PDF_COLOR_PARTIAL = 'green'
PDF_LINESTYLE = '-'
PDF_LINEWIDTH = 2
X_RANGE_MIN = -3
X_RANGE_MAX = 5
FIG_WIDTH = 6
FIG_HEIGHT = 2 # Height for first two figs
SHRINK_FIG_HEIGHT = 3 # Height for shrinkage fig
ML_FIG_HEIGHT = 3.5 # Height for multilevel fig
OUTPUT_DIR = "tex/figures"
FILE_PRIOR = "fig_003_prior.pdf"
FILE_PARTIAL = "fig_003_partial.pdf"
FILE_SHRINKAGE = "fig_003_shrinkage.pdf"
FILE_MULTILEVEL = "fig_003_multilevel.pdf" # New figure filename

# Shrinkage Plot Aesthetics
SHRINK_COLOR_POP_MEAN = 'black'
SHRINK_COLOR_OBS = 'orange'
SHRINK_COLOR_SHRUNKEN = 'purple'
SHRINK_MARKER_OBS = 'o'
SHRINK_MARKER_SHRUNKEN = 'x'
SHRINK_ARROW_COLOR = 'gray'
SHRINK_ARROW_STYLE = '-|>' # Arrow style

# Multilevel Plot Aesthetics
ML_COLOR_POP = 'black'
ML_LINEWIDTH_POP = 2.5
ML_LINESTYLE_POP = '-'
ML_LINEWIDTH_GROUP = 1.5
ML_LINESTYLE_GROUP = '--'
ML_COLORMAP = cm.viridis # Colormap for groups

# --- Ensure output directory exists ---
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Setup ---
np.random.seed(SEED) # Reset seed for consistency if needed elsewhere
plt.style.use('seaborn-v0_8-talk') # Or another preferred style

# --- Generate Data for first two plots ---
data_points = np.random.normal(loc=DATA_MEAN, scale=DATA_STD, size=NUM_POINTS)
x_values = np.linspace(X_RANGE_MIN, X_RANGE_MAX, 500)
y_jitter = np.random.uniform(-JITTER_AMOUNT, JITTER_AMOUNT, size=NUM_POINTS)

# --- Figure 1: Prior (N(0,1) overlay) ---
fig1, ax1 = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))

# Plot data points
ax1.scatter(data_points, y_jitter, color=POINT_COLOR, s=POINT_SIZE, alpha=POINT_ALPHA, zorder=2)

# Plot N(0,1) PDF
pooled_pdf = norm.pdf(x_values, loc=PRIOR_MEAN, scale=PRIOR_STD)
ax1.plot(x_values, pooled_pdf, color=PDF_COLOR_POOLED, linestyle=PDF_LINESTYLE, linewidth=PDF_LINEWIDTH, label=f'N({PRIOR_MEAN}, {PRIOR_STD}) Prior')

# Customize plot
ax1.set_yticks([]) # Remove y-axis ticks
ax1.spines['left'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)
ax1.set_xlabel("Value")
ax1.legend(loc='upper right', fontsize='small')
ax1.set_xlim(X_RANGE_MIN, X_RANGE_MAX)
ax1.set_ylim(bottom=-JITTER_AMOUNT*2, top=1.65) # Ensure points aren't cut off

plt.tight_layout()
fig1_path = os.path.join(OUTPUT_DIR, FILE_PRIOR)
plt.savefig(fig1_path, transparent=True)
print(f"Saved figure: {fig1_path}")
plt.close(fig1)

# --- Figure 2: Partial/Group (N(DATA_MEAN, DATA_STD) overlay) ---
fig2, ax2 = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))

# Plot data points
ax2.scatter(data_points, y_jitter, color=POINT_COLOR, s=POINT_SIZE, alpha=POINT_ALPHA, zorder=2)

# Plot N(DATA_MEAN, DATA_STD) PDF
partial_pdf = norm.pdf(x_values, loc=DATA_MEAN, scale=DATA_STD)
ax2.plot(x_values, partial_pdf, color=PDF_COLOR_PARTIAL, linestyle=PDF_LINESTYLE, linewidth=PDF_LINEWIDTH, label=f'N({DATA_MEAN}, {DATA_STD}) Group')

# Customize plot
ax2.set_yticks([]) # Remove y-axis ticks
ax2.spines['left'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax2.set_xlabel("Value")
ax2.legend(loc='upper right', fontsize='small')
ax2.set_xlim(X_RANGE_MIN, X_RANGE_MAX)
ax2.set_ylim(bottom=-JITTER_AMOUNT*2, top=1.65) # Ensure points aren't cut off

plt.tight_layout()
fig2_path = os.path.join(OUTPUT_DIR, FILE_PARTIAL)
plt.savefig(fig2_path, transparent=True)
print(f"Saved figure: {fig2_path}")
plt.close(fig2)

# --- Figure 3: Shrinkage Illustration ---
fig3, ax3 = plt.subplots(figsize=(FIG_WIDTH, SHRINK_FIG_HEIGHT))

# Calculate precision and weights
group_precision = SHRINK_GROUP_NS / (SHRINK_WITHIN_STD**2)
population_precision = 1 / (SHRINK_POP_STD**2)
weights = group_precision / (group_precision + population_precision)

# Calculate shrunken estimates
shrunken_means = weights * SHRINK_OBS_GROUP_MEANS + (1 - weights) * SHRINK_POP_MEAN

# Plotting
y_positions = np.arange(SHRINK_NUM_GROUPS) + 0.5 # Vertical positions for groups

# Plot population mean line
ax3.axvline(SHRINK_POP_MEAN, color=SHRINK_COLOR_POP_MEAN, linestyle=':', linewidth=2, label=f'Pop Mean ({SHRINK_POP_MEAN:.1f})', zorder=1)

# Plot observed and shrunken means for each group
for i in range(SHRINK_NUM_GROUPS):
    obs_mean = SHRINK_OBS_GROUP_MEANS[i]
    shrunk_mean = shrunken_means[i]
    y_pos = y_positions[i]
    group_n = SHRINK_GROUP_NS[i]

    # Plot arrow showing shrinkage
    ax3.annotate("", xy=(shrunk_mean, y_pos), xytext=(obs_mean, y_pos),
                 arrowprops=dict(arrowstyle=SHRINK_ARROW_STYLE, color=SHRINK_ARROW_COLOR, shrinkA=5, shrinkB=5, lw=1.5), zorder=2) # shrinkA/B prevent overlap with points

    # Plot observed mean point
    ax3.scatter(obs_mean, y_pos, color=SHRINK_COLOR_OBS, marker=SHRINK_MARKER_OBS, s=POINT_SIZE+20, zorder=3, label='Observed Estimate' if i == 0 else "")
    # Plot shrunken mean point
    ax3.scatter(shrunk_mean, y_pos, color=SHRINK_COLOR_SHRUNKEN, marker=SHRINK_MARKER_SHRUNKEN, s=POINT_SIZE+20, zorder=3, label='Shrunken Estimate' if i == 0 else "")
    # Add text for group size
    ax3.text(X_RANGE_MAX - 0.1, y_pos, f'N={group_n}', ha='right', va='center', fontsize='x-small')


# Customize plot
ax3.set_yticks(y_positions)
ax3.set_yticklabels([f'Participant {j+1}' for j in range(SHRINK_NUM_GROUPS)])
# ax3.set_xlabel("Estimated Group Mean")
# ax3.set_title("Illustration of Shrinkage")
ax3.legend(loc='upper left', fontsize='small')
ax3.set_xlim(X_RANGE_MIN, X_RANGE_MAX)
ax3.grid(axis='x', linestyle='--', alpha=0.6)
ax3.spines['right'].set_visible(False)
ax3.spines['top'].set_visible(False)

plt.tight_layout()
fig3_path = os.path.join(OUTPUT_DIR, FILE_SHRINKAGE)
plt.savefig(fig3_path, transparent=True)
print(f"Saved figure: {fig3_path}")
plt.close(fig3)

# --- Figure 4: Multilevel Structure Illustration ---
fig4, ax4 = plt.subplots(figsize=(FIG_WIDTH, ML_FIG_HEIGHT))

# X values for PDFs
ml_x_values = np.linspace(ML_X_RANGE_MIN, ML_X_RANGE_MAX, 500)

# Plot Population Distribution PDF
pop_pdf = norm.pdf(ml_x_values, loc=ML_POP_MEAN, scale=ML_POP_STD)
ax4.plot(ml_x_values, pop_pdf, color=ML_COLOR_POP, linestyle=ML_LINESTYLE_POP,
         linewidth=ML_LINEWIDTH_POP, label=f'Population Dist.\nN({ML_POP_MEAN:.1f}, {ML_POP_STD**2:.1f})', zorder=2)

# Simulate group means from population distribution
np.random.seed(SEED+1) # Use different seed
group_means = np.random.normal(loc=ML_POP_MEAN, scale=ML_POP_STD, size=ML_NUM_GROUPS)

# Get colors from colormap
group_colors = ML_COLORMAP(np.linspace(0, 1, ML_NUM_GROUPS))

# Plot Group Distribution PDFs
for i in range(ML_NUM_GROUPS):
    group_mean = group_means[i]
    group_pdf = norm.pdf(ml_x_values, loc=group_mean, scale=ML_WITHIN_STD)
    ax4.plot(ml_x_values, group_pdf, color=group_colors[i], linestyle=ML_LINESTYLE_GROUP,
             linewidth=ML_LINEWIDTH_GROUP, label=f'Group {i+1} (Mean={group_mean:.1f})' if i < 3 else "", # Label only first few
             zorder=1)
    # Mark the group mean
    ax4.axvline(group_mean, color=group_colors[i], linestyle=':', linewidth=0.8, ymax=1, zorder=0)


# Customize plot
ax4.set_ylabel("Density")
ax4.legend(loc='upper right', fontsize='x-small')
ax4.set_xlim(ML_X_RANGE_MIN, ML_X_RANGE_MAX)
ax4.set_ylim(bottom=0)
ax4.spines['right'].set_visible(False)
ax4.spines['top'].set_visible(False)
ax4.grid(axis='y', linestyle='--', alpha=0.5)

plt.tight_layout()
fig4_path = os.path.join(OUTPUT_DIR, FILE_MULTILEVEL)
plt.savefig(fig4_path, transparent=True)
print(f"Saved figure: {fig4_path}")
plt.close(fig4) 