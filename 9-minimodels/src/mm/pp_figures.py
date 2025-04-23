import pickle as pkl
import arviz as az # Still useful for InferenceData structure
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.lines as mlines
import matplotlib.patches as mpatches # Import patches
from pathlib import Path
from mpl_toolkits.axes_grid1 import make_axes_locatable # For positioning colorbars

ROOT_PATH = Path("/home/jovyan/cogs107/1-mpt/src/mm")
PKL_PATH = ROOT_PATH / "results.pkl"
FIG_PATH = ROOT_PATH
FIG_PATH.mkdir(parents=True, exist_ok=True) # Ensure the figure directory exists

# --- Load Data ---
try:
    with open(PKL_PATH, "rb") as f:
        results = pkl.load(f)
except FileNotFoundError:
    print(f"Error: {PKL_PATH} not found. Run hierarchical.mm.py first.")
    exit()
except Exception as e:
    print(f"Error loading {PKL_PATH}: {e}")
    exit()

# --- Extract InferenceData Objects ---
posterior_shared = results.get("posterior", {}).get("shared_model")
posterior_independent = results.get("posterior", {}).get("independent_model")
prior_shared = results.get("prior", {}).get("shared_model")
prior_independent = results.get("prior", {}).get("independent_model")

# --- Helper function to check data validity ---
def check_idata(idata, name, group, variables):
    if idata is None:
        print(f"Error: {name} InferenceData object not found in results.pkl.")
        return False
    if not hasattr(idata, group):
        print(f"Error: Group '{group}' not found in {name} InferenceData.")
        return False
    group_data = getattr(idata, group)
    for var in variables:
        if var not in group_data.data_vars:
            print(f"Error: Variable '{var}' not found in group '{group}' of {name} InferenceData.")
            return False
    return True

# --- Plotting ---
print("\n--- Generating Delta Distribution Plots (Prior Hist2D / Posterior Contour) ---")

# Variables to plot
delta_vars = ("delta1", "delta2")
# Define common bins for the histograms
bins = 50
plot_range = [[-1, 1], [-1, 1]] # Set the plot range for deltas

# --- Plot for Shared Model ---
print("\n--- Shared Model ---")
if (check_idata(posterior_shared, "posterior_shared", "posterior", delta_vars) and
    check_idata(prior_shared, "prior_shared", "prior", delta_vars)):

    fig_shared, ax_shared = plt.subplots(figsize=(7.5, 7)) # Adjusted width for left colorbar
    divider = make_axes_locatable(ax_shared) # Helper for colorbar positioning

    # Extract data arrays
    posterior_delta1 = posterior_shared.posterior["delta1"].values.flatten()
    posterior_delta2 = posterior_shared.posterior["delta2"].values.flatten()
    prior_delta1 = prior_shared.prior["delta1"].values.flatten()
    prior_delta2 = prior_shared.prior["delta2"].values.flatten()

    # --- Calculate and Print Prior Stats ---
    prior_data_shared = np.stack((prior_delta1, prior_delta2), axis=-1)
    prior_mean_shared = np.mean(prior_data_shared, axis=0)
    prior_cov_shared = np.cov(prior_data_shared, rowvar=False)
    print("Shared Model Prior:")
    print(f"  Mean (delta1, delta2): {prior_mean_shared}")
    print(f"  Covariance Matrix:\n{prior_cov_shared}")

    # --- Calculate and Print Posterior Stats ---
    posterior_data_shared = np.stack((posterior_delta1, posterior_delta2), axis=-1)
    posterior_mean_shared = np.mean(posterior_data_shared, axis=0)
    posterior_cov_shared = np.cov(posterior_data_shared, rowvar=False)
    print("Shared Model Posterior:")
    print(f"  Mean (delta1, delta2): {posterior_mean_shared}")
    print(f"  Covariance Matrix:\n{posterior_cov_shared}")

    # 1. Plot Prior as Grayscale 2D Histogram Heatmap
    counts_prior, xedges_prior, yedges_prior, im_prior = ax_shared.hist2d(
        prior_delta1,
        prior_delta2,
        bins=bins,
        range=plot_range,
        cmap='Greys', # Colormap for prior
        cmin=1,       # Hide bins with very few samples
        alpha=0.8     # Make slightly transparent
    )
    # Add prior colorbar to the left
    cax_prior = divider.append_axes("left", size="5%", pad=0.6)
    cbar_prior = fig_shared.colorbar(im_prior, cax=cax_prior, label='Prior Sample Count')
    cax_prior.yaxis.set_ticks_position('left')
    cax_prior.yaxis.set_label_position('left')


    # 2. Calculate Posterior Histogram data (using the same bins as prior)
    counts_post, _, _ = np.histogram2d(
        posterior_delta1,
        posterior_delta2,
        bins=[xedges_prior, yedges_prior] # Use edges from prior hist2d
    )

    # 3. Plot Posterior as Thick Contour Lines
    non_zero_counts_post = counts_post[counts_post > 0]
    if len(non_zero_counts_post) > 0:
        levels_post = np.percentile(non_zero_counts_post, [30, 60, 90]) # Adjust percentiles as needed
        levels_post = np.maximum(levels_post, 1e-9)
        levels_post = np.unique(levels_post)
        if len(levels_post) > 0:
            xcenters = (xedges_prior[:-1] + xedges_prior[1:]) / 2
            ycenters = (yedges_prior[:-1] + yedges_prior[1:]) / 2
            ax_shared.contour(
                xcenters,
                ycenters,
                counts_post.T, # Transpose counts matrix for contour
                levels=levels_post,
                cmap='Blues', # Use colormap for posterior contours
                linestyles='solid',
                linewidths=2.5 # Make lines thicker
            )
        else:
            print("Warning: Could not determine distinct posterior contour levels for shared model.")
    else:
        print("Warning: No posterior samples found in the specified range for shared model contour.")


    # --- Remove posterior hist2d plotting and its colorbar ---


    # --- Add Labels, Title, Legend ---
    ax_shared.set_title("Shared Model: Joint Distribution of Deltas")
    ax_shared.set_xlabel("delta1 (p_no_handler - p_original)")
    ax_shared.set_ylabel("delta2 (p_no_q - p_no_handler)")
    ax_shared.axvline(0, color='k', linestyle=':', alpha=0.5)
    ax_shared.axhline(0, color='k', linestyle=':', alpha=0.5)
    ax_shared.set_xlim(plot_range[0])
    ax_shared.set_ylim(plot_range[1])

    # Manual Legend Creation - Patch for prior, Line for posterior
    prior_patch = mpatches.Patch(color=plt.cm.Greys(0.7), label='Prior (Shared)')
    posterior_line = mlines.Line2D([], [], color=plt.cm.Blues(0.7), linestyle='solid', linewidth=2.5, label='Posterior (Shared)')
    ax_shared.legend(handles=[prior_patch, posterior_line])

    plt.tight_layout(rect=[0.1, 0, 1, 1]) # Adjust layout slightly for left colorbar
    save_path_shared = FIG_PATH / "shared_delta_hist2d_overlay.png"
    fig_shared.savefig(save_path_shared)
    print(f"\nSaved shared model plot to {save_path_shared}")
    plt.close(fig_shared)

else:
    print("Skipping Shared Model delta plot and stats due to missing data or variables.")


# --- Plot for Independent Model ---
print("\n--- Independent Model ---")
if (check_idata(posterior_independent, "posterior_independent", "posterior", delta_vars) and
    check_idata(prior_independent, "prior_independent", "prior", delta_vars)):

    fig_indep, ax_indep = plt.subplots(figsize=(7.5, 7)) # Adjusted width
    divider_indep = make_axes_locatable(ax_indep)

    # Extract data arrays
    posterior_delta1_ind = posterior_independent.posterior["delta1"].values.flatten()
    posterior_delta2_ind = posterior_independent.posterior["delta2"].values.flatten()
    prior_delta1_ind = prior_independent.prior["delta1"].values.flatten()
    prior_delta2_ind = prior_independent.prior["delta2"].values.flatten()

    # --- Calculate and Print Prior Stats ---
    prior_data_indep = np.stack((prior_delta1_ind, prior_delta2_ind), axis=-1)
    prior_mean_indep = np.mean(prior_data_indep, axis=0)
    prior_cov_indep = np.cov(prior_data_indep, rowvar=False)
    print("Independent Model Prior:")
    print(f"  Mean (delta1, delta2): {prior_mean_indep}")
    print(f"  Covariance Matrix:\n{prior_cov_indep}")

    # --- Calculate and Print Posterior Stats ---
    posterior_data_indep = np.stack((posterior_delta1_ind, posterior_delta2_ind), axis=-1)
    posterior_mean_indep = np.mean(posterior_data_indep, axis=0)
    posterior_cov_indep = np.cov(posterior_data_indep, rowvar=False)
    print("Independent Model Posterior:")
    print(f"  Mean (delta1, delta2): {posterior_mean_indep}")
    print(f"  Covariance Matrix:\n{posterior_cov_indep}")


    # 1. Plot Prior as Grayscale 2D Histogram Heatmap
    counts_prior_ind, xedges_prior_ind, yedges_prior_ind, im_prior_ind = ax_indep.hist2d(
        prior_delta1_ind,
        prior_delta2_ind,
        bins=bins,
        range=plot_range,
        cmap='Greys',
        cmin=1,
        alpha=0.8
    )
    # Add prior colorbar to the left
    cax_prior_indep = divider_indep.append_axes("left", size="5%", pad=0.6)
    cbar_prior_indep = fig_indep.colorbar(im_prior_ind, cax=cax_prior_indep, label='Prior Sample Count')
    cax_prior_indep.yaxis.set_ticks_position('left')
    cax_prior_indep.yaxis.set_label_position('left')


    # 2. Calculate Posterior Histogram data
    counts_post_ind, _, _ = np.histogram2d(
        posterior_delta1_ind,
        posterior_delta2_ind,
        bins=[xedges_prior_ind, yedges_prior_ind] # Use edges from prior hist2d
    )

    # 3. Plot Posterior as Thick Contour Lines
    non_zero_counts_post_ind = counts_post_ind[counts_post_ind > 0]
    if len(non_zero_counts_post_ind) > 0:
        levels_post_ind = np.percentile(non_zero_counts_post_ind, [30, 60, 90])
        levels_post_ind = np.maximum(levels_post_ind, 1e-9)
        levels_post_ind = np.unique(levels_post_ind)
        if len(levels_post_ind) > 0:
            xcenters_ind = (xedges_prior_ind[:-1] + xedges_prior_ind[1:]) / 2
            ycenters_ind = (yedges_prior_ind[:-1] + yedges_prior_ind[1:]) / 2
            ax_indep.contour(
                xcenters_ind,
                ycenters_ind,
                counts_post_ind.T,
                levels=levels_post_ind,
                cmap='Reds', # Use colormap for posterior contours
                linestyles='solid',
                linewidths=2.5 # Make lines thicker
            )
        else:
            print("Warning: Could not determine distinct posterior contour levels for independent model.")
    else:
        print("Warning: No posterior samples found in the specified range for independent model contour.")

    # --- Remove posterior hist2d plotting and its colorbar ---

    # --- Add Labels, Title, Legend ---
    ax_indep.set_title("Independent Model: Joint Distribution of Deltas")
    ax_indep.set_xlabel("delta1 (p_no_handler - p_original)")
    ax_indep.set_ylabel("delta2 (p_no_q - p_no_handler)")
    ax_indep.axvline(0, color='k', linestyle=':', alpha=0.5)
    ax_indep.axhline(0, color='k', linestyle=':', alpha=0.5)
    ax_indep.set_xlim(plot_range[0])
    ax_indep.set_ylim(plot_range[1])

    # Manual Legend Creation - Patch for prior, Line for posterior
    prior_patch_ind = mpatches.Patch(color=plt.cm.Greys(0.7), label='Prior (Independent)')
    posterior_line_ind = mlines.Line2D([], [], color=plt.cm.Reds(0.7), linestyle='solid', linewidth=2.5, label='Posterior (Independent)')
    ax_indep.legend(handles=[prior_patch_ind, posterior_line_ind])

    plt.tight_layout(rect=[0.1, 0, 1, 1]) # Adjust layout slightly
    save_path_indep = FIG_PATH / "independent_delta_hist2d_overlay.png"
    fig_indep.savefig(save_path_indep)
    print(f"\nSaved independent model plot to {save_path_indep}")
    plt.close(fig_indep)

else:
    print("Skipping Independent Model delta plot and stats due to missing data or variables.")

print("\nDelta plotting complete.")