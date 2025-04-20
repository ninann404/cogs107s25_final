"""
MPT Model for the Weapon Identification Task

The data we use is due to Rivers (2017):

    Rivers, A. M. (2017). The weapons identification task: Recommendations
    for adequately powered research. PLoS ONE, 12(6), e0177857. 
    https://doi.org/10.1371/journal.pone.0177857
    
The data are directly fetched from the OSF repository:
    https://osf.io/5jzhr/download

The models are found in Heck et al. (2023), starting on page 572 and illustrated in Figure 3:

    Heck, D. W., et al. (2023). A review of applications of the Bayes factor in psychological research. Psychological Methods, 28(3), 558-579. 
    https://doi.org/10.1037/met0000454
    
Throughout, we will ignore the neutral priming condition.
"""

import pandas as pd
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Stroop + Guessing Model
#   Parameters:
#     alpha: Probability of an automatic, stereotype-consistent response.
#     beta: Probability of controlled, accurate response.
#     gamma: Probability of guessing “tool” when guessing occurs.
#   Model Structure:
#     Sequential processing: automatic → controlled → guessing.

def fit_stroop_model(data, n_trials):
    """Fits the Stroop + Guessing MPT model and returns the trace."""

    with pm.Model() as model:
        
        alpha = pm.Uniform("alpha", 0, 1)
        beta = pm.Uniform("beta", 0, 1)
        gamma = pm.Beta("gamma", 3, 3)

        theta_wt = alpha + (1 - alpha) * beta + (1 - alpha) * (1 - beta) * gamma
        theta_bt = (1 - alpha) * beta + (1 - alpha) * (1 - beta) * gamma
        theta_wg = (1 - alpha) * beta + (1 - alpha) * (1 - beta) * (1 - gamma)
        theta_bg = alpha + (1 - alpha) * beta + (1 - alpha) * (1 - beta) * (1 - gamma)

        pm.Binomial("white_tool", n=n_trials, p=theta_wt, observed=data["white_tool"])
        pm.Binomial("black_tool", n=n_trials, p=theta_bt, observed=data["black_tool"])
        pm.Binomial("white_gun", n=n_trials, p=theta_wg, observed=data["white_gun"])
        pm.Binomial("black_gun", n=n_trials, p=theta_bg, observed=data["black_gun"])

        trace = pm.sample(1000, tune=1000, target_accept=0.9, idata_kwargs={"log_likelihood": True})

    return trace


# Process Dissociation + Guessing Model
#   Parameters:
#     A: Probability of automatic processing.
#     C: Probability of controlled processing.
#     B: Probability of guessing “tool” when guessing occurs.
#   Model Structure:
#     Automatic and controlled processing are independent branches, 
#     allowing for dual routes to correct/incorrect responses.

def fit_pd_model(data, n_trials):
    """Fits the Process Dissociation + Guessing MPT model and returns the trace."""
    pass


# Saturated MPT Model
#   Parameters:
#     theta_wt: white tool correct response
#     theta_bt: black tool correct response
#     theta_wg: white gun correct response
#     theta_bg: black gun correct response
#   Model Structure:
#     Each response type has its own independent probability of correct response.

def fit_saturated_model(data, n_trials):
    """Fits the Saturated MPT model and returns the trace."""

    with pm.Model() as model:

        theta_wt = pm.Beta("theta_wt", 1, 1)
        theta_bt = pm.Beta("theta_bt", 1, 1)
        theta_wg = pm.Beta("theta_wg", 1, 1)
        theta_bg = pm.Beta("theta_bg", 1, 1)

        pm.Binomial("white_tool", n=n_trials, p=theta_wt, observed=data["white_tool"])
        pm.Binomial("black_tool", n=n_trials, p=theta_bt, observed=data["black_tool"])
        pm.Binomial("white_gun", n=n_trials, p=theta_wg, observed=data["white_gun"])
        pm.Binomial("black_gun", n=n_trials, p=theta_bg, observed=data["black_gun"])

        trace = pm.sample(1000, tune=1000, target_accept=0.9, idata_kwargs={"log_likelihood": True})

    return trace

# Runner function
def run_all_models(conditions_data, num_trials):
    """Runs all model fits and returns a dictionary of traces."""

    model_traces = {}

    for deadline, data in conditions_data.items():
        print(f"\n--- Fitting models for {deadline} ---")
        print("Fitting Stroop model...")
        model_traces[f"stroop_{deadline}"] = fit_stroop_model(data, num_trials)
        print("Fitting PD model...")
        model_traces[f"pd_{deadline}"] = fit_pd_model(data, num_trials)
        print("Fitting Saturated model...")
        model_traces[f"saturated_{deadline}"] = fit_saturated_model(data, num_trials)

    return model_traces 

def load_rivers_data():
    """
    Loads the Rivers (2017) dataset directly from OSF.
    There's some poor formatting in the raw data, so we need to clean it up a little.

    Args:
        url (str): The direct download URL for the data file.

    Returns:
        dict: A dictionary with keys '500ms' and '1000ms', each containing a dictionary with keys 'white_tool', 'black_tool', 'white_gun', and 'black_gun'.
    """

    DATA_URL = "https://osf.io/5jzhr/download"

    # Data needs to be cleaned up a little
    df = pd.read_csv(DATA_URL, sep=',', skipinitialspace=True, header=0, engine='python')
    df.columns = ['deadline500ms', 'deadline1000ms']
    df.index = df.index.str.strip()
    
    print(df)

    data = {
        "500ms": {
            "white_tool": df.loc["white_tool_1", "deadline500ms"],
            "black_tool": df.loc["black_tool_1", "deadline500ms"],
            "white_gun": df.loc["white_gun_1", "deadline500ms"],
            "black_gun": df.loc["black_gun_1", "deadline500ms"]
        },
        "1000ms": {
            "white_tool": df.loc["white_tool_1", "deadline1000ms"],
            "black_tool": df.loc["black_tool_1", "deadline1000ms"],
            "white_gun": df.loc["white_gun_1", "deadline1000ms"],
            "black_gun": df.loc["black_gun_1", "deadline1000ms"]
        }
    }
    return data
    

# --- Constants ---
N_TRIALS = 1440

FIG_DIR = Path("1-mpt/figures") # Directory to save output figures

# --- Main execution block ---
if __name__ == "__main__":

    # --- Create Figure Directory ---
    try:
        FIG_DIR.mkdir(parents=True, exist_ok=True)
        print(f"Figures will be saved to: {FIG_DIR.resolve()}")
    except OSError as e:
        print(f"Error creating figure directory {FIG_DIR}: {e}", file=sys.stderr)

    data = load_rivers_data()

    all_results = run_all_models(data, N_TRIALS)

    # Visualization
    idata_stroop_500ms = all_results["stroop_500ms"]

    az.plot_pair(
        idata_stroop_500ms,
        var_names=["alpha", "beta", "gamma"],
        kind="kde",
        figsize=(8, 8)
    )
    plt.suptitle("Stroop Model (500ms) Posterior Pair Plot", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'stroop_500ms_pairplot.png')

    # Print summary
    print("\nStroop 500ms Model Summary:")
    print(az.summary(idata_stroop_500ms, var_names=['alpha', 'beta', 'gamma']))

    # Create Matplotlib figure and axes
    fig, ax = plt.subplots(figsize=(8, 5))

    variables_to_plot = ['alpha', 'beta', 'gamma']
    colors = ['#0000DD', '#DD0000', '#DD9500'] 

    for var_name, color in zip(variables_to_plot, colors):
        az.plot_density(
            idata_stroop_500ms,
            var_names=[var_name],
            hdi_prob=1.0,
            shade=0.2,
            point_estimate=None,
            ax=ax,
            colors = color
        )

    ax.set_title("Stroop Model (500ms) Marginal Posteriors")
    ax.set_xlabel("Parameter Value")
    ax.set_ylabel("Density")
    ax.set_xlim(0, 1)

    plt.tight_layout()
    plt.savefig(FIG_DIR / 'stroop_500ms_marginal_posteriors.png')

    