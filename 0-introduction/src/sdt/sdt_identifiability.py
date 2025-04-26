# sdt_identifiability.py
"""
Demonstration of Non-Identifiability in a Hierarchical SDT Model

This script simulates data from a Signal Detection Theory (SDT) model
where sensitivity (d') is influenced by participant effects and condition effects.
It defines two versions of the model:
1. A non-identified version with redundant intercept parameters.
2. An identified version where the redundancy is removed.

The script fits both models using PyMC and generates diagnostic plots (trace plots, pair plots)
to illustrate the symptoms of non-identifiability (poor convergence, high correlations)
and how they are resolved in the identified model.

The specific non-identifiability arises from the additive structure:
d'_ip = alpha_p + gamma_i
gamma_i = zeta_0 + zeta_1 * predictor_i
alpha_p ~ N(mu_alpha, sigma_alpha^2)

Here, mu_alpha and zeta_0 are confounded.
"""

import pymc as pm
import arviz as az
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy import stats # Use scipy.stats for norm cdf
from pathlib import Path
import sys

Phi = pm.math.invprobit

###############################################################################
### --- Model Definitions ---
###############################################################################

def build_nonid_sdt_model(sim_data, coords):
    """
    Builds the non-identified hierarchical SDT model in PyMC.

    This version includes both a population mean for person intercepts (mu_alpha)
    and a separate intercept for the condition effect (zeta0), leading to
    non-identifiability.

    Args:
        sim_data (dict): Dictionary containing the observed data ('hits', 'fas',
                         'n_signal', 'n_noise', 'person_idx', 'condition_predictor').
        coords (dict): Dictionary defining the coordinates for PyMC model dimensions.

    Returns:
        pm.Model: The non-identified PyMC model instance.
    """
    with pm.Model(coords=coords) as model_nonid:
        # --- Data Containers ---
        # Use pm.Data for flexibility, though not strictly needed if data won't change
        person_idx_data = pm.Data("person_idx_data", sim_data["person_map"], dims="obs_id")
        cond_pred_data = pm.Data("cond_pred_data", sim_data["condition_predictor"], dims="obs_id")
        n_signal_data = pm.Data("n_signal_data", sim_data["n_signal"], dims="obs_id")
        n_noise_data = pm.Data("n_noise_data", sim_data["n_noise"], dims="obs_id")

        # --- Priors for Sensitivity (d') parameters ---
        mu_alpha = pm.Normal("mu_alpha", mu=0.0, sigma=1.0e9) # Population mean person sensitivity intercept (Problem 1)
        sigma_alpha = pm.HalfNormal("sigma_alpha", sigma=10) # SD person sensitivity intercept
        zeta0 = pm.Normal("zeta0", mu=0.0, sigma=1.0e9)       # Sensitivity offset/condition intercept (Problem 2)
        zeta1 = pm.Normal("zeta1", mu=0.0, sigma=1.0e9)       # Sensitivity slope (effect of condition predictor)

        # --- Priors for Criterion (c) parameters ---
        mu_c = pm.Normal("mu_c", mu=0.0, sigma=1.0)         # Population mean criterion
        sigma_c = pm.HalfNormal("sigma_c", sigma=0.5)       # SD criterion

        # --- Person-specific parameters (Non-centered) ---
        # Sensitivity intercept deviations
        alpha_p_offset = pm.Normal("alpha_p_offset", mu=0.0, sigma=1.0, dims="person")
        alpha_p = pm.Deterministic("alpha_p", mu_alpha + alpha_p_offset * sigma_alpha, dims="person")
        # Criterion deviations
        c_p_offset = pm.Normal("c_p_offset", mu=0.0, sigma=1.0, dims="person")
        c_p = pm.Deterministic("c_p", mu_c + c_p_offset * sigma_c, dims="person")

        # --- Condition Effects on Sensitivity (gamma_i) ---
        # This includes the problematic intercept zeta0
        gamma_i = pm.Deterministic("gamma_i", zeta0 + zeta1 * cond_pred_data, dims="obs_id")

        # --- Calculate d' per observation ---
        d_prime_ip = pm.Deterministic("d_prime_ip", alpha_p[person_idx_data] + gamma_i, dims="obs_id")

        # --- SDT calculations: HR and FAR using Normal CDF (phi) ---
        # Using pm.math.phi which corresponds to stats.norm.cdf(x)
        hr_ip = pm.Deterministic("hr_ip", Phi(d_prime_ip / 2.0 - c_p[person_idx_data]), dims="obs_id")
        far_ip = pm.Deterministic("far_ip", Phi(-d_prime_ip / 2.0 - c_p[person_idx_data]), dims="obs_id")

        # --- Likelihood (Binomial for Hits and FAs) ---
        hits_like = pm.Binomial(
            "hits_like",
            n=n_signal_data,
            p=hr_ip,
            observed=sim_data["hits"], # Pass numpy array directly
            dims="obs_id"
        )
        fas_like = pm.Binomial(
            "fas_like",
            n=n_noise_data,
            p=far_ip,
            observed=sim_data["fas"], # Pass numpy array directly
            dims="obs_id"
        )
    return model_nonid


def build_semiid_sdt_model(sim_data, coords):
    """
    Builds the semi-identified hierarchical SDT model in PyMC.

    This version includes both a population mean for person intercepts (mu_alpha)
    and a separate intercept for the condition effect (zeta0), leading to
    non-identifiability.  The model has somewhat informative priors, so the lack
    of identifiability is not as obvious.

    Args:
        sim_data (dict): Dictionary containing the observed data ('hits', 'fas',
                         'n_signal', 'n_noise', 'person_idx', 'condition_predictor').
        coords (dict): Dictionary defining the coordinates for PyMC model dimensions.

    Returns:
        pm.Model: The non-identified PyMC model instance.
    """
    with pm.Model(coords=coords) as model_semiid:
        # --- Data Containers ---
        # Use pm.Data for flexibility, though not strictly needed if data won't change
        person_idx_data = pm.Data("person_idx_data", sim_data["person_map"], dims="obs_id")
        cond_pred_data = pm.Data("cond_pred_data", sim_data["condition_predictor"], dims="obs_id")
        n_signal_data = pm.Data("n_signal_data", sim_data["n_signal"], dims="obs_id")
        n_noise_data = pm.Data("n_noise_data", sim_data["n_noise"], dims="obs_id")

        # --- Priors for Sensitivity (d') parameters ---
        mu_alpha = pm.Normal("mu_alpha", mu=0.0, sigma=1.5) # Population mean person sensitivity intercept (Problem 1)
        sigma_alpha = pm.HalfNormal("sigma_alpha", sigma=1) # SD person sensitivity intercept
        zeta0 = pm.Normal("zeta0", mu=0.0, sigma=1)       # Sensitivity offset/condition intercept (Problem 2)
        zeta1 = pm.Normal("zeta1", mu=0.0, sigma=1)       # Sensitivity slope (effect of condition predictor)

        # --- Priors for Criterion (c) parameters ---
        mu_c = pm.Normal("mu_c", mu=0.0, sigma=1.0)         # Population mean criterion
        sigma_c = pm.HalfNormal("sigma_c", sigma=0.5)       # SD criterion

        # --- Person-specific parameters (Non-centered) ---
        # Sensitivity intercept deviations
        alpha_p_offset = pm.Normal("alpha_p_offset", mu=0.0, sigma=1.0, dims="person")
        alpha_p = pm.Deterministic("alpha_p", mu_alpha + alpha_p_offset * sigma_alpha, dims="person")
        # Criterion deviations
        c_p_offset = pm.Normal("c_p_offset", mu=0.0, sigma=1.0, dims="person")
        c_p = pm.Deterministic("c_p", mu_c + c_p_offset * sigma_c, dims="person")

        # --- Condition Effects on Sensitivity (gamma_i) ---
        # This includes the problematic intercept zeta0
        gamma_i = pm.Deterministic("gamma_i", zeta0 + zeta1 * cond_pred_data, dims="obs_id")

        # --- Calculate d' per observation ---
        d_prime_ip = pm.Deterministic("d_prime_ip", alpha_p[person_idx_data] + gamma_i, dims="obs_id")

        # --- SDT calculations: HR and FAR using Normal CDF (phi) ---
        # Using pm.math.phi which corresponds to stats.norm.cdf(x)
        hr_ip = pm.Deterministic("hr_ip", Phi(d_prime_ip / 2.0 - c_p[person_idx_data]), dims="obs_id")
        far_ip = pm.Deterministic("far_ip", Phi(-d_prime_ip / 2.0 - c_p[person_idx_data]), dims="obs_id")

        # --- Likelihood (Binomial for Hits and FAs) ---
        hits_like = pm.Binomial(
            "hits_like",
            n=n_signal_data,
            p=hr_ip,
            observed=sim_data["hits"], # Pass numpy array directly
            dims="obs_id"
        )
        fas_like = pm.Binomial(
            "fas_like",
            n=n_noise_data,
            p=far_ip,
            observed=sim_data["fas"], # Pass numpy array directly
            dims="obs_id"
        )
    return model_semiid


def build_id_sdt_model(sim_data, coords):
    """
    Builds the identified hierarchical SDT model in PyMC.

    This version removes the redundant intercept `zeta0`. The parameter `mu_alpha`
    is renamed to `mu_alpha_intercept` and now represents the population mean
    sensitivity intercept when the condition predictor is zero.

    Args:
        sim_data (dict): Dictionary containing the observed data.
        coords (dict): Dictionary defining the coordinates for PyMC model dimensions.

    Returns:
        pm.Model: The identified PyMC model instance.
    """
    with pm.Model(coords=coords) as model_id:
        # --- Data Containers ---
        person_idx_data = pm.Data("person_idx_data", sim_data["person_map"], dims="obs_id")
        cond_pred_data = pm.Data("cond_pred_data", sim_data["condition_predictor"], dims="obs_id")
        n_signal_data = pm.Data("n_signal_data", sim_data["n_signal"], dims="obs_id")
        n_noise_data = pm.Data("n_noise_data", sim_data["n_noise"], dims="obs_id")

        # --- Priors for Sensitivity (d') parameters ---
        # mu_alpha renamed, zeta0 removed
        mu_alpha_intercept = pm.Normal("mu_alpha_intercept", mu=0.0, sigma=1.0e9)
        sigma_alpha = pm.HalfNormal("sigma_alpha", sigma=1.0e9)
        # zeta0 is removed
        zeta1 = pm.Normal("zeta1", mu=0.0, sigma=1.0e9)

        # --- Priors for Criterion (c) parameters (same as before) ---
        mu_c = pm.Normal("mu_c", mu=0.0, sigma=1.0)
        sigma_c = pm.HalfNormal("sigma_c", sigma=0.5)

        # --- Person-specific parameters (Non-centered) ---
        # Sensitivity intercept deviations (relative to mu_alpha_intercept)
        alpha_p_offset = pm.Normal("alpha_p_offset", mu=0.0, sigma=1.0, dims="person")
        alpha_p = pm.Deterministic("alpha_p", mu_alpha_intercept + alpha_p_offset * sigma_alpha, dims="person")
        # Criterion (same as before)
        c_p_offset = pm.Normal("c_p_offset", mu=0.0, sigma=1.0, dims="person")
        c_p = pm.Deterministic("c_p", mu_c + c_p_offset * sigma_c, dims="person")

        # --- Condition Effects on Sensitivity (gamma_i - NO intercept) ---
        gamma_i = pm.Deterministic("gamma_i", zeta1 * cond_pred_data, dims="obs_id")

        # --- Calculate d' per observation (using new alpha_p interpretation) ---
        d_prime_ip = pm.Deterministic("d_prime_ip", alpha_p[person_idx_data] + gamma_i, dims="obs_id")

        # --- SDT calculations: HR and FAR (same structure) ---
        hr_ip = pm.Deterministic("hr_ip", Phi(d_prime_ip / 2.0 - c_p[person_idx_data]), dims="obs_id")
        far_ip = pm.Deterministic("far_ip", Phi(-d_prime_ip / 2.0 - c_p[person_idx_data]), dims="obs_id")

        # --- Likelihood (Binomial - same structure) ---
        hits_like = pm.Binomial(
            "hits_like",
            n=n_signal_data,
            p=hr_ip,
            observed=sim_data["hits"],
            dims="obs_id"
        )
        fas_like = pm.Binomial(
            "fas_like",
            n=n_noise_data,
            p=far_ip,
            observed=sim_data["fas"],
            dims="obs_id"
        )
    return model_id


###############################################################################
### --- Data Generation ---
###############################################################################

def generate_sdt_data(n_persons=20, n_conditions=3, n_trials_per=100, seed=1012):
    """
    Generates synthetic data for a hierarchical SDT experiment.

    Args:
        n_persons (int): Number of participants.
        n_conditions (int): Number of conditions.
        n_trials_per (int): Number of signal/noise trials per person/condition.
        seed (int): Random seed for reproducibility.

    Returns:
        dict: A dictionary containing the simulated data and parameters.
              Includes 'hits', 'fas', 'n_signal', 'n_noise', 'person_map',
              'condition_predictor', 'persons', 'conditions', etc.
    """
    np.random.seed(seed)
    phi = stats.norm.cdf # Standard normal CDF

    # True parameter values (chosen to likely exhibit the non-id issue if used naively)
    true_mu_alpha = 1.5         # True mean sensitivity intercept component
    true_sigma_alpha = 0.5      # True SD of sensitivity intercept component
    true_zeta0 = 0.5            # True sensitivity offset (REDUNDANT PARAM)
    true_zeta1 = -0.4           # True effect of condition predictor (e.g., difficulty)
    true_mu_c = 0.1             # True mean criterion
    true_sigma_c = 0.3          # True SD of criterion

    # Conditions and predictor (e.g., 0, 1, 2 for increasing difficulty)
    conditions = [f"Cond{i+1}" for i in range(n_conditions)]
    condition_predictor_vals = np.arange(n_conditions) # e.g., [0, 1, 2]

    persons = [f"P{i+1}" for i in range(n_persons)]

    # Create full design matrix indices
    person_indices = np.repeat(np.arange(n_persons), n_conditions)
    condition_indices = np.tile(np.arange(n_conditions), n_persons)
    condition_predictor = condition_predictor_vals[condition_indices]
    n_obs = len(person_indices)

    # Simulate person effects (sensitivity intercept base and criterion)
    true_alpha_p_base = np.random.normal(true_mu_alpha, true_sigma_alpha, size=n_persons)
    true_c_p = np.random.normal(true_mu_c, true_sigma_c, size=n_persons)

    # Simulate condition effect on sensitivity (gamma_i) using the redundant param
    true_gamma_i = true_zeta0 + true_zeta1 * condition_predictor

    # Calculate true d' for each observation
    # d'_ip = alpha_p + gamma_i
    true_d_prime_ip = true_alpha_p_base[person_indices] + true_gamma_i

    # Calculate true Hit Rates and False Alarm Rates
    true_hr_ip = phi(true_d_prime_ip / 2.0 - true_c_p[person_indices])
    true_far_ip = phi(-true_d_prime_ip / 2.0 - true_c_p[person_indices])

    # Ensure probabilities are valid
    true_hr_ip = np.clip(true_hr_ip, 1e-6, 1.0 - 1e-6)
    true_far_ip = np.clip(true_far_ip, 1e-6, 1.0 - 1e-6)

    # Simulate observed Hit and False Alarm counts
    n_signal_trials = np.full(n_obs, n_trials_per)
    n_noise_trials = np.full(n_obs, n_trials_per)
    observed_hits = np.random.binomial(n_signal_trials, true_hr_ip)
    observed_fas = np.random.binomial(n_noise_trials, true_far_ip)

    print(f"Generated data for {n_persons} persons, {n_conditions} conditions each.")
    print(f"N_Signal={n_trials_per}, N_Noise={n_trials_per} per obs.")
    print(f"Predictor values for conditions: {condition_predictor_vals}")

    sim_data = {
        "hits": observed_hits,
        "fas": observed_fas,
        "n_signal": n_signal_trials,
        "n_noise": n_noise_trials,
        "person_map": person_indices, # Map obs to person index (0 to P-1)
        "condition_predictor": condition_predictor, # Predictor value for each obs
        "persons": persons,
        "conditions": conditions,
        "n_persons": n_persons,
        "n_conditions": n_conditions,
        "n_obs": n_obs,
        # Store true values for reference if needed
        "true_params": {
            "mu_alpha": true_mu_alpha, "sigma_alpha": true_sigma_alpha,
            "zeta0": true_zeta0, "zeta1": true_zeta1,
            "mu_c": true_mu_c, "sigma_c": true_sigma_c
        }
    }
    return sim_data

###############################################################################
### --- Runner Function ---
###############################################################################

def run_analysis(sim_data):
    """
    Runs the fitting and analysis for both non-identified and identified models.

    Args:
        sim_data (dict): Dictionary containing the simulated data.

    Returns:
        dict: Dictionary containing the InferenceData objects for both models.
              Keys: 'non_identified', 'semi_identified', 'identified'.
    """
    results = {}
    coords = {
        "person": sim_data["persons"],
        "condition": sim_data["conditions"], # Potentially useful if grouping later
        "obs_id": np.arange(sim_data["n_obs"]),
    }

    # --- Non-Identified Model ---
    print("\n>>> --- Fitting Non-Identified Model ---")
    model_nonid = build_nonid_sdt_model(sim_data, coords)
    print("Sampling...")
    with model_nonid:
        idata_nonid = pm.sample(1000, tune=1500, chains=4, cores=4, random_seed=RANDOM_SEED,
                                target_accept=0.9, # Higher target_accept can help
                                initvals={"mu_alpha": 0.0, "sigma_alpha": 1.0, "zeta0": 0.0, "zeta1": 0.0, "mu_c": 0.0, "sigma_c": 0.3},
                                idata_kwargs={"log_likelihood": True}) # Keep loglike if comparing models
    results["non_identified"] = idata_nonid
    print("Non-Identified Model Sampling Complete.")
    print_model_summary(idata_nonid, model_type="Non-Identified SDT")
    make_diagnostic_plots(idata_nonid, model_type="nonid_sdt",
                          vars_trace=["mu_alpha", "sigma_alpha", "zeta0", "zeta1", "mu_c", "sigma_c"],
                          vars_pair=["mu_alpha", "zeta0", "zeta1"])

    # --- Semi-Identified Model ---
    print("\n>>> --- Fitting Semi-Identified Model ---")
    model_semiid = build_semiid_sdt_model(sim_data, coords)
    print("Sampling...")
    with model_semiid:
        idata_semiid = pm.sample(1000, tune=1500, chains=4, cores=4, random_seed=RANDOM_SEED,
                                target_accept=0.9, # Higher target_accept can help
                                initvals={"mu_alpha": 0.0, "sigma_alpha": 1.0, "zeta0": 0.0, "zeta1": 0.0, "mu_c": 0.0, "sigma_c": 0.3},
                                idata_kwargs={"log_likelihood": True}) # Keep loglike if comparing models
    results["semi_identified"] = idata_semiid
    print("Semi-Identified Model Sampling Complete.")
    print_model_summary(idata_semiid, model_type="Semi-Identified SDT")
    make_diagnostic_plots(idata_semiid, model_type="semiid_sdt",
                          vars_trace=["mu_alpha", "sigma_alpha", "zeta0", "zeta1", "mu_c", "sigma_c"],
                          vars_pair=["mu_alpha", "zeta0", "zeta1"])

    # --- Identified Model ---
    print("\n>>> --- Fitting Identified Model ---")
    model_id = build_id_sdt_model(sim_data, coords)
    print("Sampling...")
    with model_id:
         idata_id = pm.sample(1000, tune=1500, chains=4, cores=4, random_seed=RANDOM_SEED,
                              target_accept=0.85, # Usually easier to sample
                              initvals={"mu_alpha_intercept": 0.0, "sigma_alpha": 1.0, "zeta1": 0.0, "mu_c": 0.0, "sigma_c": 0.3},
                              idata_kwargs={"log_likelihood": True})
    results["identified"] = idata_id
    print("Identified Model Sampling Complete.")
    print_model_summary(idata_id, model_type="Identified SDT")
    make_diagnostic_plots(idata_id, model_type="id_sdt",
                          vars_trace=["mu_alpha_intercept", "sigma_alpha", "zeta1", "mu_c", "sigma_c"],
                          vars_pair=["mu_alpha_intercept", "zeta1", "mu_c"])

    return results

###############################################################################
### --- Helper Functions ---
###############################################################################

def get_var_names_from_idata(idata):
    """Helper function to get the variable names from an ArviZ IData object."""
    # Excludes coordinates and internal variables
    return [v for v in idata.posterior.variables if v not in idata.posterior.coords]

def ensure_figure_directory_exists():
    """Ensures that the figure directory exists."""
    try:
        FIG_DIR.mkdir(parents=True, exist_ok=True)
        print(f"Figures will be saved to: {FIG_DIR.resolve()}")
    except OSError as e:
        print(f"Error creating figure directory {FIG_DIR}: {e}", file=sys.stderr)


###############################################################################
### --- Postprocessing ---
###############################################################################

def make_diagnostic_plots(idata, model_type, vars_trace, vars_pair=None):
    """
    Generates and saves diagnostic trace and pair plots.

    Args:
        idata (az.InferenceData): The inference data object.
        model_type (str): String identifier for the model (e.g., 'nonid_sdt').
        vars_trace (list): List of variable names for the trace plot.
        vars_pair (list, optional): List of variable names for the pair plot. Defaults to None.
    """
    ensure_figure_directory_exists() # Ensure dir exists before saving

    # --- Trace Plot ---
    print(f"Generating trace plot for {model_type}...")
    trace_filename = FIG_DIR / f"{model_type}_trace.png"
    try:
        az.plot_trace(idata, var_names=vars_trace)
        plt.gcf().suptitle(f"Trace Plot ({model_type.replace('_', ' ').title()})", y=1.02)
        plt.tight_layout()
        plt.savefig(trace_filename)
        print(f"Saved trace plot to {trace_filename}")
        plt.close()
    except Exception as e:
        print(f"Could not generate trace plot for {model_type}: {e}")


    # --- Pair Plot ---
    if vars_pair:
        print(f"Generating pair plot for {model_type}...")
        pair_filename = FIG_DIR / f"{model_type}_pair.png"
        try:
            az.plot_pair(idata, var_names=vars_pair,
                         kind='scatter', # Use 'kde' or 'hexbin' for large samples
                         marginals=True,
                         point_estimate='mean') # Optional: plot mean on marginals
            plt.gcf().suptitle(f"Pair Plot ({model_type.replace('_', ' ').title()})", y=1.02)
            plt.tight_layout()
            plt.savefig(pair_filename)
            print(f"Saved pair plot to {pair_filename}")
            plt.close()
        except Exception as e:
            print(f"Could not generate pair plot for {model_type}: {e}")


def print_model_summary(idata, model_type):
    """
    Prints the ArviZ summary table for specified variables.

    Args:
        idata (az.InferenceData): The inference data object.
        model_type (str): String identifier for the model type.
    """
    try:
        # Automatically get top-level variables, excluding offsets etc.
        var_names = [v for v in get_var_names_from_idata(idata)
                     if not v.endswith(("_offset", "_ip", "_p", "_i")) and 'log_likelihood' not in v]
        if not var_names: # Fallback if auto-detection fails
             var_names = get_var_names_from_idata(idata)[:5] # Limit fallback

        print(f"\n--- {model_type} Model Summary ---")
        summary = az.summary(idata, var_names=var_names, hdi_prob=0.94) # Use 94% HDI
        print(summary)

        # Highlight potential issues (high R-hat)
        high_rhat = summary[summary['r_hat'] > 1.05] # Use a slightly more lenient threshold
        if not high_rhat.empty:
            print("\nPotential Convergence Issues (R-hat > 1.05):")
            print(high_rhat)

    except Exception as e:
        print(f"Could not generate summary for {model_type}: {e}")


###############################################################################
### --- Constants ---
###############################################################################

RANDOM_SEED = 1012

# Define project root and figure directory using pathlib
# Assumes script is run from its directory or a known structure
try:
    # If run directly, parent is the script's directory
    # If imported, __file__ might be elsewhere, adjust as needed
    SCRIPT_DIR = Path(__file__).parent.resolve()
except NameError:
     # Fallback if __file__ is not defined (e.g., interactive session)
     SCRIPT_DIR = Path.cwd()

ROOT_DIR = SCRIPT_DIR # Modify if your project structure is different
FIG_DIR = ROOT_DIR.parent.parent.parent / "1-mpt/slides/tex/figures" # Specific folder for these figs


###############################################################################
### --- Main execution block ---
###############################################################################

if __name__ == "__main__":

    print("Starting SDT Identifiability Analysis...")
    ensure_figure_directory_exists()

    # 1. Generate Data
    simulated_data = generate_sdt_data(n_persons=20, n_conditions=3, n_trials_per=100, seed=RANDOM_SEED)

    # 2. Run Analysis (Fits models, prints summaries, makes plots)
    analysis_results = run_analysis(simulated_data)

    print("\nAnalysis finished.")
    # Access results if needed:
    # idata_nonid = analysis_results["non_identified"]
    # idata_id = analysis_results["identified"]