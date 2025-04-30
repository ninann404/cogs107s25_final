# generate_plots.py
import pymc as pm
import arviz as az
import numpy as np
import matplotlib.pyplot as plt
import os

print("Generating data and defining model...")

# 0. Settings
RANDOM_SEED = 8927
np.random.seed(RANDOM_SEED)
az.style.use("arviz-darkgrid")
FIG_DIR = "." # Save figures in the current directory
POSTERIOR_PLOT_FILE = os.path.join(FIG_DIR, "ppc_plot.png")
PRIOR_PLOT_FILE = os.path.join(FIG_DIR, "prior_ppc_plot.png")
LOO_COMPARE_PLOT_FILE = os.path.join(FIG_DIR, "loo_compare_plot.png")

# 1. Generate some example data (e.g., slightly skewed)
true_mu = 5.0
true_sigma = 2.0
# Introduce some skew or outliers for demonstration
data = np.concatenate([
    np.random.normal(true_mu, true_sigma, size=80),
    np.random.normal(true_mu + 4, true_sigma / 2, size=20) # Some higher values
])
# data = np.random.normal(true_mu, true_sigma, size=100) # Alternative: simple normal data

# 2. Define PyMC Models
# Model 1: Normal likelihood
with pm.Model() as model_normal:
    mu = pm.Normal("mu", mu=0, sigma=10)
    sigma = pm.HalfNormal("sigma", sigma=5)
    likelihood = pm.Normal("y", mu=mu, sigma=sigma, observed=data)

# Model 2: Student-T likelihood (more robust to outliers)
with pm.Model() as model_t:
    mu = pm.Normal("mu", mu=0, sigma=10)
    sigma = pm.HalfNormal("sigma", sigma=5)
    nu = pm.Gamma("nu", alpha=2, beta=0.1) # Degrees of freedom
    likelihood = pm.StudentT("y", nu=nu, mu=mu, sigma=sigma, observed=data)

print("Sampling from models...")
# 3. Sample from posterior
with model_normal:
    idata_normal = pm.sample(1000, tune=1000, cores=1, random_seed=RANDOM_SEED, progressbar=False)
    idata_normal.extend(pm.sample_prior_predictive(random_seed=RANDOM_SEED))
    idata_normal.extend(pm.sample_posterior_predictive(idata_normal, random_seed=RANDOM_SEED))
    # Calculate LOO
    pm.compute_log_likelihood(idata_normal, model=model_normal) # Newer PyMC needs this
    az.loo(idata_normal, pointwise=True)


with model_t:
    idata_t = pm.sample(1000, tune=1000, cores=1, random_seed=RANDOM_SEED, progressbar=False)
    idata_t.extend(pm.sample_prior_predictive(random_seed=RANDOM_SEED))
    idata_t.extend(pm.sample_posterior_predictive(idata_t, random_seed=RANDOM_SEED))
    # Calculate LOO
    pm.compute_log_likelihood(idata_t, model=model_t) # Newer PyMC needs this
    az.loo(idata_t, pointwise=True)


print("Generating plots...")
# 4. Generate Posterior Predictive Plot (using the Normal model as example)
fig_ppc, ax_ppc = plt.subplots()
az.plot_ppc(idata_normal, ax=ax_ppc, num_pp_samples=100, legend=True)
ax_ppc.set_title("Posterior Predictive Check (Normal Model)")
fig_ppc.tight_layout()
fig_ppc.savefig(POSTERIOR_PLOT_FILE)
print(f"Saved posterior predictive plot to {POSTERIOR_PLOT_FILE}")
plt.close(fig_ppc)

# 5. Generate Prior Predictive Plot (using the Normal model as example)
fig_prior, ax_prior = plt.subplots()
az.plot_ppc(idata_normal, kind="prior", ax=ax_prior, num_pp_samples=100, legend=True)
ax_prior.set_title("Prior Predictive Check (Normal Model)")
fig_prior.tight_layout()
fig_prior.savefig(PRIOR_PLOT_FILE)
print(f"Saved prior predictive plot to {PRIOR_PLOT_FILE}")
plt.close(fig_prior)

# 6. Generate Model Comparison Plot (LOO)
comparison_df = az.compare({"normal": idata_normal, "student_t": idata_t}, ic="loo")
print("\nModel Comparison (LOO):")
print(comparison_df)

fig_comp, ax_comp = plt.subplots()
az.plot_compare(comparison_df, insample_dev=False, ax=ax_comp) # Changed legend=True to insample_dev=False (newer Arviz)
ax_comp.set_title("Model Comparison using LOO-CV")
fig_comp.tight_layout()
fig_comp.savefig(LOO_COMPARE_PLOT_FILE)
print(f"Saved LOO comparison plot to {LOO_COMPARE_PLOT_FILE}")
plt.close(fig_comp)

print("Plot generation complete.")