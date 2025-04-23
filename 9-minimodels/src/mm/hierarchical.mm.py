import pymc as pm
import numpy as np
import arviz as az
import matplotlib.pyplot as plt
import pickle as pkl

# Print pymc version
print(f"PyMC version: {pm.__version__}")

# Set epsilon for negligibility threshold
epsilon = 0.05

# Make data dict
data = {
    "obs_original": 4,
    "obs_no_handler": 4,
    "obs_no_q": 0,
    "N": 5
}

# Make link functions:
link  = pm.math.probit
ilink = pm.math.invprobit

def data_level(model, theta_original, theta_no_handler, theta_no_q):
    with model:
        pm.Binomial("obs_original", 
                    n=data["N"], 
                    p=theta_original, 
                    observed=data["obs_original"])
        pm.Binomial("obs_no_handler", 
                    n=data["N"], 
                    p=theta_no_handler, 
                    observed=data["obs_no_handler"])
        pm.Binomial("obs_no_q", 
                    n=data["N"], 
                    p=theta_no_q, 
                    observed=data["obs_no_q"])

        # Entailments (differences)
        delta1 = pm.Deterministic("delta1", theta_no_handler - theta_original)
        delta2 = pm.Deterministic("delta2", theta_no_q - theta_no_handler)
        
        A1 = pm.Deterministic("A1", (delta1 > -epsilon) & (delta1 < epsilon))
        A2 = pm.Deterministic("A2", (delta2 > -epsilon) & (delta2 < epsilon))
        B1 = pm.Deterministic("B1", (delta1 > -epsilon) & (delta1 < epsilon))
        B2 = pm.Deterministic("B2", (delta2 < -epsilon))
        
        A = pm.Deterministic("A", A1 & A2)
        B = pm.Deterministic("B", B1 & B2)

        return A, B

# 1. Joint Model with Shared Cue Sensitivity (theta) - Reparameterized
with pm.Model() as shared_model:
    # Shared latent cognitive capacity
    theta = pm.Beta("theta", alpha=1, beta=1, initval=0.78)

    # Transform theta to normal scale
    theta_normal = pm.Deterministic("theta_normal", link(theta))

    # Calculate standard deviation for jitter
    sigma = pm.Gamma("sigma", alpha=4, beta=0.1, initval=1)

    offset_original   = pm.Normal("offset_original",   mu=0.0, sigma=1.0)
    offset_no_handler = pm.Normal("offset_no_handler", mu=0.0, sigma=1.0)
    offset_no_q       = pm.Normal("offset_no_q",       mu=0.0, sigma=1.0)

    # Transform back to probability scale
    theta_original   = pm.Deterministic("theta_original", 
                                        ilink(theta_normal + offset_original   * sigma))
    theta_no_handler = pm.Deterministic("theta_no_handler", 
                                        ilink(theta_normal + offset_no_handler * sigma))
    theta_no_q       = pm.Deterministic("theta_no_q", 
                                        ilink(theta_normal + offset_no_q       * sigma))

    A, B = data_level(shared_model, theta_original, theta_no_handler, theta_no_q)

# Independent Model with no shared Cue Sensitivity (theta)
with pm.Model() as independent_model:
    # Shared latent cognitive capacity
    theta_original   = pm.Beta("theta_original",   alpha=1, beta=1, initval=0.78)
    theta_no_handler = pm.Beta("theta_no_handler", alpha=1, beta=1, initval=0.78)
    theta_no_q       = pm.Beta("theta_no_q",       alpha=1, beta=1, initval=0.78)

    A, B = data_level(independent_model, theta_original, theta_no_handler, theta_no_q)

def mean_sample(object, node, prior=False):
    if isinstance(object, az.InferenceData):
        if prior:
            ret = np.mean(object.prior[node].values.flatten())
        else:
            ret = np.mean(object.posterior[node].values.flatten())
    else:
        ret = np.mean(object[node].values.flatten())

    return ret
    
def postprocess_print(trace, model_name):    
    print(f"{model_name} Posterior P(Theory A): {mean_sample(trace, 'A'):.5f}")
    print(f"{model_name} Posterior P(Theory B): {mean_sample(trace, 'B'):.5f}")

def run_model(model):
    with model:
        idata = pm.sample(draws=25000,
                          tune=10000, 
                          chains=8, 
                          cores=8,
                          random_seed=0, 
                          return_inferencedata=True)
    return idata

def get_prior_predictive(model):
    with model:
        prior_idata = pm.sample_prior_predictive(draws=1000000, 
                                                 random_seed=0,
                                                 var_names=["A", "B", "A1", "A2", "B1", "B2", "delta1", "delta2"])
    return prior_idata

def get_posterior_predictive(model, idata):
    with model:
        post_idata = pm.sample_posterior_predictive(trace=idata.posterior, 
                                                    var_names=["A", "B", "A1", "A2", "B1", "B2", "delta1", "delta2"])
    return post_idata

def calculate_bayes_factor(prior_idata, posterior_idata, model_name):

    A1_prior = mean_sample(prior_idata, 'A1', prior=True)
    A2_prior = mean_sample(prior_idata, 'A2', prior=True)
    B1_prior = mean_sample(prior_idata, 'B1', prior=True)
    B2_prior = mean_sample(prior_idata, 'B2', prior=True)
    
    A1_posterior = mean_sample(posterior_idata, 'A1', prior=False)
    A2_posterior = mean_sample(posterior_idata, 'A2', prior=False)
    B1_posterior = mean_sample(posterior_idata, 'B1', prior=False)
    B2_posterior = mean_sample(posterior_idata, 'B2', prior=False)
    
    A_prior = mean_sample(prior_idata, 'A', prior=True)
    B_prior = mean_sample(prior_idata, 'B', prior=True)
    A_posterior = mean_sample(posterior_idata, 'A', prior=False)
    B_posterior = mean_sample(posterior_idata, 'B', prior=False)
    
    evidenceA = A_posterior / A_prior
    evidenceB = B_posterior / B_prior
    
    BF_AB = evidenceA / evidenceB
    BF_BA = evidenceB / evidenceA
    
    print(f"{model_name} A1_prior    : {A1_prior:.9f}")
    print(f"{model_name} A2_prior    : {A2_prior:.9f}")
    print(f"{model_name} B1_prior    : {B1_prior:.9f}")
    print(f"{model_name} B2_prior    : {B2_prior:.9f}")
    print(f"{model_name} A1_posterior: {A1_posterior:.9f}")
    print(f"{model_name} A2_posterior: {A2_posterior:.9f}")
    print(f"{model_name} B1_posterior: {B1_posterior:.9f}")
    print(f"{model_name} B2_posterior: {B2_posterior:.9f}")
    
    print(f"{model_name} A_prior    : {A_prior:.9f}")
    print(f"{model_name} B_prior    : {B_prior:.9f}")
    print(f"{model_name} A_posterior: {A_posterior:.9f}")
    print(f"{model_name} B_posterior: {B_posterior:.9f}")
    print(f"{model_name} evidenceA  : {evidenceA:.9f}")
    print(f"{model_name} evidenceB  : {evidenceB:.9f}")
    print(f"{model_name} BF_AB      : {BF_AB:.9f}")
    print(f"{model_name} BF_BA      : {BF_BA:.9f}")

    if BF_BA > 1:
        print(f"{model_name} Bayes Factor BF_BA = P(B|X)/P(B) / P(A|X)/P(A) = {BF_BA:.9f}")
    else:
        print(f"{model_name} Bayes Factor BF_AB = P(A|X)/P(A) / P(B|X)/P(B) = {BF_AB:.9f}")

    return BF_BA

# Then get the posterior
posterior_idata_shared = run_model(shared_model)
posterior_idata_independent = run_model(independent_model)

# Get prior predictive first
prior_idata_shared = get_prior_predictive(shared_model)
prior_idata_independent = get_prior_predictive(independent_model)

print("Shared Model:")
summary_shared = az.summary(posterior_idata_shared)
print(summary_shared)

print("Independent Model:")
summary_independent = az.summary(posterior_idata_independent)
print(summary_independent)

postprocess_print(posterior_idata_shared, "Shared")
postprocess_print(posterior_idata_independent, "Independent")

calculate_bayes_factor(prior_idata_shared, posterior_idata_shared, "Shared")
calculate_bayes_factor(prior_idata_independent, posterior_idata_independent, "Independent")

results = dict(posterior = dict(shared_model=posterior_idata_shared,
                                independent_model=posterior_idata_independent),
               prior = dict(shared_model=prior_idata_shared,
                            independent_model=prior_idata_independent))

with open("results.pkl", "wb") as f:
    pkl.dump(results, f)




# # 3. Prior predictive simulation
# N = 100000
# theta1_samples = np.random.beta(2, 2, size=N)
# theta2_samples = np.random.beta(2, 2, size=N)
# theta3_samples = np.random.beta(2, 2, size=N)

# # Apply degradation exponents
# p1 = stats.norm.cdf(theta1_normal_samples)
# p2 = stats.norm.cdf(theta2_normal_samples)
# p3 = stats.norm.cdf(theta3_normal_samples)

# # Compute differences
# delta1_prior = p2 - p1
# delta2_prior = p3 - p2

# # Prior probabilities of the two minimodels
# prior_A = np.mean((np.abs(delta1_prior) < epsilon) & (np.abs(delta2_prior) < epsilon))
# prior_B = np.mean((np.abs(delta1_prior) < epsilon) & (delta2_prior < -epsilon))

# print(f"Prior P(Theory A):     {prior_A:.5f}")
# print(f"Prior P(Theory B):     {prior_B:.5f}")

# # 4. Compute Bayes Factor B vs A
# BF_BA = (posterior_B / posterior_A) * (prior_A / prior_B)

# print(f"\nBayes Factor BF_BA = P(B|X)/P(A|X) × P(A)/P(B) = {BF_BA:.3f}")
