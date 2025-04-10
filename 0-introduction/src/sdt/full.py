"""
Model 4: Conditional Hierarchical SDT (P Persons, K Conditions)

This is the most comprehensive model, combining hierarchy across participants 
and conditions within participants. Each participant p has parameters (d'_pk, c_pk) 
for each condition k. These are drawn from group-level distributions that might 
also vary by condition.
"""
from setup import *  # Import common functions and libraries

def generate_sample_data():
    """Generate simulated data for P=20 participants across K=3 conditions
    
    Returns:
    --------
    dict
        Dictionary containing the generated data
    """
    # Simulate data for P=20 participants across K=3 conditions
    np.random.seed(456)
    P = 20
    conditions = ['Easy', 'Medium', 'Hard']
    K = len(conditions)

    # True group parameters PER CONDITION
    # Let's assume d' decreases and |c| increases with difficulty
    true_mu_d_k = np.array([2.5, 1.5, 0.8])  # Group mean d' for Easy, Medium, Hard
    true_sigma_d_k = np.array([0.4, 0.5, 0.6])  # Group SD d'

    true_mu_c_k = np.array([-0.1, 0.1, 0.3])  # Group mean c
    true_sigma_c_k = np.array([0.2, 0.3, 0.4])  # Group SD c

    # Individual true parameters (shape P x K)
    true_d_primes_pk = np.zeros((P, K))
    true_criteria_pk = np.zeros((P, K))
    for k in range(K):
        true_d_primes_pk[:, k] = np.random.normal(true_mu_d_k[k], true_sigma_d_k[k], P)
        true_criteria_pk[:, k] = np.random.normal(true_mu_c_k[k], true_sigma_c_k[k], P)

    # Simulate observed data
    n_signal_pk = 100  # Assume constant trials per participant per condition
    n_noise_pk = 100

    true_hr_pk = stats.norm.cdf(true_d_primes_pk / 2 - true_criteria_pk)
    true_far_pk = stats.norm.cdf(-true_d_primes_pk / 2 - true_criteria_pk)

    # Add binomial noise
    hits_pk = np.random.binomial(n_signal_pk, true_hr_pk)  # Shape P x K
    fas_pk = np.random.binomial(n_noise_pk, true_far_pk)   # Shape P x K

    # Participant IDs
    participants = [f"P{i+1}" for i in range(P)]

    print(f"Number of participants: {P}")
    print(f"Conditions: {conditions}")
    print(f"Data shape (hits, FAs): {hits_pk.shape}")  # Should be P x K
    
    return {
        'P': P,
        'conditions': conditions,
        'K': K,
        'true_mu_d_k': true_mu_d_k,
        'true_sigma_d_k': true_sigma_d_k,
        'true_mu_c_k': true_mu_c_k,
        'true_sigma_c_k': true_sigma_c_k,
        'true_d_primes_pk': true_d_primes_pk,
        'true_criteria_pk': true_criteria_pk,
        'n_signal_pk': n_signal_pk,
        'n_noise_pk': n_noise_pk,
        'hits_pk': hits_pk,
        'fas_pk': fas_pk,
        'participants': participants
    }

def build_model(data):
    """Build the conditional hierarchical SDT model with a linear predictor for d'.
    
    Assumes conditions are ordered and the effect on mean d' is linear.

    Parameters:
    -----------
    data : dict
        Dictionary containing the observed data. Must include 'conditions' in order.
        
    Returns:
    --------
    pm.Model
        PyMC model for conditional hierarchical SDT with linear d' effect.
    """
    coords = {
        "participant": data['participants'],
        "condition": data['conditions'] # Assumes this list is ordered e.g., ['Easy', 'Medium', 'Hard']
    }

    # Map conditions to numeric values (0, 1, 2...) for the linear predictor
    # Ensure the order matches the assumed linear progression
    condition_numeric = np.arange(data['K'])

    with pm.Model(coords=coords) as model_cond_hier_linear_d:
        # --- Priors for d' linear effect ---
        # Intercept: Mean d' for the first condition (index 0)
        mu_d_intercept = pm.Normal('mu_d_intercept', mu=0.0, sigma=2.0)
        # Slope: Constant change in mean d' between adjacent conditions
        delta_d = pm.Normal('delta_d', mu=0.0, sigma=1.0) # Prior allows increase or decrease

        # --- Priors for d' variability and criterion (remain the same) ---
        # Allow standard deviation of d' to vary by condition
        sigma_d_k = pm.HalfNormal('sigma_d_k', sigma=1.0, dims="condition")
        # Criterion parameters are estimated independently per condition
        mu_c_k = pm.Normal('mu_c_k', mu=0.0, sigma=0.5, dims="condition")
        sigma_c_k = pm.HalfNormal('sigma_c_k', sigma=1.0, dims="condition")

        # --- Deterministic calculation of mu_d_k based on linear model ---
        # Register numeric condition values as data for the model
        condition_idx = pm.Data("condition_idx", condition_numeric, dims="condition", mutable=False)
        # Calculate mean d' for each condition: intercept + slope * condition_index
        mu_d_k = pm.Deterministic('mu_d_k', mu_d_intercept + delta_d * condition_idx, dims="condition")

        # --- Individual-Level Parameters (per participant, per condition) ---
        # Individual d' values are drawn from the group distribution whose mean (mu_d_k)
        # is now determined by the linear relationship across conditions.
        d_prime_pk = pm.Normal('d_prime_pk', mu=mu_d_k, sigma=sigma_d_k, dims=("participant", "condition"))
        # Criterion is drawn as before
        criterion_pk = pm.Normal('criterion_pk', mu=mu_c_k, sigma=sigma_c_k, dims=("participant", "condition"))

        # --- Deterministic Transformations (remain the same) ---
        hr_D_pk = pm.Deterministic('hr_D_pk', Phi(d_prime_pk / 2 - criterion_pk), dims=("participant", "condition"))
        far_D_pk = pm.Deterministic('far_D_pk', Phi(-d_prime_pk / 2 - criterion_pk), dims=("participant", "condition"))

        # --- Likelihood (remains the same) ---
        H_obs = pm.Binomial('H_obs',
                          n=data['n_signal_pk'],
                          p=hr_D_pk,
                          observed=data['hits_pk'],
                          dims=("participant", "condition"))

        FA_obs = pm.Binomial('FA_obs',
                           n=data['n_noise_pk'],
                           p=far_D_pk,
                           observed=data['fas_pk'],
                           dims=("participant", "condition"))
                           
    return model_cond_hier_linear_d # Return the new model structure

def sample_posterior(model, draws, tune, chains, target_accept):
    """Sample from the posterior distribution
    
    Parameters:
    -----------
    model : pm.Model
        The PyMC model to sample from
    draws : int
        Number of samples per chain after tuning
    tune : int
        Number of steps to discard for tuning the sampler
    chains : int
        Number of independent chains to run
    target_accept : float
        Parameter for NUTS algorithm, higher values can help with difficult posteriors
        
    Returns:
    --------
    az.InferenceData
        Posterior samples
    """
    with model:
        # This model is more complex and may benefit from more tuning or higher target_accept
        idata = pm.sample(draws=draws, tune=tune, chains=chains, target_accept=target_accept)
    return idata

def analyze_results(idata, data):
    """Analyze and visualize the posterior samples
    
    Parameters:
    -----------
    idata : az.InferenceData
        Posterior samples
    data : dict
        Dictionary containing the observed data
    """
    # Check summary statistics
    print("Conditional Hierarchical SDT Model Summary (Group Parameters):")
    # Using HDI intervals
    summary_group = az.summary(idata, var_names=['mu_d_k', 'sigma_d_k', 'mu_c_k', 'sigma_c_k'], 
                              hdi_prob=0.94)
    print(summary_group)
    
    # Check trace plots for key parameters
    az.plot_trace(idata, var_names=['mu_d_k', 'sigma_d_k', 'mu_c_k', 'sigma_c_k'])
    plt.tight_layout()
    plt.show()
    
    # Plot posterior distributions
    az.plot_posterior(idata, var_names=['mu_d_k', 'sigma_d_k', 'mu_c_k', 'sigma_c_k'], 
                     hdi_prob=0.94, ref_val=None)
    plt.tight_layout()
    plt.show()
    
    # Forest plot for condition effects (averaged across participants)
    if 'd_prime_diff_Easy_Hard' in idata.posterior:
        az.plot_posterior(idata, var_names=['d_prime_diff_Easy_Hard'], 
                        hdi_prob=0.94, ref_val=0)
        plt.title("d'(Easy) - d'(Hard) Effect")
        plt.tight_layout()
        plt.show()
        
        # Summary of condition effect
        diff_summary = az.summary(idata, var_names=['d_prime_diff_Easy_Hard'], 
                                 hdi_prob=0.94)
        print("\nSummary of d' difference (Easy - Hard):")
        print(diff_summary)
    
    # We could go much deeper with analyzing participant-by-condition interactions,
    # examining specific participants, etc.

def run_analysis():
    """Run the complete analysis for the conditional hierarchical SDT model"""
    print_version_info()
    data = generate_sample_data()
    
    # This model is computationally expensive, so we use a smaller version for demonstration
    # This would typically be commented out in real analysis
    # data['P'] = min(data['P'], 10)  # Use fewer participants
    # data['hits_pk'] = data['hits_pk'][:data['P'], :]
    # data['fas_pk'] = data['fas_pk'][:data['P'], :]
    # data['participants'] = data['participants'][:data['P']]
    # print(f"Using reduced dataset with {data['P']} participants for demonstration")
    
    model = build_model(data)
    
    # For demonstration, use fewer draws and chains
    idata = sample_posterior(model, draws=1000, tune=1000, chains=2, target_accept=0.995)
    
    analyze_results(idata, data)
    return model, idata

if __name__ == "__main__":
    run_analysis()
