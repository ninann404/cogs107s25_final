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
    """Build the conditional hierarchical SDT model
    
    Parameters:
    -----------
    data : dict
        Dictionary containing the observed data
        
    Returns:
    --------
    pm.Model
        PyMC model for conditional hierarchical SDT
    """
    coords = {
        "participant": data['participants'],
        "condition": data['conditions']
    }

    # Reshape data for PyMC likelihood (needs to match dims)
    # ArviZ works best if observed data has named dimensions matching coords
    # We can provide data as DataFrames or xarrays, or ensure numpy arrays align
    # with the order of dims specified in the likelihood.
    # Here, hits_pk and fas_pk are already in (participant, condition) order.

    with pm.Model(coords=coords) as model_cond_hier:
        # --- Group-Level Priors (per condition) ---
        # Shape K (one value per condition)
        mu_d_k = pm.Normal('mu_d_k', mu=0.0, sigma=2.0, dims="condition")
        sigma_d_k = pm.HalfNormal('sigma_d_k', sigma=1.0, dims="condition")
        mu_c_k = pm.Normal('mu_c_k', mu=0.0, sigma=2.0, dims="condition")
        sigma_c_k = pm.HalfNormal('sigma_c_k', sigma=1.0, dims="condition")

        # --- Individual-Level Parameters (per participant, per condition) ---
        # Shape (P, K)
        # Individual d' values are drawn from the condition-specific group distribution
        # PyMC handles the broadcasting: mu_d_k (shape K) and sigma_d_k (shape K)
        # define the distribution for d_prime_pk (shape P, K)
        d_prime_pk = pm.Normal('d_prime_pk', mu=mu_d_k, sigma=sigma_d_k, dims=("participant", "condition"))
        criterion_pk = pm.Normal('criterion_pk', mu=mu_c_k, sigma=sigma_c_k, dims=("participant", "condition"))

        # --- Deterministic Transformations ---
        # Calculate HR and FAR for each participant in each condition
        # Shape (P, K)
        hr_D_pk = pm.Deterministic('hr_D_pk', Phi(d_prime_pk / 2 - criterion_pk), dims=("participant", "condition"))
        far_D_pk = pm.Deterministic('far_D_pk', Phi(-d_prime_pk / 2 - criterion_pk), dims=("participant", "condition"))

        # --- Likelihood ---
        # Binomial likelihood for each participant's hits and FAs in each condition
        # Shape (P, K)
        H_obs = pm.Binomial('H_obs',
                          n=data['n_signal_pk'],          # Can be P x K array if N varies
                          p=hr_D_pk,                     # Matrix of individual hit rates
                          observed=data['hits_pk'],       # Matrix of observed hits
                          dims=("participant", "condition"))  # Match dims order to observed data

        FA_obs = pm.Binomial('FA_obs',
                           n=data['n_noise_pk'],          # Can be P x K array if N varies
                           p=far_D_pk,                    # Matrix of individual FA rates
                           observed=data['fas_pk'],        # Matrix of observed FAs
                           dims=("participant", "condition"))  # Match dims order to observed data
                           
    return model_cond_hier

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
    data['P'] = min(data['P'], 10)  # Use fewer participants
    data['hits_pk'] = data['hits_pk'][:data['P'], :]
    data['fas_pk'] = data['fas_pk'][:data['P'], :]
    data['participants'] = data['participants'][:data['P']]
    print(f"Using reduced dataset with {data['P']} participants for demonstration")
    
    model = build_model(data)
    
    # For demonstration, use fewer draws and chains
    idata = sample_posterior(model, draws=1000, tune=1000, chains=2, target_accept=0.995)
    
    analyze_results(idata, data)
    return model, idata

if __name__ == "__main__":
    run_analysis()
