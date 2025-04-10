"""
Model 3: Hierarchical SDT (P Persons, One Condition)

This model estimates parameters for each individual participant while assuming
these individual parameters are drawn from a group-level distribution. This allows
participants' data to inform each other (pooling information / shrinkage),
leading to more stable estimates, especially for participants with noisy data.
"""
from setup import *  # Import common functions and libraries

def generate_sample_data():
    """Generate hypothetical data for P=20 participants
    
    Returns:
    --------
    dict
        Dictionary containing the generated data
    """
    # Simulate data for P=20 participants
    np.random.seed(123)  # for reproducibility
    P = 20
    # True group parameters (for simulation)
    true_mu_d = 1.5
    true_sigma_d = 0.5
    true_mu_c = 0.2
    true_sigma_c = 0.3

    # Individual true parameters
    true_d_primes = np.random.normal(true_mu_d, true_sigma_d, P)
    true_criteria = np.random.normal(true_mu_c, true_sigma_c, P)

    # Simulate observed data
    n_signal_p = 100  # Assume constant number of trials for simplicity
    n_noise_p = 100

    true_hr = stats.norm.cdf(true_d_primes / 2 - true_criteria)
    true_far = stats.norm.cdf(-true_d_primes / 2 - true_criteria)

    # Add binomial noise
    hits_p = np.random.binomial(n_signal_p, true_hr)
    fas_p = np.random.binomial(n_noise_p, true_far)

    # We don't need misses and CRs directly for the model,
    # but they are implied:
    # misses_p = n_signal_p - hits_p
    # crs_p = n_noise_p - fas_p

    # Participant IDs
    participants = [f"P{i+1}" for i in range(P)]

    print(f"Number of participants: {P}")
    
    return {
        'P': P,
        'true_mu_d': true_mu_d,
        'true_sigma_d': true_sigma_d,
        'true_mu_c': true_mu_c,
        'true_sigma_c': true_sigma_c,
        'true_d_primes': true_d_primes,
        'true_criteria': true_criteria,
        'n_signal_p': n_signal_p,
        'n_noise_p': n_noise_p,
        'hits_p': hits_p,
        'fas_p': fas_p,
        'participants': participants
    }

def build_model(data):
    """Build the hierarchical SDT model
    
    Parameters:
    -----------
    data : dict
        Dictionary containing the observed data
        
    Returns:
    --------
    pm.Model
        PyMC model for hierarchical SDT
    """
    coords = {"participant": data['participants']}

    with pm.Model(coords=coords) as model_hierarchical:
        # --- Group-Level Priors (Hyper-priors) ---
        # Prior for the mean d' across participants
        mu_d = pm.Normal('mu_d', mu=0.0, sigma=2.0)
        # Prior for the standard deviation of d' across participants
        # HalfNormal is common for scale parameters (must be positive)
        sigma_d = pm.HalfNormal('sigma_d', sigma=1.0)

        # Prior for the mean criterion across participants
        mu_c = pm.Normal('mu_c', mu=0.0, sigma=0.5)
        # Prior for the standard deviation of criterion across participants
        sigma_c = pm.HalfNormal('sigma_c', sigma=1.0)

        # --- Individual-Level Parameters ---
        # Individual d' values are drawn from a Normal distribution
        # defined by the group parameters mu_d and sigma_d.
        # Dims links this to participant coordinate.
        d_prime_p = pm.Normal('d_prime_p', mu=mu_d, sigma=sigma_d, dims="participant")

        # Individual criterion values are drawn from a Normal distribution
        # defined by the group parameters mu_c and sigma_c.
        criterion_p = pm.Normal('criterion_p', mu=mu_c, sigma=sigma_c, dims="participant")

        # --- Deterministic Transformations ---
        # Calculate HR and FAR for each participant
        hr_D_p = pm.Deterministic('hr_D_p', Phi(d_prime_p / 2 - criterion_p), dims="participant")
        far_D_p = pm.Deterministic('far_D_p', Phi(-d_prime_p / 2 - criterion_p), dims="participant")

        # --- Likelihood ---
        # Binomial likelihood for each participant's hits and FAs
        # Ensure trial numbers are broadcast correctly if they differ per participant
        # Here we assume they are constant (n_signal_p, n_noise_p)
        H_obs = pm.Binomial('H_obs',
                          n=data['n_signal_p'],      # Can be an array if N varies
                          p=hr_D_p,                 # Vector of individual hit rates
                          observed=data['hits_p'],   # Vector of observed hits
                          dims="participant")

        FA_obs = pm.Binomial('FA_obs',
                           n=data['n_noise_p'],     # Can be an array if N varies
                           p=far_D_p,               # Vector of individual FA rates
                           observed=data['fas_p'],  # Vector of observed FAs
                           dims="participant")
                           
    return model_hierarchical

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
        # Hierarchical models might require more tuning steps or higher target_accept
        idata = pm.sample(draws=draws, tune=tune, chains=chains, target_accept=target_accept)
    return idata

def analyze_results(idata):
    """Analyze and visualize the posterior samples
    
    Parameters:
    -----------
    idata : az.InferenceData
        Posterior samples
    """
    # Check summary statistics
    print("Hierarchical SDT Model Summary:")
    # Using HDI intervals
    summary_hierarchical = az.summary(idata, var_names=['mu_d', 'sigma_d', 'mu_c', 'sigma_c', 
                                                      'd_prime_p', 'criterion_p'], 
                                    hdi_prob=0.94)
    print(summary_hierarchical)
    
    # Check trace plots for key parameters
    az.plot_trace(idata, var_names=['mu_d', 'sigma_d', 'mu_c', 'sigma_c'])
    plt.tight_layout()
    plt.show()
    
    # Plot posterior distributions
    az.plot_posterior(idata, var_names=['mu_d', 'sigma_d', 'mu_c', 'sigma_c'], 
                     hdi_prob=0.94, ref_val=None)
    plt.tight_layout()
    plt.show()
    
    # Forest plot for individual participant parameters
    pm.plot_forest(idata, var_names=['d_prime_p'], 
                  combined=True, hdi_prob=0.94)
    plt.axvline(x=idata.posterior['mu_d'].mean().item(), color='r', linestyle='--')
    plt.title("d' by participant (red line is group mean)")
    plt.tight_layout()
    plt.show()
    
    pm.plot_forest(idata, var_names=['criterion_p'], 
                  combined=True, hdi_prob=0.94)
    plt.axvline(x=idata.posterior['mu_c'].mean().item(), color='r', linestyle='--')
    plt.title("criterion by participant (red line is group mean)")
    plt.tight_layout()
    plt.show()

def run_analysis():
    """Run the complete analysis for the hierarchical SDT model"""
    print_version_info()
    data = generate_sample_data()
    model = build_model(data)
    idata = sample_posterior(model, draws=2000, tune=2000, chains=4, target_accept=0.95)
    analyze_results(idata)
    return model, idata

if __name__ == "__main__":
    run_analysis()
