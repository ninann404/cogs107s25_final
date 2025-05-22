# who_stroops.py
#
# This script performs a hierarchical EZ-diffusion analysis of the Stroop task data using PyMC.
# It loads the data, selects participants, calculates summary statistics, and fits a hierarchical EZ diffusion model.


import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, List
import pickle

class StroopAnalysis:
    def __init__(self, data_dir: str = 'stroop'):
        """Initialize with directory containing CSV files."""
        self.data_dir = Path(data_dir)
        self.data = self._load_data()
        self.selected_participants = None
        
    def _load_data(self) -> np.ndarray:
        """Load all CSV files from the data directory."""
        files = list(self.data_dir.glob('*.csv'))
        if not files:
            raise ValueError(f"No CSV files found in {self.data_dir}")
            
        data_list = []
        for file in files:
            data = np.loadtxt(file, delimiter=',')
            # Reorder columns to match expected format:
            # [block, trial, ?, condition, accuracy, RT]
            data_list.append(data)
            
        return np.dstack(data_list)
    
    def select_participants(self, participant_indices: List[int]):
        """Select specific participants for analysis."""
        self.selected_participants = participant_indices
                
    def return_summary_as_df(self):
        """Return summary statistics as a pandas DataFrame."""
        if self.selected_participants is None:
            raise ValueError("Please select participants first using select_participants()")
            
        data = self.data[:,:,self.selected_participants]
        conditions = [0, 2]  # Neutral and Incongruent
        
        # Create summary DataFrame
        records = []
        for p in range(len(self.selected_participants)):
            for c, cond in enumerate(conditions):
                mask = data[:, 3, p] == cond  # condition is in column 3
                acc = data[mask, 4, p]  # accuracy is in column 4
                rt = data[mask, 5, p]  # RT is in column 5
                
                records.append({
                    'cond': c + 1,
                    'pnum': p + 1,
                    'N': len(rt),
                    'sumacc': np.sum(acc),
                    'meanrt': np.mean(rt),
                    'varrt': np.var(rt)
                })
                
        return pd.DataFrame(records)
    
    def write_summary_to_csv(self, output_file: str = 'stroopdemo.csv'):
        """Write summary statistics to CSV file."""
        df = self.return_summary_as_df()
        df.to_csv(output_file, index=False)
    
    def save_pickle(self, output_file: str = 'stroopdemo.pkl'):
        """Write entire object to pickle file."""
        with open(output_file, 'wb') as f:
            pickle.dump(self, f)
    
    @staticmethod
    def load_pickle(input_file: str = 'stroopdemo.pkl'):
        """Load entire object from pickle file."""
        with open(input_file, 'rb') as f:
            return pickle.load(f)


def create_hierarchical_ez_model(data_df: pd.DataFrame):
    """Create a hierarchical PyMC model for EZ diffusion across all participants and conditions"""
    N_participants = len(data_df['pnum'].unique())
    
    # Convert pandas data to numpy arrays
    N = data_df['N'].values
    sumacc = data_df['sumacc'].values
    meanrt = data_df['meanrt'].values
    varrt = data_df['varrt'].values
    pnum = data_df['pnum'].values - 1  # Convert to 0-based indexing
    cond = data_df['cond'].values - 1  # Convert to 0-based indexing
    
    with pm.Model() as model:
        # Hyperpriors for group-level parameters
        mu_alpha = pm.TruncatedNormal("mu_alpha", mu=1, sigma=.3, lower=0)
        sigma_alpha = pm.HalfNormal("sigma_alpha", sigma=0.5)
        
        mu_tau = pm.TruncatedNormal("mu_tau", mu=.3, sigma=.1, lower=0)
        sigma_tau = pm.HalfNormal("sigma_tau", sigma=0.2)
        
        # Global condition effects
        mu_delta_neutral = pm.Normal("mu_delta_neutral", mu=1, sigma=1)  # Neutral condition mean
        sigma_delta_neutral = pm.HalfNormal("sigma_delta_neutral", sigma=0.5)
        
        mu_delta_diff = pm.Normal("mu_delta_diff", mu=0, sigma=1)  # Mean condition difference
        sigma_delta_diff = pm.HalfNormal("sigma_delta_diff", sigma=0.5)
        
        # Person-specific parameters
        alpha_p = pm.TruncatedNormal("alpha_p", 
                          mu=mu_alpha,
                          sigma=sigma_alpha,
                          lower=0,
                          shape=N_participants)
        
        tau_p = pm.TruncatedNormal("tau_p",
                          mu=mu_tau,
                          sigma=sigma_tau,
                          lower=0,
                          shape=N_participants)
        
        # Person-specific condition parameters
        delta_neutral_p = pm.Normal("delta_neutral_p",
                                  mu=mu_delta_neutral,
                                  sigma=sigma_delta_neutral,
                                  shape=N_participants)
        
        delta_diff_p = pm.Normal("delta_diff_p",
                               mu=mu_delta_diff,
                               sigma=sigma_delta_diff,
                               shape=N_participants)
        
        # Map person-specific parameters to observations
        alpha = alpha_p[pnum]
        tau = tau_p[pnum]
        delta = pm.math.switch(cond == 0,
                             delta_neutral_p[pnum],
                             delta_neutral_p[pnum] + delta_diff_p[pnum])
        
        # Forward equations from EZ Diffusion
        y = -alpha * delta
        Pc = pm.math.invlogit(-y)
        
        # Calculate precision (PRT) and mean decision time (MDT)
        VRT = alpha * (pm.math.exp(y) + 1)**2 * \
            (2 * y * pm.math.exp(y) - pm.math.exp(2*y) + 1) * delta**(-3) / 2
        
        MDT = (alpha / (2 * delta)) * \
              (1 - pm.math.exp(y)) / (1 + pm.math.exp(y))
        
        MRT = MDT + tau
        
        # Likelihoods
        pm.Normal("rt_mean", 
                 mu=MRT, 
                 sigma=pm.math.sqrt(VRT / sumacc), 
                 observed=meanrt)
        
        pm.Normal("rt_var",
                 mu=VRT,
                 sigma=pm.math.sqrt(2 / sumacc) * VRT,
                 observed=varrt)
        
        pm.Binomial("accuracy",
                   n=N,
                   p=Pc,
                   observed=sumacc)
        
        # Add deviance calculation
        pm.Deterministic(
            "deviance",
            -2 * (
                pm.logp(pm.Binomial.dist(n=N, p=Pc), sumacc).sum() +  # Log likelihood of accuracy data
                pm.logp(pm.Normal.dist(mu=MRT, sigma=pm.math.sqrt(VRT / sumacc)), meanrt).sum() +  # RT mean
                pm.logp(pm.Normal.dist(mu=VRT, sigma=pm.math.sqrt(2 / sumacc) * VRT), varrt).sum()  # RT variance
            )
        )
    
    return model


def main():
    # Set a seed for reproducibility
    seed = 0
    
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    # Use Pathlib to get the data directory
    data_dir = Path(__file__).parent.parent / 'data'
    stroop_data_dir = data_dir / 'stroop'
    
    # Load and prepare data
    analysis = StroopAnalysis(data_dir=stroop_data_dir)

    # Get all unique participant numbers and sort them
    all_participants = list(range(analysis.data.shape[2]))  # Each participant is a slice in the 3rd dimension
    print(f"Found {len(all_participants)} unique participants")
    
    # Select some participants, chosen at random
    selected_participants = rng.choice(all_participants, size=20, replace=False)
    print(f"Selected participants: {selected_participants}")
    analysis.select_participants(selected_participants)

    # Get summary statistics
    analysis.write_summary_to_csv(data_dir / 'stroop_processed.csv')
    data_df = analysis.return_summary_as_df()
    
    # Verify data before model creation
    print("\nData summary:")
    print(f"Number of observations: {len(data_df)}")
    print(f"Number of unique participants: {len(data_df['pnum'].unique())}")
    print(f"Conditions: {sorted(data_df['cond'].unique())}")
    
    # Create and fit model
    model = create_hierarchical_ez_model(data_df)
    
    with model:
        trace = pm.sample(
            draws=500,
            tune=250,
            target_accept=0.95,
            return_inferencedata=True,
            chains=4,
            cores=4,
            random_seed=rng
        )
    
    # Print summary statistics
    print("\nGroup-level Parameters:")
    
    print(az.summary(trace, var_names=['mu_alpha', 'sigma_alpha', 'mu_tau', 'sigma_tau',
                                        'mu_delta_neutral', 'sigma_delta_neutral',
                                        'mu_delta_diff', 'sigma_delta_diff']))
    
    print("\nIndividual Parameters:")
    
    print("\n  alpha_p: person-specific alpha parameters")
    print(az.summary(trace, var_names=['alpha_p']))
    
    print("\n  tau_p: person-specific tau parameters")
    print(az.summary(trace, var_names=['tau_p']))
    
    print("\n  delta_neutral_p: person-specific delta parameters for the neutral condition")
    print(az.summary(trace, var_names=['delta_neutral_p']))
    
    print("\n  delta_diff_p: person-specific delta effect size parameters")
    print(az.summary(trace, var_names=['delta_diff_p']))
    
    # Print deviance statistics per chain
    print("\nDeviance Statistics by Chain:")
    deviance = trace.posterior['deviance'].values
    print("\nChain |   Mean   |   Std    |    Min   |    Max   ")
    print("---------------------------------------------")
    for chain in range(deviance.shape[0]):
        print(f"{chain+1:5d} | {deviance[chain].mean():8.2f} | {deviance[chain].std():8.2f} | {deviance[chain].min():8.2f} | {deviance[chain].max():8.2f}")
    
    # Also print overall deviance statistics
    print("\nOverall Deviance Statistics:")
    print(az.summary(trace, var_names=['deviance']))
    

if __name__ == "__main__":
    main()