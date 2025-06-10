"""
Signal Detection Theory (SDT) and Delta Plot Analysis for Response Time Data
"""

import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import os

# Mapping dictionaries for categorical variables
# These convert categorical labels to numeric codes for analysis
MAPPINGS = {
    'stimulus_type': {'simple': 0, 'complex': 1},
    'difficulty': {'easy': 0, 'hard': 1},
    'signal': {'present': 0, 'absent': 1}
}

# Descriptive names for each experimental condition
CONDITION_NAMES = {
    0: 'Easy Simple',
    1: 'Easy Complex',
    2: 'Hard Simple',
    3: 'Hard Complex'
}

# Percentiles used for delta plot analysis
PERCENTILES = [10, 30, 50, 70, 90]

def read_data(file_path, prepare_for='sdt', display=False):
    """Read and preprocess data from a CSV file into SDT format.
    
    Args:
        file_path: Path to the CSV file containing raw response data
        prepare_for: Type of analysis to prepare data for ('sdt' or 'delta plots')
        display: Whether to print summary statistics
        
    Returns:
        DataFrame with processed data in the requested format
    """
    # Read and preprocess data
    data = pd.read_csv(file_path)
    
    # Convert categorical variables to numeric codes
    for col, mapping in MAPPINGS.items():
        data[col] = data[col].map(mapping)
    
    # Create participant number and condition index
    data['pnum'] = data['participant_id']
    data['condition'] = data['stimulus_type'] + data['difficulty'] * 2
    data['accuracy'] = data['accuracy'].astype(int)
    
    if display:
        print("\nRaw data sample:")
        print(data.head())
        print("\nUnique conditions:", data['condition'].unique())
        print("Signal values:", data['signal'].unique())
    
    # Transform to SDT format if requested
    if prepare_for == 'sdt':
        # Group data by participant, condition, and signal presence
        grouped = data.groupby(['pnum', 'condition', 'signal']).agg({
            'accuracy': ['count', 'sum']
        }).reset_index()
        
        # Flatten column names
        grouped.columns = ['pnum', 'condition', 'signal', 'nTrials', 'correct']
        
        if display:
            print("\nGrouped data:")
            print(grouped.head())
        
        # Transform into SDT format (hits, misses, false alarms, correct rejections)
        sdt_data = []
        for pnum in grouped['pnum'].unique():
            p_data = grouped[grouped['pnum'] == pnum]
            for condition in p_data['condition'].unique():
                c_data = p_data[p_data['condition'] == condition]
                
                # Get signal and noise trials
                signal_trials = c_data[c_data['signal'] == 0]
                noise_trials = c_data[c_data['signal'] == 1]
                
                if not signal_trials.empty and not noise_trials.empty:
                    sdt_data.append({
                        'pnum': pnum,
                        'condition': condition,
                        'hits': signal_trials['correct'].iloc[0],
                        'misses': signal_trials['nTrials'].iloc[0] - signal_trials['correct'].iloc[0],
                        'false_alarms': noise_trials['nTrials'].iloc[0] - noise_trials['correct'].iloc[0],
                        'correct_rejections': noise_trials['correct'].iloc[0],
                        'nSignal': signal_trials['nTrials'].iloc[0],
                        'nNoise': noise_trials['nTrials'].iloc[0]
                    })
        
        data = pd.DataFrame(sdt_data)
        
        if display:
            print("\nSDT summary:")
            print(data)
            if data.empty:
                print("\nWARNING: Empty SDT summary generated!")
                print("Number of participants:", len(data['pnum'].unique()))
                print("Number of conditions:", len(data['condition'].unique()))
            else:
                print("\nSummary statistics:")
                print(data.groupby('condition').agg({
                    'hits': 'sum',
                    'misses': 'sum',
                    'false_alarms': 'sum',
                    'correct_rejections': 'sum',
                    'nSignal': 'sum',
                    'nNoise': 'sum'
                }).round(2))
    
    # Prepare data for delta plot analysis
    if prepare_for == 'delta plots':
        # Initialize list for delta plot data
        dp_data_list = []
        
        # Process data for each participant and condition
        for pnum in data['pnum'].unique():
            for condition in data['condition'].unique():
                # Get data for this participant and condition
                c_data = data[(data['pnum'] == pnum) & (data['condition'] == condition)]
                
                if len(c_data) == 0:
                    continue
                
                # Calculate percentiles for overall RTs
                overall_rt = c_data['rt']
                if len(overall_rt) > 0:
                    row = {'pnum': pnum, 'condition': condition, 'mode': 'overall'}
                    for p in PERCENTILES:
                        row[f'p{p}'] = np.percentile(overall_rt, p)
                    dp_data_list.append(row)
                
                # Calculate percentiles for accurate responses
                accurate_rt = c_data[c_data['accuracy'] == 1]['rt']
                if len(accurate_rt) > 0:
                    row = {'pnum': pnum, 'condition': condition, 'mode': 'accurate'}
                    for p in PERCENTILES:
                        row[f'p{p}'] = np.percentile(accurate_rt, p)
                    dp_data_list.append(row)
                
                # Calculate percentiles for error responses
                error_rt = c_data[c_data['accuracy'] == 0]['rt']
                if len(error_rt) > 0:
                    row = {'pnum': pnum, 'condition': condition, 'mode': 'error'}
                    for p in PERCENTILES:
                        row[f'p{p}'] = np.percentile(error_rt, p)
                    dp_data_list.append(row)
        
        data = pd.DataFrame(dp_data_list)
        
        if display:
            print("\nDelta plots data:")
            print(data.head())

    return data


def apply_hierarchical_sdt_model(data):
    """Apply a hierarchical Signal Detection Theory model using PyMC.
    
    This function implements a Bayesian hierarchical model for SDT analysis,
    allowing for both group-level and individual-level parameter estimation.
    
    Args:
        data: DataFrame containing SDT summary statistics
        
    Returns:
        PyMC model object
    """
    # Get unique participants and conditions
    P = len(data['pnum'].unique())
    C = len(data['condition'].unique())
    
    print(f"Model setup: {P} participants, {C} conditions")
    
    # Define the hierarchical model
    with pm.Model() as sdt_model:
        # Group-level parameters for d-prime (sensitivity)
        mean_d_prime = pm.Normal('mean_d_prime', mu=0.0, sigma=1.0, shape=C)
        stdev_d_prime = pm.HalfNormal('stdev_d_prime', sigma=1.0)
        
        # Group-level parameters for criterion (bias)
        mean_criterion = pm.Normal('mean_criterion', mu=0.0, sigma=1.0, shape=C)
        stdev_criterion = pm.HalfNormal('stdev_criterion', sigma=1.0)
        
        # Individual-level parameters
        d_prime = pm.Normal('d_prime', mu=mean_d_prime, sigma=stdev_d_prime, shape=(P, C))
        criterion = pm.Normal('criterion', mu=mean_criterion, sigma=stdev_criterion, shape=(P, C))
        
        # Calculate hit and false alarm rates using SDT
        hit_rate = pm.math.invlogit(d_prime - criterion)
        false_alarm_rate = pm.math.invlogit(-criterion)
                
        # Likelihood for signal trials
        pm.Binomial('hit_obs', 
                   n=data['nSignal'], 
                   p=hit_rate[data['pnum']-1, data['condition']], 
                   observed=data['hits'])
        
        # Likelihood for noise trials
        pm.Binomial('false_alarm_obs', 
                   n=data['nNoise'], 
                   p=false_alarm_rate[data['pnum']-1, data['condition']], 
                   observed=data['false_alarms'])
    
    return sdt_model

def draw_comprehensive_delta_plots(data):
    """Draw comprehensive delta plots for all participants in a single figure."""
    
    # Get all participants
    participants = sorted(data['pnum'].unique())
    n_participants = len(participants)
    
    # Create a large figure with subplots for each participant
    fig = plt.figure(figsize=(20, 5*n_participants))
    
    # Create output directory
    OUTPUT_DIR = Path(__file__).parent / 'output'
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    for p_idx, pnum in enumerate(participants):
        # Filter data for this participant
        p_data = data[data['pnum'] == pnum]
        conditions = sorted(p_data['condition'].unique())
        
        if len(conditions) < 2:
            continue
            
        # Create subplot for this participant (2x3 grid for condition comparisons)
        gs = fig.add_gridspec(n_participants, 6, hspace=0.3, wspace=0.3)
        
        # Define condition pairs to compare
        comparisons = [
            (0, 1, 'Easy: Complex - Simple'),  # Stimulus type effect (easy)
            (2, 3, 'Hard: Complex - Simple'),  # Stimulus type effect (hard)
            (0, 2, 'Simple: Hard - Easy'),     # Difficulty effect (simple)
            (1, 3, 'Complex: Hard - Easy'),    # Difficulty effect (complex)
            (0, 3, 'Hard Complex - Easy Simple'),  # Combined effect
            (1, 2, 'Hard Simple - Easy Complex')   # Cross comparison
        ]
        
        for comp_idx, (cond1, cond2, title) in enumerate(comparisons):
            ax = fig.add_subplot(gs[p_idx, comp_idx])
            
            # Check if both conditions exist for this participant
            if cond1 not in conditions or cond2 not in conditions:
                ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'P{pnum}: {title}', fontsize=10)
                continue
            
            # Get data for both conditions
            data1 = p_data[p_data['condition'] == cond1]
            data2 = p_data[p_data['condition'] == cond2]
            
            # Plot overall RT differences
            overall1 = data1[data1['mode'] == 'overall']
            overall2 = data2[data2['mode'] == 'overall']
            
            if len(overall1) > 0 and len(overall2) > 0:
                delta_overall = []
                for p in PERCENTILES:
                    delta_overall.append(overall2[f'p{p}'].iloc[0] - overall1[f'p{p}'].iloc[0])
                ax.plot(PERCENTILES, delta_overall, 'k-o', linewidth=2, markersize=6, label='Overall')
            
            # Plot accurate RT differences
            acc1 = data1[data1['mode'] == 'accurate']
            acc2 = data2[data2['mode'] == 'accurate']
            
            if len(acc1) > 0 and len(acc2) > 0:
                delta_acc = []
                for p in PERCENTILES:
                    delta_acc.append(acc2[f'p{p}'].iloc[0] - acc1[f'p{p}'].iloc[0])
                ax.plot(PERCENTILES, delta_acc, 'g-s', linewidth=2, markersize=5, label='Accurate', alpha=0.7)
            
            # Plot error RT differences
            err1 = data1[data1['mode'] == 'error']
            err2 = data2[data2['mode'] == 'error']
            
            if len(err1) > 0 and len(err2) > 0:
                delta_err = []
                for p in PERCENTILES:
                    delta_err.append(err2[f'p{p}'].iloc[0] - err1[f'p{p}'].iloc[0])
                ax.plot(PERCENTILES, delta_err, 'r-^', linewidth=2, markersize=5, label='Error', alpha=0.7)
            
            # Formatting
            ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            ax.set_ylim(-0.4, 0.6)
            ax.set_xlabel('Percentile', fontsize=9)
            ax.set_ylabel('RT Difference (s)', fontsize=9)
            ax.set_title(f'P{pnum}: {title}', fontsize=10)
            ax.grid(True, alpha=0.3)
            
            # Add legend only to first subplot of each row
            if comp_idx == 0:
                ax.legend(fontsize=8, loc='upper left')
    
    plt.suptitle('Delta Plots: Response Time Distribution Comparisons Across All Participants', 
                 fontsize=16, y=0.98)
    plt.savefig(OUTPUT_DIR / 'comprehensive_delta_plots.png', dpi=150, bbox_inches='tight')
    plt.show()

def display_descriptive_statistics(raw_data, sdt_data):
    """Display descriptive statistics of the data."""
    print("="*60)
    print("DESCRIPTIVE STATISTICS")
    print("="*60)
    
    # Raw data statistics
    print("\n1. RAW DATA SUMMARY:")
    print(f"   Total participants: {raw_data['pnum'].nunique()}")
    print(f"   Total trials: {len(raw_data)}")
    print(f"   Conditions: {sorted(raw_data['condition'].unique())}")
    
    # Accuracy by condition
    print("\n2. ACCURACY BY CONDITION:")
    accuracy_stats = raw_data.groupby('condition')['accuracy'].agg(['mean', 'std', 'count']).round(3)
    for cond in accuracy_stats.index:
        print(f"   {CONDITION_NAMES[cond]}: {accuracy_stats.loc[cond, 'mean']:.3f} ± {accuracy_stats.loc[cond, 'std']:.3f} (n={accuracy_stats.loc[cond, 'count']})")
    
    # RT statistics
    print("\n3. RESPONSE TIME BY CONDITION:")
    rt_stats = raw_data.groupby('condition')['rt'].agg(['mean', 'std', 'median']).round(3)
    for cond in rt_stats.index:
        print(f"   {CONDITION_NAMES[cond]}: M={rt_stats.loc[cond, 'mean']:.3f}s, SD={rt_stats.loc[cond, 'std']:.3f}s, Median={rt_stats.loc[cond, 'median']:.3f}s")
    
    # SDT summary
    print("\n4. SDT DATA SUMMARY:")
    sdt_summary = sdt_data.groupby('condition').agg({
        'hits': 'sum',
        'misses': 'sum', 
        'false_alarms': 'sum',
        'correct_rejections': 'sum'
    })
    
    for cond in sdt_summary.index:
        hits = sdt_summary.loc[cond, 'hits']
        misses = sdt_summary.loc[cond, 'misses']
        fas = sdt_summary.loc[cond, 'false_alarms']
        crs = sdt_summary.loc[cond, 'correct_rejections']
        hit_rate = hits / (hits + misses)
        fa_rate = fas / (fas + crs)
        print(f"   {CONDITION_NAMES[cond]}: Hit Rate={hit_rate:.3f}, FA Rate={fa_rate:.3f}")

def check_convergence_and_display_results(trace, model):
    """Check model convergence and display results."""
    print("\n" + "="*60)
    print("CONVERGENCE DIAGNOSTICS")
    print("="*60)
    
    # Summary statistics with R-hat
    summary = az.summary(trace)
    print("\nR-hat values (should be < 1.1 for convergence):")
    print(summary[['mean', 'sd', 'r_hat']].round(3))
    
    # Check for convergence issues
    high_rhat = summary[summary['r_hat'] > 1.1]
    if len(high_rhat) > 0:
        print(f"\nWARNING: {len(high_rhat)} parameters have R-hat > 1.1:")
        print(high_rhat[['r_hat']].round(3))
    else:
        print("\n✓ All parameters show good convergence (R-hat < 1.1)")
    
    # Create output directory
    OUTPUT_DIR = Path(__file__).parent / 'output'
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Plot trace plots for key parameters
    print("\nGenerating trace plots...")
    az.plot_trace(trace, var_names=['mean_d_prime', 'mean_criterion'], compact=True)
    plt.savefig(OUTPUT_DIR / 'trace_plots.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Plot posterior distributions
    print("\nGenerating posterior plots...")
    az.plot_posterior(trace, var_names=['mean_d_prime', 'mean_criterion'], 
                     hdi_prob=0.95, point_estimate='mean')
    plt.savefig(OUTPUT_DIR / 'posterior_distributions.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Pair plot for correlations
    print("\nGenerating pair plot...")
    az.plot_pair(trace, var_names=['mean_d_prime', 'mean_criterion'], 
                 kind='hexbin', marginals=True)
    plt.savefig(OUTPUT_DIR / 'pair_plot.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return summary

def display_parameter_estimates(summary):
    """Display population-level and individual parameter estimates."""
    print("\n" + "="*60)
    print("PARAMETER ESTIMATES")
    print("="*60)
    
    # Population-level estimates
    print("\n1. POPULATION-LEVEL ESTIMATES:")
    print("\nSensitivity (d-prime) by condition:")
    for i in range(4):  # 4 conditions
        if f'mean_d_prime[{i}]' in summary.index:
            mean_val = summary.loc[f'mean_d_prime[{i}]', 'mean']
            # Use the correct column names from az.summary
            if 'hdi_2.5%' in summary.columns:
                hdi_lower = summary.loc[f'mean_d_prime[{i}]', 'hdi_2.5%']
                hdi_upper = summary.loc[f'mean_d_prime[{i}]', 'hdi_97.5%']
            else:
                # Fallback to mean ± sd if HDI not available
                sd_val = summary.loc[f'mean_d_prime[{i}]', 'sd']
                hdi_lower = mean_val - 1.96 * sd_val
                hdi_upper = mean_val + 1.96 * sd_val
            print(f"   {CONDITION_NAMES[i]}: {mean_val:.3f} [95% CI: {hdi_lower:.3f}, {hdi_upper:.3f}]")
    
    print("\nResponse Bias (criterion) by condition:")
    for i in range(4):  # 4 conditions
        if f'mean_criterion[{i}]' in summary.index:
            mean_val = summary.loc[f'mean_criterion[{i}]', 'mean']
            if 'hdi_2.5%' in summary.columns:
                hdi_lower = summary.loc[f'mean_criterion[{i}]', 'hdi_2.5%']
                hdi_upper = summary.loc[f'mean_criterion[{i}]', 'hdi_97.5%']
            else:
                sd_val = summary.loc[f'mean_criterion[{i}]', 'sd']
                hdi_lower = mean_val - 1.96 * sd_val
                hdi_upper = mean_val + 1.96 * sd_val
            print(f"   {CONDITION_NAMES[i]}: {mean_val:.3f} [95% CI: {hdi_lower:.3f}, {hdi_upper:.3f}]")
    
    # Effect comparisons
    print("\n2. EFFECT COMPARISONS:")
    
    # Extract condition means for comparisons
    d_prime_means = {}
    criterion_means = {}
    
    for i in range(4):
        if f'mean_d_prime[{i}]' in summary.index:
            d_prime_means[i] = summary.loc[f'mean_d_prime[{i}]', 'mean']
            criterion_means[i] = summary.loc[f'mean_criterion[{i}]', 'mean']
    
    if len(d_prime_means) >= 4:
        # Stimulus Type effects (Simple vs Complex)
        easy_simple_vs_complex = d_prime_means[1] - d_prime_means[0]  # Easy Complex - Easy Simple
        hard_simple_vs_complex = d_prime_means[3] - d_prime_means[2]  # Hard Complex - Hard Simple
        
        print(f"\nStimulus Type Effect on Sensitivity (Complex - Simple):")
        print(f"   Easy trials: {easy_simple_vs_complex:.3f}")
        print(f"   Hard trials: {hard_simple_vs_complex:.3f}")
        
        # Difficulty effects (Hard vs Easy)
        simple_easy_vs_hard = d_prime_means[2] - d_prime_means[0]  # Hard Simple - Easy Simple
        complex_easy_vs_hard = d_prime_means[3] - d_prime_means[1]  # Hard Complex - Easy Complex
        
        print(f"\nDifficulty Effect on Sensitivity (Hard - Easy):")
        print(f"   Simple stimuli: {simple_easy_vs_hard:.3f}")
        print(f"   Complex stimuli: {complex_easy_vs_hard:.3f}")
        
        # Bias effects
        print(f"\nStimulus Type Effect on Bias (Complex - Simple):")
        easy_bias_effect = criterion_means[1] - criterion_means[0]
        hard_bias_effect = criterion_means[3] - criterion_means[2]
        print(f"   Easy trials: {easy_bias_effect:.3f}")
        print(f"   Hard trials: {hard_bias_effect:.3f}")

# Main execution
if __name__ == "__main__":
    # Create output directory
    OUTPUT_DIR = Path(__file__).parent / 'output'
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("="*60)
    print("SDT AND DELTA PLOT ANALYSIS")
    print("="*60)
    
    # Read and analyze data
    print("\nReading data...")
    raw_data = read_data('data.csv', prepare_for='raw', display=False)
    sdt_data = read_data('data.csv', prepare_for='sdt', display=True)
    delta_data = read_data('data.csv', prepare_for='delta plots', display=False)
    
    # Display descriptive statistics
    display_descriptive_statistics(raw_data, sdt_data)
    
    # Fit SDT model
    print("\n" + "="*60)
    print("FITTING HIERARCHICAL SDT MODEL")
    print("="*60)
    
    model = apply_hierarchical_sdt_model(sdt_data)
    
    print("\nSampling from posterior...")
    with model:
        trace = pm.sample(2000, tune=1000, chains=4, cores=1, 
                         target_accept=0.9, random_seed=42)
    
    # Check convergence and display results
    summary = check_convergence_and_display_results(trace, model)
    
    # Display parameter estimates
    display_parameter_estimates(summary)
    
    # Generate comprehensive delta plots
    print("\n" + "="*60)
    print("GENERATING COMPREHENSIVE DELTA PLOTS")
    print("="*60)
    
    print("Generating comprehensive delta plots for all participants...")
    draw_comprehensive_delta_plots(delta_data)
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print(f"\nAll figures saved to: {OUTPUT_DIR}")
    print("\nFiles generated:")
    print("- trace_plots.png: MCMC convergence diagnostics")
    print("- posterior_distributions.png: Parameter posterior distributions")  
    print("- pair_plot.png: Parameter correlations")
    print("- comprehensive_delta_plots.png: RT distribution comparisons for all participants")