# Generate samples from a diffusion model using PyDDM

import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path

# Try to import PyDDM and its components
# If PyDDM is not installed, you can also install it manually
try:
    import pyddm as ddm
except ImportError as e:
    print(f"Failed to import PyDDM. Let's install it first...")
    os.system("pip install pyddm")
    import pyddm as ddm


def print_pyddm_version():
    print(f"PyDDM version: {ddm.__version__}")


def print_parameters(parameters):
    # PyDDM uses slightly different parameter conventions
    pyddm_x0 = parameters['boundary_sep_a'] * (parameters['relative_bias_b'] - 0.5)
    pyddm_bound = parameters['boundary_sep_a'] / 2.0
    # Print everything
    print(f"Parameters set for PyDDM simulation:")
    print(f"  + Drift Rate (v): {parameters['drift_rate_v']}")
    print(f"  + Boundary Separation (a): {parameters['boundary_sep_a']}")
    print(f"  + Non-decision Time (t0): {parameters['non_decision_t0']} s")
    print(f"  + Relative Bias (b): {parameters['relative_bias_b']}")
    print(f"PyDDM Conversions:")
    print(f"  -> PyDDM Bound Parameter (B for +/-B): {pyddm_bound}")
    print(f"  -> PyDDM Initial Condition (x0 relative to 0): {pyddm_x0}")


def print_summary(sim_rts, sim_choices):
    mean_upper_rt = np.mean(sim_rts[sim_choices == 1])
    mean_lower_rt = np.mean(sim_rts[sim_choices == 0])
    prop_upper = np.mean(sim_choices)

    print(f"Summary of PyDDM Simulation:")
    print(f"  + Mean Upper Boundary Response Time: {mean_upper_rt:.4f} seconds")
    print(f"  + Mean Lower Boundary Response Time: {mean_lower_rt:.4f} seconds")
    print(f"  + Proportion of Upper Boundary Choices: {prop_upper:.4f}")
    

def plot_rt_histogram(sim_rts, sim_choices, parameters, save_to_file=None):
    plt.figure(figsize=(10, 6))
    plt.hist(sim_rts[sim_choices == 1], 
             bins=50, 
             density=True, 
             alpha=0.75, 
             label="Simulated Upper Boundary RTs (PyDDM)", 
             color='mediumseagreen', 
             edgecolor='black')
    plt.hist(sim_rts[sim_choices == 0], 
             bins=50, 
             density=True, 
             alpha=0.75, 
             label="Simulated Lower Boundary RTs (PyDDM)", 
             color='lightcoral', 
             edgecolor='black')
    plt.axvline(np.mean(sim_rts[sim_choices == 1]), 
                color='red', 
                linestyle='dashed', 
                linewidth=3, 
                label=f"Mean Upper RT: {np.mean(sim_rts[sim_choices == 1]):.2f}s")
    plt.axvline(np.mean(sim_rts[sim_choices == 0]), 
                color='blue', 
                linestyle='dashed', 
                linewidth=3, 
                label=f"Mean Lower RT: {np.mean(sim_rts[sim_choices == 0]):.2f}s")
    title_str = (
        f"PyDDM: Simulated DDM RTs ({len(sim_rts)} Trials)\n"
        f"v={parameters['drift_rate_v']}, "
        f"a={parameters['boundary_sep_a']}, "
        f"t0={parameters['non_decision_t0']}, "
        f"b={parameters['relative_bias_b']}"
    )
    plt.title(title_str, fontsize=12)
    plt.xlabel("Response Time (s)", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    
    # Better practice would be to prepare the figure in one function and then call that function
    # from two wrappers, one for saving and one for showing
    if save_to_file is not None:
        plt.savefig(save_to_file)
        print(f"\nResponse time histogram saved to '{save_to_file}'")
    else:
        plt.show()


def simulate_ddm(parameters, num_sim_trials):
    if parameters.get('simulation_seed', None) is not None:
        np.random.seed(parameters["simulation_seed"])

    # Make PyDDM parameters
    pyddm_drift = ddm.DriftConstant(drift=parameters["drift_rate_v"])
    pyddm_bound = ddm.BoundConstant(B=parameters['boundary_sep_a'] / 2.0)
    pyddm_x0 = ddm.ICPoint(x0=parameters['boundary_sep_a'] * (parameters['relative_bias_b'] - 0.5))
    pyddm_nondectime = ddm.OverlayNonDecision(nondectime=parameters["non_decision_t0"])
    pyddm_dt = 0.001

    # Create PyDDM Model instance
    model = ddm.Model(
        name    = "DDM_via_PyDDM",
        drift   = pyddm_drift,
        bound   = pyddm_bound,
        IC      = pyddm_x0,
        overlay = pyddm_nondectime,
        dt      = pyddm_dt
    )

    # Simulate multiple trials using model.solve() and solution.resample()
    solution = model.solve() 
    sample_obj = solution.resample(num_sim_trials)
    
    # Extract correct and incorrect RTs 
    correct_rts = sample_obj.choice_upper
    incorrect_rts = sample_obj.choice_lower
    
    # Combine into a single array
    rts = np.concatenate([correct_rts, incorrect_rts])
    choices = np.concatenate([np.ones(len(correct_rts)), np.zeros(len(incorrect_rts))])

    return rts, choices


# --- Main execution block for demonstration ---
if __name__ == "__main__":
    print("--- PyDDM Simulator Demonstration ---")

    print_pyddm_version()

    parameters = {
        "drift_rate_v": 1.0,
        "boundary_sep_a": 2.0,
        "non_decision_t0": 0.25,
        "relative_bias_b": 0.35
    }

    print_parameters(parameters)

    sim_rts, sim_choices = simulate_ddm(parameters, 10000)

    print_summary(sim_rts, sim_choices)

    save_plot_name = Path(__file__).parent.parent / "plots" / "pyddm_simulation_rts_histogram.png"
    plot_rt_histogram(sim_rts, 
                      sim_choices,
                      parameters, 
                      save_to_file=save_plot_name)
    
