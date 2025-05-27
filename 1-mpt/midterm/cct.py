import pymc as pm
import numpy as np
import pandas as pd
import arviz as az
import matplotlib.pyplot as plt
from pathlib import Path

print(f"Running on PyMC v{pm.__version__}")
print(f"Running on ArviZ v{az.__version__}")


class Cct_Data_Set:
    def __init__(self, file_path):
        self._file_path = Path(file_path)
        self._X_data = None
        self._informants = None
        self._questions = None
        self._df = None
        self.load_data()
        
    def __str__(self):
        return f"CCT Data Set: * File: {self._file_path.name}\n\
              * Informants: {self.get_N_informants()}\n\
              * Questions: {self.get_N_questions()}"
    
    # Getters
    def get_N_informants(self):
        if self._informants is None:
            self.load_data()
        return len(self._informants)
    
    def get_N_questions(self):
        if self._questions is None:
            self.load_data()
        return len(self._questions)
    
    def get_X_data(self):
        return self._X_data
    
    def get_informants(self):
        return self._informants
    
    def get_questions(self):
        return self._questions
    
    def get_df(self):
        return self._df
    
    def get_coords(self):
        return {"informant": self._informants, "question": self._questions}
    
    def load_data(self):
        """
        Loads Cultural Consensus Theory data from a CSV file.

        Assumes the CSV has an 'Informant' column (or similar first column)
        and subsequent columns represent questions with 0/1 answers.

        Needs:
            file_path (str or Path): The path to the CSV file.

        After:
            tuple: A tuple containing:
                - np.ndarray: The data matrix (informants x questions) as integers.
                - list: A list of informant identifiers.
                - list: A list of question identifiers.
                - pd.DataFrame: The original DataFrame with informant index.
        """
        if not self._file_path.is_file():
            raise FileNotFoundError(f"Data file not found at: {self._file_path}")

        # Read the CSV
        df_raw = pd.read_csv(self._file_path)

        informant_col = df_raw.columns[0]
        self._df = df_raw.set_index(informant_col)
        self._informants = self._df.index.tolist()
        self._questions = self._df.columns.tolist()

        self._X_data = self._df.values.astype(np.int64)

        if np.isnan(self._X_data).any():
            raise ValueError(f"Data in {self._file_path.name} contains NaN values after loading. Please check the CSV file.")

        print(f"Successfully loaded data from: {self._file_path.name}")
        print(f"Found {len(self._informants)} informants and {len(self._questions)} questions.")

        return self

def fit_cct_model(data_set):
    coords = data_set.get_coords()

    with pm.Model(coords=coords) as cct_model:
        # Priors
        # Competence: Uniform(0.5, 1.0) prior - assumes each informant has some competence.
        D = pm.Uniform("D", lower=0.5, upper=1.0, dims="informant")

        # Consensus Answer Key: Bernoulli(0.5) prior - assumes each answer is equally likely T or F a priori.
        Z = pm.Bernoulli("Z", p=0.5, dims="question")

        # Calculate probability p_ij = Z_j * D_i + (1 - Z_j) * (1 - D_i)
        # Need to reshape D and Z for broadcasting
        # D has shape (N,), Z has shape (M,)
        # We want p to have shape (N, M)
        # D[:, None] gives shape (N, 1), Z[None, :] gives shape (1, M) - broadcasting not direct
        # Let's use explicit indexing or meshgrid approach if needed, or rely on PyMC's broadcasting if simpler forms work
        # Simpler in PyMC: pm.math.switch(Z, D, 1-D) where D needs proper dims
        
        # Reshape D to (N, 1) so it broadcasts correctly with Z (M,) -> p (N, M)
        D_reshaped = D[:, None] 
        # p_ij = Z * D_i + (1 - Z) * (1 - D_i)
        p = Z * D_reshaped + (1 - Z) * (1 - D_reshaped) 
        
        # Alternatively using pm.math.switch or pm.math.where
        # p = pm.math.where(Z, D_reshaped, 1 - D_reshaped) # If Z is 1, use D, else use 1-D

        # Likelihood
        X_obs = pm.Bernoulli("X_obs", p=p, observed=data.get_X_data(), dims=("informant", "question"))

    with cct_model:
        trace = pm.sample(2000, tune=1500, chains=4, cores=4, target_accept=0.90)

    return trace, cct_model


# --- Main Script ---

# 1. Load Data
data_choice = 'plants'

if data_choice == 'birds':
    data_file = Path(__file__).parent.parent / 'data' / 'bird_knowledge.csv'
elif data_choice == 'plants':
    data_file = Path(__file__).parent.parent / 'data' / 'plant_knowledge.csv'
else:
    raise ValueError(f"Invalid data choice: {data_choice}. Please choose 'birds' or 'plants'.")

data = Cct_Data_Set(data_file)

print(data)

# 2. Define Model in PyMC
trace, cct_model = fit_cct_model(data)

print(cct_model.basic_RVs)


# Check convergence
summary = az.summary(trace, var_names=["D", "Z"])

# Check for R-hat values > 1.01 which might indicate convergence issues
rhat_values = summary['r_hat']
if (rhat_values > 1.01).any():
    print("\nWarning: Some R-hat values are > 1.01, indicating potential convergence issues.")
    print(summary)
else:
    print(f"\nAll R-hat values look good (highest R-hat was {rhat_values.max():.2f}).")


# Plot posteriors
FIG_DIR = Path(__file__).parent / 'figures'
FIG_DIR.mkdir(parents=True, exist_ok=True)

az.plot_posterior(trace, var_names=["D"])
plt.suptitle("Posterior Distributions for Informant Competence (D)")
plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap
plt.savefig(FIG_DIR / 'cct_informant_competence.png')

az.plot_posterior(trace, var_names=["Z"])
plt.suptitle("Posterior Probabilities for Consensus Answers (Z)")
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(FIG_DIR / 'cct_consensus_answers.png')


# Extract and Report Competence Estimates
# Using posterior mean as point estimate
competence_means = trace.posterior['D'].mean(dim=['chain', 'draw']).values
competence_df = pd.DataFrame({'Informant': data.get_informants(), 'Mean Competence': competence_means})
competence_df = competence_df.sort_values(by='Mean Competence', ascending=False)
print("\nEstimated Informant Competence (Posterior Mean):")
print(competence_df)

print(f"Highest competence: {competence_df.iloc[0]['Informant']} with {competence_df.iloc[0]['Mean Competence']:.2f}")
print(f"Lowest competence : {competence_df.iloc[-1]['Informant']} with {competence_df.iloc[-1]['Mean Competence']:.2f}")


# Estimate Consensus Answer Key
# Using posterior mean probability > 0.5 as 'True' (1)
consensus_probs = trace.posterior['Z'].mean(dim=['chain', 'draw']).values
consensus_key_estimated = (consensus_probs > 0.5).astype(int)

# Compare with Naive Aggregation (Majority Vote)
# Calculate majority vote for each question
majority_vote = (data.get_X_data().mean(axis=0) > 0.5).astype(int)
comparison = pd.DataFrame({
    'Question': data.get_questions(),
    'Majority Vote': majority_vote,
    'CCT Estimate': consensus_key_estimated,
    'Posterior Prob (Z=1)': consensus_probs
})
comparison['Match'] = comparison['Majority Vote'] == comparison['CCT Estimate']

print("\nComparison: Majority Vote vs CCT Consensus Mismatches:")
print(comparison[~comparison['Match']])
