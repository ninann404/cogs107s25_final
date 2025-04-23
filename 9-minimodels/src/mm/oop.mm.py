#!/usr/bin/env python3
# clever_hans_bayes_factor_oop.py
# OOP implementation of Bayes factor calculation for Clever Hans example

import numpy as np
from scipy import stats
import math


class NormalDistribution:
    """Class to represent a normal distribution and calculate probabilities of regions."""
    
    def __init__(self, mean, sd):
        """Initialize with a mean and standard deviation."""
        self.mean = mean
        self.sd = sd
    
    def probability_less_than(self, value):
        """Calculate P(X < value)."""
        return stats.norm.cdf(value, loc=self.mean, scale=self.sd)
    
    def probability_greater_than(self, value):
        """Calculate P(X > value)."""
        return 1 - stats.norm.cdf(value, loc=self.mean, scale=self.sd)
    
    def probability_between(self, lower, upper):
        """Calculate P(lower < X < upper)."""
        return (stats.norm.cdf(upper, loc=self.mean, scale=self.sd) - 
                stats.norm.cdf(lower, loc=self.mean, scale=self.sd))
    
    def calculate_regions(self, epsilon):
        """Calculate probabilities for the three regions: negative, zero, positive."""
        p_neg = self.probability_less_than(-epsilon)
        p_pos = self.probability_greater_than(epsilon)
        p_zero = 1 - p_neg - p_pos
        
        return {
            'neg': p_neg,
            'zero': p_zero,
            'pos': p_pos
        }


class ConjugateNormalModel:
    """Class to handle the conjugate normal-normal Bayesian model."""
    
    @staticmethod
    def calculate_posterior(observed, se, prior_mean, prior_sd):
        """Calculate posterior distribution parameters for normal-normal model."""
        prior_precision = 1 / (prior_sd ** 2)
        data_precision = 1 / (se ** 2)
        posterior_precision = prior_precision + data_precision
        posterior_mean = ((prior_mean * prior_precision) + 
                          (observed * data_precision)) / posterior_precision
        posterior_sd = math.sqrt(1 / posterior_precision)
        
        return NormalDistribution(posterior_mean, posterior_sd)


class Theory:
    """Class to represent a theory with predictions."""
    
    def __init__(self, name, predictions):
        """Initialize with a name and dictionary of predictions."""
        self.name = name
        self.predictions = predictions  # Dictionary: {effect_name: prediction_type}
        
    def calculate_evidence(self, posteriors, priors, epsilon):
        """Calculate evidence for this theory based on the data."""
        evidence = 1.0
        
        for effect_name, prediction_type in self.predictions.items():
            posterior = posteriors[effect_name].calculate_regions(epsilon)
            prior = priors[effect_name].calculate_regions(epsilon)
            
            if prediction_type == 'zero':
                evidence *= posterior['zero'] / prior['zero']
            elif prediction_type == 'neg':
                evidence *= posterior['neg'] / prior['neg']
            elif prediction_type == 'pos':
                evidence *= posterior['pos'] / prior['pos']
                
        return evidence


class BayesFactorAnalysis:
    """Class to manage the Bayes factor analysis."""
    
    def __init__(self, epsilon=0.1, prior_mean=0, prior_sd=0.25):
        """Initialize the analysis with parameters."""
        self.epsilon = epsilon
        self.prior_mean = prior_mean
        self.prior_sd = prior_sd
        self.theories = {}
        self.effects = {}
        self.posteriors = {}
        self.priors = {}
        
    def add_theory(self, name, predictions):
        """Add a theory to the analysis."""
        self.theories[name] = Theory(name, predictions)
        
    def add_effect(self, name, observed, se):
        """Add an observed effect to the analysis."""
        self.effects[name] = {
            'observed': observed,
            'se': se
        }
        
    def calculate_posteriors(self):
        """Calculate posterior distributions for all effects."""
        for name, effect in self.effects.items():
            self.posteriors[name] = ConjugateNormalModel.calculate_posterior(
                effect['observed'], 
                effect['se'], 
                self.prior_mean, 
                self.prior_sd
            )
            self.priors[name] = NormalDistribution(self.prior_mean, self.prior_sd)
        
    def calculate_bayes_factors(self):
        """Calculate Bayes factors for all pairs of theories."""
        evidence = {}
        bayes_factors = {}
        
        # Calculate evidence for each theory
        for name, theory in self.theories.items():
            evidence[name] = theory.calculate_evidence(
                self.posteriors, 
                self.priors, 
                self.epsilon
            )
        
        # Calculate Bayes factors
        for name1 in self.theories:
            for name2 in self.theories:
                if name1 != name2:
                    key = f"{name1}:{name2}"
                    bayes_factors[key] = evidence[name1] / evidence[name2]
        
        return {
            'evidence': evidence,
            'bayes_factors': bayes_factors
        }
        
    def report(self):
        """Generate a report of the analysis."""
        self.calculate_posteriors()
        results = self.calculate_bayes_factors()
        
        # Header for the report
        report = []
        report.append("\nClever Hans Bayes Factor Analysis (OOP Implementation)")
        report.append("=" * 60)
        
        # Observed effects
        report.append("\nObserved Effects")
        report.append("-" * 16)
        for name, effect in self.effects.items():
            report.append(f"{name:15} {effect['observed']:.3f}")
        
        # Posterior distributions
        report.append("\nPosterior Distribution Parameters")
        report.append("-" * 32)
        for name, posterior in self.posteriors.items():
            report.append(f"{name} mean: {posterior.mean:.3f}, sd: {posterior.sd:.3f}")
        
        # Posterior probabilities
        report.append("\nPosterior Probabilities")
        report.append("-" * 22)
        for name, posterior in self.posteriors.items():
            regions = posterior.calculate_regions(self.epsilon)
            report.append(f"P(|{name}| < {self.epsilon:.2f} | data): {regions['zero']:.3f}")
            report.append(f"P({name} < -{self.epsilon:.2f} | data): {regions['neg']:.3f}")
        
        # Prior probabilities
        report.append("\nPrior Probabilities")
        report.append("-" * 19)
        for name, prior in self.priors.items():
            regions = prior.calculate_regions(self.epsilon)
            report.append(f"P(|{name}| < {self.epsilon:.2f}): {regions['zero']:.3f}")
            report.append(f"P({name} < -{self.epsilon:.2f}): {regions['neg']:.3f}")
        
        # Evidence
        report.append("\nEvidence")
        report.append("-" * 8)
        for name, value in results['evidence'].items():
            report.append(f"Evidence for Theory {name}: {value:.3f}")
        
        # Bayes factors
        report.append("\nBayes Factors")
        report.append("-" * 13)
        max_bf = 0
        max_pair = ""
        for pair, bf in results['bayes_factors'].items():
            report.append(f"Bayes factor {pair}: {bf:.3f}")
            if bf > max_bf:
                max_bf = bf
                max_pair = pair
        
        # Interpretation
        winning = max_pair.split(':')[0]
        report.append(f"\nInterpretation: Evidence favors Theory {winning} by a factor of 10^{math.log10(max_bf):.0f}")
        
        return "\n".join(report)


def main():
    """Run the Clever Hans example analysis."""
    # Data from experiment
    base_accuracy = 0.9        # Accuracy with handler and questioner
    handler_removed = 0.9      # Accuracy with only handler removed
    both_removed = 0.0         # Accuracy with both handler and questioner removed
    
    # Calculate effect sizes
    delta1 = handler_removed - base_accuracy
    delta2 = both_removed - handler_removed
    
    # Calculate standard errors (approximation for this example)
    se1 = math.sqrt(base_accuracy * (1 - base_accuracy) / 10)
    se2 = math.sqrt(handler_removed * (1 - handler_removed) / 10)
    
    # Create and configure the analysis
    analysis = BayesFactorAnalysis(epsilon=0.1, prior_mean=0, prior_sd=0.25)
    
    # Add observed effects
    analysis.add_effect('delta1', delta1, se1)
    analysis.add_effect('delta2', delta2, se2)
    
    # Define theories
    # Theory A: Hans can do arithmetic - both effects should be negligible
    analysis.add_theory('A', {'delta1': 'zero', 'delta2': 'zero'})
    
    # Theory B: Hans responds to cues - first effect negligible, second negative
    analysis.add_theory('B', {'delta1': 'zero', 'delta2': 'neg'})
    
    # Generate and print the report
    print(analysis.report())


if __name__ == "__main__":
    main() 