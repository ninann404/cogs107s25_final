"""
Run all Signal Detection Theory (SDT) models in sequence

This script runs all four SDT models:
1. Basic SDT (One Person, One Condition)
2. Conditional SDT (One Person, K Conditions)
3. Hierarchical SDT (P Persons, One Condition)
4. Full Conditional Hierarchical SDT (P Persons, K Conditions)

"""
from setup import print_version_info
import basic
import conditional
import hierarchical
import full
import time

def run_all_models():
    """Run all four SDT models in sequence and report execution times"""
    print_version_info()
    print("\n" + "="*80)
    print("RUNNING ALL SDT MODELS")
    print("="*80)
    print("Using HDI intervals with compatibility for ArviZ 0.21.0")
    print("(Newer versions would support quantile intervals for better transformation invariance)")
    
    # Track execution times
    start_times = {}
    end_times = {}
    
    # Model 1: Basic SDT
    print("\n" + "="*80)
    print("Model 1: Basic SDT (One Person, One Condition)")
    print("="*80)
    start_times[1] = time.time()
    model1, idata1 = basic.run_analysis()
    end_times[1] = time.time()
    
    # Model 2: Conditional SDT
    print("\n" + "="*80)
    print("Model 2: Conditional SDT (One Person, K Conditions)")
    print("="*80)
    start_times[2] = time.time()
    model2, idata2 = conditional.run_analysis()
    end_times[2] = time.time()
    
    # Model 3: Hierarchical SDT
    print("\n" + "="*80)
    print("Model 3: Hierarchical SDT (P Persons, One Condition)")
    print("="*80)
    start_times[3] = time.time()
    model3, idata3 = hierarchical.run_analysis()
    end_times[3] = time.time()
    
    # Model 4: Conditional Hierarchical SDT
    print("\n" + "="*80)
    print("Model 4: Conditional Hierarchical SDT (P Persons, K Conditions)")
    print("="*80)
    start_times[4] = time.time()
    model4, idata4 = full.run_analysis()
    end_times[4] = time.time()
    
    # Report execution times
    print("\n" + "="*80)
    print("EXECUTION TIMES")
    print("="*80)
    for i in range(1, 5):
        execution_time = end_times[i] - start_times[i]
        print(f"Model {i}: {execution_time:.2f} seconds ({execution_time/60:.2f} minutes)")
    
    # Return all models and inference data
    return {
        'basic': (model1, idata1),
        'conditional': (model2, idata2),
        'hierarchical': (model3, idata3),
        'full': (model4, idata4)
    }

if __name__ == "__main__":
    all_results = run_all_models() 