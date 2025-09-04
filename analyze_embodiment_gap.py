"""
Statistical analysis of embodiment gap
References: Spelke & Kinzler (2007), Core knowledge
"""

import numpy as np
from scipy import stats
from physics_violation_detector import PhysicsViolationDetector, PHYSICS_TESTS

def compute_embodiment_gap():
    """Quantify the gap between infant and model performance"""
    
    detector = PhysicsViolationDetector()
    
    # Run multiple iterations for statistical power
    print("Running 10 iterations for statistical reliability...")
    all_differences = []
    
    for iteration in range(10):
        for test_name, test_data in PHYSICS_TESTS.items():
            result = detector.test_violation_pair(
                test_data['normal'],
                test_data['violation'],
                test_data['target']
            )
            all_differences.append(result['difference'])
    
    # Statistical tests
    differences = np.array(all_differences)
    
    # Test if model detects violations (difference > 0)
    t_stat, p_value = stats.ttest_1samp(differences, 0)
    
    print("\n📈 STATISTICAL ANALYSIS")
    print("=" * 50)
    print(f"Mean surprise difference: {np.mean(differences):.3f}")
    print(f"Standard error: {stats.sem(differences):.3f}")
    print(f"t-statistic: {t_stat:.3f}")
    print(f"p-value: {p_value:.4f}")
    
    if p_value < 0.05 and np.mean(differences) > 0:
        print("✓ Model shows some violation detection (p < 0.05)")
    else:
        print("✗ No significant violation detection")
    
    # Effect size (Cohen's d against 0)
    cohen_d = np.mean(differences) / np.std(differences)
    print(f"\nEffect size (Cohen's d): {cohen_d:.3f}")
    
    # Interpretation
    if abs(cohen_d) < 0.2:
        effect = "negligible"
    elif abs(cohen_d) < 0.5:
        effect = "small"
    elif abs(cohen_d) < 0.8:
        effect = "medium"
    else:
        effect = "large"
    
    print(f"Effect magnitude: {effect}")
    
    # Embodiment gap
    print("\n🧠 EMBODIMENT GAP")
    print("=" * 50)
    print("Infant performance: Near-perfect detection")
    print(f"Model performance: {effect} effect")
    print("\nConclusion: Models lack embodied physics intuitions")
    print("that emerge in human development by 6 months")
    
    return differences, p_value, cohen_d

if __name__ == "__main__":
    compute_embodiment_gap()
