"""
TRUE META-ANALYSIS: What's really happening?
"""

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# All our actual results
results = {
    'Original Tests': {
        'object_permanence': 0.848,
        'gravity': -0.506,
        'support': 1.199,
        'containment': 1.915,
        'continuity': 4.460
    },
    'Novel Physics': {
        'liquid_in_mesh': 0.103,
        'shadow_physics': 0.550,
        'momentum_transfer': 0.591,
        'thermal_direction': -1.051,
        'center_of_mass': -1.933,
        'fluid_dynamics': 0.110
    },
    'Implicit Physics': {
        'cooking_physics': 3.017,
        'sports_trajectory': -0.843,
        'furniture_stability': 0.501,
        'clothing_physics': 7.782,
        'bathing_physics': -0.016,
        'walking_physics': -0.019
    },
    'Cross-Modal': {
        'tower_stability': 0.475,
        'pendulum_motion': 1.597,
        'domino_effect': -1.185,
        'seesaw_balance': -0.378,
        'rolling_direction': -1.139
    }
}

print("=" * 70)
print("COMPREHENSIVE EMBODIMENT GAP ANALYSIS")
print("=" * 70)

# 1. Overall statistics
all_values = []
for category, tests in results.items():
    all_values.extend(tests.values())

all_values = np.array(all_values)

print("\n📊 OVERALL STATISTICS")
print(f"Total tests: {len(all_values)}")
print(f"Successful detection (>0): {sum(all_values > 0)}/{len(all_values)} = {sum(all_values > 0)/len(all_values)*100:.1f}%")
print(f"Mean difference: {np.mean(all_values):.3f}")
print(f"Std deviation: {np.std(all_values):.3f}")
print(f"Range: [{np.min(all_values):.3f}, {np.max(all_values):.3f}]")

# 2. Category comparison
print("\n📈 CATEGORY COMPARISON")
for category, tests in results.items():
    values = list(tests.values())
    print(f"\n{category}:")
    print(f"  Detection rate: {sum(v > 0 for v in values)}/{len(values)}")
    print(f"  Mean: {np.mean(values):.3f}")
    print(f"  Std: {np.std(values):.3f}")
    print(f"  95% CI: [{np.mean(values) - 1.96*np.std(values)/np.sqrt(len(values)):.3f}, "
          f"{np.mean(values) + 1.96*np.std(values)/np.sqrt(len(values)):.3f}]")

# 3. Find patterns in failures vs successes
print("\n🔍 FAILURE ANALYSIS")
failures = [(name, val) for cat in results.values() for name, val in cat.items() if val < 0]
successes = [(name, val) for cat in results.values() for name, val in cat.items() if val > 1]

print(f"\nSTRONGEST FAILURES (<0):")
for name, val in sorted(failures, key=lambda x: x[1])[:5]:
    print(f"  {name}: {val:.3f}")

print(f"\nSTRONGEST SUCCESSES (>1):")
for name, val in sorted(successes, key=lambda x: x[1], reverse=True)[:5]:
    print(f"  {name}: {val:.3f}")

# 4. Statistical test: Is there ANY consistent physics understanding?
# H0: Detection is random (mean = 0)
t_stat, p_value = stats.ttest_1samp(all_values, 0)
print("\n📊 HYPOTHESIS TEST: Is detection better than chance?")
print(f"t-statistic: {t_stat:.3f}")
print(f"p-value: {p_value:.6f}")
print(f"Result: {'Significant' if p_value < 0.05 else 'Not significant'}")

# 5. Variance analysis - is the inconsistency itself the finding?
print("\n🎲 VARIANCE ANALYSIS")
# Compare variance across categories
f_stat, p_anova = stats.f_oneway(
    list(results['Original Tests'].values()),
    list(results['Novel Physics'].values()),
    list(results['Implicit Physics'].values()),
    list(results['Cross-Modal'].values())
)
print(f"ANOVA F-statistic: {f_stat:.3f}")
print(f"ANOVA p-value: {p_anova:.6f}")
print(f"Categories {'differ' if p_anova < 0.05 else 'do not differ'} significantly")

# 6. The key insight
print("\n💡 KEY INSIGHT")
print("The model doesn't have physics intuition - it has LINGUISTIC ASSOCIATIONS")
print(f"Coefficient of variation: {np.std(all_values)/abs(np.mean(all_values)):.3f}")
print("(Values >1 indicate extreme inconsistency)")

# 7. What predicts success?
print("\n🎯 PATTERN DETECTION")
# Words that appear in successful vs failed tests
success_words = set()
failure_words = set()

for category, tests in results.items():
    for test_name, value in tests.items():
        words = test_name.split('_')
        if value > 0.5:
            success_words.update(words)
        elif value < -0.5:
            failure_words.update(words)

print(f"Words in successful tests: {success_words}")
print(f"Words in failed tests: {failure_words}")
print(f"Overlap: {success_words & failure_words}")
