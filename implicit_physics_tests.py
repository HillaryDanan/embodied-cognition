"""
Test physics understanding when not explicitly framed as physics
Based on McCloskey (1983) naive physics misconceptions
"""

from physics_violation_detector import PhysicsViolationDetector
import numpy as np

detector = PhysicsViolationDetector()

print("=" * 60)
print("IMPLICIT PHYSICS IN EVERYDAY CONTEXTS")
print("=" * 60)

# Everyday scenarios that implicitly require physics understanding
IMPLICIT_TESTS = {
    'cooking_physics': {
        'normal': "She put the pot on the stove to boil",
        'violation': "She put the pot in the freezer to boil",
        'target': 'pot'
    },
    'sports_trajectory': {
        'normal': "He threw the basketball in an arc toward the hoop",
        'violation': "He threw the basketball in zigzags toward the hoop",
        'target': 'basketball'
    },
    'furniture_stability': {
        'normal': "They placed the vase in the table's center",
        'violation': "They placed the vase beyond the table's edge",
        'target': 'vase'
    },
    'clothing_physics': {
        'normal': "Her scarf fluttered behind her in the wind",
        'violation': "Her scarf stayed rigid against the wind",
        'target': 'scarf'
    },
    'bathing_physics': {
        'normal': "The bathtub filled from bottom to top gradually",
        'violation': "The bathtub filled from top to bottom gradually",
        'target': 'bathtub'
    },
    'walking_physics': {
        'normal': "She leaned forward slightly while walking uphill",
        'violation': "She leaned backward strongly while walking uphill",
        'target': 'She'
    }
}

implicit_results = {}
for test_name, test_data in IMPLICIT_TESTS.items():
    print(f"\n{test_name.replace('_', ' ').upper()}")
    result = detector.test_violation_pair(
        test_data['normal'],
        test_data['violation'],
        test_data['target']
    )
    implicit_results[test_name] = result['difference']
    print(f"Difference: {result['difference']:.3f} ({'✓' if result['difference'] > 0 else '✗'})")

print("\n" + "=" * 60)
print("IMPLICIT VS EXPLICIT COMPARISON")
print(f"Detection rate: {sum(1 for v in implicit_results.values() if v > 0)}/{len(implicit_results)}")
print(f"Mean difference: {np.mean(list(implicit_results.values())):.3f}")
