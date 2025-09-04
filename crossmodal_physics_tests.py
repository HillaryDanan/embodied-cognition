"""
Describe visual scenarios that require embodied understanding
Based on Battaglia et al. (2016) on physical scene understanding
"""

from physics_violation_detector import PhysicsViolationDetector
import numpy as np

detector = PhysicsViolationDetector()

print("=" * 60)
print("CROSS-MODAL PHYSICS (Visual->Linguistic)")
print("=" * 60)

# Scenarios that are obvious visually but hard linguistically
VISUAL_TESTS = {
    'tower_stability': {
        'normal': "The narrow tower with wide base stood firmly",
        'violation': "The wide tower with narrow base stood firmly",
        'target': 'tower'
    },
    'pendulum_motion': {
        'normal': "The pendulum swung back and forth symmetrically",
        'violation': "The pendulum swung only forward repeatedly",
        'target': 'pendulum'
    },
    'domino_effect': {
        'normal': "The dominoes fell sequentially from first to last",
        'violation': "The dominoes fell randomly starting from middle",
        'target': 'dominoes'
    },
    'seesaw_balance': {
        'normal': "The seesaw tilted toward the heavier child",
        'violation': "The seesaw tilted toward the lighter child",
        'target': 'seesaw'
    },
    'rolling_direction': {
        'normal': "The sphere rolled straight down the incline",
        'violation': "The sphere rolled upward on the incline",
        'target': 'sphere'
    }
}

visual_results = {}
for test_name, test_data in VISUAL_TESTS.items():
    print(f"\n{test_name.replace('_', ' ').upper()}")
    result = detector.test_violation_pair(
        test_data['normal'],
        test_data['violation'],
        test_data['target']
    )
    visual_results[test_name] = result['difference']
    print(f"Difference: {result['difference']:.3f} ({'✓' if result['difference'] > 0 else '✗'})")

print("\n" + "=" * 60)
print(f"Visual scenario detection: {sum(1 for v in visual_results.values() if v > 0)}/{len(visual_results)}")
print(f"Mean difference: {np.mean(list(visual_results.values())):.3f}")
