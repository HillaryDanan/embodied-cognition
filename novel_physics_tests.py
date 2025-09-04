"""
Novel physics scenarios unlikely to appear in training data
Based on Battaglia et al. (2013) PNAS - intuitive physics engine
"""

from physics_violation_detector import PhysicsViolationDetector
import numpy as np

detector = PhysicsViolationDetector()

print("=" * 60)
print("NOVEL PHYSICS SCENARIOS")
print("=" * 60)

# These combine multiple physics principles in novel ways
NOVEL_TESTS = {
    'liquid_in_mesh': {
        'normal': "The water pooled in the solid bowl completely",
        'violation': "The water pooled in the mesh basket completely",
        'target': 'water'
    },
    'shadow_physics': {
        'normal': "The shadow appeared behind the object away from light",
        'violation': "The shadow appeared between the object and light",
        'target': 'shadow'
    },
    'momentum_transfer': {
        'normal': "The marble stopped after hitting the heavy block",
        'violation': "The marble reversed without hitting any surface",
        'target': 'marble'
    },
    'thermal_direction': {
        'normal': "The ice melted faster near the heater",
        'violation': "The ice melted faster near the freezer",
        'target': 'ice'
    },
    'center_of_mass': {
        'normal': "The L-shaped block tipped over the edge",
        'violation': "The L-shaped block balanced past the edge",
        'target': 'block'
    },
    'fluid_dynamics': {
        'normal': "The syrup flowed slower than the water",
        'violation': "The syrup flowed faster than the water",
        'target': 'syrup'
    }
}

results = {}
for test_name, test_data in NOVEL_TESTS.items():
    print(f"\n{test_name.upper()}")
    print(f"Normal:    {test_data['normal']}")
    print(f"Violation: {test_data['violation']}")
    
    result = detector.test_violation_pair(
        test_data['normal'],
        test_data['violation'],
        test_data['target']
    )
    
    results[test_name] = result['difference']
    print(f"Difference: {result['difference']:.3f} ({'✓' if result['difference'] > 0 else '✗'})")

print("\n" + "=" * 60)
print(f"NOVEL PHYSICS DETECTION RATE: {sum(1 for v in results.values() if v > 0)}/{len(results)}")
print(f"Mean difference: {np.mean(list(results.values())):.3f}")
print(f"Std deviation: {np.std(list(results.values())):.3f}")
