"""
Gravity failure investigation - systematic variable isolation
Based on methodology from Tenenbaum et al. (2011) Science
"""

from physics_violation_detector import PhysicsViolationDetector
import numpy as np

detector = PhysicsViolationDetector()

print("=" * 60)
print("GRAVITY FAILURE INVESTIGATION")
print("=" * 60)

# Test 1: Remove ALL modifiers
print("\n1. PURE DIRECTION WORDS")
tests = [
    ("fell down", "fell up"),
    ("dropped down", "dropped up"),
    ("sank down", "sank up"),
]

for normal, violation in tests:
    # Test with different objects
    objects = ["rock", "ball", "apple", "stone"]
    differences = []
    
    for obj in objects:
        normal_sent = f"The {obj} {normal}"
        violation_sent = f"The {obj} {violation}"
        result = detector.test_violation_pair(normal_sent, violation_sent, obj)
        differences.append(result['difference'])
    
    mean_diff = np.mean(differences)
    print(f"{normal} vs {violation}: {mean_diff:.3f} ({'✓' if mean_diff > 0 else '✗'})")

# Test 2: Word frequency confound?
print("\n2. WORD FREQUENCY ANALYSIS")
direction_pairs = [
    ("downward", "upward"),  # Both common
    ("descended", "ascended"),  # More formal
    ("plummeted", "soared"),  # More dramatic
    ("fell", "rose"),  # Simple past
]

for down_word, up_word in direction_pairs:
    normal = f"The object {down_word}"
    violation = f"The object {up_word}"
    result = detector.test_violation_pair(normal, violation, "object")
    print(f"{down_word} vs {up_word}: {result['difference']:.3f}")

# Test 3: Context matters?
print("\n3. CONTEXT DEPENDENCY")
contexts = [
    ("", ""),  # No context
    ("Naturally, ", "Naturally, "),  # Physical context
    ("In Earth's gravity, ", "In Earth's gravity, "),  # Explicit physics
    ("Without support, ", "Without support, "),  # Causal context
]

for prefix, suffix in contexts:
    normal = f"{prefix}the rock fell down{suffix}"
    violation = f"{prefix}the rock fell up{suffix}"
    result = detector.test_violation_pair(normal, violation, "rock")
    print(f"Context '{prefix}...{suffix}': {result['difference']:.3f}")
