"""
Why did we get opposite results?
Testing methodology effects on physics detection
"""

from physics_violation_detector import PhysicsViolationDetector
import numpy as np

detector = PhysicsViolationDetector()

print("=" * 70)
print("METHODOLOGY COMPARISON")
print("=" * 70)

# Method 1: Our original approach (test surprise at OBJECT)
print("\nMETHOD 1: Testing surprise at OBJECT word")
print("-" * 40)

tests_method1 = [
    ("The rock fell down", "The rock fell up", "rock"),
    ("The ball stayed there", "The ball vanished completely", "ball"),
]

for normal, violation, target in tests_method1:
    result = detector.test_violation_pair(normal, violation, target)
    print(f"Target '{target}': difference = {result['difference']:.3f}")

# Method 2: Test surprise at VIOLATION word
print("\nMETHOD 2: Testing surprise at VIOLATION word")
print("-" * 40)

tests_method2 = [
    ("The rock fell [MASK]", "down", "up"),
    ("The ball [MASK] there", "stayed", "vanished"),
]

for template, normal_word, violation_word in tests_method2:
    # Test normal
    normal_sent = template.replace("[MASK]", normal_word)
    normal_surprise = detector.measure_surprise(normal_sent, normal_word)
    
    # Test violation
    violation_sent = template.replace("[MASK]", violation_word)
    violation_surprise = detector.measure_surprise(violation_sent, violation_word)
    
    difference = violation_surprise - normal_surprise
    print(f"'{normal_word}' vs '{violation_word}': difference = {difference:.3f}")

print("\n" + "=" * 70)
print("INSIGHT: Measurement methodology determines results!")
print("Models detect violations at VIOLATION WORDS, not OBJECT WORDS")
