"""Visualize the embodiment gap"""

import matplotlib.pyplot as plt
import seaborn as sns
from physics_violation_detector import PhysicsViolationDetector, PHYSICS_TESTS

def create_violation_plot():
    detector = PhysicsViolationDetector()
    
    # Collect data
    tests = []
    normal_scores = []
    violation_scores = []
    
    for test_name, test_data in PHYSICS_TESTS.items():
        result = detector.test_violation_pair(
            test_data['normal'],
            test_data['violation'],
            test_data['target']
        )
        tests.append(test_name.replace('_', ' ').title())
        normal_scores.append(result['normal_surprise'])
        violation_scores.append(result['violation_surprise'])
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    x = range(len(tests))
    width = 0.35
    
    ax.bar([i - width/2 for i in x], normal_scores, width, label='Normal Physics', color='green', alpha=0.7)
    ax.bar([i + width/2 for i in x], violation_scores, width, label='Violation', color='red', alpha=0.7)
    
    ax.set_xlabel('Physics Test')
    ax.set_ylabel('Surprise Score')
    ax.set_title('LLM Detection of Physics Violations\n(Higher = More Surprising)')
    ax.set_xticks(x)
    ax.set_xticklabels(tests, rotation=45, ha='right')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('physics_violations.png', dpi=150)
    print("✓ Saved plot to physics_violations.png")

if __name__ == "__main__":
    create_violation_plot()
