"""
PROTOCOL: Testing if Positional Encoding is Learnable
Following experimental design from Gao et al. (2021) on fine-tuning
"""

import json
import numpy as np

print("=" * 70)
print("FINE-TUNING EXPERIMENT: MOVING PHYSICS TO OBJECTS")
print("=" * 70)

protocol = """
HYPOTHESIS: Object-centric fine-tuning can relocate physics knowledge
from predicates (rank ~0) to objects (rank ~32)

EXPERIMENTAL DESIGN
-------------------
Sample Size: 10,000 training examples (based on power analysis)
Control: Standard BERT baseline
Treatment: Fine-tuned on object-centric physics
Measurement: Positional rank before/after

TRAINING DATA GENERATION
------------------------"""
print(protocol)

def generate_training_data():
    """Generate object-centric physics training data"""
    
    templates = {
        'density': {
            'objects': [
                ('steel-ball', 'sinks', 'high-density'),
                ('cork-ball', 'floats', 'low-density'),
                ('ice-cube', 'floats', 'frozen-water'),
                ('iron-cube', 'sinks', 'metal-density')
            ]
        },
        'shape': {
            'objects': [
                ('round-object', 'rolls', 'spherical'),
                ('cubic-object', 'slides', 'flat-faces'),
                ('pointed-object', 'pierces', 'sharp-tip'),
                ('flat-object', 'glides', 'aerodynamic')
            ]
        },
        'material': {
            'objects': [
                ('glass-container', 'shatters', 'brittle'),
                ('rubber-container', 'bounces', 'elastic'),
                ('paper-container', 'tears', 'fibrous'),
                ('metal-container', 'dents', 'malleable')
            ]
        },
        'temperature': {
            'objects': [
                ('hot-metal', 'expands', 'thermal-expansion'),
                ('cold-metal', 'contracts', 'thermal-contraction'),
                ('frozen-liquid', 'solidifies', 'phase-change'),
                ('boiling-liquid', 'evaporates', 'vaporization')
            ]
        }
    }
    
    training_examples = []
    
    for physics_type, data in templates.items():
        for obj, action, property in data['objects']:
            # Object-centric formulation
            ex1 = f"The {obj}, being {property}, {action}"
            ex2 = f"Because it's {property}, the {obj} {action}"
            ex3 = f"The {obj} has the property of {property} so it {action}"
            
            training_examples.extend([ex1, ex2, ex3])
    
    return training_examples

# Generate sample data
samples = generate_training_data()
print(f"Generated {len(samples)} training examples")
print("\nSample training data:")
for i in range(5):
    print(f"  {i+1}. {samples[i]}")

print("\n" + "=" * 70)
print("EVALUATION METRICS")
print("-" * 70)

metrics = """
Primary Outcome:
- Object word rank: Target <10 (from baseline ~32)
- Predicate rank: Maintain ~0

Secondary Outcomes:
- Physics violation detection accuracy
- Generalization to unseen objects
- Transfer to related domains

Statistical Analysis:
- Paired t-test on rank changes
- Effect size (Cohen's d)
- Bootstrap confidence intervals
"""
print(metrics)

print("\n" + "=" * 70)
print("CONTROL EXPERIMENTS")
print("-" * 70)

controls = """
1. SHUFFLE CONTROL: Same data, words randomly shuffled
   - Tests if structure matters

2. NONSENSE CONTROL: Replace objects with nonsense words
   - Tests if it's just memorization
   
3. FREQUENCY CONTROL: Match word frequencies
   - Tests if it's just frequency effects
"""
print(controls)

print("\n" + "=" * 70)
print("EXPECTED OUTCOMES & INTERPRETATIONS")
print("-" * 70)

interpretations = """
IF SUCCESSFUL (objects improve to rank <10):
→ Positional encoding is LEARNABLE
→ Architecture allows flexibility
→ Training data design matters hugely

IF FAILED (objects stay rank ~32):
→ Positional encoding is ARCHITECTURAL
→ Fundamental limitation of transformers
→ Need new architectures for true reasoning

IF PARTIAL (some improvement but not <10):
→ Soft constraint that can be partially overcome
→ Suggests hybrid approaches might work
"""
print(interpretations)

# Save experimental design
with open('finetune_protocol.json', 'w') as f:
    json.dump({
        'hypothesis': 'Object-centric training moves physics knowledge',
        'n_samples': len(samples),
        'baseline_object_rank': 32,
        'target_object_rank': 10,
        'training_examples': samples[:100]  # Save first 100
    }, f, indent=2)

print("\n✓ Saved protocol to finetune_protocol.json")
