"""
Generate full training dataset for fine-tuning experiment
Based on our positional encoding findings
"""

import json
import random

print("=" * 70)
print("GENERATING OBJECT-CENTRIC PHYSICS TRAINING DATA")
print("=" * 70)

# Physical properties dictionary
physics_properties = {
    'density': {
        'heavy': ['sinks', 'falls-quickly', 'drops'],
        'light': ['floats', 'falls-slowly', 'drifts']
    },
    'temperature': {
        'hot': ['expands', 'melts', 'steams'],
        'cold': ['contracts', 'freezes', 'solidifies']
    },
    'shape': {
        'spherical': ['rolls', 'rotates', 'spins'],
        'flat': ['slides', 'glides', 'slips']
    },
    'state': {
        'liquid': ['flows', 'pours', 'splashes'],
        'solid': ['maintains-shape', 'resists', 'holds']
    },
    'elasticity': {
        'elastic': ['bounces', 'springs', 'rebounds'],
        'rigid': ['shatters', 'cracks', 'breaks']
    }
}

# Generate contrastive pairs
def generate_contrastive_pairs(n=5000):
    pairs = []
    
    for _ in range(n):
        prop_type = random.choice(list(physics_properties.keys()))
        properties = physics_properties[prop_type]
        
        prop1, actions1 = random.choice(list(properties.items()))
        prop2, actions2 = [(k,v) for k,v in properties.items() if k != prop1][0]
        
        action1 = random.choice(actions1)
        action2 = random.choice(actions2)
        
        # Generate object names
        obj_base = random.choice(['ball', 'cube', 'rod', 'sheet', 'block'])
        
        # Contrastive sentences
        sent1 = f"The {prop1}-{obj_base} {action1}"
        sent2 = f"The {prop2}-{obj_base} {action2}"
        
        pairs.append({
            'property_type': prop_type,
            'sentence_1': sent1,
            'sentence_2': sent2,
            'contrast': f"{prop1} vs {prop2}"
        })
    
    return pairs

# Generate training data
train_data = generate_contrastive_pairs(5000)
val_data = generate_contrastive_pairs(500)
test_data = generate_contrastive_pairs(500)

print(f"Generated:")
print(f"  Training: {len(train_data)} pairs")
print(f"  Validation: {len(val_data)} pairs")
print(f"  Test: {len(test_data)} pairs")

print("\nSample contrastive pairs:")
for i in range(3):
    pair = train_data[i]
    print(f"\nPair {i+1} ({pair['contrast']}):")
    print(f"  A: {pair['sentence_1']}")
    print(f"  B: {pair['sentence_2']}")

# Save datasets
with open('finetune_train.json', 'w') as f:
    json.dump(train_data, f)
with open('finetune_val.json', 'w') as f:
    json.dump(val_data, f)
with open('finetune_test.json', 'w') as f:
    json.dump(test_data, f)

print("\n✓ Saved datasets to finetune_*.json")

print("\n" + "=" * 70)
print("NEXT STEPS FOR ACTUAL FINE-TUNING:")
print("1. Load data into HuggingFace datasets")
print("2. Fine-tune BERT with object-masking objective")
print("3. Test positional ranks before/after")
print("4. Measure physics violation detection")
