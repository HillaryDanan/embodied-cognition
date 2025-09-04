"""
Test physics chaos across your LOCAL models
You have: BERT, RoBERTa, ALBERT, DistilBERT
"""

from transformers import AutoModelForMaskedLM, AutoTokenizer
import torch
import numpy as np

MODELS_TO_TEST = [
    'bert-base-uncased',
    'roberta-base',
    'albert-base-v2',
    'distilbert-base-uncased'
]

print("=" * 70)
print("TESTING PHYSICS CHAOS ACROSS LOCAL MODELS")
print("=" * 70)

# Our key diagnostic tests
physics_tests = {
    'gravity': {
        'template': "The rock fell [MASK] to the ground",
        'normal': 'down',
        'violation': 'up'
    },
    'permanence': {
        'template': "The ball rolled behind the wall and [MASK] there",
        'normal': 'stayed',
        'violation': 'vanished'
    },
    'support': {
        'template': "The book [MASK] on the table",
        'normal': 'rested',
        'violation': 'floated'
    }
}

all_results = {}

for model_name in MODELS_TO_TEST:
    print(f"\n{'='*50}")
    print(f"MODEL: {model_name}")
    print('='*50)
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name)
    model.eval()
    
    model_results = {}
    
    for test_name, test in physics_tests.items():
        # Handle different mask tokens
        if 'roberta' in model_name:
            mask_token = '<mask>'
        else:
            mask_token = '[MASK]'
        
        sentence = test['template'].replace('[MASK]', mask_token)
        
        inputs = tokenizer(sentence, return_tensors='pt')
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Get mask position
        if 'roberta' in model_name:
            mask_id = tokenizer.mask_token_id
        else:
            mask_id = tokenizer.convert_tokens_to_ids([mask_token])[0]
            
        mask_idx = (inputs['input_ids'] == mask_id).nonzero()[0,1]
        probs = torch.softmax(outputs.logits[0, mask_idx], dim=-1)
        
        # Get probabilities for normal vs violation
        try:
            normal_id = tokenizer.encode(test['normal'], add_special_tokens=False)[0]
            violation_id = tokenizer.encode(test['violation'], add_special_tokens=False)[0]
            
            normal_prob = probs[normal_id].item()
            violation_prob = probs[violation_id].item()
            
            # Calculate surprise difference
            normal_surprise = -np.log(normal_prob + 1e-10)
            violation_surprise = -np.log(violation_prob + 1e-10)
            difference = violation_surprise - normal_surprise
            
            print(f"\n{test_name.upper()}:")
            print(f"  '{test['normal']}': p={normal_prob:.4f}, surprise={normal_surprise:.3f}")
            print(f"  '{test['violation']}': p={violation_prob:.4f}, surprise={violation_surprise:.3f}")
            print(f"  Difference: {difference:.3f} ({'✓ detects' if difference > 0 else '✗ misses'} violation)")
            
            model_results[test_name] = difference
            
        except Exception as e:
            print(f"  Error with {test_name}: {e}")
    
    all_results[model_name] = model_results

# Summary statistics
print("\n" + "=" * 70)
print("CROSS-MODEL COMPARISON")
print("=" * 70)

for model_name, results in all_results.items():
    values = list(results.values())
    mean = np.mean(values) if values else 0
    std = np.std(values) if values else 0
    detected = sum(1 for v in values if v > 0)
    
    print(f"\n{model_name}:")
    print(f"  Detection rate: {detected}/{len(values)}")
    print(f"  Mean difference: {mean:.3f}")
    print(f"  Std deviation: {std:.3f}")

print("\n" + "=" * 70)
print("CONCLUSION:")
if all(np.std(list(r.values())) > 0.5 for r in all_results.values() if r):
    print("✓ ALL models show physics chaos - it's UNIVERSAL!")
else:
    print("Models vary in consistency - architecture matters")
