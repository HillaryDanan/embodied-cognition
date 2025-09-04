"""
Can we hack prompts to make object prediction work?
Testing various strategies to privilege object words
"""

from transformers import AutoModelForMaskedLM, AutoTokenizer
import torch
import numpy as np

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = AutoModelForMaskedLM.from_pretrained('bert-base-uncased')
model.eval()

print("=" * 70)
print("HUNTING FOR OBJECT-PRIVILEGED PROMPT PATTERNS")
print("=" * 70)

# Different strategies to make objects predictable
strategies = {
    'BASELINE': {
        'template': "The [MASK] fell down",
        'target': 'rock',
        'description': 'Standard object position (fails)'
    },
    'EMPHASIS': {
        'template': "The heavy [MASK] fell",
        'target': 'rock',
        'description': 'Adjective before object'
    },
    'LISTING': {
        'template': "Things that fall: apple, ball, [MASK]",
        'target': 'rock',
        'description': 'List context'
    },
    'DEFINITION': {
        'template': "A [MASK] is a hard mineral object",
        'target': 'rock',
        'description': 'Definition frame'
    },
    'CONTRAST': {
        'template': "Not paper, not feather, but [MASK]",
        'target': 'rock',
        'description': 'Contrastive context'
    },
    'CATEGORY': {
        'template': "Types of stones: pebble, boulder, [MASK]",
        'target': 'rock',
        'description': 'Category membership'
    },
    'COMPOUND': {
        'template': "rock-solid, rock-hard, [MASK]-heavy",
        'target': 'rock',
        'description': 'Compound word pattern'
    }
}

results = []

for strategy_name, strategy in strategies.items():
    inputs = tokenizer(strategy['template'], return_tensors='pt')
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    mask_idx = (inputs['input_ids'] == tokenizer.mask_token_id).nonzero()[0,1]
    probs = torch.softmax(outputs.logits[0, mask_idx], dim=-1)
    
    # Check if target word is predicted
    target_id = tokenizer.encode(strategy['target'], add_special_tokens=False)[0]
    target_prob = probs[target_id].item()
    target_rank = (probs.argsort(descending=True) == target_id).nonzero().item()
    
    # Get top prediction
    top_pred_id = probs.argmax().item()
    top_pred_word = tokenizer.decode([top_pred_id])
    
    results.append({
        'strategy': strategy_name,
        'description': strategy['description'],
        'target_rank': target_rank,
        'target_prob': target_prob,
        'top_pred': top_pred_word
    })
    
    print(f"\n{strategy_name}: {strategy['description']}")
    print(f"  Template: '{strategy['template']}'")
    print(f"  Target '{strategy['target']}' rank: {target_rank}")
    print(f"  Target probability: {target_prob:.4f}")
    print(f"  Top prediction: '{top_pred_word}'")

# Find best strategy
best = min(results, key=lambda x: x['target_rank'])
print("\n" + "=" * 70)
print(f"BEST STRATEGY: {best['strategy']}")
print(f"Achieved rank {best['target_rank']} (baseline was ~32)")

if best['target_rank'] < 10:
    print("✓ Found object-privileged prompt pattern!")
else:
    print("✗ Objects remain hard to predict even with prompt engineering")
