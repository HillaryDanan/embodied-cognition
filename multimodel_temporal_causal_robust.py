"""
Test temporal-causal correlation across multiple architectures
Following methodology from Rogers et al. (2020) - A Primer on Neural Network Architectures for NLP
ROBUST VERSION: Better test cases and error handling
"""

from transformers import AutoModelForMaskedLM, AutoTokenizer
import torch
import numpy as np
from scipy import stats
import json

models_to_test = [
    'bert-base-uncased',
    'roberta-base',
    'albert-base-v2',
    'distilbert-base-uncased'
]

# More comprehensive test set
test_pairs = [
    # (temporal_test, temporal_answer, causal_test, causal_answer)
    ("Yesterday comes before [MASK]", "today", "Heat melts [MASK]", "ice"),
    ("Spring follows [MASK]", "winter", "Rain makes things [MASK]", "wet"),
    ("Past, present, [MASK]", "future", "Gravity pulls objects [MASK]", "down"),
    ("Monday, Tuesday, [MASK]", "wednesday", "Fire causes [MASK]", "smoke"),
    ("First, second, [MASK]", "third", "Seeds grow into [MASK]", "plants"),
]

results_by_model = {}

print("=" * 70)
print("MULTI-MODEL TEMPORAL-CAUSAL ANALYSIS")
print("=" * 70)
print(f"Testing {len(test_pairs)} paired temporal-causal tests")
print(f"Models: {', '.join(models_to_test)}")
print("=" * 70)

for model_name in models_to_test:
    print(f"\nTesting {model_name}...")
    print("-" * 40)
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForMaskedLM.from_pretrained(model_name)
        model.eval()
        
        temporal_correct = 0
        causal_correct = 0
        temporal_punct = 0
        causal_punct = 0
        temporal_ranks = []
        causal_ranks = []
        
        for temp_test, temp_ans, caus_test, caus_ans in test_pairs:
            # Adjust mask token for different models
            mask_token = '<mask>' if 'roberta' in model_name else '[MASK]'
            temp_test = temp_test.replace('[MASK]', mask_token)
            caus_test = caus_test.replace('[MASK]', mask_token)
            
            # Test temporal
            inputs = tokenizer(temp_test, return_tensors='pt')
            with torch.no_grad():
                outputs = model(**inputs)
            
            mask_idx = (inputs['input_ids'] == tokenizer.mask_token_id).nonzero()[0,1]
            probs = torch.softmax(outputs.logits[0, mask_idx], dim=-1)
            top_id = probs.argmax().item()
            top_word = tokenizer.decode([top_id]).strip()
            
            # Get rank of expected word
            try:
                temp_ans_id = tokenizer.encode(temp_ans, add_special_tokens=False)[0]
                temp_rank = (probs.argsort(descending=True) == temp_ans_id).nonzero().item()
                temporal_ranks.append(temp_rank)
            except:
                temporal_ranks.append(999)
            
            if top_word.lower() == temp_ans.lower():
                temporal_correct += 1
                print(f"  ✓ Temporal: '{temp_test}' → '{top_word}'")
            elif top_word in '.,;!?:':
                temporal_punct += 1
                print(f"  PUNCT Temporal: '{temp_test}' → '{top_word}'")
            else:
                print(f"  ✗ Temporal: '{temp_test}' → '{top_word}' (expected '{temp_ans}')")
            
            # Test causal
            inputs = tokenizer(caus_test, return_tensors='pt')
            with torch.no_grad():
                outputs = model(**inputs)
            
            mask_idx = (inputs['input_ids'] == tokenizer.mask_token_id).nonzero()[0,1]
            probs = torch.softmax(outputs.logits[0, mask_idx], dim=-1)
            top_id = probs.argmax().item()
            top_word = tokenizer.decode([top_id]).strip()
            
            # Get rank of expected word
            try:
                caus_ans_id = tokenizer.encode(caus_ans, add_special_tokens=False)[0]
                caus_rank = (probs.argsort(descending=True) == caus_ans_id).nonzero().item()
                causal_ranks.append(caus_rank)
            except:
                causal_ranks.append(999)
            
            if top_word.lower() == caus_ans.lower():
                causal_correct += 1
                print(f"  ✓ Causal: '{caus_test}' → '{top_word}'")
            elif top_word in '.,;!?:':
                causal_punct += 1
                print(f"  PUNCT Causal: '{caus_test}' → '{top_word}'")
            else:
                print(f"  ✗ Causal: '{caus_test}' → '{top_word}' (expected '{caus_ans}')")
        
        results_by_model[model_name] = {
            'temporal_acc': temporal_correct / len(test_pairs),
            'causal_acc': causal_correct / len(test_pairs),
            'temporal_punct': temporal_punct / len(test_pairs),
            'causal_punct': causal_punct / len(test_pairs),
            'temporal_avg_rank': np.mean(temporal_ranks),
            'causal_avg_rank': np.mean(causal_ranks)
        }
        
    except Exception as e:
        print(f"  Error loading model: {e}")

# Statistical analysis
print("\n" + "=" * 70)
print("RESULTS SUMMARY")
print("-" * 40)

summary_table = []
for model_name, results in results_by_model.items():
    print(f"\n{model_name}:")
    print(f"  Temporal: {results['temporal_acc']:.0%} correct, {results['temporal_punct']:.0%} punctuation")
    print(f"    Average rank: {results['temporal_avg_rank']:.1f}")
    print(f"  Causal: {results['causal_acc']:.0%} correct, {results['causal_punct']:.0%} punctuation")
    print(f"    Average rank: {results['causal_avg_rank']:.1f}")
    
    summary_table.append([
        model_name,
        results['temporal_acc'],
        results['causal_acc'],
        results['temporal_punct'],
        results['causal_punct']
    ])

# Test correlation across models
print("\n" + "=" * 70)
print("CROSS-MODEL CORRELATION ANALYSIS")
print("-" * 40)

temporal_scores = [r['temporal_acc'] for r in results_by_model.values()]
causal_scores = [r['causal_acc'] for r in results_by_model.values()]
temporal_puncts = [r['temporal_punct'] for r in results_by_model.values()]
causal_puncts = [r['causal_punct'] for r in results_by_model.values()]

if len(temporal_scores) > 1 and np.std(temporal_scores) > 0 and np.std(causal_scores) > 0:
    correlation, p_value = stats.pearsonr(temporal_scores, causal_scores)
    print(f"Accuracy correlation (temporal vs causal):")
    print(f"  Pearson r = {correlation:.3f}, p = {p_value:.4f}")
else:
    print("Cannot compute correlation (no variance in scores)")

# Check if all models fail similarly
if all(t < 0.2 for t in temporal_scores) and all(c < 0.2 for c in causal_scores):
    print("\n✓ ALL models fail at both temporal AND causal reasoning")
    print("  Universal failure pattern detected")

if all(t > 0.7 for t in temporal_puncts) and all(c > 0.7 for c in causal_puncts):
    print("\n✓ ALL models default to punctuation for both types")
    print("  Systematic syntactic default behavior")

# Save comprehensive results
all_results = {
    'models': results_by_model,
    'summary': {
        'mean_temporal_acc': np.mean(temporal_scores),
        'mean_causal_acc': np.mean(causal_scores),
        'mean_temporal_punct': np.mean(temporal_puncts),
        'mean_causal_punct': np.mean(causal_puncts),
        'all_fail': all(t < 0.2 for t in temporal_scores) and all(c < 0.2 for c in causal_scores)
    }
}

with open('multimodel_temporal_causal_results.json', 'w') as f:
    json.dump(all_results, f, indent=2)

print("\n" + "=" * 70)
print("EMBODIED TIME HYPOTHESIS EVALUATION:")
print("-" * 40)
if all_results['summary']['all_fail']:
    print("✓ STRONG SUPPORT: Universal failure across architectures")
    print("  No model can handle temporal OR causal reasoning")
    print("  Suggests fundamental limitation, not architecture-specific")
else:
    print("✗ MIXED SUPPORT: Some models perform better")
    print("  Needs further investigation")

print("\nResults saved to multimodel_temporal_causal_results.json")
