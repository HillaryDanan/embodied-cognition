"""
Statistical test: Do models that fail at temporal reasoning also fail at causal reasoning?
Based on Pearl (2009) - Causality: Models, Reasoning, and Inference
"""

from transformers import AutoModelForMaskedLM, AutoTokenizer
import torch
import numpy as np
from scipy import stats
import json

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = AutoModelForMaskedLM.from_pretrained('bert-base-uncased')
model.eval()

print("=" * 70)
print("QUANTITATIVE TEMPORAL-CAUSAL CORRELATION TEST")
print("=" * 70)

# Temporal understanding tests (based on Reichenbach, 1947 - Elements of Symbolic Logic)
temporal_tests = [
    # (sentence, position_of_mask, expected_word, temporal_type)
    ("Yesterday comes before [MASK]", 3, "today", "sequence"),
    ("Spring follows [MASK]", 2, "winter", "sequence"),
    ("Past, present, [MASK]", 3, "future", "sequence"),
    ("Monday, Tuesday, [MASK]", 3, "wednesday", "sequence"),
    ("First, second, [MASK]", 3, "third", "ordinal"),
    ("Morning precedes [MASK]", 2, "afternoon", "precedence"),
    ("Birth comes before [MASK]", 3, "death", "lifecycle"),
    ("Dawn happens before [MASK]", 3, "dusk", "daily"),
    ("January precedes [MASK]", 2, "february", "calendar"),
    ("Alpha, beta, [MASK]", 3, "gamma", "sequence")
]

# Causal understanding tests (based on Hume, 1748 - Enquiry Concerning Human Understanding)
causal_tests = [
    ("Heat causes ice to [MASK]", 5, "melt", "physical"),
    ("Rain makes ground [MASK]", 3, "wet", "physical"),
    ("Gravity pulls objects [MASK]", 3, "down", "physical"),
    ("Seeds grow into [MASK]", 3, "plants", "biological"),
    ("Study leads to [MASK]", 3, "knowledge", "cognitive"),
    ("Exercise improves [MASK]", 2, "health", "health"),
    ("Practice increases [MASK]", 2, "skill", "learning"),
    ("Smoking causes [MASK]", 2, "cancer", "medical"),
    ("Sleep reduces [MASK]", 2, "fatigue", "biological"),
    ("Force creates [MASK]", 2, "motion", "physics")
]

def test_completion(sentence, mask_pos, expected):
    """Test model's ability to complete sentence correctly"""
    words = sentence.split()
    
    inputs = tokenizer(sentence, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**inputs)
    
    mask_idx = (inputs['input_ids'] == tokenizer.mask_token_id).nonzero()[0,1]
    probs = torch.softmax(outputs.logits[0, mask_idx], dim=-1)
    
    # Get top prediction
    top_id = probs.argmax().item()
    top_word = tokenizer.decode([top_id])
    confidence = probs[top_id].item()
    
    # Check if it's punctuation
    is_punctuation = top_word in '.,;!?:'
    
    # Get rank of expected word
    try:
        expected_id = tokenizer.encode(expected, add_special_tokens=False)[0]
        expected_prob = probs[expected_id].item()
        all_probs_sorted = probs.argsort(descending=True)
        expected_rank = (all_probs_sorted == expected_id).nonzero().item()
    except:
        expected_rank = 999  # Not in vocabulary
        expected_prob = 0.0
    
    return {
        'correct': top_word.lower() == expected.lower(),
        'confidence': confidence,
        'is_punctuation': is_punctuation,
        'expected_rank': expected_rank,
        'expected_prob': expected_prob,
        'top_prediction': top_word
    }

# Test both types
temporal_results = []
causal_results = []

print("\nTEMPORAL TESTS:")
print("-" * 40)
for sentence, pos, expected, ttype in temporal_tests:
    result = test_completion(sentence, pos, expected)
    temporal_results.append(result)
    status = "✓" if result['correct'] else ("PUNCT" if result['is_punctuation'] else "✗")
    print(f"{status} '{sentence}' → '{result['top_prediction']}' (rank {result['expected_rank']})")

print("\nCAUSAL TESTS:")
print("-" * 40)
for sentence, pos, expected, ctype in causal_tests:
    result = test_completion(sentence, pos, expected)
    causal_results.append(result)
    status = "✓" if result['correct'] else ("PUNCT" if result['is_punctuation'] else "✗")
    print(f"{status} '{sentence}' → '{result['top_prediction']}' (rank {result['expected_rank']})")

# Statistical analysis
print("\n" + "=" * 70)
print("STATISTICAL ANALYSIS")
print("-" * 40)

# Calculate scores
temporal_score = sum(1 for r in temporal_results if r['correct']) / len(temporal_results)
temporal_punct = sum(1 for r in temporal_results if r['is_punctuation']) / len(temporal_results)
temporal_avg_rank = np.mean([r['expected_rank'] for r in temporal_results])

causal_score = sum(1 for r in causal_results if r['correct']) / len(causal_results)
causal_punct = sum(1 for r in causal_results if r['is_punctuation']) / len(causal_results)
causal_avg_rank = np.mean([r['expected_rank'] for r in causal_results])

print(f"\nTEMPORAL UNDERSTANDING:")
print(f"  Accuracy: {temporal_score:.1%}")
print(f"  Punctuation defaults: {temporal_punct:.1%}")
print(f"  Avg expected word rank: {temporal_avg_rank:.1f}")

print(f"\nCAUSAL UNDERSTANDING:")
print(f"  Accuracy: {causal_score:.1%}")
print(f"  Punctuation defaults: {causal_punct:.1%}")
print(f"  Avg expected word rank: {causal_avg_rank:.1f}")

# Test correlation between temporal and causal performance
temporal_binary = [1 if r['correct'] else 0 for r in temporal_results]
causal_binary = [1 if r['correct'] else 0 for r in causal_results]

# Chi-square test for independence
contingency = np.array([
    [sum(temporal_binary), len(temporal_binary) - sum(temporal_binary)],
    [sum(causal_binary), len(causal_binary) - sum(causal_binary)]
])
chi2, p_value, dof, expected = stats.chi2_contingency(contingency)

print(f"\nCORRELATION TEST:")
print(f"  Chi-square: {chi2:.3f}")
print(f"  p-value: {p_value:.4f}")
print(f"  Correlation: {'Significant' if p_value < 0.05 else 'Not significant'}")

# Save results for cross-model comparison
results = {
    'model': 'bert-base-uncased',
    'temporal_accuracy': temporal_score,
    'causal_accuracy': causal_score,
    'temporal_punct_rate': temporal_punct,
    'causal_punct_rate': causal_punct,
    'correlation_p': p_value
}

with open('temporal_causal_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\n" + "=" * 70)
print("HYPOTHESIS TEST:")
if temporal_punct > 0.5 and causal_punct > 0.5:
    print("✓ Model fails at BOTH temporal AND causal reasoning")
    print("  Supporting embodied time hypothesis")
elif temporal_score < 0.3 and causal_score < 0.3:
    print("✓ Low performance on both dimensions")
    print("  Suggests linked cognitive capacity")
else:
    print("✗ Mixed results - needs further investigation")
