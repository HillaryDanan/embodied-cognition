"""
Does word frequency predict positional encoding success?
Using Zipf frequency as proxy for corpus frequency
"""

from transformers import AutoModelForMaskedLM, AutoTokenizer
import torch
import numpy as np
from scipy import stats

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = AutoModelForMaskedLM.from_pretrained('bert-base-uncased')
model.eval()

print("=" * 70)
print("LEXICAL FREQUENCY VS POSITIONAL ENCODING")
print("=" * 70)

# Test words of different frequencies
test_sets = {
    'HIGH_FREQ_OBJECTS': ['man', 'time', 'day', 'thing', 'world'],
    'LOW_FREQ_OBJECTS': ['saxophone', 'molecule', 'asteroid', 'paradox', 'genome'],
    'HIGH_FREQ_VERBS': ['is', 'have', 'do', 'make', 'go'],
    'LOW_FREQ_VERBS': ['hypothesize', 'crystallize', 'metabolize', 'polarize', 'oxidize']
}

def test_word_prediction(word, context_template):
    """Test how well a word is predicted in context"""
    sentence = context_template.replace('[WORD]', word)
    masked = sentence.replace(word, '[MASK]')
    
    inputs = tokenizer(masked, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**inputs)
    
    mask_idx = (inputs['input_ids'] == tokenizer.mask_token_id).nonzero()[0,1]
    target_id = tokenizer.encode(word, add_special_tokens=False)[0]
    
    probs = torch.softmax(outputs.logits[0, mask_idx], dim=-1)
    rank = (probs.argsort(descending=True) == target_id).nonzero().item()
    
    # Use token ID as frequency proxy (lower ID = higher frequency generally)
    frequency_proxy = 30000 - target_id
    
    return rank, frequency_proxy

print("\nTesting different frequency words:\n")

results = {}
for category, words in test_sets.items():
    print(f"{category}:")
    ranks = []
    frequencies = []
    
    # Choose appropriate context
    if 'OBJECTS' in category:
        context = "The [WORD] was there"
    else:  # VERBS
        context = "They [WORD] quickly"
    
    for word in words:
        rank, freq = test_word_prediction(word, context)
        ranks.append(rank)
        frequencies.append(freq)
        print(f"  {word:15s}: rank {rank:4d}, freq_proxy {freq:5d}")
    
    results[category] = {'ranks': ranks, 'frequencies': frequencies}
    
    # Calculate correlation
    if len(ranks) > 1:
        correlation, p_value = stats.pearsonr(frequencies, ranks)
        print(f"  Correlation(freq,rank): r={correlation:.3f}, p={p_value:.3f}")

print("\n" + "=" * 70)
print("HYPOTHESIS TEST: Higher frequency = Better prediction (negative correlation)")

# Aggregate analysis
all_ranks = []
all_freqs = []
for data in results.values():
    all_ranks.extend(data['ranks'])
    all_freqs.extend(data['frequencies'])

overall_corr, overall_p = stats.pearsonr(all_freqs, all_ranks)
print(f"Overall correlation: r={overall_corr:.3f}, p={overall_p:.3f}")

if overall_corr < -0.3:
    print("✓ Frequency strongly affects prediction")
elif overall_corr < 0:
    print("✓ Frequency weakly affects prediction")
else:
    print("✗ No frequency effect or reverse effect")
