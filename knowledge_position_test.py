"""
Is positional encoding unique to physics or universal?
Testing social, mathematical, and causal knowledge
"""

from transformers import AutoModelForMaskedLM, AutoTokenizer
import torch

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = AutoModelForMaskedLM.from_pretrained('bert-base-uncased')
model.eval()

def test_position(sentence, word_type_labels):
    """Test which positions are well-predicted"""
    words = sentence.split()
    results = {}
    
    for i, (word, label) in enumerate(zip(words, word_type_labels)):
        masked = ' '.join([w if j != i else '[MASK]' for j, w in enumerate(words)])
        
        inputs = tokenizer(masked, return_tensors='pt')
        with torch.no_grad():
            outputs = model(**inputs)
        
        mask_idx = (inputs['input_ids'] == tokenizer.mask_token_id).nonzero()[0,1]
        target_id = tokenizer.encode(word, add_special_tokens=False)[0]
        
        probs = torch.softmax(outputs.logits[0, mask_idx], dim=-1)
        rank = (probs.argsort(descending=True) == target_id).nonzero().item()
        
        results[label] = results.get(label, []) + [rank]
        print(f"  {word:15s} ({label:10s}): rank {rank}")
    
    return results

print("=" * 70)
print("TESTING POSITIONAL ENCODING ACROSS KNOWLEDGE TYPES")
print("=" * 70)

# Test different knowledge domains
tests = {
    'PHYSICS': (
        "The ball rolled down the hill",
        ['det', 'object', 'verb', 'direction', 'det', 'location']
    ),
    'SOCIAL': (
        "John helped Mary with her homework",
        ['agent', 'verb', 'patient', 'prep', 'det', 'object']
    ),
    'MATHEMATICAL': (
        "Five plus three equals eight exactly",
        ['number', 'operator', 'number', 'verb', 'number', 'modifier']
    ),
    'CAUSAL': (
        "Rain caused flooding in the valley",
        ['cause', 'verb', 'effect', 'prep', 'det', 'location']
    )
}

all_results = {}
for domain, (sentence, labels) in tests.items():
    print(f"\n{domain}:")
    print(f"Sentence: {sentence}")
    results = test_position(sentence, labels)
    all_results[domain] = results

print("\n" + "=" * 70)
print("SUMMARY BY WORD TYPE")
print("-" * 70)

# Aggregate by word type
word_types = {}
for domain_results in all_results.values():
    for word_type, ranks in domain_results.items():
        if word_type not in word_types:
            word_types[word_type] = []
        word_types[word_type].extend(ranks)

for word_type, ranks in sorted(word_types.items()):
    mean_rank = sum(ranks) / len(ranks) if ranks else 0
    print(f"{word_type:12s}: mean rank = {mean_rank:6.1f} (n={len(ranks)})")

print("\n" + "=" * 70)
print("HYPOTHESIS: Verbs and operators are privileged across ALL domains")
