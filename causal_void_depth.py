"""
How deep is the causal reasoning failure?
Testing increasingly simple causal structures
"""

from transformers import AutoModelForMaskedLM, AutoTokenizer
import torch

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = AutoModelForMaskedLM.from_pretrained('bert-base-uncased')
model.eval()

print("=" * 70)
print("MEASURING THE CAUSAL VOID")
print("=" * 70)

# Increasingly simple causal structures
causal_tests = {
    'ULTRA_SIMPLE': [
        ("Rain makes things [MASK]", "wet"),
        ("Fire makes things [MASK]", "hot"),
        ("Ice is [MASK]", "cold"),
    ],
    'TEMPORAL_SEQUENCE': [
        ("First A, then [MASK]", "B"),
        ("Before sunrise comes [MASK]", "dawn"),
        ("After winter comes [MASK]", "spring"),
    ],
    'CAUSAL_COMPLETION': [
        ("Cause: rain. Effect: [MASK]", "wet"),
        ("If hot, then [MASK]", "melt"),
        ("Because tired, therefore [MASK]", "sleep"),
    ],
    'BIDIRECTIONAL_TEST': [
        ("A causes B and B causes [MASK]", "A"),
        ("The loop continues: X leads to [MASK]", "X"),
        ("Chicken, egg, chicken, [MASK]", "egg"),
    ]
}

def test_completion(sentence, expected):
    """Test if model can complete basic causal sentence"""
    inputs = tokenizer(sentence, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**inputs)
    
    mask_idx = (inputs['input_ids'] == tokenizer.mask_token_id).nonzero()[0,1]
    probs = torch.softmax(outputs.logits[0, mask_idx], dim=-1)
    
    # Get top prediction
    top_id = probs.argmax().item()
    top_word = tokenizer.decode([top_id])
    confidence = probs[top_id].item()
    
    # Check if expected word is in top 10
    top10 = torch.topk(probs, 10)
    top10_words = [tokenizer.decode([idx]) for idx in top10.indices]
    
    expected_rank = None
    if expected in top10_words:
        expected_rank = top10_words.index(expected)
    
    return {
        'top_pred': top_word,
        'confidence': confidence,
        'expected_rank': expected_rank,
        'is_punctuation': top_word in '.,;!?',
        'top10': top10_words
    }

results_summary = {
    'total_tests': 0,
    'punctuation_defaults': 0,
    'correct_predictions': 0,
    'expected_in_top10': 0
}

for test_type, tests in causal_tests.items():
    print(f"\n{test_type}:")
    print("-" * 40)
    
    for sentence, expected in tests:
        result = test_completion(sentence, expected)
        results_summary['total_tests'] += 1
        
        if result['is_punctuation']:
            results_summary['punctuation_defaults'] += 1
        if result['top_pred'] == expected:
            results_summary['correct_predictions'] += 1
        if result['expected_rank'] is not None:
            results_summary['expected_in_top10'] += 1
        
        print(f"'{sentence}'")
        print(f"  Expected: '{expected}'")
        print(f"  Got: '{result['top_pred']}' (conf: {result['confidence']:.3f})")
        
        if result['expected_rank'] is not None:
            print(f"  ✓ Expected word at rank {result['expected_rank']}")
        else:
            print(f"  ✗ Expected word not in top 10")
        
        if result['is_punctuation']:
            print(f"  ⚠️ DEFAULTED TO PUNCTUATION")

print("\n" + "=" * 70)
print("SUMMARY:")
print(f"Punctuation defaults: {results_summary['punctuation_defaults']}/{results_summary['total_tests']} ({results_summary['punctuation_defaults']/results_summary['total_tests']*100:.1f}%)")
print(f"Correct predictions: {results_summary['correct_predictions']}/{results_summary['total_tests']} ({results_summary['correct_predictions']/results_summary['total_tests']*100:.1f}%)")
print(f"Expected in top 10: {results_summary['expected_in_top10']}/{results_summary['total_tests']} ({results_summary['expected_in_top10']/results_summary['total_tests']*100:.1f}%)")

if results_summary['punctuation_defaults'] > results_summary['total_tests'] * 0.5:
    print("\n⚠️ MODEL HAS NO CAUSAL REASONING - JUST SYNTACTIC PUNCTUATION PATTERNS")
