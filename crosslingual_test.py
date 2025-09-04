"""
Test positional encoding in multilingual models
Different languages have different word orders!
"""

from transformers import AutoModelForMaskedLM, AutoTokenizer
import torch

print("=" * 70)
print("CROSS-LINGUISTIC POSITIONAL ENCODING TEST")
print("=" * 70)

# We can test with mBERT (multilingual BERT)
model_name = 'bert-base-multilingual-cased'

try:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name)
    model.eval()
    print(f"✓ Loaded {model_name}")
except:
    print(f"✗ Need to download {model_name} first")
    print("This would test patterns across languages with different word orders")
    
    # Show what we WOULD test
    test_plan = """
    PLANNED TESTS:
    
    English (SVO): "The cat [MASK] the mouse"
    Spanish (SVO): "El gato [MASK] el ratón"
    Japanese (SOV): "猫が ネズミを [MASK]"
    Arabic (VSO): "[MASK] القط الفأر"
    
    Hypothesis: Function words still > content words
    regardless of word order
    """
    print(test_plan)
    exit()

# Test sentences in different languages
test_sentences = {
    'English': {
        'sentence': 'The ball rolled [MASK] the hill',
        'expected': 'down',
        'type': 'preposition'
    },
    'Spanish': {
        'sentence': 'La pelota rodó [MASK] la colina',
        'expected': 'por',
        'type': 'preposition'
    },
    'French': {
        'sentence': 'Le ballon a roulé [MASK] la colline',
        'expected': 'sur',
        'type': 'preposition'
    },
    'German': {
        'sentence': 'Der Ball rollte [MASK] den Hügel',
        'expected': 'über',
        'type': 'preposition'
    }
}

print("\nTesting function words across languages:")
print("-" * 40)

for lang, test_data in test_sentences.items():
    sentence = test_data['sentence']
    
    inputs = tokenizer(sentence, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**inputs)
    
    mask_idx = (inputs['input_ids'] == tokenizer.mask_token_id).nonzero()[0,1]
    probs = torch.softmax(outputs.logits[0, mask_idx], dim=-1)
    
    top5 = torch.topk(probs, 5)
    predictions = [tokenizer.decode([idx]) for idx in top5.indices]
    
    print(f"\n{lang}:")
    print(f"  Sentence: {sentence}")
    print(f"  Top predictions: {', '.join(predictions[:3])}")
    print(f"  Confidence: {top5.values[0]:.3f}")

print("\n" + "=" * 70)
print("If function words dominate across languages,")
print("it suggests UNIVERSAL architectural bias")
