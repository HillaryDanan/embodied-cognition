"""
If LLMs are syntax machines, they should:
1. Preserve syntactic structure even with nonsense
2. Fail at semantic tasks requiring object understanding
"""

from transformers import AutoModelForMaskedLM, AutoTokenizer
import torch

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = AutoModelForMaskedLM.from_pretrained('bert-base-uncased')
model.eval()

print("=" * 70)
print("TESTING: SYNTAX PRESERVATION VS SEMANTIC UNDERSTANDING")
print("=" * 70)

tests = {
    "NORMAL": "The ball rolled [MASK] the hill",
    "NONSENSE_OBJECTS": "The gribble rolled [MASK] the florp",
    "NONSENSE_PREDICATES": "The ball xyzqed [MASK] the hill",
}

for test_type, sentence in tests.items():
    inputs = tokenizer(sentence, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**inputs)
    
    mask_idx = (inputs['input_ids'] == tokenizer.mask_token_id).nonzero()[0,1]
    probs = torch.softmax(outputs.logits[0, mask_idx], dim=-1)
    
    top5 = torch.topk(probs, 5)
    predictions = [tokenizer.decode([idx]) for idx in top5.indices]
    
    print(f"\n{test_type}:")
    print(f"  Sentence: {sentence}")
    print(f"  Predictions: {', '.join(predictions)}")
    print(f"  Top confidence: {top5.values[0]:.3f}")

print("\n" + "=" * 70)
print("HYPOTHESIS: Syntax preserved even with nonsense objects")
print("but broken with nonsense predicates")
