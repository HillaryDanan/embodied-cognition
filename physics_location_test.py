"""
Where does physics knowledge live in sentences?
Testing different word positions
"""

from transformers import AutoModelForMaskedLM, AutoTokenizer
import torch
import numpy as np

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = AutoModelForMaskedLM.from_pretrained('bert-base-uncased')
model.eval()

print("=" * 70)
print("WHERE DOES PHYSICS KNOWLEDGE LIVE?")
print("=" * 70)

sentence = "The heavy rock naturally fell down toward the ground"

# Test each word position
words = sentence.split()
for i, word in enumerate(words):
    masked = ' '.join([w if j != i else '[MASK]' for j, w in enumerate(words)])
    
    inputs = tokenizer(masked, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**inputs)
    
    mask_idx = (inputs['input_ids'] == tokenizer.mask_token_id).nonzero()[0,1]
    target_id = tokenizer.encode(word, add_special_tokens=False)[0]
    
    probs = torch.softmax(outputs.logits[0, mask_idx], dim=-1)
    
    # Get top 3 predictions
    top3 = torch.topk(probs, 3)
    top_words = [tokenizer.decode([idx]) for idx in top3.indices]
    
    print(f"\nPosition {i}: '{word}'")
    print(f"  Top predictions: {', '.join(top_words)}")
    print(f"  Original word rank: {(probs.argsort(descending=True) == target_id).nonzero().item()}")
