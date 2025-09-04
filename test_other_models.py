"""
Test if chaos is BERT-specific or universal
Using models we can run locally on Mac
"""

from transformers import AutoModelForMaskedLM, AutoTokenizer
import torch

MODELS_TO_TEST = [
    'bert-base-uncased',  # Our baseline
    'roberta-base',       # Different training
    'albert-base-v2',     # Different architecture
    'distilbert-base-uncased'  # Distilled
]

print("=" * 70)
print("CROSS-MODEL PHYSICS CHAOS TEST")
print("=" * 70)

# Our diagnostic test cases
test_cases = [
    ("The rock fell [MASK] to the ground", "down", "up"),
    ("The water flowed [MASK] the hill", "down", "up"),
    ("The ball rolled [MASK] the screen", "behind", "through"),
]

for model_name in MODELS_TO_TEST:
    print(f"\n{model_name.upper()}")
    print("-" * 40)
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForMaskedLM.from_pretrained(model_name)
        model.eval()
        
        for sentence, normal_word, violation_word in test_cases:
            inputs = tokenizer(sentence, return_tensors='pt')
            
            with torch.no_grad():
                outputs = model(**inputs)
            
            # Get mask position
            mask_id = tokenizer.mask_token_id if hasattr(tokenizer, 'mask_token_id') else tokenizer.convert_tokens_to_ids(['[MASK]'])[0]
            mask_idx = (inputs['input_ids'] == mask_id).nonzero()[0,1]
            
            # Get probabilities for both words
            probs = torch.softmax(outputs.logits[0, mask_idx], dim=-1)
            
            normal_id = tokenizer.encode(normal_word, add_special_tokens=False)[0]
            violation_id = tokenizer.encode(violation_word, add_special_tokens=False)[0]
            
            normal_prob = probs[normal_id].item()
            violation_prob = probs[violation_id].item()
            
            print(f"  '{normal_word}' vs '{violation_word}': {normal_prob:.3f} vs {violation_prob:.3f}")
            
    except Exception as e:
        print(f"  Failed to load: {e}")

print("\n" + "=" * 70)
print("If all models show chaos -> fundamental to current architectures")
print("If only BERT -> training-specific artifact")
