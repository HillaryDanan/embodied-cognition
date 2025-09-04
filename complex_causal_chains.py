"""
Test if models can handle multi-step causal reasoning
Based on Pearl (2009) causal graphs
"""

from transformers import AutoModelForMaskedLM, AutoTokenizer
import torch

models = ['bert-base-uncased', 'roberta-base', 'gpt2']

complex_chains = [
    # (test, expected, chain_length)
    ("A causes B, B causes C, therefore A causes [MASK]", "c", 3),
    ("Rain causes wet, wet causes slippery, therefore rain causes [MASK]", "slippery", 3),
    ("Study leads to knowledge, knowledge leads to success, therefore study leads to [MASK]", "success", 3),
    ("If P then Q, if Q then R, therefore if P then [MASK]", "r", 3),
]

print("=" * 70)
print("COMPLEX CAUSAL CHAIN TEST")
print("=" * 70)

for model_name in models:
    if model_name == 'gpt2':
        print(f"\n{model_name}: [Generation model - skipping MLM test]")
        continue
        
    print(f"\n{model_name}:")
    print("-" * 40)
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForMaskedLM.from_pretrained(model_name)
        model.eval()
        
        correct = 0
        for test, expected, chain_len in complex_chains:
            # Adjust mask token
            mask_token = '<mask>' if 'roberta' in model_name else '[MASK]'
            test = test.replace('[MASK]', mask_token)
            
            inputs = tokenizer(test, return_tensors='pt')
            with torch.no_grad():
                outputs = model(**inputs)
            
            mask_idx = (inputs['input_ids'] == tokenizer.mask_token_id).nonzero()[0,1]
            probs = torch.softmax(outputs.logits[0, mask_idx], dim=-1)
            top_word = tokenizer.decode([probs.argmax().item()]).strip()
            
            is_correct = top_word.lower() == expected.lower()
            if is_correct:
                correct += 1
            
            symbol = "✓" if is_correct else "✗"
            print(f"  {symbol} Chain-{chain_len}: got '{top_word}' (expected '{expected}')")
        
        print(f"  Score: {correct}/{len(complex_chains)}")
        
    except Exception as e:
        print(f"  Error: {e}")

print("\n" + "=" * 70)
print("If models fail at multi-step chains, they lack")
print("compositional causal reasoning, not just simple causation")
