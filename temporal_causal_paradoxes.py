"""
Testing if LLMs understand temporal-causal relationships
or just pattern match "if...then" syntax

IF P, THEN Q implies P happens BEFORE Q
But what if Q causes P? What breaks?
"""

from transformers import AutoModelForMaskedLM, AutoTokenizer
import torch

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = AutoModelForMaskedLM.from_pretrained('bert-base-uncased')
model.eval()

print("=" * 70)
print("TEMPORAL-CAUSAL PARADOX EXPERIMENTS")
print("=" * 70)

# Test different temporal-causal structures
paradoxes = {
    'NORMAL_CAUSATION': [
        "Because it rained, the ground got [MASK]",  # wet (normal)
        "If you drop it, it will [MASK]",  # fall (normal)
        "After eating, she felt [MASK]",  # full (normal)
    ],
    
    'REVERSE_CAUSATION': [
        "Because the ground got wet, it [MASK]",  # rained? (reversed)
        "If it fell, you must have [MASK] it",  # dropped (backward inference)
        "After feeling full, she [MASK]",  # ate? (effect before cause)
    ],
    
    'CIRCULAR_CAUSATION': [
        "It happened because it [MASK]",  # happened? (circular)
        "The cause of X is [MASK]",  # X? (self-causing)
        "A leads to B which leads to [MASK]",  # A? (causal loop)
    ],
    
    'TEMPORAL_PARADOX': [
        "Before it started, it had already [MASK]",  # ended? (temporal impossibility)
        "Tomorrow's event caused yesterday's [MASK]",  # ??? (backwards time)
        "The future [MASK] the past",  # determines? (retrocausation)
    ]
}

print("\nTesting how models handle temporal-causal paradoxes:\n")

for paradox_type, sentences in paradoxes.items():
    print(f"\n{paradox_type}:")
    print("-" * 40)
    
    for sentence in sentences:
        inputs = tokenizer(sentence, return_tensors='pt')
        with torch.no_grad():
            outputs = model(**inputs)
        
        mask_idx = (inputs['input_ids'] == tokenizer.mask_token_id).nonzero()[0,1]
        probs = torch.softmax(outputs.logits[0, mask_idx], dim=-1)
        
        # Get top 3 predictions
        top3 = torch.topk(probs, 3)
        predictions = [tokenizer.decode([idx]) for idx in top3.indices]
        confidence = top3.values[0].item()
        
        print(f"'{sentence}'")
        print(f"  Predictions: {predictions}")
        print(f"  Confidence: {confidence:.3f}")
        
        # Check if model shows confusion (low confidence or weird predictions)
        if confidence < 0.1:
            print("  ⚠️ Model confused (low confidence)")
        if '.' in predictions or ',' in predictions:
            print("  ⚠️ Model defaulting to punctuation (no semantic understanding)")

print("\n" + "=" * 70)
print("HYPOTHESIS: Models will handle normal causation fine")
print("but fail catastrophically on paradoxes, revealing")
print("they don't actually understand temporal-causal relations")
