"""
Where does the model 'look' when predicting different word types?
Analyzing attention patterns for function vs content words
"""

from transformers import AutoModel, AutoTokenizer
import torch
import numpy as np
import matplotlib.pyplot as plt

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = AutoModel.from_pretrained('bert-base-uncased', output_attentions=True)
model.eval()

print("=" * 70)
print("ATTENTION PATTERN ANALYSIS")
print("=" * 70)

def get_attention_patterns(sentence, target_idx):
    """Get attention patterns for a specific token"""
    inputs = tokenizer(sentence, return_tensors='pt')
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Average attention across all heads and layers
    attentions = outputs.attentions  # tuple of tensors (one per layer)
    avg_attention = torch.stack(attentions).mean(dim=[0, 1, 2])  # Average over layers, heads, batch
    
    # Get attention FROM the target position
    attention_from_target = avg_attention[target_idx].numpy()
    
    # Get attention TO the target position
    attention_to_target = avg_attention[:, target_idx].numpy()
    
    return attention_from_target, attention_to_target

# Test sentences with clear function/content positions
test_cases = [
    ("The cat sat on the mat", 3, "on", "function"),
    ("The cat sat on the mat", 1, "cat", "content"),
    ("The ball rolled down", 3, "down", "function"),
    ("The ball rolled down", 1, "ball", "content"),
]

print("\nAttention Distribution Analysis:")
print("-" * 40)

for sentence, idx, word, word_type in test_cases:
    tokens = tokenizer.tokenize(sentence)
    attention_from, attention_to = get_attention_patterns(sentence, idx + 1)  # +1 for CLS token
    
    # Calculate attention entropy (higher = more distributed)
    entropy_from = -np.sum(attention_from * np.log(attention_from + 1e-10))
    entropy_to = -np.sum(attention_to * np.log(attention_to + 1e-10))
    
    # Find peak attention
    peak_from_idx = np.argmax(attention_from)
    peak_to_idx = np.argmax(attention_to)
    
    print(f"\n'{word}' ({word_type} word) at position {idx}:")
    print(f"  Attention FROM {word}:")
    print(f"    Entropy: {entropy_from:.3f}")
    print(f"    Peak attention to: position {peak_from_idx}")
    print(f"  Attention TO {word}:")
    print(f"    Entropy: {entropy_to:.3f}")
    print(f"    Peak attention from: position {peak_to_idx}")

print("\n" + "=" * 70)
print("HYPOTHESIS: Function words have more distributed attention")
print("(higher entropy) while content words have focused attention")

# Summary statistics
function_entropies = []
content_entropies = []

for _, idx, _, word_type in test_cases:
    sentence = test_cases[0][0]  # Use first sentence for consistency
    attention_from, _ = get_attention_patterns(sentence, idx + 1)
    entropy = -np.sum(attention_from * np.log(attention_from + 1e-10))
    
    if word_type == "function":
        function_entropies.append(entropy)
    else:
        content_entropies.append(entropy)

if len(function_entropies) > 0 and len(content_entropies) > 0:
    print(f"\nMean entropy - Function words: {np.mean(function_entropies):.3f}")
    print(f"Mean entropy - Content words: {np.mean(content_entropies):.3f}")
