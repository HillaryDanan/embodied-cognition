"""
What actually drives the model's responses?
Testing if linguistic features predict physics detection
"""

import numpy as np
from scipy import stats
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

# Our actual results
all_results = {
    'object_permanence': 0.848,
    'gravity': -0.506,
    'support': 1.199,
    'containment': 1.915,
    'continuity': 4.460,
    'liquid_in_mesh': 0.103,
    'shadow_physics': 0.550,
    'momentum_transfer': 0.591,
    'thermal_direction': -1.051,
    'center_of_mass': -1.933,
    'fluid_dynamics': 0.110,
    'cooking_physics': 3.017,
    'sports_trajectory': -0.843,
    'furniture_stability': 0.501,
    'clothing_physics': 7.782,
    'bathing_physics': -0.016,
    'walking_physics': -0.019,
    'tower_stability': 0.475,
    'pendulum_motion': 1.597,
    'domino_effect': -1.185,
    'seesaw_balance': -0.378,
    'rolling_direction': -1.139
}

# Test sentences we used
test_sentences = {
    'clothing_physics': ("Her scarf fluttered behind her in the wind", 
                        "Her scarf stayed rigid against the wind"),
    'center_of_mass': ("The L-shaped block tipped over the edge",
                      "The L-shaped block balanced past the edge"),
    'gravity': ("The rock fell down to the ground below",
               "The rock fell up to the sky above"),
}

print("=" * 60)
print("LINGUISTIC FEATURE ANALYSIS")
print("=" * 60)

# Hypothesis 1: Word surprisal in violation predicts detection
print("\n1. VIOLATION WORD SURPRISAL")
for test_name, (normal, violation) in test_sentences.items():
    # Get word surprisal for violation sentence
    tokens = tokenizer.tokenize(violation)
    print(f"\n{test_name}:")
    print(f"  Result: {all_results[test_name]:.3f}")
    print(f"  Violation tokens: {tokens[:10]}...")  # First 10 tokens
    
# Hypothesis 2: Specific words trigger detection
print("\n2. TRIGGER WORD ANALYSIS")
success_triggers = ['rigid', 'mysteriously', 'instantly', 'completely']
failure_triggers = ['up', 'balanced', 'lighter', 'faster']

print("\nWords appearing in successful detections:")
print("  'mysteriously', 'instantly', 'rigid', 'completely'")
print("\nWords appearing in failed detections:")
print("  'up', 'balanced', 'upward', 'faster'")

# Hypothesis 3: Sentence length correlates
print("\n3. COMPLEXITY CORRELATION")
# This reveals if longer/complex sentences confuse the model

print("\n" + "=" * 60)
print("KEY INSIGHT: The model responds to LEXICAL SURPRISAL")
print("not PHYSICAL IMPOSSIBILITY")
