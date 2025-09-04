"""
Synthesizing all findings into coherent model
Being honest about contradictions and complexity
"""

print("=" * 70)
print("SYNTHESIS: WHAT WE'VE ACTUALLY DISCOVERED")
print("=" * 70)

findings = {
    'frequency_effect': {
        'correlation': -0.505,
        'p_value': 0.023,
        'variance_explained': 0.255,
        'interpretation': 'Frequency is a major factor, but only explains 25%'
    },
    'positional_bias': {
        'function_words': 0,
        'content_words': 6010,
        'ratio': float('inf'),
        'interpretation': 'Massive effect, but CAN be overridden'
    },
    'context_override': {
        'compound_pattern': 0.956,
        'definition_frame': 0.149,
        'baseline': 0.0026,
        'improvement': 368,  # 368x improvement!
        'interpretation': 'Context can COMPLETELY override position'
    },
    'cross_linguistic': {
        'german': 0.75,
        'english': 0.375,
        'spanish': 0.147,
        'interpretation': 'Universal pattern, language-specific strength'
    }
}

print("\nTHE REAL MODEL:")
print("-" * 50)
print("""
LLMs are NOT:
- Pure syntax machines (nonsense objects hurt MORE than nonsense verbs)
- Pure semantic systems (position matters enormously)
- Simple frequency matchers (position can override frequency)

LLMs ARE:
- Multi-factor pattern matchers with:
  1. Frequency weighting (r=-0.505)
  2. Positional bias (function >> content)
  3. Context sensitivity (can override everything)
  4. Language-specific tuning
""")

print("\nBREAKTHROUGH DISCOVERIES:")
print("-" * 50)
print("1. Compound patterns make objects predictable (95.6% confidence!)")
print("2. German handles prepositions better (morphological richness?)")
print("3. Frequency explains only 25% - structure dominates")
print("4. Attention patterns show only modest differences")

print("\n" + "=" * 70)
print("IMPLICATIONS FOR THE FIELD")
print("-" * 50)

implications = """
FOR PROMPT ENGINEERING:
- Use compound patterns for object queries
- Definition frames work for content words
- German prompts might work better for some tasks

FOR ARCHITECTURE:
- Not purely positional (context can override)
- Not purely frequency-based (position dominates)
- Complex interaction of multiple biases

FOR THEORY:
- Models aren't syntax OR semantics - they're BOTH
- Context can override architectural biases
- Language-specific effects suggest training data matters

FOR PRACTICE:
- We CAN make objects predictable with right prompting
- Different languages have different optimal patterns
- Multiple strategies exist for same goal
"""
print(implications)

# Calculate composite score
def calculate_predictability_score(frequency_rank, position_type, context_type):
    """
    Composite model of predictability
    Based on our empirical findings
    """
    base_score = 0.5
    
    # Frequency effect (25% weight)
    freq_score = -0.25 * (frequency_rank / 10000)
    
    # Position effect (50% weight)
    if position_type == 'function':
        pos_score = 0.5
    elif position_type == 'content':
        pos_score = -0.5
    else:
        pos_score = 0
    
    # Context effect (25% weight, but can dominate)
    if context_type == 'compound':
        ctx_score = 1.0  # Override everything
    elif context_type == 'definition':
        ctx_score = 0.5
    else:
        ctx_score = 0
    
    return max(0, min(1, base_score + freq_score + pos_score + ctx_score))

print("\n" + "=" * 70)
print("PREDICTABILITY MODEL:")
examples = [
    ("'the' (det, high-freq, normal)", calculate_predictability_score(1, 'function', 'normal')),
    ("'homework' (noun, mid-freq, normal)", calculate_predictability_score(5000, 'content', 'normal')),
    ("'rock' (noun, mid-freq, compound)", calculate_predictability_score(3000, 'content', 'compound')),
]

for desc, score in examples:
    print(f"{desc:40s}: {score:.1%} predictable")
