"""
Why did nonsense objects hurt MORE than nonsense predicates?
Let's dig deeper into what's happening
"""

print("=" * 70)
print("ANALYZING UNEXPECTED NONSENSE RESULTS")
print("=" * 70)

results = {
    'Normal': {
        'confidence': 0.837,
        'predictions': ['down', 'up', 'over', 'off', 'into']
    },
    'Nonsense_Objects': {
        'confidence': 0.147,
        'predictions': ['under', 'into', 'on', 'off', 'with']
    },
    'Nonsense_Predicates': {
        'confidence': 0.226,
        'predictions': ['on', 'up', 'over', 'down', 'in']
    }
}

print("CONFIDENCE DROPS:")
normal_conf = results['Normal']['confidence']
for condition, data in results.items():
    if condition != 'Normal':
        drop = (1 - data['confidence']/normal_conf) * 100
        print(f"{condition}: {drop:.1f}% confidence drop")

print("\nPREDICTION OVERLAP WITH NORMAL:")
normal_preds = set(results['Normal']['predictions'])
for condition, data in results.items():
    if condition != 'Normal':
        overlap = len(set(data['predictions']) & normal_preds)
        print(f"{condition}: {overlap}/5 predictions match")

print("\n" + "=" * 70)
print("REVISED UNDERSTANDING:")
print("""
The model relies on BOTH lexical semantics AND syntax:
1. "gribble" and "florp" provide NO semantic context → 82% drop
2. "xyzqed" breaks verb expectations but maintains position → 73% drop
3. Both matter, but SEMANTIC VACANCY hurts more than SYNTACTIC NOVELTY

This suggests models use:
- Lexical co-occurrence (ball-roll-hill)
- Syntactic frames (NOUN VERB PREP NOUN)
- BOTH are needed for high confidence
""")

print("\n" + "=" * 70)
print("IMPLICATION: Models aren't pure syntax machines")
print("They're syntax-guided semantic associators")
