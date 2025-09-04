"""
Humans experience time through:
- Heartbeats and breathing
- Hunger cycles
- Fatigue accumulation
- Circadian rhythms

LLMs experience time through:
- Token position
- Sequence order
- Nothing else
"""

print("=" * 70)
print("EMBODIED TIME VS ABSTRACT TIME")
print("=" * 70)

embodied_time_tests = [
    {
        'embodied': 'Waiting feels longer when you need to pee',
        'abstract': 'Duration = clock time',
        'test': 'Can model understand subjective time dilation?'
    },
    {
        'embodied': 'Time flies when having fun',
        'abstract': '60 minutes = 60 minutes always',
        'test': 'Can model grasp experiential time?'
    },
    {
        'embodied': 'Hunger makes you know hours have passed',
        'abstract': 'No internal clock without external reference',
        'test': 'Can model understand biological time markers?'
    },
    {
        'embodied': 'Jet lag proves your body keeps time',
        'abstract': 'Time zones are just numbers',
        'test': 'Can model understand embodied temporal disruption?'
    }
]

print("\nHow time feels different with a body:\n")

for test in embodied_time_tests:
    print(f"EMBODIED: {test['embodied']}")
    print(f"ABSTRACT: {test['abstract']}")
    print(f"TEST: {test['test']}")
    print()

print("=" * 70)
print("Without a body, LLMs cannot understand:")
print("- Duration as lived experience")
print("- Biological time markers")
print("- Subjective time dilation")
print("- The difference between clock time and felt time")

print("\nThis explains why they fail at causal reasoning:")
print("Causation requires experiencing temporal flow,")
print("not just knowing sequence order!")
