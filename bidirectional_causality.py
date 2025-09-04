"""
Can models handle bidirectional causality?
Or does their unidirectional training break?

"The chicken exists because of the egg"
"The egg exists because of the chicken"
BOTH TRUE SIMULTANEOUSLY
"""

print("=" * 70)
print("BIDIRECTIONAL CAUSALITY: BREAKING UNIDIRECTIONAL MODELS")
print("=" * 70)

bidirectional_pairs = [
    ("Supply increases because demand increases",
     "Demand increases because supply increases"),
    
    ("She's happy because she smiles",
     "She smiles because she's happy"),
    
    ("The present determines the future",
     "The future determines the present"),
    
    ("Evolution shaped consciousness",
     "Consciousness shaped evolution"),
    
    ("Language creates thought",
     "Thought creates language")
]

print("\nTesting if models can handle P↔Q relationships:\n")

for forward, backward in bidirectional_pairs:
    print(f"Forward:  {forward}")
    print(f"Backward: {backward}")
    print(f"Can model understand both are true? [WOULD TEST COHERENCE]\n")

print("=" * 70)
print("This gets at whether models have true causal models")
print("or just learned directional correlations")
