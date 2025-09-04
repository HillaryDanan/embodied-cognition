"""
Experiment design: Can we teach models to encode physics in objects?
This is just the DESIGN - actual fine-tuning would need GPU
"""

print("=" * 70)
print("EXPERIMENT: MOVING PHYSICS KNOWLEDGE TO OBJECTS")
print("=" * 70)

training_data_design = """
HYPOTHESIS: Fine-tuning on object-centric physics can move knowledge

TRAINING DATA STRUCTURE:
1. Standard: "The ball rolled down" (predicate-focused)
2. Object-centric: "The ball, being spherical, rolled" (object-focused)
3. Property-explicit: "The heavy-ball fell, the light-ball fell slowly"

CONTRASTIVE PAIRS:
- "The steel-ball sank" vs "The wooden-ball floated"
- "The ice-cube melted" vs "The rock-cube remained"
- "The paper-plane glided" vs "The paper-ball fell"

EXPECTED OUTCOME:
Before: ball (rank 32), rolled (rank 0)
After: ball (rank <10), rolled (rank ~0)

MEASUREMENT:
Same positional encoding test before/after fine-tuning
"""

print(training_data_design)

print("\nTRAINING EXAMPLES TO GENERATE:")
examples = [
    ("Heavy-rocks fall quickly", "Light-feathers fall slowly"),
    ("Solid-ice floats", "Solid-iron sinks"),
    ("Round-balls roll", "Square-blocks slide"),
    ("Hot-metal expands", "Cold-metal contracts"),
]

for positive, negative in examples:
    print(f"  + {positive}")
    print(f"  - {negative}")

print("\n" + "=" * 70)
print("This would test if positional encoding is LEARNABLE or ARCHITECTURAL")
