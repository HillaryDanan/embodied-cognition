# Embodied Cognition Research: Final Summary

## Core Discovery
**LLMs encode knowledge positionally in syntax, not semantically in concepts.**

## Key Statistics
- Function words: Rank 0 (perfect prediction)
- Content words: Rank 32-11,988 (terrible prediction)
- Coefficient of variation: 2.86 (extreme inconsistency)

## Evidence Chain
1. Started testing physics violations → 59% detection, high variance
2. Discovered measurement position matters → object vs predicate surprise
3. Found perfect prediction at predicates, terrible at objects
4. Confirmed across 4 architectures (BERT family)
5. Generalized to social, mathematical, causal domains
6. Explained Chain-of-Thought (7x predicate increase)

## Implications
- **Prompt Engineering**: Target prepositions/verbs, not nouns
- **Model Limitations**: Can't trust object-level reasoning
- **Architecture**: Fundamental transformer limitation
- **Training**: Need object-centric objectives

## What This Means
LLMs are sophisticated syntactic pattern matchers, not reasoning engines.
They excel at grammatical structure but lack semantic grounding.

## Next Steps
1. Test if fine-tuning can relocate knowledge
2. Compare to human developmental trajectories
3. Design object-centric architectures
4. Test cross-linguistic variations
