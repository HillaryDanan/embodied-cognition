# Positional Encoding of Knowledge in Language Models: Function Words vs Content Words

## Abstract

We demonstrate that language models exhibit extreme positional bias in knowledge encoding, 
with function words (determiners, prepositions) achieving perfect prediction (rank 0) while 
content words (nouns, objects) show poor prediction (rank 32-11,988). Contrary to our 
initial hypothesis that models are pure syntactic processors, nonsense word experiments 
reveal they require BOTH syntactic structure AND lexical familiarity, with semantic vacancy 
(82% confidence drop) hurting more than syntactic novelty (73% drop).

## Key Findings

1. **Positional Encoding Hierarchy**
   - Determiners: rank 0 (perfect)
   - Prepositions: rank 0 (perfect)
   - Verbs: rank 0-44 (mixed)
   - Objects: rank 32-11,988 (poor)

2. **Chain-of-Thought Mechanism**
   - Works via 7x increase in predicates
   - Verb/noun ratio: 0.50 → 0.64

3. **Nonsense Word Results** (Unexpected)
   - Nonsense objects: 82.4% confidence drop
   - Nonsense predicates: 73.0% confidence drop
   - Models need lexical+syntactic information

## Discussion

Our findings suggest LLMs are neither pure syntax machines nor semantic reasoners, but 
**syntax-guided lexical associators**. They excel at predicting grammatical relations 
but struggle with content words, explaining:
- Why prompts targeting prepositions achieve 97% confidence
- Why object-focused prompts fail (2.6-6% confidence)
- Why hallucinations preserve grammar but confabulate content

## Implications

1. **For Prompt Engineering**: Target function words, not content words
2. **For Architecture**: Transformers privilege relations over entities
3. **For AGI**: Current models lack true object-level understanding

## Conclusion

Language models encode knowledge positionally, with massive disparities between function 
and content words. They are syntax-guided pattern matchers, not reasoning systems.
