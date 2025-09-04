# Positional Encoding of Knowledge in Language Models

## Abstract
We demonstrate that LLMs encode knowledge in syntactic positions, with function words 
(rank 0) vastly outperforming content words (rank 32-11,988). This explains numerous 
phenomena including Chain-of-Thought effectiveness and prompt engineering successes.

## 1. Introduction
- The embodiment gap hypothesis
- Initial chaos in physics detection
- Discovery of positional encoding

## 2. Methods
- Masked prediction ranking
- Physics violation detection
- Cross-domain testing
- Cross-model validation

## 3. Results
### 3.1 Physics Knowledge is Positional
- Objects: mean rank 32
- Predicates: mean rank 0-2
- Adverbs: rank 438

### 3.2 Universal Pattern
- BERT, RoBERTa, ALBERT, DistilBERT all show pattern
- GPT-3.5, Claude, Gemini generate predicates correctly

### 3.3 Function vs Content
- Determiners: rank 0
- Prepositions: rank 0
- Objects: rank 6010 (mean)

### 3.4 Chain-of-Thought Analysis
- 7x increase in predicates
- Verb/noun ratio: 0.50 → 0.64

## 4. Implications
### 4.1 For Prompt Engineering
- Target function words for reliability
- Avoid object-focused queries

### 4.2 For Model Architecture
- Fundamental transformer limitation?
- Need for object-centric architectures?

### 4.3 For AI Safety
- Models can't be trusted for object-level reasoning
- Syntactic competence ≠ semantic understanding

## 5. Future Work
- Fine-tuning to relocate knowledge
- Cross-linguistic studies
- Developmental comparisons

## 6. Conclusion
LLMs are syntactic engines, not reasoning systems.
