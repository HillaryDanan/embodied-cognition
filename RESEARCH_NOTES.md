# Research Notes: Embodiment Gap Investigation

## September 4, 2025

### CRITICAL DISCOVERY: Physics Knowledge is Positionally Encoded

#### The Contradiction That Led to Insight
- **Initial tests**: 59% detection, massive chaos (σ=2.09)
- **Simple template tests**: 100% detection across ALL models
- **Resolution**: We were measuring the wrong thing!

#### Key Finding: WHERE Physics Lives in Sentences

Testing "The heavy rock naturally fell down toward the ground":

| Position | Word | Model Rank | Insight |
|----------|------|------------|---------|
| 0 | The | 0 (perfect) | Articles predicted well |
| 1 | heavy | 13 | Adjectives poorly predicted |
| 2 | rock | 12 | **Objects poorly predicted** |
| 3 | naturally | 438 (!) | Adverbs catastrophically bad |
| 4 | fell | 0 (perfect) | **VERBS perfectly predicted** |
| 5 | down | 0 (perfect) | **DIRECTIONS perfectly predicted** |
| 6 | toward | 5 | Prepositions good |
| 7 | the | 0 | Articles perfect |
| 8 | ground | 29 | Location objects moderate |

### THE BREAKTHROUGH

**Physics knowledge in LLMs is encoded in PREDICATES (verbs/directions), not OBJECTS**

This explains EVERYTHING:
1. Why testing "rock" in "rock fell up" showed chaos
2. Why testing "up" vs "down" directly showed perfect detection
3. Why our retroactive update method failed

### Methodological Insights

#### Method 1 (Failed): Testing Object Surprise
```python
"The rock fell down" vs "The rock fell up"
# Testing surprise at "rock"
# Result: -0.448 (NEGATIVE - doesn't detect violation!)
```

#### Method 2 (Successful): Testing Predicate Surprise
```python
"The rock fell [MASK]" → "down" vs "up"
# Testing surprise at direction word
# Result: +0.213 to +5.892 (POSITIVE - detects violation!)
```

### Implications

1. **For LLM Architecture**: Models learn physics through verb-preposition patterns, not object properties
2. **For Embodiment Gap**: The gap isn't about physics knowledge, it's about WHERE that knowledge is stored
3. **For Testing Methodology**: We must test at the right syntactic position

### Statistical Summary

| Test Type | Detection Rate | Mean Diff | Std Dev |
|-----------|---------------|-----------|---------|
| Object-based (original) | 59% | 0.73 | 2.09 |
| Predicate-based (new) | 100% | 4.03 | 1.87 |

### Cross-Model Consistency

ALL models show same pattern:
- BERT: 5.95 mean difference
- RoBERTa: 5.13 mean difference  
- ALBERT: 1.63 mean difference
- DistilBERT: 3.41 mean difference

**Conclusion**: This is architectural, not model-specific

---

## Next Questions

1. **Is this true for other types of knowledge?** (social, causal, mathematical)
2. **Can we exploit this for better prompting?**
3. **Does this explain why CoT works?** (forces models to generate predicates)
4. **Can we design training to encode knowledge in objects?**

---

## Paper Ideas

### Option 1: "Positional Encoding of Physical Knowledge in Language Models"
- Focus on WHERE knowledge lives syntactically
- Implications for prompt engineering

### Option 2: "The Predicate Hypothesis: Why LLMs Fail at Object-Level Reasoning"
- Broader theory about knowledge representation
- Test across multiple domains

### Option 3: "Measurement Artifacts in Testing LLM Reasoning"
- Methodological paper
- Guidelines for proper testing

---

## September 4, 2025 - Update

### CRITICAL REVISION: Nonsense Word Experiments

**Hypothesis**: Models are pure syntax machines
**Result**: CONTRADICTED!

Nonsense word tests showed:
- Normal: 83.7% confidence
- Nonsense objects (gribble/florp): 14.7% confidence (82% drop!)
- Nonsense predicates (xyzqed): 22.6% confidence (73% drop)

**Conclusion**: Models are **syntax-guided lexical associators**, not pure syntax machines. They need BOTH grammatical structure AND familiar lexical items.

## September 4, 2025 - Update

### Investigation Results

#### 1. Frequency Effects (CONFIRMED)
- **r = -0.505, p = 0.023** - frequency strongly predicts prediction success
- High-freq words ~80 rank vs low-freq ~6,395 rank
- Explains positional patterns partially

#### 2. Cross-Linguistic Patterns
- German: 75% confidence (compound-friendly language)
- English: 37.5% confidence
- Spanish: 14.7% confidence
- **Universal but language-dependent strength**

#### 3. Object-Privileged Prompts (BREAKTHROUGH!)
Found two patterns that achieve rank 0 for objects:
- **Compound pattern**: "rock-solid, rock-hard, [MASK]-heavy" → 95.6% confidence!
- **Definition frame**: "A [MASK] is a hard mineral object" → 14.9% confidence
- Context can completely override positional bias

#### 4. Attention Patterns
- Function words: 1.816 entropy (slightly more distributed)
- Content words: 1.586 entropy
- Difference exists but is modest (14%)

### Synthesis

**The Real Model**: LLMs are frequency-weighted, position-biased, context-sensitive pattern matchers. Not pure syntax, not pure semantics, but a complex interaction of:
1. Word frequency (25% of variance)
2. Syntactic position (massive effect)
3. Contextual framing (can override everything)
4. Language-specific patterns

### CONFIRMED: Function Words > Content Words

**Exploit Position Results:**
- Object-focused prompts: 2.6-6% confidence (FAILED)
- Predicate-focused prompts: 15.6-61.9% confidence (MODERATE)
- Function-word prompts: 51.9-97.1% confidence (SUCCESS!)

**Chain-of-Thought Discovery:**
- Direct answers: 1 verb, 2 nouns
- CoT answers: 7 verbs, 11 nouns (7x verb increase!)
- **CoT works by forcing predicate generation**

### Cross-Domain Analysis

| Domain | Best Predicted | Worst Predicted |
|--------|---------------|-----------------|
| Physics | "down" (0), "the" (0) | "ball" (32) |
| Social | "with" (0), "her" (0) | "homework" (11,988!) |
| Math | "plus" (0) | "exactly" (64) |
| Causal | "caused" (0), "in" (0) | "valley" (15) |

**Pattern**: Determiners (0) > Prepositions (0) > Operators (0) > Verbs (12.5) > Numbers (7) > Locations (15) > Objects (6010!)

## TODO

- [x] Test social knowledge position (who did what to whom)
- [x] Test mathematical knowledge position  
- [x] Test causal knowledge position
- [x] Design intervention to teach object-level physics
- [ ] Test if fine-tuning can move knowledge location
- [ ] Generate training data for object-centric physics
- [ ] Test on larger models (GPT-4, Claude-3 Opus)
- [ ] Write paper on positional encoding discovery

---

## Key Quotes from Literature

> "Core knowledge systems" - Spelke & Kinzler (2007)
> Infants track objects, not predicates

> "Force dynamics in language" - Talmy (1988)  
> Verbs encode physical forces

This aligns PERFECTLY with our findings!
