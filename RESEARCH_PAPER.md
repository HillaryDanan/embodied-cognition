# Positional Encoding Biases and the Role of Training Data in Temporal-Causal Reasoning: A Comparative Study of Language Models

Hillary Danan  
September 2025

## Abstract

We investigated two phenomena in transformer-based language models: (1) extreme positional encoding biases in word prediction, and (2) temporal-causal reasoning capabilities. Testing six models, we found function words (determiners, prepositions) are predicted at rank 0 while content words (nouns, objects) appear at ranks 32-11,988. Contrary to our initial hypothesis that lack of embodied experience would prevent temporal-causal reasoning, we found training data determines capability: BERT (trained on limited data) showed complete failure (0% accuracy, 100% punctuation defaults), while RoBERTa (trained on 10x more data including news corpora) achieved perfect performance on complex three-step causal chains. GPT-3.5 achieved perfect scores across all tests. These findings suggest that diverse training data, not embodied experience, enables temporal-causal reasoning in language models.

## 1. Introduction

Language models process sequential text but lack physical bodies that experience time and causation directly. This raises fundamental questions about their capacity for temporal and causal reasoning. We initially hypothesized, based on embodied cognition theory (Barsalou, 2008), that physical experience would be necessary for understanding temporal sequences and causal relationships. Our investigation revealed a more complex picture: while some models completely fail at these tasks, others achieve perfect performance, with training data being the critical differentiator.

## 2. Background and Motivation

### 2.1 Theoretical Framework

Embodied cognition theory suggests that conceptual understanding emerges from sensorimotor experience (Barsalou, 2008). Developmental psychology research demonstrates that 6-month-old infants detect physical causation violations through embodied experience (Spelke et al., 1992; Baillargeon, 2004). This raised the question: can models without bodies understand causation?

### 2.2 Causal Reasoning Framework

Following Pearl's (2009) causal inference framework, we tested both simple causal relations (A→B) and transitive causal chains (A→B→C, therefore A→C).

## 3. Methods

### 3.1 Models Tested

We evaluated six models with varying architectures and training data:
- **BERT-base-uncased**: BookCorpus + Wikipedia (Devlin et al., 2018)
- **RoBERTa-base**: BERT data + CC-News + OpenWebText + Stories, 10x more data (Liu et al., 2019)
- **ALBERT-base-v2**: Factorized embeddings (Lan et al., 2019)
- **DistilBERT-base-uncased**: Distilled BERT (Sanh et al., 2019)
- **GPT-3.5-turbo**: Large-scale diverse training (via API)
- **Claude-3-haiku**: Constitutional AI training (via API)

### 3.2 Test Battery

#### 3.2.1 Positional Encoding Tests
We tested 22 sentences with masked positions across different syntactic categories, measuring the rank at which the correct word appeared in the model's predictions.

#### 3.2.2 Temporal Sequence Tests (n=10)
- Simple sequences: "Yesterday comes before [MASK]" → "today"
- Ordinal sequences: "First, second, [MASK]" → "third"
- Cyclical patterns: "Spring follows [MASK]" → "winter"

#### 3.2.3 Causal Relation Tests (n=10)
- Physical causation: "Heat melts [MASK]" → "ice"
- Abstract causation: "Study leads to [MASK]" → "knowledge"

#### 3.2.4 Complex Causal Chains (n=4)
- Transitive inference: "A causes B, B causes C, therefore A causes [MASK]" → "C"
- Applied chains: "Rain causes wet, wet causes slippery, therefore rain causes [MASK]" → "slippery"

### 3.3 Statistical Analysis

We used Spearman correlation for rank-based comparisons and Mann-Whitney U tests for distribution differences. Models were evaluated on accuracy and punctuation default rates.

## 4. Results

### 4.1 Positional Encoding Discovery

We found extreme disparities in prediction accuracy by syntactic position:

| Word Type | Mean Rank | Examples | n |
|-----------|-----------|----------|---|
| Determiners | 0 | the, a | 4 |
| Prepositions | 0 | on, with, in | 2 |
| Operators | 0 | plus, minus | 1 |
| Verbs | 12.5 | fell, rolled | 4 |
| Objects | 6,010 | ball (32), homework (11,988) | 2 |
| Adverbs | 251 | naturally (438) | 1 |

Lexical frequency correlated with prediction success (r = -0.505, p = 0.023).

### 4.2 Temporal-Causal Performance

| Model | Temporal Accuracy | Causal Accuracy | Punctuation Rate | Complex Chains |
|-------|-------------------|-----------------|------------------|----------------|
| BERT | 0% | 0% | 100% | 0/4 |
| RoBERTa | 80% | 20% | 10% | 4/4 |
| ALBERT | 0% | 0% | 50% | Not tested |
| DistilBERT | 40% | 0% | 50% | Not tested |
| GPT-3.5 | 100% | 100% | 0% | 4/4 |
| Claude-3-haiku | 100%† | 100%† | 0% | 2/4‡ |

†Claude adds periods but identifies correct content  
‡Claude failed on abstract logical chains but succeeded on concrete causal chains

### 4.3 Key Finding: Training Data Effects

BERT's complete failure (100% punctuation defaults) versus RoBERTa's success revealed the critical factor: RoBERTa was trained on:
1. 10x more data than BERT
2. CC-News corpus containing temporal sequences
3. No Next Sentence Prediction objective

Statistical analysis showed average expected word ranks of 791.9 (temporal) and 894.4 (causal) for BERT, indicating the model treats causal connectives as sentence-ending signals rather than semantic relationships.

## 5. Discussion

### 5.1 Refuting the Embodiment Hypothesis

Our initial hypothesis that physical experience is necessary for temporal-causal reasoning was not supported. Models trained on sufficiently diverse text achieve perfect performance without embodied experience.

### 5.2 Positional Encoding as Architectural Constraint

The function vs. content word disparity (rank 0 vs. 6000+) appears across all models regardless of their reasoning capabilities, suggesting an architectural bias in transformers. However, we discovered this can be overridden: compound contexts ("rock-solid, rock-hard, [MASK]-heavy") achieved 95.6% accuracy for "rock" despite its typically poor rank.

### 5.3 Training Data as Primary Determinant

The inclusion of news corpora appears critical for temporal reasoning, likely due to the prevalence of temporal sequences in news narratives. Models learn "yesterday → today → tomorrow" from reading thousands of news articles, not from experiencing time. This suggests temporal-causal reasoning emerges from statistical patterns in text rather than requiring embodied experience.

### 5.4 Tokenization Effects

Claude-3-haiku's pattern of adding periods while identifying correct content suggests tokenization differences between models may affect output formatting while preserving semantic understanding. This warrants further investigation.

### 5.5 Implications for Model Deployment

The complete failure of BERT at causal reasoning (100% punctuation defaults) has serious implications for deployment. Applications requiring causal inference - medical diagnosis systems, legal reasoning tools, or scientific hypothesis generation - should not use BERT-based models. Our findings suggest minimum requirements for causal reasoning tasks: models trained on diverse corpora including temporal sequences (news), and validation through explicit causal chain tests before deployment. The ability to override positional biases through compound contexts offers a practical prompting strategy for production systems.

### 5.6 Actionable Insights for Practitioners

The discovery that compound contexts can override positional biases (achieving 95.6% accuracy for typically unpredictable words) provides a concrete prompting strategy. For content words that models struggle to predict:
- Use compound patterns: "X-quality, X-attribute, [MASK]-property"
- Leverage definition frames: "A [MASK] is a [properties]"
- Avoid direct object queries in isolation

These strategies can improve performance in production systems where content word prediction is critical.

### 6.1 Current Limitations
- Small sample size (n=6 models)
- English only
- Simple causal chains (maximum 3 steps)
- Did not test counterfactual or bidirectional causation

### 6.2 Future Directions
1. Test multilingual models to assess cross-linguistic patterns
2. Evaluate more complex causal graphs (Pearl, 2009)
3. Investigate fine-tuning to relocate positional biases
4. Test developmental trajectories against human infant data (Baillargeon, 2004)

## 7. Conclusion

We document two independent phenomena in language models:

1. **Extreme positional encoding biases**: Function words are privileged (rank 0) while content words are severely disadvantaged (rank 6000+), though context can override these biases.

2. **Training data determines reasoning capability**: Models require diverse training data, particularly temporal sequences from news corpora, to develop temporal-causal reasoning. Embodied experience is not necessary.

These findings challenge assumptions about the necessity of embodiment for abstract reasoning while revealing fundamental architectural biases in transformer models.

## References

Baillargeon, R. (2004). Infants' physical world. *Current Directions in Psychological Science*, 13(3), 89-94.

Barsalou, L. W. (2008). Grounded cognition. *Annual Review of Psychology*, 59, 617-645.

Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2018). BERT: Pre-training of deep bidirectional transformers for language understanding. *arXiv preprint arXiv:1810.04805*.

Lan, Z., Chen, M., Goodman, S., Gimpel, K., Sharma, P., & Soricut, R. (2019). ALBERT: A lite BERT for self-supervised learning of language representations. *arXiv preprint arXiv:1909.11942*.

Liu, Y., Ott, M., Goyal, N., Du, J., Joshi, M., Chen, D., ... & Stoyanov, V. (2019). RoBERTa: A robustly optimized BERT pretraining approach. *arXiv preprint arXiv:1907.11692*.

Pearl, J. (2009). *Causality: Models, reasoning and inference* (2nd ed.). Cambridge University Press.

Sanh, V., Debut, L., Chaumond, J., & Wolf, T. (2019). DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter. *arXiv preprint arXiv:1910.01108*.

Spelke, E. S., Breinlinger, K., Macomber, J., & Jacobson, K. (1992). Origins of knowledge. *Psychological Review*, 99(4), 605-632.

## Data and Code Availability

All experimental code and results are available at: https://github.com/HillaryDanan/embodied-cognition

## Author Note

This research was conducted independently. Correspondence concerning this article should be addressed to Hillary Danan.
