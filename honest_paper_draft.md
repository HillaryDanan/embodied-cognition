# From Positional Encoding to Causal Chains: How Training Data Determines Reasoning Capabilities in Language Models

## Abstract
We investigated temporal and causal reasoning across multiple transformer architectures, initially hypothesizing that lack of embodied experience would prevent such reasoning. Our results contradicted this hypothesis: while BERT showed complete failure (0% on both), RoBERTa achieved perfect performance on complex causal chains (100%). The key differentiator was training data - models trained on news corpora and larger datasets successfully reason about time and causation, while those with limited training cannot. We also document extreme positional encoding biases, with function words predicted at rank 0 and content words at rank 6000+.

## 1. Introduction
Language models lack bodies and thus physical experience of time and causation. We hypothesized this would prevent temporal-causal reasoning. We were wrong.

## 2. Key Findings

### 2.1 Positional Encoding Discovery
- Function words (the, in, with): rank 0
- Content words (homework, ball): rank 32-11,988
- Compound contexts can override bias (95.6% accuracy)

### 2.2 Temporal-Causal Performance by Model
| Model | Training Data | Temporal | Causal | Complex Chains |
|-------|--------------|----------|---------|----------------|
| BERT | BookCorpus+Wiki | 0% | 0% | 0/4 |
| RoBERTa | +CC-News, 10x more | 80% | 20% | 4/4 |
| GPT-3.5 | Massive diverse | 100% | 100% | Not tested |

### 2.3 The Training Data Hypothesis
Models trained on news (containing temporal sequences and causal narratives) learn these patterns. Models without such data default to punctuation.

## 3. Methods
- Masked prediction tasks for temporal sequences
- Causal completion tests based on Pearl (2009)
- Complex 3-step causal chain reasoning
- Statistical analysis using Spearman correlation and Mann-Whitney U

## 4. Results That Contradicted Our Hypothesis
- RoBERTa perfectly solves "A causes B, B causes C, therefore A causes ___" → C
- GPT-3.5 handles all temporal and causal tasks perfectly
- Training data, not embodiment, determines capability

## 5. Discussion
We must acknowledge our embodied cognition hypothesis was not supported. Instead, we found:
1. Sufficient diverse training data enables temporal-causal reasoning
2. News corpora specifically help (temporal sequences)
3. Architectural differences matter less than data

## 6. Limitations
- Small sample size (6 models)
- Primarily tested English
- Did not test larger models systematically

## 7. Conclusion
Transformers can learn temporal-causal reasoning from text alone given sufficient diverse training data. The lack of embodiment is not a fundamental barrier. However, positional encoding biases remain strong across all models tested.

## References
- Baillargeon, R. (2004). Infants' physical world. Current Directions in Psychological Science.
- Pearl, J. (2009). Causality: Models, Reasoning and Inference.
- Rogers, A., Kovaleva, O., & Rumshisky, A. (2020). A primer on neural network architectures for NLP.
- Spelke, E. S., & Kinzler, K. D. (2007). Core knowledge. Developmental Science.

## Data Availability
All code and results available at: https://github.com/HillaryDanan/embodied-cognition
