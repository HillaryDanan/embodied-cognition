# Context Overrides Position: How Prompting Can Defeat Architectural Biases in Language Models

## Abstract
We demonstrate that while LLMs exhibit extreme positional bias (function words: rank 0, 
content words: rank 6010), specific prompt patterns can completely override these 
architectural constraints, achieving 95.6% confidence for traditionally "unpredictable" 
object words through compound framing.

## Key Finding
**Compound patterns** ("rock-solid, rock-hard, [MASK]-heavy") achieve 368x improvement 
in object prediction, suggesting that architectural biases are soft constraints that 
can be overcome with proper prompting strategies.
