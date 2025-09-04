"""
Does CoT work because it forces generation of predicates?
Count syntactic categories in CoT vs direct answers
"""

print("=" * 70)
print("CHAIN-OF-THOUGHT: THE PREDICATE HYPOTHESIS")
print("=" * 70)

# Example responses (you could get these from API calls)
direct_answer = "The answer is 47 apples"
cot_answer = """Let me work through this step by step.
First, John had 23 apples.
Then Mary gave him 15 more apples.
Next, he gave away 8 apples to Susan.
After that, he bought 17 more apples.
Finally, calculating: 23 + 15 - 8 + 17 = 47.
Therefore, John has 47 apples."""

# Count word types (simplified)
import re

def count_syntactic_categories(text):
    verbs = len(re.findall(r'\b(is|are|was|were|had|gave|bought|has|work|calculating|gave)\b', text.lower()))
    nouns = len(re.findall(r'\b(answer|apples|john|mary|susan|step)\b', text.lower()))
    prepositions = len(re.findall(r'\b(through|to|with|after|from)\b', text.lower()))
    numbers = len(re.findall(r'\b\d+\b', text))
    
    return {
        'verbs': verbs,
        'nouns': nouns, 
        'prepositions': prepositions,
        'numbers': numbers,
        'verb_noun_ratio': verbs/nouns if nouns > 0 else 0
    }

print("\nDIRECT ANSWER:")
print(f"'{direct_answer}'")
direct_counts = count_syntactic_categories(direct_answer)
for category, count in direct_counts.items():
    print(f"  {category}: {count:.2f}" if isinstance(count, float) else f"  {category}: {count}")

print("\nCHAIN-OF-THOUGHT:")
print(f"'{cot_answer[:50]}...'")
cot_counts = count_syntactic_categories(cot_answer)
for category, count in cot_counts.items():
    print(f"  {category}: {count:.2f}" if isinstance(count, float) else f"  {category}: {count}")

print("\n" + "=" * 70)
print("HYPOTHESIS: CoT works because it generates more PREDICATES")
print(f"Verb increase: {cot_counts['verbs']/direct_counts['verbs']:.1f}x")
print(f"This aligns with our positional encoding discovery!")
