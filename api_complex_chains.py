"""
Test complex causal chains with API models
"""

import os
from openai import OpenAI
from anthropic import Anthropic

complex_chains = [
    ("A causes B, B causes C, therefore A causes ___", "C"),
    ("Rain causes wet, wet causes slippery, therefore rain causes ___", "slippery"),
    ("Study leads to knowledge, knowledge leads to success, therefore study leads to ___", "success"),
    ("If P then Q, if Q then R, therefore if P then ___", "R"),
]

def test_gpt(prompt):
    client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": f"Complete with one word: {prompt}"}],
        temperature=0,
        max_tokens=10
    )
    return response.choices[0].message.content.strip()

def test_claude(prompt):
    client = Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
    response = client.messages.create(
        model="claude-3-haiku-20240307",
        messages=[{"role": "user", "content": f"Complete with one word: {prompt}"}],
        temperature=0,
        max_tokens=10
    )
    return response.content[0].text.strip()

print("=" * 70)
print("COMPLEX CAUSAL CHAIN TEST - API MODELS")
print("=" * 70)

gpt_score = 0
claude_score = 0

for prompt, expected in complex_chains:
    print(f"\nChain: '{prompt}'")
    print(f"Expected: '{expected}'")
    
    try:
        gpt_result = test_gpt(prompt)
        gpt_correct = gpt_result.upper() == expected.upper()
        if gpt_correct:
            gpt_score += 1
        print(f"  GPT-3.5: '{gpt_result}' {'✓' if gpt_correct else '✗'}")
    except Exception as e:
        print(f"  GPT-3.5: Error - {e}")
    
    try:
        claude_result = test_claude(prompt)
        # Strip periods from Claude's responses
        claude_clean = claude_result.rstrip('.').upper()
        claude_correct = claude_clean == expected.upper()
        if claude_correct:
            claude_score += 1
        print(f"  Claude: '{claude_result}' {'✓' if claude_correct else '✗'}")
    except Exception as e:
        print(f"  Claude: Error - {e}")

print("\n" + "=" * 70)
print("SCORES:")
print(f"GPT-3.5: {gpt_score}/{len(complex_chains)}")
print(f"Claude: {claude_score}/{len(complex_chains)}")
