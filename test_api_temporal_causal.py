"""
Test if conversational models handle temporal-causal better
Using same test pairs for consistency
"""

import os
from openai import OpenAI
from anthropic import Anthropic

test_pairs = [
    ("Yesterday comes before ___", "today", "Heat melts ___", "ice"),
    ("Spring follows ___", "winter", "Rain makes things ___", "wet"),
    ("Past, present, ___", "future", "Gravity pulls objects ___", "down"),
]

def test_gpt(prompt):
    client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": f"Complete with one word: {prompt}"}],
        temperature=0,
        max_tokens=5
    )
    return response.choices[0].message.content.strip().lower()

def test_claude(prompt):
    client = Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
    response = client.messages.create(
        model="claude-3-haiku-20240307",
        messages=[{"role": "user", "content": f"Complete with one word: {prompt}"}],
        temperature=0,
        max_tokens=5
    )
    return response.content[0].text.strip().lower()

print("=" * 70)
print("API MODEL TEMPORAL-CAUSAL TEST")
print("=" * 70)

for temp_prompt, temp_ans, caus_prompt, caus_ans in test_pairs:
    print(f"\nTemporal: '{temp_prompt}'")
    print(f"Expected: '{temp_ans}'")
    
    try:
        gpt_temp = test_gpt(temp_prompt)
        print(f"  GPT-3.5: '{gpt_temp}' {'✓' if gpt_temp == temp_ans else '✗'}")
    except:
        print(f"  GPT-3.5: [API error]")
    
    try:
        claude_temp = test_claude(temp_prompt)
        print(f"  Claude: '{claude_temp}' {'✓' if claude_temp == temp_ans else '✗'}")
    except:
        print(f"  Claude: [API error]")
    
    print(f"\nCausal: '{caus_prompt}'")
    print(f"Expected: '{caus_ans}'")
    
    try:
        gpt_caus = test_gpt(caus_prompt)
        print(f"  GPT-3.5: '{gpt_caus}' {'✓' if gpt_caus == caus_ans else '✗'}")
    except:
        print(f"  GPT-3.5: [API error]")
    
    try:
        claude_caus = test_claude(caus_prompt)
        print(f"  Claude: '{claude_caus}' {'✓' if claude_caus == caus_ans else '✗'}")
    except:
        print(f"  Claude: [API error]")
