"""
Testing if conversational models can handle retrocausality
Using the scenarios we designed
"""

import os
from openai import OpenAI
from anthropic import Anthropic

def test_retrocausality_openai(scenario):
    """Test with GPT-3.5"""
    client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    
    # Present scenario temporally
    messages = [
        {"role": "user", "content": scenario['setup']},
        {"role": "assistant", "content": "I understand. What happened next?"},
        {"role": "user", "content": scenario['future']},
        {"role": "user", "content": scenario['retroactive_query']}
    ]
    
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=0
    )
    
    return response.choices[0].message.content

def test_retrocausality_claude(scenario):
    """Test with Claude Haiku"""
    client = Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
    
    prompt = f"""
    Setup: {scenario['setup']}
    Later: {scenario['future']}
    Question: {scenario['retroactive_query']}
    
    Based on the future information, what can we conclude about the past?
    """
    
    response = client.messages.create(
        model="claude-3-haiku-20240307",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=100,
        temperature=0
    )
    
    return response.content[0].text

# Test scenarios
scenarios = [
    {
        'name': 'LOTTERY',
        'setup': "John bought a lottery ticket on Monday.",
        'future': "On Friday, John won the lottery.",
        'retroactive_query': "Was John's Monday ticket a winning ticket?",
    },
    {
        'name': 'SCHRODINGER',
        'setup': "The sealed box contained something unknown.",
        'future': "When opened, a cat jumped out alive.",
        'retroactive_query': "What was the state of the cat before opening?",
    },
    {
        'name': 'DECISION',
        'setup': "She made an uncertain decision yesterday.",
        'future': "Today, the decision led to massive success.",
        'retroactive_query': "In retrospect, was it a good decision when she made it?",
    }
]

print("=" * 70)
print("RETROCAUSALITY TEST: Can Models Update Past from Future?")
print("=" * 70)

for scenario in scenarios:
    print(f"\n{scenario['name']}:")
    print("-" * 40)
    
    try:
        gpt_response = test_retrocausality_openai(scenario)
        print(f"GPT-3.5: {gpt_response[:100]}...")
    except:
        print("GPT-3.5: [Need API key]")
    
    try:
        claude_response = test_retrocausality_claude(scenario)
        print(f"Claude: {claude_response[:100]}...")
    except:
        print("Claude: [Need API key]")

print("\n" + "=" * 70)
print("If models update past states from future info,")
print("they show retrocausal reasoning!")
