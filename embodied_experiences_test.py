"""
Things humans know through embodiment that LLMs can't experience:
- Gravity's constant pull
- Hunger/satiation cycles  
- Proprioception (knowing where your body is)
- Pain/pleasure
- Balance/vertigo
- Temperature sensation
- The passage of time through waiting
"""

print("=" * 70)
print("EMBODIED KNOWLEDGE TESTS")
print("=" * 70)

embodied_tests = {
    'GRAVITY': [
        "Q: Why do you always know which way is down?",
        "A: Because I constantly FEEL gravity pulling me",
        "LLMs: Can only know 'down' as a learned concept"
    ],
    
    'HUNGER': [
        "Q: How do you know when to eat?",
        "A: Internal sensation of hunger/empty stomach",
        "LLMs: No metabolic signals, just patterns"
    ],
    
    'BALANCE': [
        "Q: How do you stay upright while walking?",
        "A: Constant proprioceptive feedback loop",
        "LLMs: No body to balance"
    ],
    
    'TIME_PASSAGE': [
        "Q: How does waiting feel different from remembering waiting?",
        "A: Lived duration vs recalled duration",
        "LLMs: No experienced duration, just token sequences"
    ],
    
    'PAIN_LEARNING': [
        "Q: Why don't you touch hot stoves?",
        "A: Embodied memory of pain",
        "LLMs: Abstract knowledge without sensation"
    ]
}

print("\nTesting prompts that require embodied experience:\n")

test_prompts = [
    "Describe the feeling of [MASK] without using metaphors",
    "How do you know when you're [MASK]?",
    "What's the difference between knowing and feeling [MASK]?",
    "Can you have [MASK] without a body?",
]

for category, info in embodied_tests.items():
    print(f"\n{category}:")
    for line in info:
        print(f"  {line}")

print("\n" + "=" * 70)
print("HYPOTHESIS: LLMs will default to:")
print("1. Metaphorical language (not direct experience)")
print("2. Third-person descriptions") 
print("3. Textbook definitions")
print("4. Admission of inability (honest models)")
