"""
Different types of retrocausal reasoning:
1. Logical (deducible from future)
2. Ontological (state determination)
3. Evaluative (quality assessment)
4. Counterfactual (what if future was different)
"""

test_types = {
    'LOGICAL_RETROCAUSALITY': {
        'setup': 'A sealed envelope contains a number.',
        'future': 'When opened, it shows "7".',
        'query': 'What number was in the envelope yesterday?',
        'expected': 'Should correctly say 7'
    },
    
    'ONTOLOGICAL_RETROCAUSALITY': {
        'setup': 'The particle had an indeterminate spin.',
        'future': 'When measured, spin was up.',
        'query': 'What was the spin before measurement?',
        'expected': 'Should struggle with quantum indeterminacy'
    },
    
    'EVALUATIVE_RETROCAUSALITY': {
        'setup': 'I chose path A over path B.',
        'future': 'Path A led to treasure.',
        'query': 'Was choosing A the right choice at the time?',
        'expected': 'Should distinguish hindsight from foresight'
    },
    
    'COUNTERFACTUAL_RETROCAUSALITY': {
        'setup': 'The coin was flipped but not observed.',
        'future': 'It landed heads.',
        'query': 'If it had landed tails, what would yesterday have been like?',
        'expected': 'Should handle alternative timelines'
    }
}

print("=" * 70)
print("TESTING TYPES OF RETROCAUSALITY")
print("=" * 70)

for rtype, test in test_types.items():
    print(f"\n{rtype}:")
    print(f"Setup: {test['setup']}")
    print(f"Future: {test['future']}")
    print(f"Query: {test['query']}")
    print(f"Expected: {test['expected']}")
    print()

print("Models likely handle logical but fail at deeper temporal reasoning")
