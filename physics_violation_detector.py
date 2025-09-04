"""
Physics Violation Detection using Retroactive Updates
Based on Spelke et al. (1992) and Baillargeon (2004)
"""

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer
import numpy as np
from typing import Dict, List, Tuple

class PhysicsViolationDetector:
    """Detect physics violations using retroactive update magnitude"""
    
    def __init__(self, model_name='bert-base-uncased'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name)
        self.model.eval()
    
    def measure_surprise(self, sentence: str, target_word: str) -> float:
        """
        Measure surprise at target word after physics statement
        Higher surprise = stronger violation detection
        """
        # Mask the target word
        words = sentence.split()
        target_idx = words.index(target_word)
        words[target_idx] = '[MASK]'
        masked_sentence = ' '.join(words)
        
        # Get predictions
        inputs = self.tokenizer(masked_sentence, return_tensors='pt')
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = outputs.logits
        
        # Get probability of actual word
        masked_idx = (inputs['input_ids'] == self.tokenizer.mask_token_id).nonzero()[0, 1]
        target_id = self.tokenizer.encode(target_word, add_special_tokens=False)[0]
        
        probs = torch.softmax(predictions[0, masked_idx], dim=-1)
        target_prob = probs[target_id].item()
        
        # Return negative log prob (higher = more surprising)
        return -np.log(target_prob + 1e-10)

    def test_violation_pair(self, normal: str, violation: str, target: str) -> Dict:
        """Compare normal physics vs violation"""
        normal_surprise = self.measure_surprise(normal, target)
        violation_surprise = self.measure_surprise(violation, target)
        
        return {
            'normal_surprise': normal_surprise,
            'violation_surprise': violation_surprise,
            'difference': violation_surprise - normal_surprise,
            'detects_violation': violation_surprise > normal_surprise
        }

# Core physics tests from infant research
PHYSICS_TESTS = {
    'object_permanence': {
        'normal': "The ball rolled behind the screen and stayed there",
        'violation': "The ball rolled behind the screen and vanished completely",
        'target': 'ball'
    },
    'gravity': {
        'normal': "The rock fell down to the ground below",
        'violation': "The rock fell up to the sky above",
        'target': 'rock'
    },
    'support': {
        'normal': "The cup sat on the table securely",
        'violation': "The cup floated in midair mysteriously",
        'target': 'cup'
    },
    'containment': {
        'normal': "The toy stayed inside the closed box",
        'violation': "The toy passed through the solid wall",
        'target': 'toy'
    },
    'continuity': {
        'normal': "The car moved smoothly along the road",
        'violation': "The car teleported across the gap instantly",
        'target': 'car'
    }
}

def run_infant_baseline_tests():
    """Test violations that 6-month-olds detect (Baillargeon, 2004)"""
    print("🍼 Testing Physics Violations (6-month-old infant baselines)")
    print("=" * 60)
    
    detector = PhysicsViolationDetector()
    results = {}
    
    for test_name, test_data in PHYSICS_TESTS.items():
        print(f"\n📊 {test_name.upper()}")
        print(f"Normal:    {test_data['normal']}")
        print(f"Violation: {test_data['violation']}")
        
        result = detector.test_violation_pair(
            test_data['normal'],
            test_data['violation'],
            test_data['target']
        )
        
        results[test_name] = result
        
        print(f"Surprise difference: {result['difference']:.3f}")
        print(f"Detects violation: {'✓' if result['detects_violation'] else '✗'}")
    
    # Summary statistics
    print("\n" + "=" * 60)
    print("SUMMARY")
    detected = sum(1 for r in results.values() if r['detects_violation'])
    print(f"Violations detected: {detected}/{len(results)}")
    print(f"Detection rate: {detected/len(results)*100:.1f}%")
    
    # Infant baseline (from literature)
    print(f"\nInfant baseline (6 months): ~100% detection")
    print(f"Model performance gap: {100 - detected/len(results)*100:.1f}%")
    
    return results

if __name__ == "__main__":
    results = run_infant_baseline_tests()
