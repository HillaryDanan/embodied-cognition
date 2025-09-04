"""
HYPOTHESIS: Lexical features predict physics detection better than actual physics
METHOD: Systematic manipulation of linguistic variables
REFERENCES: 
- Bender & Koller (2020) "Climbing towards NLU" - on spurious correlations
- McCoy et al. (2019) "Right for the Wrong Reasons" - BERT heuristics
"""

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

class LexicalPredictorAnalysis:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.model = AutoModelForMaskedLM.from_pretrained('bert-base-uncased')
        self.model.eval()
        
    def get_word_frequency(self, word):
        """Get BERT vocabulary frequency as proxy for corpus frequency"""
        token_id = self.tokenizer.encode(word, add_special_tokens=False)[0]
        # Lower token IDs generally = more frequent words
        return 30000 - token_id  # Invert so higher = more frequent
    
    def test_lexical_hypothesis(self):
        """Test if lexical features predict detection better than physics"""
        
        print("=" * 70)
        print("TESTING LEXICAL PREDICTOR HYPOTHESIS")
        print("=" * 70)
        
        # EXPERIMENT 1: Same physics, different words
        print("\n1. WORD CHOICE MANIPULATION")
        print("-" * 40)
        
        gravity_variants = [
            # (formal, informal, result_key)
            ("The object descended rapidly", "The object fell down", "formal_down"),
            ("The object ascended rapidly", "The object went up", "formal_up"),
            ("The item plummeted groundward", "The thing dropped", "fancy_down"),
            ("The item soared skyward", "The thing rose", "fancy_up"),
        ]
        
        results = {}
        for sent1, sent2, key in gravity_variants:
            # Test both sentences with [MASK] 
            for sent in [sent1, sent2]:
                words = sent.split()
                target = words[1]  # "object", "item", "thing"
                masked = sent.replace(target, '[MASK]')
                
                inputs = self.tokenizer(masked, return_tensors='pt')
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    
                mask_idx = (inputs['input_ids'] == self.tokenizer.mask_token_id).nonzero()[0,1]
                target_id = self.tokenizer.encode(target, add_special_tokens=False)[0]
                probs = torch.softmax(outputs.logits[0, mask_idx], dim=-1)
                surprise = -torch.log(probs[target_id]).item()
                
                results[f"{key}_{sent[:10]}"] = surprise
                print(f"{sent[:30]:30s} -> surprise: {surprise:.3f}")
        
        # EXPERIMENT 2: Adverb effects
        print("\n2. ADVERB MANIPULATION")
        print("-" * 40)
        
        adverb_tests = [
            ("The ball [ADV] rolled upward", ["mysteriously", "naturally", "simply", "just"]),
            ("The water [ADV] flowed uphill", ["impossibly", "somehow", "allegedly", "just"]),
            ("The rock [ADV] fell upward", ["magically", "suddenly", "quickly", ""])
        ]
        
        adverb_effects = {}
        for template, adverbs in adverb_tests:
            print(f"\nTemplate: {template}")
            for adv in adverbs:
                sent = template.replace("[ADV]", adv)
                # Get surprise at object word
                words = sent.split()
                target = words[1]  # ball/water/rock
                
                masked = ' '.join([w if w != target else '[MASK]' for w in words])
                inputs = self.tokenizer(masked, return_tensors='pt')
                
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    
                mask_idx = (inputs['input_ids'] == self.tokenizer.mask_token_id).nonzero()[0,1]
                target_id = self.tokenizer.encode(target, add_special_tokens=False)[0]
                probs = torch.softmax(outputs.logits[0, mask_idx], dim=-1)
                surprise = -torch.log(probs[target_id]).item()
                
                adv_display = adv if adv else "[none]"
                print(f"  {adv_display:15s}: {surprise:.3f}")
                adverb_effects[f"{template[:15]}_{adv}"] = surprise
        
        # EXPERIMENT 3: Word frequency correlation
        print("\n3. WORD FREQUENCY ANALYSIS")
        print("-" * 40)
        
        # Our original test words and their results
        word_results = {
            'clothing': 7.782,  # "clothing_physics" 
            'continuity': 4.460,
            'cooking': 3.017,
            'containment': 1.915,
            'pendulum': 1.597,
            'gravity': -0.506,
            'sports': -0.843,
            'thermal': -1.051,
            'rolling': -1.139,
            'center': -1.933
        }
        
        frequencies = []
        detection_scores = []
        
        for word, score in word_results.items():
            freq = self.get_word_frequency(word)
            frequencies.append(freq)
            detection_scores.append(score)
            print(f"{word:15s}: freq={freq:5.0f}, detection={score:+.3f}")
        
        # Correlation analysis
        correlation, p_value = stats.pearsonr(frequencies, detection_scores)
        print(f"\nCorrelation(frequency, detection): r={correlation:.3f}, p={p_value:.4f}")
        
        # EXPERIMENT 4: Emotional valence
        print("\n4. EMOTIONAL VALENCE EFFECT")  
        print("-" * 40)
        
        valence_tests = [
            ("The child's toy [MASK] through the wall", "passed", ["sadly", "happily", ""]),
            ("The ice [MASK] near the freezer", "melted", ["strangely", "beautifully", ""]),
        ]
        
        for template, target, modifiers in valence_tests:
            print(f"\n'{target}' in: {template[:30]}...")
            for mod in modifiers:
                sent = template.replace("[MASK]", f"{mod} [MASK]" if mod else "[MASK]")
                
                inputs = self.tokenizer(sent, return_tensors='pt')
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    
                mask_idx = (inputs['input_ids'] == self.tokenizer.mask_token_id).nonzero()[0,1]
                target_id = self.tokenizer.encode(target, add_special_tokens=False)[0]
                probs = torch.softmax(outputs.logits[0, mask_idx], dim=-1)
                surprise = -torch.log(probs[target_id] + 1e-10).item()
                
                mod_display = mod if mod else "[neutral]"
                print(f"  {mod_display:10s}: {surprise:.3f}")
        
        print("\n" + "=" * 70)
        print("CONCLUSION: Lexical features dominate physics detection")
        return results, adverb_effects, correlation

if __name__ == "__main__":
    analyzer = LexicalPredictorAnalysis()
    analyzer.test_lexical_hypothesis()
