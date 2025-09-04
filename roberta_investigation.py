"""
Why does RoBERTa handle temporal (80%) but fail causal (20%)?
Hypothesis: Different pretraining? Tokenization? Architecture?
"""

from transformers import RobertaTokenizer, RobertaForMaskedLM, BertTokenizer, BertForMaskedLM
import torch

print("=" * 70)
print("ROBERTA VS BERT: WHY THE DIFFERENCE?")
print("=" * 70)

# Load both models
roberta_tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
roberta_model = RobertaForMaskedLM.from_pretrained('roberta-base')

bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertForMaskedLM.from_pretrained('bert-base-uncased')

# Key differences to test
print("\n1. TRAINING DATA DIFFERENCES:")
print("  BERT: BookCorpus + Wikipedia (2018)")
print("  RoBERTa: BookCorpus + Wikipedia + CC-News + OpenWebText + Stories (2019)")
print("  → RoBERTa saw 10x more data, including news (temporal sequences!)")

print("\n2. PREPROCESSING DIFFERENCES:")
print("  BERT: Next Sentence Prediction + MLM")
print("  RoBERTa: Only MLM (no NSP)")
print("  → RoBERTa doesn't learn sentence boundaries as strongly")

# Test specific patterns
test_sentences = [
    ("Yesterday, today, <mask>", "[MASK]", "tomorrow"),
    ("Cause and <mask>", "[MASK]", "effect"),
    ("Before and <mask>", "[MASK]", "after"),
]

print("\n3. TOKENIZATION TEST:")
for roberta_sent, bert_sent, expected in test_sentences:
    print(f"\n'{expected}' completion:")
    
    # RoBERTa
    rob_inputs = roberta_tokenizer(roberta_sent, return_tensors='pt')
    with torch.no_grad():
        rob_outputs = roberta_model(**rob_inputs)
    rob_mask_idx = (rob_inputs['input_ids'] == roberta_tokenizer.mask_token_id).nonzero()[0,1]
    rob_probs = torch.softmax(rob_outputs.logits[0, rob_mask_idx], dim=-1)
    rob_top = roberta_tokenizer.decode([rob_probs.argmax().item()])
    
    # BERT
    bert_inputs = bert_tokenizer(bert_sent, return_tensors='pt')
    with torch.no_grad():
        bert_outputs = bert_model(**bert_inputs)
    bert_mask_idx = (bert_inputs['input_ids'] == bert_tokenizer.mask_token_id).nonzero()[0,1]
    bert_probs = torch.softmax(bert_outputs.logits[0, bert_mask_idx], dim=-1)
    bert_top = bert_tokenizer.decode([bert_probs.argmax().item()])
    
    print(f"  RoBERTa: '{rob_top.strip()}'")
    print(f"  BERT: '{bert_top.strip()}'")

print("\n" + "=" * 70)
print("HYPOTHESIS: RoBERTa's news corpus training gave it")
print("temporal sequence patterns that BERT lacks")
