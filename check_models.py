"""
Check which models are available locally
"""

from transformers import AutoModelForMaskedLM, AutoTokenizer
import os

print("=" * 60)
print("CHECKING LOCALLY AVAILABLE MODELS")
print("=" * 60)

# Models to check
MODELS_TO_CHECK = [
    'bert-base-uncased',
    'roberta-base',
    'albert-base-v2',
    'distilbert-base-uncased',
    'bert-large-uncased'
]

# Check cache directory
cache_dir = os.path.expanduser("~/.cache/huggingface/hub/")
print(f"\nHuggingFace cache directory: {cache_dir}")

if os.path.exists(cache_dir):
    cached = os.listdir(cache_dir)
    print(f"Found {len(cached)} items in cache")
else:
    print("Cache directory doesn't exist yet")

print("\nChecking specific models:")
print("-" * 40)

available = []
for model_name in MODELS_TO_CHECK:
    try:
        # Try to load just the tokenizer first (smaller)
        tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
        print(f"✓ {model_name}: AVAILABLE LOCALLY")
        available.append(model_name)
    except:
        print(f"✗ {model_name}: Not downloaded")

print("\n" + "=" * 60)
if available:
    print(f"You have {len(available)} models ready to test locally!")
else:
    print("No models downloaded yet. Models will download on first use (~450MB each)")
    print("Run 'python3 test_other_models.py' to auto-download and test")
