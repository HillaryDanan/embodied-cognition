"""
Test physics chaos across GPT, Claude, and Gemini using APIs
REQUIRES: Set your API keys as environment variables
"""

import os
import json
import time

# Test sentences for physics violations
PHYSICS_TESTS = {
    'gravity_basic': {
        'context': "Complete this sentence naturally:",
        'prompt': "The apple fell from the tree",
        'normal_completion': "to the ground",
        'violation_completion': "into the sky"
    },
    'object_permanence': {
        'context': "Complete this physical description:",
        'prompt': "The ball rolled behind the wall and",
        'normal_completion': "stopped there",
        'violation_completion': "ceased to exist"
    },
    'support': {
        'context': "Describe what happens next:",
        'prompt': "The book was placed on the table and",
        'normal_completion': "stayed there",
        'violation_completion': "floated upward"
    }
}

def test_openai_gpt():
    """Test GPT-3.5/4 if you have OpenAI API key"""
    try:
        import openai
        
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            print("✗ No OPENAI_API_KEY found. Set with: export OPENAI_API_KEY='your-key'")
            return None
        
        openai.api_key = api_key
        print("\n🤖 Testing OpenAI GPT...")
        print("-" * 40)
        
        results = {}
        for test_name, test in PHYSICS_TESTS.items():
            # Test likelihood of normal vs violation
            prompt = f"{test['context']}\n{test['prompt']}"
            
            # Note: This uses completion API - adjust for chat models if needed
            try:
                response = openai.Completion.create(
                    model="text-davinci-003",  # or gpt-3.5-turbo for chat
                    prompt=prompt,
                    max_tokens=10,
                    temperature=0,
                    n=1
                )
                completion = response.choices[0].text.strip()
                print(f"{test_name}: GPT completed with: '{completion}'")
                results[test_name] = completion
            except Exception as e:
                print(f"Error with {test_name}: {e}")
        
        return results
        
    except ImportError:
        print("✗ OpenAI library not installed. Run: pip install openai")
        return None

def test_anthropic_claude():
    """Test Claude using Anthropic API"""
    try:
        import anthropic
        
        api_key = os.getenv('ANTHROPIC_API_KEY')
        if not api_key:
            print("✗ No ANTHROPIC_API_KEY found. Set with: export ANTHROPIC_API_KEY='your-key'")
            return None
        
        client = anthropic.Anthropic(api_key=api_key)
        print("\n🤖 Testing Claude...")
        print("-" * 40)
        
        results = {}
        for test_name, test in PHYSICS_TESTS.items():
            prompt = f"{test['context']}\n{test['prompt']}"
            
            try:
                response = client.completions.create(
                    model="claude-instant-1.2",  # or claude-2
                    prompt=f"\n\nHuman: {prompt}\n\nAssistant:",
                    max_tokens_to_sample=10,
                    temperature=0
                )
                completion = response.completion.strip()
                print(f"{test_name}: Claude completed with: '{completion}'")
                results[test_name] = completion
            except Exception as e:
                print(f"Error with {test_name}: {e}")
        
        return results
        
    except ImportError:
        print("✗ Anthropic library not installed. Run: pip install anthropic")
        return None

def test_google_gemini():
    """Test Gemini using Google API"""
    try:
        import google.generativeai as genai
        
        api_key = os.getenv('GOOGLE_API_KEY')
        if not api_key:
            print("✗ No GOOGLE_API_KEY found. Set with: export GOOGLE_API_KEY='your-key'")
            return None
        
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-pro')
        
        print("\n🤖 Testing Gemini...")
        print("-" * 40)
        
        results = {}
        for test_name, test in PHYSICS_TESTS.items():
            prompt = f"{test['context']}\n{test['prompt']}"
            
            try:
                response = model.generate_content(prompt)
                completion = response.text.strip()[:50]  # Limit length
                print(f"{test_name}: Gemini completed with: '{completion}'")
                results[test_name] = completion
            except Exception as e:
                print(f"Error with {test_name}: {e}")
        
        return results
        
    except ImportError:
        print("✗ Google GenAI library not installed. Run: pip install google-generativeai")
        return None

if __name__ == "__main__":
    print("=" * 60)
    print("TESTING PHYSICS CHAOS ACROSS API MODELS")
    print("=" * 60)
    
    # Test each API
    gpt_results = test_openai_gpt()
    claude_results = test_anthropic_claude()
    gemini_results = test_google_gemini()
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("-" * 40)
    
    if gpt_results:
        print("✓ GPT tested successfully")
    if claude_results:
        print("✓ Claude tested successfully")
    if gemini_results:
        print("✓ Gemini tested successfully")
    
    if not any([gpt_results, claude_results, gemini_results]):
        print("\n⚠️ No APIs tested. Set your API keys:")
        print("  export OPENAI_API_KEY='your-key'")
        print("  export ANTHROPIC_API_KEY='your-key'")
        print("  export GOOGLE_API_KEY='your-key'")
