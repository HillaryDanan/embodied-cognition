"""
FIXED API testing for OpenAI v1.0+, Claude Haiku, Gemini 1.5
"""

import os
import json

def test_openai_gpt():
    """Test GPT with modern OpenAI library"""
    try:
        from openai import OpenAI
        
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            print("✗ No OPENAI_API_KEY found")
            return None
        
        client = OpenAI(api_key=api_key)
        
        print("\n🤖 Testing OpenAI GPT-3.5...")
        print("-" * 40)
        
        tests = [
            "The rock fell",
            "The ball rolled behind the wall and",
            "The water flowed"
        ]
        
        for prompt in tests:
            try:
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "user", "content": f"Complete this physics sentence with 2-3 words: {prompt}"}
                    ],
                    temperature=0,
                    max_tokens=10
                )
                completion = response.choices[0].message.content
                print(f"'{prompt}...' → '{completion}'")
            except Exception as e:
                print(f"Error: {e}")
        
    except ImportError:
        print("✗ Run: pip install openai")

def test_anthropic_claude():
    """Test Claude HAIKU (since you paid for it!)"""
    try:
        from anthropic import Anthropic
        
        api_key = os.getenv('ANTHROPIC_API_KEY')
        if not api_key:
            print("✗ No ANTHROPIC_API_KEY found")
            return None
        
        client = Anthropic(api_key=api_key)
        
        print("\n🤖 Testing Claude 3 Haiku...")
        print("-" * 40)
        
        tests = [
            "The rock fell",
            "The ball rolled behind the wall and", 
            "The water flowed"
        ]
        
        for prompt in tests:
            try:
                response = client.messages.create(
                    model="claude-3-haiku-20240307",  # HAIKU!
                    max_tokens=10,
                    temperature=0,
                    messages=[
                        {"role": "user", "content": f"Complete this physics sentence with 2-3 words: {prompt}"}
                    ]
                )
                completion = response.content[0].text
                print(f"'{prompt}...' → '{completion}'")
            except Exception as e:
                print(f"Error: {e}")
                
    except ImportError:
        print("✗ Run: pip install anthropic")

def test_google_gemini():
    """Test Gemini 1.5 Flash"""
    try:
        import google.generativeai as genai
        
        api_key = os.getenv('GOOGLE_API_KEY')
        if not api_key:
            print("✗ No GOOGLE_API_KEY found")
            return None
        
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')  # Updated model
        
        print("\n🤖 Testing Gemini 1.5 Flash...")
        print("-" * 40)
        
        tests = [
            "The rock fell",
            "The ball rolled behind the wall and",
            "The water flowed"
        ]
        
        for prompt in tests:
            try:
                response = model.generate_content(
                    f"Complete this physics sentence with 2-3 words: {prompt}"
                )
                completion = response.text.strip()[:30]
                print(f"'{prompt}...' → '{completion}'")
            except Exception as e:
                print(f"Error: {e}")
                
    except ImportError:
        print("✗ Run: pip install google-generativeai")

if __name__ == "__main__":
    print("=" * 60)
    print("TESTING PHYSICS COMPLETIONS ACROSS APIs")
    print("=" * 60)
    
    test_openai_gpt()
    test_anthropic_claude()
    test_google_gemini()
