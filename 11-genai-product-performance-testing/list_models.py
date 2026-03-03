#!/usr/bin/env python3
"""
List all available models from OpenAI and Google Gemini
"""

import os

# API keys should be set as environment variables before running:
# export OPENAI_API_KEY="your-key-here"
# export GOOGLE_API_KEY="your-key-here"

print("=" * 80)
print("AVAILABLE OPENAI MODELS")
print("=" * 80)

try:
    from openai import OpenAI
    client = OpenAI()

    models = client.models.list()

    # Filter for GPT and O1 models
    relevant_models = []
    for model in models.data:
        model_id = model.id
        if any(x in model_id.lower() for x in ['gpt', 'o1', 'chatgpt']):
            relevant_models.append(model_id)

    relevant_models.sort()

    print("\nChatGPT and O1 models:")
    for model in relevant_models:
        print(f"  - {model}")

    print(f"\nTotal models found: {len(relevant_models)}")

except Exception as e:
    print(f"Error listing OpenAI models: {e}")

print("\n" + "=" * 80)
print("AVAILABLE GEMINI MODELS")
print("=" * 80)

try:
    import google.generativeai as genai

    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

    models = genai.list_models()

    print("\nAll Gemini models:")
    for model in models:
        # Only show models that support generateContent
        if 'generateContent' in model.supported_generation_methods:
            print(f"  - {model.name}")
            print(f"    Display Name: {model.display_name}")
            print(f"    Description: {model.description[:80]}..." if len(model.description) > 80 else f"    Description: {model.description}")
            print()

except Exception as e:
    print(f"Error listing Gemini models: {e}")

print("=" * 80)
