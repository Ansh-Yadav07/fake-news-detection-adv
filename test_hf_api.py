#!/usr/bin/env python3
from huggingface_hub import InferenceClient
import os

# Set token if available
HF_TOKEN = os.environ.get("HF_TOKEN")
print(f"Token present: {bool(HF_TOKEN)}")

# Try to initialize client
try:
    client = InferenceClient(api_key=HF_TOKEN)
    print("✓ Client initialized")
    
    # Try classification with our model
    print("\nTesting with fake-news-detector model...")
    result = client.text_classification("This is a test", model="anshy047/fake-news-detector-transformer")
    print(f"✓ Classification result: {result}")
except Exception as e:
    print(f"✗ Error: {type(e).__name__}: {str(e)}")

# Also try with a known public model
print("\n\nTesting with public model...")
try:
    client2 = InferenceClient(api_key=HF_TOKEN)
    result2 = client2.text_classification("This is great", model="distilbert-base-uncased-finetuned-sst-2-english")
    print(f"✓ Public model works: {result2}")
except Exception as e:
    print(f"✗ Error with public model: {type(e).__name__}: {str(e)}")
