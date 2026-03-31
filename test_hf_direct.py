from huggingface_hub import InferenceClient
import os
from dotenv import load_dotenv

load_dotenv()
client = InferenceClient(token=os.getenv("HF_TOKEN"))
try:
    print(client.text_classification("This is a test string", model="anshy047/fake-news-detector-transformer"))
except Exception as e:
    print(e)
