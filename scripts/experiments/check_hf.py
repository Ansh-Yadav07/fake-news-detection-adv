# Run from the repo root: python scripts/experiments/check_hf.py
import os
from transformers import pipeline
local_transformer = pipeline("text-classification", model="models/transformer", tokenizer="models/transformer", max_length=512, truncation=True)
print("Real news:", local_transformer("The latest economic report released by the Bureau of Labor Statistics on Friday indicated that the United States added 250,000 jobs."))
print("Fake news:", local_transformer("BREAKING: Secret documents expose aliens on Mars!"))
