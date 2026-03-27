import torch
import numpy as np
import sys
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import torch.nn.functional as F

# Load model + tokenizer
model_path = "models/transformer"

tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)
model = DistilBertForSequenceClassification.from_pretrained(model_path)

model.eval()

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model.to(device)

def predict(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    probs = F.softmax(logits, dim=1)

    predicted_class = torch.argmax(probs, dim=1).item()
    confidence = probs[0][predicted_class].item()

    label_map = {0: "FAKE", 1: "REAL"}

    return label_map[predicted_class], confidence


if __name__ == "__main__":
    if len(sys.argv) > 1:
        text = " ".join(sys.argv[1:])
        label, confidence = predict(text)
        print(f"\nPrediction: {label}")
        print(f"Confidence: {confidence:.2%}")
    else:
        while True:
            try:
                text = input("\nEnter news text (or type 'exit'): ")
                if not text or text.lower().strip() == "exit":
                    break
            except EOFError:
                break

            label, confidence = predict(text)
            print(f"\nPrediction: {label}")
            print(f"Confidence: {confidence:.2%}")
    