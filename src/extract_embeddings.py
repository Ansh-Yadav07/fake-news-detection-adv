import torch
import pandas as pd
import numpy as np
from transformers import DistilBertTokenizerFast, DistilBertModel
from tqdm import tqdm

# Load dataset
df = pd.read_csv("data/final_english_dataset.csv")
df = df.dropna(subset=["text", "label"])
df["label"] = df["label"].map({"FAKE": 0, "REAL": 1})

# Load tokenizer + model (not classification head)
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
model = DistilBertModel.from_pretrained("models/transformer")

model.eval()
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model.to(device)

embeddings = []

for text in tqdm(df["text"]):  # process all data
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=256
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    cls_embedding = outputs.last_hidden_state[:, 0, :]
    embeddings.append(cls_embedding.cpu().numpy().flatten())

embeddings = np.array(embeddings)

np.save("data/bert_embeddings.npy", embeddings)
df[["label"]].to_csv("data/embedding_labels.csv", index=False)

print("Embeddings saved successfully.")
