import os
import numpy as np
import pandas as pd

from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments
)

# ==============================
# 1. Load final unified dataset
# ==============================
df = pd.read_csv("data/final_english_dataset.csv")
df = df.dropna(subset=["text", "label"])
df["label"] = df["label"].map({"FAKE": 0, "REAL": 1})

print("Total samples:", len(df))
print(df["label"].value_counts())

# ==============================
# 2. 70–20–10 split
# ==============================
train_df, test_df = train_test_split(
    df,
    test_size=0.10,
    stratify=df["label"],
    random_state=42
)

train_df, val_df = train_test_split(
    train_df,
    test_size=0.2222,   # 20% of total
    stratify=train_df["label"],
    random_state=42
)

# ==============================
# 3. Convert to HuggingFace Dataset
# ==============================
train_ds = Dataset.from_pandas(train_df[["text", "label"]])
val_ds   = Dataset.from_pandas(val_df[["text", "label"]])
test_ds  = Dataset.from_pandas(test_df[["text", "label"]])

# ==============================
# 4. Tokenizer
# ==============================
tokenizer = DistilBertTokenizerFast.from_pretrained(
    "distilbert-base-uncased"
)

def tokenize(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=256
    )

train_ds = train_ds.map(tokenize, batched=True)
val_ds   = val_ds.map(tokenize, batched=True)
test_ds  = test_ds.map(tokenize, batched=True)

cols = ["input_ids", "attention_mask", "label"]
train_ds.set_format("torch", columns=cols)
val_ds.set_format("torch", columns=cols)
test_ds.set_format("torch", columns=cols)

# ==============================
# 5. Model
# ==============================

model = DistilBertForSequenceClassification.from_pretrained(
    "models/transformer"
)

# ==============================
# 6. Minimal training args (for evaluation only)
# ==============================
training_args = TrainingArguments(
    output_dir="models/transformer",
    per_device_eval_batch_size=8,
    disable_tqdm=False,
    report_to="none"
)

# ==============================
# 7. Create trainer (no training dataset needed)
# ==============================
trainer = Trainer(
    model=model,
    args=training_args
)
# ==============================
# 8. Evaluate ONLY on test set
# ==============================
predictions = trainer.predict(test_ds)

from sklearn.metrics import classification_report

preds = np.argmax(predictions.predictions, axis=1)
labels = predictions.label_ids

print("\nClassification Report:")
print(classification_report(labels, preds, target_names=["FAKE", "REAL"]))