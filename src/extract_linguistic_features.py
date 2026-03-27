import pandas as pd
import numpy as np
import string
from nltk.corpus import stopwords
from tqdm import tqdm

import nltk
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download("stopwords")

stop_words = set(stopwords.words("english"))

# Load all data
df = pd.read_csv("data/final_english_dataset.csv")
df = df.dropna(subset=["text", "label"])
# df = df.iloc[:20000]

features = []

for text in tqdm(df["text"]):
    words = text.split()
    total_words = len(words)

    if total_words == 0:
        features.append([0]*8)
        continue

    exclam_count = text.count("!")
    question_count = text.count("?")
    uppercase_ratio = sum(1 for c in text if c.isupper()) / len(text)
    digit_ratio = sum(1 for c in text if c.isdigit()) / len(text)
    punctuation_ratio = sum(1 for c in text if c in string.punctuation) / len(text)
    stopword_ratio = sum(1 for w in words if w.lower() in stop_words) / total_words
    avg_word_length = np.mean([len(w) for w in words])
    text_length = len(text)

    features.append([
        exclam_count,
        question_count,
        uppercase_ratio,
        digit_ratio,
        punctuation_ratio,
        stopword_ratio,
        avg_word_length,
        text_length
    ])

features = np.array(features)

np.save("data/linguistic_features.npy", features)

print("Linguistic features saved.")
