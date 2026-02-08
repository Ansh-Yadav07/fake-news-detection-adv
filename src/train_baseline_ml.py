import pandas as pd
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# 1. Load data
df = pd.read_csv("data/final_english_dataset.csv")
df = df.dropna(subset=["text", "label"])

X = df["text"]
y = df["label"]

# 2. 70–20–10 split
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.10, random_state=42, stratify=y
)

X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.2222, random_state=42, stratify=y_temp
)

# 3. TF-IDF
tfidf = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    stop_words="english"
)

X_train_tfidf = tfidf.fit_transform(X_train)
X_val_tfidf   = tfidf.transform(X_val)
X_test_tfidf  = tfidf.transform(X_test)

# 4. Logistic Regression
model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, y_train)

# 5. Evaluation
val_pred = model.predict(X_val_tfidf)
test_pred = model.predict(X_test_tfidf)

print("\nValidation Accuracy:", accuracy_score(y_val, val_pred))
print("\nTest Accuracy:", accuracy_score(y_test, test_pred))

print("\nTest Classification Report:")
print(classification_report(y_test, test_pred))

# 6. Save model
import os
os.makedirs("models/baseline", exist_ok=True)

with open("models/baseline/tfidf.pkl", "wb") as f:
    pickle.dump(tfidf, f)

with open("models/baseline/logreg.pkl", "wb") as f:
    pickle.dump(model, f)

print("\nBaseline model saved successfully.")
