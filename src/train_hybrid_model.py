import numpy as np
import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from hybrid_features import get_hybrid_features

# Load data
df = pd.read_csv("data/news.csv").dropna()
df["label"] = df["label"].map({"FAKE": 0, "REAL": 1})

X = np.vstack(df["text"].apply(get_hybrid_features).values)
y = df["label"].values

# 70–20–10 split
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.10, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.2222, random_state=42)

# Train classifier
clf = LogisticRegression(max_iter=2000)
clf.fit(X_train, y_train)

# Evaluate
val_acc = accuracy_score(y_val, clf.predict(X_val))
test_acc = accuracy_score(y_test, clf.predict(X_test))

print(f"Hybrid Validation Accuracy: {val_acc*100:.2f}%")
print(f"Hybrid Test Accuracy: {test_acc*100:.2f}%")

# Save model
import os
os.makedirs("models/hybrid", exist_ok=True)
pickle.dump(clf, open("models/hybrid/hybrid_clf.pkl", "wb"))
