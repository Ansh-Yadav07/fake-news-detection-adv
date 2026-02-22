import numpy as np
import pandas as pd
import pickle
import os

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

from hybrid_features import get_hybrid_features

# ==============================
# 1. Load Data
# ==============================
df = pd.read_csv("data/final_english_dataset.csv").dropna()
df["label"] = df["label"].map({"FAKE": 0, "REAL": 1})

print("Total samples:", len(df))
print(df["label"].value_counts())

# ==============================
# 2. Extract Hybrid Features
# ==============================
print("\nExtracting hybrid features...")
X = np.vstack(df["text"].apply(get_hybrid_features).values)
y = df["label"].values

print("Feature shape:", X.shape)

# ==============================
# 3. Scale Features
# ==============================
scaler = StandardScaler()
X = scaler.fit_transform(X)

# ==============================
# 4. 70–20–10 Stratified Split
# ==============================
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y,
    test_size=0.10,
    random_state=42,
    stratify=y
)

X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp,
    test_size=0.2222,  # 20% of total
    random_state=42,
    stratify=y_temp
)

# ==============================
# 5. Train Classifier (Balanced)
# ==============================
clf = LogisticRegression(
    max_iter=2000,
    class_weight="balanced"
)

print("\nTraining hybrid classifier...")
clf.fit(X_train, y_train)

# ==============================
# 6. Evaluation
# ==============================
y_val_pred = clf.predict(X_val)
y_test_pred = clf.predict(X_test)

val_acc = accuracy_score(y_val, y_val_pred)
test_acc = accuracy_score(y_test, y_test_pred)

print(f"\nHybrid Validation Accuracy: {val_acc*100:.2f}%")
print(f"Hybrid Test Accuracy: {test_acc*100:.2f}%")

print("\nHybrid Classification Report (Test):")
print(classification_report(
    y_test,
    y_test_pred,
    target_names=["FAKE", "REAL"]
))

# ==============================
# 7. Save Model + Scaler
# ==============================
os.makedirs("models/hybrid", exist_ok=True)

pickle.dump(clf, open("models/hybrid/hybrid_clf.pkl", "wb"))
pickle.dump(scaler, open("models/hybrid/scaler.pkl", "wb"))

print("\n✅ Hybrid model and scaler saved successfully.")