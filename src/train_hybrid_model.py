import os
import numpy as np
import pandas as pd
import pickle

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

def main():
    # ==============================
    # 1. Load Precomputed Data
    # ==============================
    print("Loading precomputed features...")
    try:
        bert_embeddings = np.load("data/bert_embeddings.npy")
        linguistic_features = np.load("data/linguistic_features.npy")
        labels_df = pd.read_csv("data/embedding_labels.csv")
    except FileNotFoundError as e:
        print(f"Error loading files: {e}")
        print("Please ensure 'data/bert_embeddings.npy', 'data/linguistic_features.npy', and 'data/embedding_labels.csv' exist.")
        return

    # Ensure all have the same number of samples
    n_samples = min(len(bert_embeddings), len(linguistic_features), len(labels_df))
    
    # Truncate to the minimum length found to ensure safe alignment
    bert_embeddings = bert_embeddings[:n_samples]
    linguistic_features = linguistic_features[:n_samples]
    y = labels_df["label"].values[:n_samples]
    
    print(f"\nDataset Size: {n_samples} samples")

    # ==============================
    # 2. Combine Features
    # ==============================
    print("Combining features...")
    if len(linguistic_features.shape) == 1:
        linguistic_features = linguistic_features.reshape(-1, 1)

    # Stack BERT features (Nx768) and Linguistic features (Nx8) -> (Nx776)
    X = np.hstack((bert_embeddings, linguistic_features))

    print(f"Feature shape: {X.shape}")
    
    unique_labels, counts = np.unique(y, return_counts=True)
    print("Label distribution:")
    for label, count in zip(unique_labels, counts):
        label_name = "REAL" if label == 1 else "FAKE"
        print(f"  {label_name} ({label}): {count}")

    # ==============================
    # 3. 70-20-10 Stratified Split
    # ==============================
    print("\nSplitting dataset into 70% Train, 20% Val, 10% Test...")
    # First split into Train+Val (90%) and Test (10%)
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y,
        test_size=0.10,
        random_state=42,
        stratify=y
    )

    # Split Train+Val (90%) into Train (70% of total) and Val (20% of total)
    # The proportion of val relative to temp is 20/90
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=(0.20 / 0.90),
        random_state=42,
        stratify=y_temp
    )

    print(f"Split Sizes -> Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # ==============================
    # 4. Scale Features (NO LEAKAGE)
    # ==============================
    print("\nApplying StandardScaler (fit on train only)...")
    scaler = StandardScaler()
    
    # Fit ONLY on training data, then transform all
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # ==============================
    # 5. Train Classifier
    # ==============================
    print("\nTraining Logistic Regression model...")
    clf = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=42
    )
    clf.fit(X_train, y_train)

    # ==============================
    # 6. Evaluate Model
    # ==============================
    print("\nEvaluating model...")
    y_val_pred = clf.predict(X_val)
    y_test_pred = clf.predict(X_test)

    val_acc = accuracy_score(y_val, y_val_pred)
    test_acc = accuracy_score(y_test, y_test_pred)

    print(f"\nValidation Accuracy: {val_acc * 100:.2f}%")
    print(f"Test Accuracy: {test_acc * 100:.2f}%")

    print("\nClassification Report (Test Data):")
    label_names = ["FAKE", "REAL"]
    print(classification_report(y_test, y_test_pred, target_names=label_names))

    print("Confusion Matrix (Test Data):")
    print(confusion_matrix(y_test, y_test_pred))

    # ==============================
    # 7. Save Model & Scaler
    # ==============================
    os.makedirs("models/hybrid", exist_ok=True)

    clf_path = "models/hybrid/hybrid_clf.pkl"
    scaler_path = "models/hybrid/scaler.pkl"
    
    with open(clf_path, "wb") as f:
        pickle.dump(clf, f)
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)

    print(f"\n✅ Model saved to {clf_path}")
    print(f"✅ Scaler saved to {scaler_path}")

if __name__ == "__main__":
    main()
