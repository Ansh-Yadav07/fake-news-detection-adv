import pandas as pd 
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
try:
    from preprocess import clean_text
except ImportError:
    import re
    def clean_text(text):
        return text.lower()

import os

def main():
    # 1. Load Data
    data_path = 'data/news.csv'
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found. Please run src/create_dataset.py first.")
        return

    print("Loading dataset...")
    df = pd.read_csv(data_path)
    
    # Handle missing values if any
    df = df.dropna()

    # 2. Preprocess
    print("Cleaning text (this may take a moment)...")
    # Apply cleaning - ensuring text is string
    df['text'] = df['text'].astype(str).apply(clean_text)

    # 3. Split Data (60% Train, 30% Validation, 10% Test)
    # First split: 10% for Test, 90% for Train+Val
    X_temp, X_test, y_temp, y_test = train_test_split(
        df['text'], df['label'], test_size=0.10, random_state=42
    )

    # Second split: From the 90% remaining, we want 30% of TOTAL to be Validation.
    # 30% of total is (30/90) = 1/3 = 0.3333... of the temp data.
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.3333, random_state=42
    )

    # 4. Vectorize
    print("Vectorizing...")
    tfidf_vectorizer = TfidfVectorizer(max_df=0.7)

    tfidf_train = tfidf_vectorizer.fit_transform(X_train)
    tfidf_val = tfidf_vectorizer.transform(X_val)
    tfidf_test = tfidf_vectorizer.transform(X_test)

    models = {
        "Passive Aggressive": PassiveAggressiveClassifier(max_iter=50),
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Naive Bayes": MultinomialNB()
    }

    results = []
    best_model = None
    best_model_name = None
    best_test_score = 0.0

    print("\nTraining and evaluating models...")

    for name, model in models.items():
        model.fit(tfidf_train, y_train)

        val_pred = model.predict(tfidf_val)
        val_score = accuracy_score(y_val, val_pred)

        test_pred = model.predict(tfidf_test)
        test_score = accuracy_score(y_test, test_pred)

        results.append((name, val_score, test_score))

        print(f"{name:20s} | Validation: {val_score*100:.2f}% | Test: {test_score*100:.2f}%")

        if test_score > best_test_score:
            best_test_score = test_score
            best_model = model
            best_model_name = name

    print(f"\nBest Model (by Test Accuracy): {best_model_name}")
    print(f"Best Test Accuracy: {best_test_score*100:.2f}%")

    best_test_pred = best_model.predict(tfidf_test)

    print("\nClassification Report (Best Model):")
    print(classification_report(y_test, best_test_pred))

    print("\nConfusion Matrix (Best Model):")
    print(confusion_matrix(y_test, best_test_pred))
    
    # 7. Save best model and vectorizer
    os.makedirs("models", exist_ok=True)

    with open("models/best_model.pkl", "wb") as f:
        pickle.dump(best_model, f)

    with open("models/tfidf.pkl", "wb") as f:
        pickle.dump(tfidf_vectorizer, f)

    print("\nBest model and TF-IDF vectorizer saved.")

if __name__ == "__main__":
    main()
