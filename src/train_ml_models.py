import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
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

    # 3. Split Data
    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)

    # 4. Vectorize
    print("Vectorizing...")
    tfidf_vectorizer = TfidfVectorizer(max_df=0.7)
    tfidf_train = tfidf_vectorizer.fit_transform(X_train) 
    tfidf_test = tfidf_vectorizer.transform(X_test)

    # 5. Train Model
    print("Training model...")
    pac = PassiveAggressiveClassifier(max_iter=50)
    pac.fit(tfidf_train, y_train)

    # 6. Evaluate
    y_pred = pac.predict(tfidf_test)
    score = accuracy_score(y_test, y_pred)
    print(f'\nAccuracy: {round(score*100,2)}%')
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

if __name__ == "__main__":
    main()
