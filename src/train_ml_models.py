import pandas as pd
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from preprocess import clean_text


def main():
    # 1. Load dataset
    df = pd.read_csv("data/news.csv")

    # 2. Clean text
    df["clean_text"] = df["text"].apply(clean_text)

    X = df["clean_text"]
    y = df["label"]

    # 3. TF-IDF Vectorization
    vectorizer = TfidfVectorizer(
        max_features=7000,
        ngram_range=(1, 2)
    )
    X_vec = vectorizer.fit_transform(X)

    # 4. Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_vec, y, test_size=0.2, random_state=42
    )

    # 5. Train Logistic Regression
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # 6. Evaluate model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Logistic Regression Accuracy: {accuracy:.4f}")

    # 7. Save model & vectorizer
    pickle.dump(model, open("models/logistic.pkl", "wb"))
    pickle.dump(vectorizer, open("models/tfidf.pkl", "wb"))


if __name__ == "__main__":
    main()