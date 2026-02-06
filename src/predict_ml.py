import pickle
from preprocess import clean_text

def predict_news(text):
    # Load trained model & vectorizer
    with open("models/best_model.pkl", "rb") as f:
        model = pickle.load(f)

    with open("models/tfidf.pkl", "rb") as f:
        vectorizer = pickle.load(f)

    # Preprocess input
    cleaned_text = clean_text(text)

    # Vectorize
    text_vector = vectorizer.transform([cleaned_text])

    # Predict
    prediction = model.predict(text_vector)[0]

    return prediction


if __name__ == "__main__":
    print("=== Fake News Detector ===")
    user_input = input("\nEnter news text:\n")

    result = predict_news(user_input)

    print("\nPrediction:", result)
    if result == "REAL":
        print("This news is likely REAL.")
    else:
        print("This news is likely FAKE.")