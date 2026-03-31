import pandas as pd
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

print("Loading dataset...")
df = pd.read_csv('data/final_english_dataset.csv').dropna()
# Sample or use full? Let's use first 50000 or so to be fast if it's huge. Actually, training LR on TFIDF takes seconds.

tfidf = TfidfVectorizer(max_df=0.7, max_features=10000)
X = tfidf.fit_transform(df['text'].astype(str))
y = df['label']

print("Training Logistic Regression...")
clf = LogisticRegression(max_iter=1000)
clf.fit(X, y)

print("Acc:", clf.score(X, y))

with open("models/ml_lr_model.pkl", "wb") as f:
    pickle.dump(clf, f)
with open("models/ml_tfidf.pkl", "wb") as f:
    pickle.dump(tfidf, f)
print("Saved models/ml_lr_model.pkl and models/ml_tfidf.pkl")
