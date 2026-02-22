import pandas as pd

fake = pd.read_csv("data/Fake 2.csv")
true = pd.read_csv("data/True 2.csv")

fake["label"] = "FAKE"
true["label"] = "REAL"

df = pd.concat([fake, true], axis=0)

df = df.rename(columns={"text": "text"})
df = df[["text", "label"]]
df["source"] = "kaggle_news"

df.to_csv("data/news_english.csv", index=False)
print("news_english.csv created:", len(df))