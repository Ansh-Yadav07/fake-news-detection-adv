import pandas as pd

true_df = pd.read_csv("data/True.csv")
fake_df = pd.read_csv("data/Fake.csv")

true_df["label"] = "REAL"
fake_df["label"] = "FAKE"

df = pd.concat([true_df, fake_df], axis=0)
df = df[["text", "label"]]

df.to_csv("data/news.csv", index=False)

print("news.csv created successfully")
