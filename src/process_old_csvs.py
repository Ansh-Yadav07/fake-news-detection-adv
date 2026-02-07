import pandas as pd

rows = []

# ---- True / False datasets ----
files = [
    ("data/True.csv", "REAL", "old_true"),
    ("data/False.csv", "FAKE", "old_false"),
    ("data/True 2.csv", "REAL", "old_true2"),
    ("data/Fake 2.csv", "FAKE", "old_fake2"),
]

for path, label, source in files:
    try:
        df = pd.read_csv(path)

        # Guess text column
        if "text" in df.columns:
            texts = df["text"]
        elif "content" in df.columns:
            texts = df["content"]
        else:
            continue

        for t in texts.dropna():
            rows.append({
                "text": str(t),
                "label": label,
                "source": source
            })
    except FileNotFoundError:
        pass

out_df = pd.DataFrame(rows)

# Remove duplicates (VERY IMPORTANT)
out_df.drop_duplicates(subset=["text"], inplace=True)

out_df.to_csv("data/old_news_english.csv", index=False)

print("Old CSVs processed")
print("Total samples:", len(out_df))
print(out_df["label"].value_counts())
