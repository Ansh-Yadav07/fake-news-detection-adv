import pandas as pd

files = [
    "data/news_english.csv",
    "data/liar_english.csv",
    "data/json_real_english.csv",
    "data/old_news_english.csv"
]

dfs = []

for file in files:
    df = pd.read_csv(file)
    dfs.append(df)

# Merge everything
final_df = pd.concat(dfs, axis=0)

# Safety cleaning
final_df.dropna(subset=["text", "label"], inplace=True)
final_df.drop_duplicates(subset=["text"], inplace=True)

# Save final dataset
final_df.to_csv("data/final_english_dataset.csv", index=False)

print("Final dataset created")
print("Total samples:", len(final_df))
print(final_df["label"].value_counts())
