import pandas as pd

FILES = {
    "train": "data/train.tsv",
    "valid": "data/valid.tsv",
    "test":  "data/test.tsv"
}

label_map = {
    "pants-fire": "FAKE",
    "false": "FAKE",
    "barely-true": "FAKE",
    "half-true": "FAKE",
    "mostly-true": "REAL",
    "true": "REAL"
}

rows = []

for split, path in FILES.items():
    df = pd.read_csv(path, sep="\t", header=None)

    # LIAR has at least 3 columns: label, id, statement
    for _, r in df.iterrows():
        label = str(r[1]).strip().lower()
        text  = r[2]

        if label in label_map and isinstance(text, str) and len(text.strip()) > 0:
            rows.append({
                "text": text.strip(),
                "label": label_map[label],
                "source": f"liar_{split}"
            })

out_df = pd.DataFrame(rows)
out_df.to_csv("data/liar_english.csv", index=False)

print("LIAR English dataset created successfully")
print("Total samples:", len(out_df))
print(out_df["label"].value_counts())
