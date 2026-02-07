import json
import pandas as pd

INPUT_JSON = "data/News_Category_Dataset_V3.json"
OUTPUT_CSV = "data/json_real_english.csv"

rows = []

with open(INPUT_JSON, "r", encoding="utf-8") as f:
    for line in f:
        item = json.loads(line)

        headline = item.get("headline", "")
        short_desc = item.get("short_description", "")

        text = (headline + ". " + short_desc).strip()

        # filter very short or empty entries
        if len(text) > 50:
            rows.append({
                "text": text,
                "label": "REAL",
                "source": "news_category_json"
            })

df = pd.DataFrame(rows)
df.to_csv(OUTPUT_CSV, index=False)

print("JSON real-news dataset created")
print("Total samples:", len(df))
