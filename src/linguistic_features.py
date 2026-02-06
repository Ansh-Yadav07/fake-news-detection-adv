import re
import numpy as np

CLICKBAIT_WORDS = [
    "shocking", "breaking", "exposed", "truth",
    "secret", "you won’t believe", "they don’t want you"
]

def extract_linguistic_features(text: str):
    if not isinstance(text, str) or len(text.strip()) == 0:
        return np.zeros(8)

    text_lower = text.lower()

    # 1. Length-based features
    char_len = len(text)
    word_count = len(text.split())
    avg_word_len = np.mean([len(w) for w in text.split()]) if word_count > 0 else 0

    # 2. Style-based features
    exclamation_count = text.count("!")
    question_count = text.count("?")
    capital_ratio = sum(1 for c in text if c.isupper()) / max(1, char_len)

    # 3. Clickbait signal
    clickbait_count = sum(1 for w in CLICKBAIT_WORDS if w in text_lower)

    # 4. Numeric exaggeration
    number_count = len(re.findall(r"\d+", text))

    return np.array([
        char_len,
        word_count,
        avg_word_len,
        exclamation_count,
        question_count,
        capital_ratio,
        clickbait_count,
        number_count
    ], dtype=float)
    