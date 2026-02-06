import re
from typing import Set

# Offline-safe stopword list (no NLTK download required)
STOP_WORDS: Set[str] = {
    "the", "is", "in", "and", "to", "of", "a", "an", "for", "on", "with",
    "that", "this", "it", "as", "at", "by", "from", "be", "are", "was",
    "were", "has", "have", "had", "but", "not", "or", "if", "then",
    "so", "because", "about", "into", "over", "after", "before",
    "between", "out", "up", "down", "again", "once",
    # Common artifacts/leaks in this dataset
    "reuters", "washington", "breaking", "image", "via",
    # Recurring stylistic leaks (Standard journalism uses days heavily, fakes might not)
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
    # Specific source leaks
    "breitbart", "gop"
}

def clean_text(text: str) -> str:
    """
    Cleans raw news text for ML models.
    Fully offline-safe (no NLTK dependency).
    """
    if not isinstance(text, str):
        return ""

    text = text.lower()
    
    # Remove common dataset leakage patterns like "WASHINGTON (Reuters) -"
    # Matches "city (reuters) -" at the start of string
    text = re.sub(r"^.*?\(reuters\)\s*-\s*", "", text)
    
    text = re.sub(r"[^a-z\s]", "", text)
    words = [w for w in text.split() if w not in STOP_WORDS]
    return " ".join(words)
    