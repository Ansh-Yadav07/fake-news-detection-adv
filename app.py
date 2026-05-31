import os
import numpy as np
import string
import re
import joblib
import requests
import time
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine_similarity
from flask import Flask, request, jsonify
from flask_cors import CORS
import nltk
from nltk.corpus import stopwords
from transformers import pipeline
from dotenv import load_dotenv

load_dotenv()

# ---- Initialization and Setup ----
app = Flask(__name__)
CORS(app)

# Force NLTK download for stopwords
try:
    nltk.download('stopwords', quiet=True)
    stop_words = set(stopwords.words("english"))
except Exception as e:
    stop_words = set()

# ---- Constants & API Config ----
HF_TOKEN = os.environ.get("HF_TOKEN", "")
GNEWS_API_KEY = os.environ.get("GNEWS_API_KEY", "")
HF_MODEL_ID = "anshy047/fake-news-detector-transformer"

# Local model paths
TRANSFORMER_MODEL_PATH = "models/transformer"
MODEL_PATH = "models/ml_lr_model.pkl"
TFIDF_PATH = "models/ml_tfidf.pkl"

# ---- Clickbait Detection Words ----
CLICKBAIT_TRIGGERS = [
    "breaking", "shocking", "you won't believe", "exposed", "secret",
    "urgent", "exclusive", "bombshell", "scandal", "horrifying",
    "jaw-dropping", "mind-blowing", "unbelievable", "insane", "crazy",
    "amazing", "must see", "gone wrong", "what happens next",
    "doctors hate", "one weird trick", "this changes everything",
    "you need to see", "never before seen", "finally revealed"
]

# ---- Trusted News Sources ----
TRUSTED_SOURCES = [
    "reuters", "ap news", "bbc", "the hindu",
    "indian express", "times of india", "hindustan times", "ndtv",
    "associated press", "bbc news"
]

# ---- News/Event Keywords for Input Classification ----
EVENT_KEYWORDS = [
    "election", "government", "minister", "president", "prime minister",
    "launched", "announced", "yesterday", "today", "recently",
    "breaking", "reported", "according to", "sources say",
    "company", "organization", "incident", "attack", "killed",
    "arrested", "protest", "rally", "vote", "parliament",
    "congress", "bjp", "nasa", "isro", "satellite", "war",
    "pandemic", "covid", "earthquake", "flood", "storm"
]

# ---- Load Models ----
print("Loading local models...")
hybrid_clf = None
tfidf = None
local_transformer = None

try:
    hybrid_clf = joblib.load(MODEL_PATH)
    tfidf = joblib.load(TFIDF_PATH)
    print(f"Local ML models loaded successfully!")
except FileNotFoundError as e:
    print(f"ERROR: Model file not found: {e}")
except Exception as e:
    print(f"ERROR loading ML models: {type(e).__name__}: {str(e)}")

try:
    local_transformer = pipeline(
        "text-classification",
        model=TRANSFORMER_MODEL_PATH,
        tokenizer=TRANSFORMER_MODEL_PATH,
        max_length=512,
        truncation=True
    )
    print("Local Transformer loaded successfully!")
except Exception as e:
    print(f"WARNING: Could not load local transformer: {type(e).__name__}: {str(e)}")
    print("Will attempt HuggingFace Inference API as fallback.")

if not hybrid_clf:
    print("WARNING: ML model will not be available")
if not tfidf:
    print("WARNING: TFIDF vectorizer not available")


# ======================================================================
# INPUT CLASSIFICATION
# ======================================================================

def classify_input(text):
    """
    Classify user input to determine verification routing.
    Returns: 'fact_claim', 'news_article', or 'mixed'
    
    - fact_claim: Short factual statements (< 25 words, no event keywords)
    - news_article: Longer text with event/news keywords
    - mixed: Uncertain — run both Wikipedia and GNews
    """
    words = text.split()
    word_count = len(words)
    text_lower = text.lower()

    # Check for event/news keywords
    event_keyword_count = sum(1 for kw in EVENT_KEYWORDS if kw in text_lower)

    # Short text with no event keywords → likely a factual claim
    if word_count < 25 and event_keyword_count == 0:
        return "fact_claim"

    # Short text with event keywords → mixed (could be headline)
    if word_count < 25 and event_keyword_count >= 1:
        return "mixed"

    # Longer text with event keywords → news article
    if word_count >= 25 and event_keyword_count >= 1:
        return "news_article"

    # Longer text without event keywords → mixed
    if word_count >= 25:
        return "mixed"

    return "mixed"


# ======================================================================
# FEATURE EXTRACTION
# ======================================================================

def compute_clickbait_score(text, uppercase_ratio, punct_ratio):
    """
    Compute a continuous 0-1 clickbait probability based on multiple linguistic heuristics.
    This replaces the old hardcoded binary 0.85/0.24 value.
    """
    score = 0.0
    text_lower = text.lower()
    words = text.split()
    total_words = len(words)
    if total_words == 0:
        return 0.0

    # 1. Sensationalist trigger words (up to 0.40)
    trigger_count = sum(1 for trigger in CLICKBAIT_TRIGGERS if trigger in text_lower)
    score += min(trigger_count * 0.15, 0.40)

    # 2. Excessive punctuation — !!! or ??? patterns (up to 0.20)
    exclamation_runs = len(re.findall(r'!{2,}', text))
    question_runs = len(re.findall(r'\?{2,}', text))
    score += min((exclamation_runs + question_runs) * 0.10, 0.20)

    # 3. General punctuation density (up to 0.10)
    if punct_ratio > 0.05:
        score += min((punct_ratio - 0.05) * 2.0, 0.10)

    # 4. Excessive uppercase ratio (up to 0.20)
    if uppercase_ratio > 0.08:
        score += min((uppercase_ratio - 0.08) * 2.0, 0.20)

    # 5. ALL CAPS words pattern — very strong clickbait signal (up to 0.25)
    caps_words = sum(1 for w in words if w.isupper() and len(w) > 1)
    caps_ratio = caps_words / total_words
    if caps_ratio > 0.3:
        score += min(caps_ratio * 0.35, 0.25)
    elif caps_ratio > 0.15:
        score += min((caps_ratio - 0.15) * 0.5, 0.10)

    # 6. Very short text (headlines are more likely clickbait) (up to 0.10)
    if total_words < 20:
        score += 0.05
    if total_words < 10:
        score += 0.05

    return round(min(score, 1.0), 4)


def extract_features(text):
    """Extract linguistic features from text for ML model and analysis."""
    words = text.split()
    total_words = len(words)
    if total_words == 0:
        return [0] * 8, 0, 0

    exclam_count = text.count("!")
    question_count = text.count("?")
    uppercase_ratio = sum(1 for c in text if c.isupper()) / max(len(text), 1)
    digit_ratio = sum(1 for c in text if c.isdigit()) / max(len(text), 1)
    punctuation_ratio = sum(1 for c in text if c in string.punctuation) / max(len(text), 1)
    stopword_ratio = sum(1 for w in words if w.lower() in stop_words) / total_words
    avg_word_length = np.mean([len(w) for w in words])
    text_length = len(text)

    return [
        exclam_count,
        question_count,
        uppercase_ratio,
        digit_ratio,
        punctuation_ratio,
        stopword_ratio,
        avg_word_length,
        text_length
    ], uppercase_ratio, punctuation_ratio


# ======================================================================
# HUGGINGFACE INFERENCE API (FALLBACK)
# ======================================================================

def get_hf_classification(text, max_retries=3):
    """
    Call HuggingFace Inference API as a fallback when local transformer is unavailable.
    Uses the model: anshy047/fake-news-detector-transformer
    """
    if not HF_TOKEN:
        raise Exception("HF_TOKEN is not configured")

    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    API_URL = f"https://api-inference.huggingface.co/models/{HF_MODEL_ID}"

    for attempt in range(max_retries):
        try:
            response = requests.post(
                API_URL,
                headers=headers,
                json={"inputs": text[:512]},  # Truncate to model's max length
                timeout=30
            )

            if response.status_code == 200:
                result = response.json()

                # HF returns [[{label, score}, ...]] or [{label, score}, ...]
                if isinstance(result, list) and len(result) > 0:
                    # Unwrap nested list if needed
                    predictions = result[0] if isinstance(result[0], list) else result
                    if isinstance(predictions, list) and len(predictions) > 0:
                        best = max(predictions, key=lambda x: x.get('score', 0))
                        return best.get('label', 'UNKNOWN'), best.get('score', 0.5)

                # If result is a dict with error
                if isinstance(result, dict) and "error" in result:
                    if "loading" in result["error"].lower() or "estimated_time" in result:
                        wait_time = min(result.get("estimated_time", 20), 30)
                        print(f"HF model loading. Waiting {wait_time}s... (Attempt {attempt+1}/{max_retries})")
                        time.sleep(wait_time)
                        continue
                    else:
                        raise Exception(f"HF API error: {result['error']}")

            elif response.status_code == 503:
                try:
                    error_data = response.json()
                    wait_time = min(error_data.get("estimated_time", 20), 30)
                    print(f"HF model loading (503). Waiting {wait_time}s... (Attempt {attempt+1}/{max_retries})")
                    time.sleep(wait_time)
                    continue
                except ValueError:
                    pass
                raise Exception(f"HF API 503: Service Unavailable")

            else:
                raise Exception(f"HF API error {response.status_code}: {response.text[:200]}")

        except requests.exceptions.Timeout:
            print(f"HF API timeout (attempt {attempt+1}/{max_retries})")
            if attempt < max_retries - 1:
                time.sleep(5)
            continue
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"HF API error: {str(e)}, retrying...")
                time.sleep(3)
                continue
            raise

    raise Exception(f"Failed to get HF API response after {max_retries} attempts")


# ======================================================================
# WIKIPEDIA VERIFICATION
# ======================================================================

def extract_entity_for_wiki(text):
    """
    Extract the most relevant entity/topic from text for Wikipedia lookup.
    Focuses on proper nouns, capitalized words, and key subjects.
    Returns a search query string.
    """
    # Remove common noise words and punctuation
    clean = re.sub(r'[^\w\s]', '', text)
    words = clean.split()

    # Strategy 1: Find capitalized words (likely proper nouns/entities)
    capitalized = [w for w in words if w[0].isupper() and w.lower() not in stop_words and len(w) > 1]

    if capitalized:
        # Use up to 3 capitalized words as the query
        return ' '.join(capitalized[:3])

    # Strategy 2: Remove stopwords and use remaining content words
    content_words = [w for w in words if w.lower() not in stop_words and len(w) > 2]
    if content_words:
        return ' '.join(content_words[:3])

    # Strategy 3: Fallback — first 5 words
    return ' '.join(words[:5])


def search_wikipedia(query, timeout=1.5):
    """
    Search Wikipedia REST API for a summary of the given query.
    Uses the page summary endpoint.
    Timeout: 1.5 seconds (hard limit per spec).
    
    Returns dict with title, extract, description, url or None on failure.
    """
    if not query or not query.strip():
        return None

    # URL-encode the query for the REST API
    encoded_query = urllib.parse.quote(query.strip().replace(' ', '_'))
    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{encoded_query}"

    try:
        response = requests.get(
            url,
            headers={"User-Agent": "FakeNewsDetector/1.0"},
            timeout=timeout
        )

        if response.status_code == 200:
            data = response.json()
            if data.get("type") == "disambiguation":
                # Try the first word only for disambiguation pages
                first_word = query.strip().split()[0]
                if first_word != query.strip():
                    return search_wikipedia(first_word, timeout=timeout)
                return None

            return {
                "title": data.get("title", ""),
                "extract": data.get("extract", ""),
                "description": data.get("description", ""),
                "url": data.get("content_urls", {}).get("desktop", {}).get("page", ""),
                "found": True
            }
        elif response.status_code == 404:
            # Try Wikipedia search API as fallback
            return _wikipedia_search_fallback(query, timeout)
        else:
            return None

    except requests.exceptions.Timeout:
        print(f"Wikipedia API timed out for query: '{query}'")
        return None
    except Exception as e:
        print(f"Wikipedia API error: {e}")
        return None


def _wikipedia_search_fallback(query, timeout=1.5):
    """
    Fallback: use Wikipedia search API to find the best matching article.
    """
    search_url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "list": "search",
        "srsearch": query,
        "srlimit": 1,
        "format": "json"
    }

    try:
        resp = requests.get(search_url, params=params, timeout=timeout,
                            headers={"User-Agent": "FakeNewsDetector/1.0"})
        if resp.status_code == 200:
            data = resp.json()
            results = data.get("query", {}).get("search", [])
            if results:
                title = results[0].get("title", "")
                # Now fetch the summary for this title
                return search_wikipedia(title, timeout=timeout)
        return None
    except Exception:
        return None


def verify_against_wikipedia(user_text, wiki_data):
    """
    Compare user's claim against Wikipedia data using TF-IDF cosine similarity.
    Also checks if key terms from the claim appear in the Wikipedia extract.
    
    Returns verification result dict.
    """
    if not wiki_data or not wiki_data.get("found"):
        return {
            "verification_score": 0,
            "status": "NOT FOUND",
            "message": "No Wikipedia article found for this claim.",
            "insights": ["Could not find a relevant Wikipedia article for verification."],
            "wiki_title": "",
            "wiki_url": "",
            "wiki_extract": ""
        }

    extract = wiki_data.get("extract", "")
    description = wiki_data.get("description", "")
    wiki_text = f"{description} {extract}"

    if not wiki_text.strip():
        return {
            "verification_score": 0,
            "status": "NOT FOUND",
            "message": "Wikipedia article has no content.",
            "insights": ["Wikipedia article found but contains no useful extract."],
            "wiki_title": wiki_data.get("title", ""),
            "wiki_url": wiki_data.get("url", ""),
            "wiki_extract": ""
        }

    # Method 1: TF-IDF cosine similarity
    try:
        vectorizer = TfidfVectorizer(
            stop_words='english',
            max_features=5000,
            ngram_range=(1, 2)
        )
        tfidf_matrix = vectorizer.fit_transform([user_text, wiki_text])
        similarity = sklearn_cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        tfidf_score = round(float(similarity) * 100, 1)
    except Exception:
        tfidf_score = 0

    # Method 2: Key term overlap — check if important words from claim appear in Wikipedia
    user_words = set(w.lower() for w in re.sub(r'[^\w\s]', '', user_text).split()
                     if w.lower() not in stop_words and len(w) > 2)
    wiki_words = set(w.lower() for w in re.sub(r'[^\w\s]', '', wiki_text).split()
                     if w.lower() not in stop_words and len(w) > 2)

    if user_words:
        overlap = len(user_words & wiki_words)
        overlap_ratio = overlap / len(user_words)
        term_score = round(overlap_ratio * 100, 1)
    else:
        term_score = 0

    # Combined score: weighted average (TF-IDF 60%, term overlap 40%)
    verification_score = round(tfidf_score * 0.6 + term_score * 0.4, 1)

    # Determine status
    if verification_score > 70:
        status = "VERIFIED FACT"
        message = f"Wikipedia confirms this claim. Article: \"{wiki_data['title']}\"."
    elif verification_score > 45:
        status = "PARTIALLY VERIFIED"
        message = f"Wikipedia contains related information but not a full match. Article: \"{wiki_data['title']}\"."
    elif verification_score > 20:
        status = "WEAK MATCH"
        message = f"Wikipedia has some related content but the claim could not be strongly verified."
    else:
        status = "NOT VERIFIED"
        message = f"The claim does not match Wikipedia content for \"{wiki_data['title']}\"."

    # Generate insights
    insights = []
    if status == "VERIFIED FACT":
        insights.append(f"Wikipedia confirms the statement.")
        insights.append(f"Fact matches trusted knowledge sources.")
    elif status == "PARTIALLY VERIFIED":
        insights.append(f"Wikipedia contains related information in the article \"{wiki_data['title']}\".")
    else:
        insights.append(f"Wikipedia article \"{wiki_data['title']}\" found but claim could not be fully verified.")

    insights.append(f"Wikipedia verification score: {verification_score}%.")

    # Add description if it directly relates
    if description:
        insights.append(f"Wikipedia describes \"{wiki_data['title']}\" as: {description}.")

    return {
        "verification_score": verification_score,
        "status": status,
        "message": message,
        "insights": insights,
        "wiki_title": wiki_data.get("title", ""),
        "wiki_url": wiki_data.get("url", ""),
        "wiki_extract": extract[:300] if extract else "",
        "tfidf_score": tfidf_score,
        "term_overlap_score": term_score
    }


# ======================================================================
# GNEWS VERIFICATION (updated with faster timeout)
# ======================================================================

def extract_main_claim(text):
    """
    Extract the main claim/headline from the input text.
    Short text (<30 words) is used directly.
    Longer text: extract the first sentence as the search query.
    """
    words = text.split()
    if len(words) < 30:
        return text.strip()

    # Try to extract first sentence
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    if sentences:
        first_sentence = sentences[0].strip()
        # If first sentence is reasonable length, use it
        if 5 <= len(first_sentence.split()) <= 50:
            return first_sentence

    # Fallback: use first 25 words
    return ' '.join(words[:25])


def search_gnews(query, max_results=10, timeout=2.0):
    """
    Search GNews API for articles matching the query.
    Returns a list of article dicts with title, description, source, url, publishedAt.
    Timeout: 2 seconds (hard limit per spec).
    """
    if not GNEWS_API_KEY:
        raise Exception("GNEWS_API_KEY is not configured")

    # Clean query for search — remove all punctuation that breaks GNews API syntax
    clean_query = re.sub(r'[^\w\s]', '', query)
    clean_query = re.sub(r'\s+', ' ', clean_query).strip()[:200]  # GNews query length limit

    url = "https://gnews.io/api/v4/search"
    params = {
        "q": clean_query,
        "lang": "en",
        "max": max_results,
        "apikey": GNEWS_API_KEY
    }

    try:
        response = requests.get(url, params=params, timeout=timeout)
        if response.status_code == 200:
            data = response.json()
            articles = data.get("articles", [])
            return [{ 
                "title": a.get("title", ""),
                "description": a.get("description", ""),
                "source": a.get("source", {}).get("name", "Unknown"),
                "url": a.get("url", ""),
                "published_at": a.get("publishedAt", "")
            } for a in articles]
        elif response.status_code == 403:
            raise Exception("GNews API key is invalid or quota exceeded")
        elif response.status_code == 429:
            raise Exception("GNews API rate limit exceeded")
        else:
            raise Exception(f"GNews API error {response.status_code}: {response.text[:200]}")
    except requests.exceptions.Timeout:
        raise Exception("GNews API request timed out (2s limit)")
    except requests.exceptions.ConnectionError:
        raise Exception("Could not connect to GNews API")


def compute_article_similarity(user_text, articles):
    """
    Compute TF-IDF cosine similarity between user text and each article.
    Returns list of similarity scores (0-100) for each article.
    """
    if not articles:
        return []

    # Build corpus: user text + each article's title+description
    corpus = [user_text]
    for article in articles:
        article_text = f"{article['title']} {article.get('description', '')}".strip()
        corpus.append(article_text)

    try:
        vectorizer = TfidfVectorizer(
            stop_words='english',
            max_features=5000,
            ngram_range=(1, 2)
        )
        tfidf_matrix = vectorizer.fit_transform(corpus)

        # Compute cosine similarity between user text (index 0) and each article
        user_vector = tfidf_matrix[0:1]
        article_vectors = tfidf_matrix[1:]
        similarities = sklearn_cosine_similarity(user_vector, article_vectors)[0]

        # Convert to 0-100 scale
        return [round(float(s) * 100, 1) for s in similarities]
    except Exception as e:
        print(f"Similarity computation error: {e}")
        return [0.0] * len(articles)


def is_trusted_source(source_name):
    """Check if a source name matches any trusted source."""
    source_lower = source_name.lower().strip()
    return any(trusted in source_lower or source_lower in trusted for trusted in TRUSTED_SOURCES)


def analyze_verification_results(user_text, articles, similarity_scores):
    """
    Analyze verification results to determine status and generate insights.
    Returns verification_score, trusted_source_count, status, and insights.
    """
    if not articles or not similarity_scores:
        return {
            "verification_score": 0,
            "supporting_articles": 0,
            "trusted_source_count": 0,
            "status": "UNVERIFIED",
            "message": "No sufficient online evidence available.",
            "insights": ["No supporting articles found online for this claim."]
        }

    # Count supporting articles (similarity > 20%)
    supporting_threshold = 20
    supporting = [(articles[i], similarity_scores[i])
                  for i in range(len(articles))
                  if similarity_scores[i] > supporting_threshold]

    supporting_count = len(supporting)

    # Count trusted sources among supporting articles
    trusted_supporting = [(a, s) for a, s in supporting if is_trusted_source(a['source'])]
    trusted_count = len(trusted_supporting)

    # Calculate verification score (weighted average of top similarities)
    if similarity_scores:
        # Weight top scores more heavily
        sorted_scores = sorted(similarity_scores, reverse=True)
        top_scores = sorted_scores[:5]  # Top 5
        verification_score = round(sum(top_scores) / len(top_scores), 1) if top_scores else 0
    else:
        verification_score = 0

    # Determine status
    if supporting_count == 0:
        status = "UNVERIFIED"
        message = "No sufficient online evidence available."
    elif verification_score > 80 and trusted_count >= 3:
        status = "VERIFIED"
        message = "Multiple trusted news organizations independently confirm this claim."
    elif verification_score > 60 and trusted_count >= 1:
        status = "LIKELY SUPPORTED"
        message = "Evidence from trusted sources partially supports this claim."
    elif verification_score > 40 and supporting_count >= 3:
        status = "PARTIALLY SUPPORTED"
        message = "Some online sources report similar content."
    else:
        status = "UNVERIFIED"
        message = "Insufficient evidence from trusted sources to verify this claim."

    # Generate insights
    insights = []
    insights.append(f"{supporting_count} supporting article{'s' if supporting_count != 1 else ''} found online.")

    # Add top trusted source matches
    for article, score in sorted(trusted_supporting, key=lambda x: x[1], reverse=True)[:4]:
        insights.append(f"{article['source']} supports the claim ({score}%).")

    if trusted_count > 0:
        insights.append(f"{trusted_count} trusted source{'s' if trusted_count != 1 else ''} independently support{'s' if trusted_count == 1 else ''} the claim.")

    insights.append(f"Online verification confidence: {verification_score}%.")

    if status == "VERIFIED":
        insights.append("Real-world evidence strongly supports this article.")
    elif status == "UNVERIFIED" and supporting_count == 0:
        insights.append("This does not automatically mean the article is fake — it may be too new or niche for online coverage.")

    return {
        "verification_score": verification_score,
        "supporting_articles": supporting_count,
        "trusted_source_count": trusted_count,
        "status": status,
        "message": message,
        "insights": insights
    }


# ======================================================================
# PARALLEL TASK RUNNERS (wrapped for safe execution)
# ======================================================================

def _run_transformer(text):
    """Run transformer model prediction. Returns dict or None."""
    try:
        t_label = None
        t_conf = None
        source = "none"

        # Try local transformer first
        if local_transformer:
            hf_result = local_transformer(text[:512])
            best_pred = max(hf_result, key=lambda x: x['score'])
            t_label_raw = best_pred['label']
            t_conf = best_pred['score']

            if t_label_raw in ["LABEL_1", "1", "REAL"]:
                t_label = "REAL"
            elif t_label_raw in ["LABEL_0", "0", "FAKE"]:
                t_label = "FAKE"
            else:
                t_label = "FAKE"

            source = "local"

        # Fallback to HuggingFace API
        if t_label is None:
            raw_label, raw_conf = get_hf_classification(text)
            t_conf = raw_conf
            if raw_label in ["LABEL_1", "1", "REAL"]:
                t_label = "REAL"
            elif raw_label in ["LABEL_0", "0", "FAKE"]:
                t_label = "FAKE"
            else:
                t_label = "FAKE"
            source = "remote"

        return {"label": t_label, "confidence": round(float(t_conf), 4), "source": source}
    except Exception as e:
        print(f"Transformer task error: {e}")
        return None


def _run_hybrid_ml(text):
    """Run LR + TF-IDF model prediction. Returns dict or None."""
    try:
        if hybrid_clf and tfidf:
            vectorized = tfidf.transform([text])
            h_probs = hybrid_clf.predict_proba(vectorized)[0]
            h_pred_class = np.argmax(h_probs)
            h_conf = float(h_probs[h_pred_class])
            h_label = hybrid_clf.classes_[h_pred_class]
            return {"label": h_label, "confidence": round(h_conf, 4)}
        return None
    except Exception as e:
        print(f"Hybrid ML task error: {e}")
        return None


def _run_linguistic(text):
    """Run linguistic feature extraction. Returns dict or None."""
    try:
        ling_feats, uppercase_ratio, punct_ratio = extract_features(text)
        clickbait_score = compute_clickbait_score(text, uppercase_ratio, punct_ratio)
        word_count = len(text.split())
        avg_word_length = np.mean([len(w) for w in text.split()]) if word_count > 0 else 0

        return {
            "uppercase": round(float(uppercase_ratio), 4),
            "punctuation": round(float(punct_ratio), 4),
            "clickbait": round(float(clickbait_score), 4),
            "complexity": round(float(avg_word_length), 2),
            "word_count": word_count
        }
    except Exception as e:
        print(f"Linguistic task error: {e}")
        return None


def _run_wikipedia_verification(text, input_type):
    """Run Wikipedia verification. Returns dict or None."""
    try:
        # Skip Wikipedia for pure news articles (GNews is primary)
        if input_type == "news_article":
            return None

        entity = extract_entity_for_wiki(text)
        print(f"[Wikipedia] Searching for entity: '{entity}'")

        wiki_data = search_wikipedia(entity, timeout=1.5)
        if not wiki_data:
            return {
                "verification_score": 0,
                "status": "NOT FOUND",
                "message": "No Wikipedia article found.",
                "insights": ["No relevant Wikipedia article found for verification."],
                "wiki_title": "",
                "wiki_url": "",
                "wiki_extract": ""
            }

        result = verify_against_wikipedia(text, wiki_data)
        return result
    except Exception as e:
        print(f"Wikipedia task error: {e}")
        return None


def _run_gnews_verification(text, input_type):
    """Run GNews verification. Returns dict or None."""
    try:
        # Skip GNews for pure fact claims (Wikipedia is primary)
        if input_type == "fact_claim":
            return None

        claim = extract_main_claim(text)
        print(f"[GNews] Searching for claim: '{claim[:80]}'")

        articles = search_gnews(claim, timeout=2.0)
        similarity_scores = compute_article_similarity(text, articles)
        analysis = analyze_verification_results(text, articles, similarity_scores)

        # Build enriched article list
        enriched_articles = []
        for i, article in enumerate(articles):
            score = similarity_scores[i] if i < len(similarity_scores) else 0
            enriched_articles.append({
                "title": article["title"],
                "source": article["source"],
                "url": article["url"],
                "similarity": score,
                "is_trusted": is_trusted_source(article["source"]),
                "published_at": article["published_at"]
            })

        enriched_articles.sort(key=lambda x: x["similarity"], reverse=True)

        return {
            "verification_score": analysis["verification_score"],
            "supporting_articles": analysis["supporting_articles"],
            "trusted_source_count": analysis["trusted_source_count"],
            "status": analysis["status"],
            "message": analysis["message"],
            "articles": enriched_articles,
            "insights": analysis["insights"],
            "claim_searched": claim
        }
    except Exception as e:
        print(f"GNews task error: {e}")
        return None


# ======================================================================
# API ENDPOINTS
# ======================================================================

@app.route('/', methods=['GET'])
def health():
    return jsonify({
        "status": "API is running",
        "models": {
            "lr_tfidf": hybrid_clf is not None,
            "transformer_local": local_transformer is not None,
            "transformer_remote": bool(HF_TOKEN)
        }
    }), 200


@app.route('/analyze', methods=['POST'])
def analyze():
    """
    Unified analysis endpoint — runs ALL tasks in parallel.
    
    Tasks:
      1. Transformer Model
      2. Hybrid ML Model
      3. Linguistic Analysis
      4. Clickbait Analysis (part of linguistic)
      5. Wikipedia Verification
      6. GNews Verification
    
    All tasks execute concurrently via ThreadPoolExecutor.
    Returns a unified response with all results.
    """
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({"error": "No text provided"}), 400

    text = data['text']
    if len(text.strip()) == 0:
        return jsonify({"error": "Empty text provided"}), 400

    start_time = time.time()

    try:
        # Step 1: Classify input type
        input_type = classify_input(text)
        print(f"\n{'='*60}")
        print(f"[Analyze] Input type: {input_type} | Length: {len(text.split())} words")
        print(f"{'='*60}")

        # Step 2: Run ALL tasks in parallel
        timings = {}
        results = {}

        with ThreadPoolExecutor(max_workers=6) as executor:
            task_starts = {}
            futures = {}

            # Submit all tasks
            task_starts['transformer'] = time.time()
            futures['transformer'] = executor.submit(_run_transformer, text)

            task_starts['hybrid'] = time.time()
            futures['hybrid'] = executor.submit(_run_hybrid_ml, text)

            task_starts['linguistic'] = time.time()
            futures['linguistic'] = executor.submit(_run_linguistic, text)

            task_starts['wikipedia'] = time.time()
            futures['wikipedia'] = executor.submit(_run_wikipedia_verification, text, input_type)

            task_starts['gnews'] = time.time()
            futures['gnews'] = executor.submit(_run_gnews_verification, text, input_type)

            # Collect results with timeout
            for task_name, future in futures.items():
                try:
                    results[task_name] = future.result(timeout=5)
                    timings[task_name] = round(time.time() - task_starts[task_name], 3)
                except Exception as e:
                    print(f"[Analyze] Task '{task_name}' failed: {e}")
                    results[task_name] = None
                    timings[task_name] = round(time.time() - task_starts[task_name], 3)

        # Step 3: Handle missing model results with fallbacks
        transformer = results.get('transformer')
        hybrid = results.get('hybrid')
        linguistic = results.get('linguistic')
        wikipedia = results.get('wikipedia')
        gnews = results.get('gnews')

        # Fallback: if transformer failed, use hybrid result
        if transformer is None:
            if hybrid is not None:
                transformer = {"label": hybrid["label"], "confidence": hybrid["confidence"], "source": "fallback_lr"}
            else:
                transformer = {"label": "UNKNOWN", "confidence": 0.5, "source": "none"}

        # Fallback: if hybrid failed, use transformer result
        if hybrid is None:
            if transformer is not None:
                hybrid = {"label": transformer["label"], "confidence": transformer["confidence"]}
            else:
                hybrid = {"label": "UNKNOWN", "confidence": 0.5}

        # Fallback: if linguistic failed
        if linguistic is None:
            linguistic = {
                "uppercase": 0, "punctuation": 0,
                "clickbait": 0, "complexity": 5, "word_count": len(text.split())
            }

        # Step 4: Build response
        total_time = round(time.time() - start_time, 3)
        print(f"[Analyze] Total time: {total_time}s | Timings: {timings}")

        payload = {
            "input_type": input_type,
            "transformer": transformer,
            "hybrid": hybrid,
            "raw_features": linguistic,
            "wikipedia": wikipedia,
            "verification": gnews,
            "timings": timings,
            "total_time": total_time
        }

        return jsonify(payload)

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Analyze Error: {str(e)}")
        return jsonify({"error": str(e)}), 500


# ---- Legacy Endpoints (backward compatible) ----

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({"error": "No text provided"}), 400

    text = data['text']
    if len(text.strip()) == 0:
        return jsonify({"error": "Empty text provided"}), 400

    try:
        # 1. TRANSFORMER MODEL PREDICTION
        t_label = None
        t_conf = None
        transformer_source = "none"

        try:
            if local_transformer:
                hf_result = local_transformer(text[:512])
                best_pred = max(hf_result, key=lambda x: x['score'])
                t_label_raw = best_pred['label']
                t_conf = best_pred['score']

                if t_label_raw in ["LABEL_1", "1", "REAL"]:
                    t_label = "REAL"
                elif t_label_raw in ["LABEL_0", "0", "FAKE"]:
                    t_label = "FAKE"
                else:
                    t_label = "FAKE"

                transformer_source = "local"
        except Exception as e:
            print(f"Local transformer error: {str(e)}")

        if t_label is None:
            try:
                raw_label, raw_conf = get_hf_classification(text)
                t_conf = raw_conf
                if raw_label in ["LABEL_1", "1", "REAL"]:
                    t_label = "REAL"
                elif raw_label in ["LABEL_0", "0", "FAKE"]:
                    t_label = "FAKE"
                else:
                    t_label = "FAKE"
                transformer_source = "remote"
            except Exception as e:
                print(f"HF API fallback also failed: {str(e)}")

        # 2. LR + TF-IDF MODEL PREDICTION
        h_label = None
        h_conf = None

        if hybrid_clf and tfidf:
            vectorized = tfidf.transform([text])
            h_probs = hybrid_clf.predict_proba(vectorized)[0]
            h_pred_class = np.argmax(h_probs)
            h_conf = float(h_probs[h_pred_class])
            h_label = hybrid_clf.classes_[h_pred_class]

        # 3. EXTRACT LINGUISTIC FEATURES
        ling_feats, uppercase_ratio, punct_ratio = extract_features(text)
        clickbait_score = compute_clickbait_score(text, uppercase_ratio, punct_ratio)
        word_count = len(text.split())
        avg_word_length = np.mean([len(w) for w in text.split()]) if word_count > 0 else 0

        # 4. HANDLE MISSING MODELS
        if t_label is None:
            if h_label is not None:
                t_label = h_label
                t_conf = h_conf
                transformer_source = "fallback_lr"
            else:
                t_label = "UNKNOWN"
                t_conf = 0.5

        if h_label is None:
            if t_label is not None:
                h_label = t_label
                h_conf = t_conf
            else:
                h_label = "UNKNOWN"
                h_conf = 0.5

        # 5. BUILD RESPONSE
        payload = {
            "transformer": {
                "label": t_label,
                "confidence": round(float(t_conf), 4),
                "source": transformer_source
            },
            "hybrid": {
                "label": h_label,
                "confidence": round(float(h_conf), 4)
            },
            "raw_features": {
                "uppercase": round(float(uppercase_ratio), 4),
                "punctuation": round(float(punct_ratio), 4),
                "clickbait": round(float(clickbait_score), 4),
                "complexity": round(float(avg_word_length), 2),
                "word_count": word_count
            }
        }

        return jsonify(payload)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route('/verify', methods=['POST'])
def verify():
    """Legacy verify endpoint — kept for backward compatibility."""
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({"error": "No text provided"}), 400

    text = data['text']
    if len(text.strip()) == 0:
        return jsonify({"error": "Empty text provided"}), 400

    try:
        claim = extract_main_claim(text)
        articles = search_gnews(claim, timeout=2.0)
        similarity_scores = compute_article_similarity(text, articles)
        analysis = analyze_verification_results(text, articles, similarity_scores)

        enriched_articles = []
        for i, article in enumerate(articles):
            score = similarity_scores[i] if i < len(similarity_scores) else 0
            enriched_articles.append({
                "title": article["title"],
                "source": article["source"],
                "url": article["url"],
                "similarity": score,
                "is_trusted": is_trusted_source(article["source"]),
                "published_at": article["published_at"]
            })

        enriched_articles.sort(key=lambda x: x["similarity"], reverse=True)

        payload = {
            "verification_score": analysis["verification_score"],
            "supporting_articles": analysis["supporting_articles"],
            "trusted_source_count": analysis["trusted_source_count"],
            "status": analysis["status"],
            "message": analysis["message"],
            "articles": enriched_articles,
            "insights": analysis["insights"],
            "claim_searched": claim
        }

        return jsonify(payload)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            "error": str(e),
            "verification_score": 0,
            "supporting_articles": 0,
            "trusted_source_count": 0,
            "status": "UNVERIFIED",
            "message": "Verification service unavailable.",
            "articles": [],
            "insights": ["Online verification could not be completed."],
            "claim_searched": ""
        }), 200


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    app.run(host='0.0.0.0', port=port, debug=False)