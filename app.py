import os
import numpy as np
import string
import re
import joblib
import requests
import time
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


# ---- Feature Extraction ----

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


# ---- HuggingFace Inference API (Fallback) ----

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


# ---- Prediction Endpoint ----

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


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({"error": "No text provided"}), 400

    text = data['text']
    if len(text.strip()) == 0:
        return jsonify({"error": "Empty text provided"}), 400

    try:
        # ========================================
        # 1. TRANSFORMER MODEL PREDICTION
        # ========================================
        t_label = None
        t_conf = None
        transformer_source = "none"

        # Try local transformer first
        try:
            if local_transformer:
                hf_result = local_transformer(text[:512])
                best_pred = max(hf_result, key=lambda x: x['score'])
                t_label_raw = best_pred['label']
                t_conf = best_pred['score']

                # Map labels — handles both LABEL_0/LABEL_1 and FAKE/REAL
                if t_label_raw in ["LABEL_1", "1", "REAL"]:
                    t_label = "REAL"
                elif t_label_raw in ["LABEL_0", "0", "FAKE"]:
                    t_label = "FAKE"
                else:
                    t_label = "FAKE"  # Default unknown labels to FAKE for safety

                transformer_source = "local"
                print(f"Transformer (local): {t_label} ({t_conf:.4f})")
        except Exception as e:
            print(f"Local transformer error: {str(e)}")

        # Fallback to HuggingFace API if local failed
        if t_label is None:
            try:
                raw_label, raw_conf = get_hf_classification(text)
                t_conf = raw_conf

                # Map HF API labels
                if raw_label in ["LABEL_1", "1", "REAL"]:
                    t_label = "REAL"
                elif raw_label in ["LABEL_0", "0", "FAKE"]:
                    t_label = "FAKE"
                else:
                    t_label = "FAKE"

                transformer_source = "remote"
                print(f"Transformer (HF API): {t_label} ({t_conf:.4f})")
            except Exception as e:
                print(f"HF API fallback also failed: {str(e)}")

        # ========================================
        # 2. LR + TF-IDF MODEL PREDICTION
        # ========================================
        h_label = None
        h_conf = None

        if hybrid_clf and tfidf:
            vectorized = tfidf.transform([text])
            h_probs = hybrid_clf.predict_proba(vectorized)[0]
            h_pred_class = np.argmax(h_probs)
            h_conf = float(h_probs[h_pred_class])
            h_label = hybrid_clf.classes_[h_pred_class]
            print(f"LR Model: {h_label} ({h_conf:.4f})")

        # ========================================
        # 3. EXTRACT LINGUISTIC FEATURES
        # ========================================
        ling_feats, uppercase_ratio, punct_ratio = extract_features(text)
        clickbait_score = compute_clickbait_score(text, uppercase_ratio, punct_ratio)
        word_count = len(text.split())
        avg_word_length = np.mean([len(w) for w in text.split()]) if word_count > 0 else 0

        # ========================================
        # 4. HANDLE MISSING MODELS
        # ========================================
        # If transformer failed completely, use LR result
        if t_label is None:
            if h_label is not None:
                t_label = h_label
                t_conf = h_conf
                transformer_source = "fallback_lr"
            else:
                t_label = "UNKNOWN"
                t_conf = 0.5

        # If LR failed, use transformer result
        if h_label is None:
            if t_label is not None:
                h_label = t_label
                h_conf = t_conf
            else:
                h_label = "UNKNOWN"
                h_conf = 0.5

        # ========================================
        # 5. BUILD RESPONSE
        # ========================================
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
        print("Prediction Error:", str(e))
        return jsonify({"error": str(e)}), 500


# ---- News Verification Functions ----

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


def search_gnews(query, max_results=10):
    """
    Search GNews API for articles matching the query.
    Returns a list of article dicts with title, description, source, url, publishedAt.
    """
    if not GNEWS_API_KEY:
        raise Exception("GNEWS_API_KEY is not configured")

    # Clean query for search — remove excessive punctuation and caps
    clean_query = re.sub(r'[!?]{2,}', '', query)
    clean_query = clean_query.strip()[:200]  # GNews query length limit

    url = "https://gnews.io/api/v4/search"
    params = {
        "q": clean_query,
        "lang": "en",
        "max": max_results,
        "apikey": GNEWS_API_KEY
    }

    try:
        response = requests.get(url, params=params, timeout=15)
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
        raise Exception("GNews API request timed out")
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
        insights.append(f"{article['source']} reports a highly similar story ({score}%).")

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


@app.route('/verify', methods=['POST'])
def verify():
    """Verify submitted text against online news sources using GNews API."""
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({"error": "No text provided"}), 400

    text = data['text']
    if len(text.strip()) == 0:
        return jsonify({"error": "Empty text provided"}), 400

    try:
        # Step 1: Extract main claim
        claim = extract_main_claim(text)
        print(f"Verification: Searching for claim: '{claim[:80]}...'")

        # Step 2: Search GNews
        articles = search_gnews(claim)
        print(f"Verification: Found {len(articles)} articles")

        # Step 3: Compute similarity
        similarity_scores = compute_article_similarity(text, articles)

        # Step 4: Analyze results
        analysis = analyze_verification_results(text, articles, similarity_scores)

        # Build enriched article list with similarity scores
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

        # Sort by similarity descending
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
        print(f"Verification Error: {str(e)}")
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
        }), 200  # Return 200 with error info so frontend can still show partial results


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    app.run(host='0.0.0.0', port=port, debug=False)