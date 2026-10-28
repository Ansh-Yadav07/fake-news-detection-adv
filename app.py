import os
import numpy as np
import string
import re
import joblib
import requests
import time
import logging
import threading
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, wait as futures_wait
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from cachetools import TTLCache
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine_similarity
from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from werkzeug.middleware.proxy_fix import ProxyFix
from transformers import pipeline
from dotenv import load_dotenv

load_dotenv()

# ---- Logging ----
if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s"
    )
logger = logging.getLogger(__name__)

# ---- Initialization and Setup ----
app = Flask(__name__)

# Reject oversized request bodies early (413) — 64KB is plenty for 10,000 chars of text
app.config['MAX_CONTENT_LENGTH'] = 64 * 1024

# Trust exactly one proxy hop for X-Forwarded-For (so rate limiting sees the real client IP)
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1)

# ---- CORS (API routes only, origins from ALLOWED_ORIGINS) ----
# Comma-separated list. When unset, falls back to local dev origins plus the
# deployed frontend origins found in the repo (none at the time of writing).
# PRODUCTION: set ALLOWED_ORIGINS to your deployed frontend origin(s).
DEFAULT_ALLOWED_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]
ALLOWED_ORIGINS = [
    o.strip().rstrip('/')
    for o in os.environ.get("ALLOWED_ORIGINS", "").split(",")
    if o.strip()
] or DEFAULT_ALLOWED_ORIGINS
CORS(
    app,
    resources={r"/(analyze|predict|verify)": {"origins": ALLOWED_ORIGINS}},
    methods=["POST", "OPTIONS"],
    allow_headers=["Content-Type"],
    max_age=600,
)

# ---- Rate limiting (in-memory, per client IP) ----
RATE_LIMIT = os.environ.get("RATE_LIMIT", "20 per minute")
limiter = Limiter(
    key_func=get_remote_address,
    app=app,
    default_limits=[],
    storage_uri="memory://",
)

# Stopwords (Hardcoded to avoid NLTK download hangs on startup in production)
ENGLISH_STOPWORDS = {
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 
    "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 
    'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 
    'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 
    'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 
    'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"
}
stop_words = ENGLISH_STOPWORDS

# ---- Constants & API Config ----
HF_TOKEN = os.environ.get("HF_TOKEN", "")
GNEWS_API_KEY = os.environ.get("GNEWS_API_KEY", "")
NEWSDATA_API_KEY = os.environ.get("NEWSDATA_API_KEY", "")
NEWSAPI_KEY = os.environ.get("NEWSAPI_KEY", "")
SERPAPI_KEY = os.environ.get("SERPAPI_KEY", "")
HF_MODEL_ID = "anshy047/fake-news-detector-transformer"

# ---- Timing budgets / concurrency (env-overridable) ----
ANALYZE_BUDGET_S = float(os.environ.get("ANALYZE_BUDGET_S", "8"))
ANALYZE_WORKERS = int(os.environ.get("ANALYZE_WORKERS", "16"))
HF_TIMEOUT_S = float(os.environ.get("HF_TIMEOUT_S", "6"))
MAX_TEXT_CHARS = 10000
CACHE_TTL_S = int(os.environ.get("CACHE_TTL_S", "900"))
CACHE_MAXSIZE = int(os.environ.get("CACHE_MAXSIZE", "512"))

# Shared executors: never created per request and never shut down with wait=True
# in the request path, so a slow external call can't block a response.
_EXECUTOR = ThreadPoolExecutor(max_workers=ANALYZE_WORKERS, thread_name_prefix="analyze")
# Separate pool for the nested per-source news lookups (avoids pool self-deadlock)
_IO_EXECUTOR = ThreadPoolExecutor(max_workers=ANALYZE_WORKERS, thread_name_prefix="io")

# ---- Shared HTTP session (connection pooling + one retry on transient gateway errors) ----
_http = requests.Session()
_retry = Retry(total=1, backoff_factor=0, status_forcelist=[502, 503, 504])
_adapter = HTTPAdapter(max_retries=_retry, pool_connections=16, pool_maxsize=32)
_http.mount("https://", _adapter)
_http.mount("http://", _adapter)

# ---- Thread-safe TTL cache for external lookups ----
_cache = TTLCache(maxsize=CACHE_MAXSIZE, ttl=CACHE_TTL_S)
_cache_lock = threading.Lock()
_MISS = object()


def _cache_get(key):
    with _cache_lock:
        return _cache.get(key, _MISS)


def _cache_set(key, value):
    with _cache_lock:
        _cache[key] = value


# ---- Safe logging of exceptions (API keys travel as query params) ----
_SECRET_QS_RE = re.compile(r'(?i)\b(api[_-]?key|apikey|token|key|access_token)=[^&\s\'")]+')


def _safe_err(e):
    """Exception type + sanitized message, safe to log (strips API keys from URLs)."""
    msg = _SECRET_QS_RE.sub(r'\1=***', str(e))
    for secret in (HF_TOKEN, GNEWS_API_KEY, NEWSDATA_API_KEY, NEWSAPI_KEY, SERPAPI_KEY):
        if secret:
            msg = msg.replace(secret, '***')
    return f"{type(e).__name__}: {msg[:200]}"


def _normalize_label(raw):
    """Map a raw model label to REAL / FAKE / UNKNOWN (case-insensitive)."""
    key = str(raw).strip().upper()
    if key in ("LABEL_1", "1", "REAL"):
        return "REAL"
    if key in ("LABEL_0", "0", "FAKE"):
        return "FAKE"
    return "UNKNOWN"

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
    "associated press", "bbc news",
    # Compact forms used in domains and NewsData source_ids (e.g. "apnews.com", "timesofindia")
    "apnews", "thehindu", "indianexpress", "timesofindia", "hindustantimes",
]
# Whole-phrase match on a normalized name, so generic fragments ("News", "The", "") never match
_TRUSTED_SOURCE_PATTERNS = [re.compile(rf"\b{re.escape(t)}\b") for t in TRUSTED_SOURCES]

# ---- News/Event Keywords for Input Classification ----
EVENT_KEYWORDS = [
    # Government & Politics
    "election", "government", "minister", "president", "prime minister",
    "parliament", "congress", "bjp", "senate", "legislation", "bill",
    "vote", "ballot", "campaign", "governor", "mayor", "chancellor",
    "diplomatic", "embassy", "sanctions", "treaty", "summit",
    # Events & Actions
    "launched", "announced", "yesterday", "today", "recently",
    "breaking", "reported", "according to", "sources say",
    "approved", "rejected", "signed", "proposed", "proposes",
    "introduced", "unveiled", "revealed", "confirmed", "denied",
    "ordered", "banned", "lifted", "suspended", "revoked",
    # Business & Economy
    "company", "organization", "ceo", "stock", "shares", "market",
    "tariff", "tariffs", "duties", "import", "export", "trade",
    "gdp", "inflation", "recession", "economy", "economic",
    "billion", "million", "trillion", "revenue", "profit",
    "merger", "acquisition", "ipo", "startup", "investment",
    # Security & Incidents
    "incident", "attack", "killed", "arrested", "protest", "rally",
    "conflict", "ceasefire", "troops", "military", "defense",
    "terror", "bombing", "shooting", "hostage", "crisis",
    # Science & Tech
    "nasa", "isro", "satellite", "space", "rocket", "mission",
    "research", "study", "scientists", "discovered", "breakthrough",
    "ai", "artificial intelligence", "technology", "cyber",
    # Disasters & Health
    "war", "pandemic", "covid", "earthquake", "flood", "storm",
    "hurricane", "wildfire", "tsunami", "drought", "outbreak",
    "vaccine", "virus", "disease", "health", "hospital",
    # Sports
    "match", "tournament", "championship", "olympics", "world cup",
    "league", "cricket", "football", "medal", "finals",
    # Legal
    "court", "judge", "verdict", "lawsuit", "indicted", "charged",
    "convicted", "sentenced", "bail", "appeal", "ruling",
    # International
    "countries", "nations", "global", "international", "foreign",
    "violations", "compliance", "regulation", "policy", "reform",
    # Organization acronyms
    "ustr", "un", "who", "imf", "nato", "eu", "wto", "fbi", "cia",
    "sebi", "rbi", "fda", "sec", "epa"
]

# ---- Load Models ----
logger.info("Loading local models...")
hybrid_clf = None
tfidf = None
local_transformer = None

try:
    hybrid_clf = joblib.load(MODEL_PATH)
    tfidf = joblib.load(TFIDF_PATH)
    logger.info("Local ML models loaded successfully")
except FileNotFoundError as e:
    logger.error("Model file not found: %s", e)
except Exception as e:
    logger.error("Error loading ML models: %s: %s", type(e).__name__, e)

try:
    if os.path.exists(TRANSFORMER_MODEL_PATH):
        local_transformer = pipeline(
            "text-classification",
            model=TRANSFORMER_MODEL_PATH,
            tokenizer=TRANSFORMER_MODEL_PATH,
            max_length=512,
            truncation=True
        )
        logger.info("Local Transformer loaded successfully")
    else:
        logger.warning("Local transformer path '%s' not found. Will attempt HuggingFace Inference API as fallback.", TRANSFORMER_MODEL_PATH)
except Exception as e:
    logger.warning("Could not load local transformer (%s: %s). Will attempt HuggingFace Inference API as fallback.", type(e).__name__, e)

if not hybrid_clf:
    logger.warning("ML model will not be available")
if not tfidf:
    logger.warning("TFIDF vectorizer not available")


# ======================================================================
# INPUT CLASSIFICATION
# ======================================================================

def classify_input(text):
    """
    Classify user input to determine verification routing.
    Returns: 'fact_claim', 'news_article', or 'mixed'
    
    - fact_claim: Short factual statements like "Earth orbits the Sun" (no event signals)
    - news_article: Headlines or articles about current events
    - mixed: Uncertain — run both Wikipedia and GNews
    """
    words = text.split()
    word_count = len(words)
    text_lower = text.lower()

    # Check for event/news keywords
    event_keyword_count = sum(1 for kw in EVENT_KEYWORDS if kw in text_lower)

    # Check for proper nouns (capitalized words that aren't first word)
    # Headlines like "USTR proposes..." or "India announces..." have proper nouns
    proper_nouns = [w for w in words[1:] if w[0].isupper() and w.lower() not in stop_words and len(w) > 1]
    has_multiple_proper_nouns = len(proper_nouns) >= 2
    has_proper_nouns = len(proper_nouns) >= 1
    
    # Check for numbers with % or specific figures (common in news)
    has_numbers = bool(re.search(r'\d+\.?\d*%|\$\d|\d+\s*(billion|million|crore|lakh)', text_lower))
    
    # Check if first word is all-caps acronym (like USTR, NASA, WHO)
    first_word_acronym = len(words) > 0 and words[0].isupper() and len(words[0]) >= 2 and words[0].isalpha()
    
    # News signal score — multiple signals compound
    news_signals = event_keyword_count + (2 if has_multiple_proper_nouns else (1 if has_proper_nouns and event_keyword_count > 0 else 0)) + (1 if has_numbers else 0) + (2 if first_word_acronym else 0)

    # Strong news signals → news_article regardless of length
    if news_signals >= 3:
        return "news_article"
    
    # Short text with at least some news signals → mixed (headline)
    if word_count < 25 and news_signals >= 1:
        return "mixed"

    # Short text with no signals at all → fact claim
    if word_count < 25 and news_signals == 0:
        return "fact_claim"

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

def get_hf_classification(text, max_retries=1):
    """
    Call HuggingFace Inference API as a fallback when local transformer is unavailable.
    Uses the model: anshy047/fake-news-detector-transformer

    Fails fast: overall time is capped at HF_TIMEOUT_S (default 6s, inside the analyze
    budget), at most `max_retries` retry, and no sleeping. A loading model (503) or a
    timeout is not retried.
    """
    if not HF_TOKEN:
        raise RuntimeError("HF_TOKEN is not configured")

    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    API_URL = f"https://api-inference.huggingface.co/models/{HF_MODEL_ID}"
    deadline = time.time() + HF_TIMEOUT_S

    for attempt in range(max_retries + 1):
        remaining = deadline - time.time()
        if remaining <= 0.25:
            raise RuntimeError("HF API time budget exhausted")
        try:
            response = _http.post(
                API_URL,
                headers=headers,
                json={"inputs": text[:512]},  # Truncate to model's max length
                timeout=remaining
            )
        except requests.exceptions.Timeout:
            raise RuntimeError("HF API timeout")
        except requests.exceptions.RequestException as e:
            if attempt < max_retries:
                logger.warning("HF API request error (attempt %d/%d): %s", attempt + 1, max_retries + 1, _safe_err(e))
                continue
            raise RuntimeError(f"HF API request failed: {type(e).__name__}")

        if response.status_code == 200:
            result = response.json()

            # HF returns [[{label, score}, ...]] or [{label, score}, ...]
            if isinstance(result, list) and len(result) > 0:
                # Unwrap nested list if needed
                predictions = result[0] if isinstance(result[0], list) else result
                if isinstance(predictions, list) and len(predictions) > 0:
                    best = max(predictions, key=lambda x: x.get('score', 0))
                    return best.get('label', 'UNKNOWN'), best.get('score', 0.5)

            if isinstance(result, dict) and "error" in result:
                # Includes "model is loading" — fail fast rather than sleeping
                raise RuntimeError("HF API returned an error response")
            raise RuntimeError("HF API returned an unexpected response")

        if response.status_code == 503:
            # Model loading / unavailable: fail fast (no sleep in the request path)
            raise RuntimeError("HF API 503: model loading or unavailable")

        if response.status_code >= 500 and attempt < max_retries:
            logger.warning("HF API error %d (attempt %d/%d)", response.status_code, attempt + 1, max_retries + 1)
            continue
        raise RuntimeError(f"HF API error {response.status_code}")

    raise RuntimeError("Failed to get HF API response")


# ======================================================================
# WIKIPEDIA VERIFICATION
# ======================================================================

def extract_entity_for_wiki(text):
    """
    Extract the most relevant entity/topic from text for Wikipedia lookup.
    Focuses on proper nouns, capitalized words, and key subjects.
    
    For relational claims like "X is capital of Y", returns BOTH entities
    separately so we can verify from multiple angles.
    
    Returns a search query string (primary entity) or a list of queries
    if multiple lookups are needed.
    """
    # Remove common noise words and punctuation
    clean = re.sub(r'[^\w\s]', '', text)
    words = clean.split()
    text_lower = text.lower().strip()

    # Strategy 0: Detect relational claims like "X is capital of Y"
    # For these, we need to look up BOTH the subject and the object separately
    relational_match = re.match(
        r'^(.+?)\s+is\s+(?:the\s+)?(?:capital|president|prime minister|king|queen|leader|founder|ceo|chairman|head)\s+of\s+(.+?)$',
        text_lower.rstrip('.')
    )
    if relational_match:
        subject = relational_match.group(1).strip()
        obj = relational_match.group(2).strip()
        # Return the subject as primary query — we want to check what the subject actually is
        # e.g., for "India is capital of London", look up "India" to see it's a country, not a city
        # The caller (_run_wikipedia_verification) will also try the object
        return subject.title()

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
    Search Wikipedia REST API for a summary of the given query (cached).
    Uses the page summary endpoint.
    Timeout: 1.5 seconds (hard limit per spec).

    Returns dict with title, extract, description, url or None on failure / not found.
    Only definitive answers (found / genuinely not found) are cached, never errors.
    """
    if not query or not query.strip():
        return None

    key = ("wiki", query.strip().lower())
    cached = _cache_get(key)
    if cached is not _MISS:
        return cached

    result, definitive = _search_wikipedia_uncached(query, timeout)
    if definitive:
        _cache_set(key, result)
    return result


def _search_wikipedia_uncached(query, timeout=1.5):
    """Returns (result_or_None, definitive: bool). definitive=False means a lookup error."""
    # URL-encode the query for the REST API
    encoded_query = urllib.parse.quote(query.strip().replace(' ', '_'))
    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{encoded_query}"

    try:
        response = _http.get(
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
                    return _search_wikipedia_uncached(first_word, timeout=timeout)
                return None, True

            return {
                "title": data.get("title", ""),
                "extract": data.get("extract", ""),
                "description": data.get("description", ""),
                "url": data.get("content_urls", {}).get("desktop", {}).get("page", ""),
                "found": True
            }, True
        elif response.status_code == 404:
            # Try Wikipedia search API as fallback
            return _wikipedia_search_fallback(query, timeout)
        else:
            logger.warning("Wikipedia API returned status %d", response.status_code)
            return None, False

    except requests.exceptions.Timeout:
        logger.warning("Wikipedia API timed out for query: '%s'", query)
        return None, False
    except Exception as e:
        logger.warning("Wikipedia API error: %s", _safe_err(e))
        return None, False


def _wikipedia_search_fallback(query, timeout=1.5):
    """
    Fallback: use Wikipedia search API to find the best matching article.
    Returns (result_or_None, definitive: bool).
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
        resp = _http.get(search_url, params=params, timeout=timeout,
                         headers={"User-Agent": "FakeNewsDetector/1.0"})
        if resp.status_code == 200:
            data = resp.json()
            results = data.get("query", {}).get("search", [])
            if results:
                title = results[0].get("title", "")
                # Now fetch the summary for this title
                return _search_wikipedia_uncached(title, timeout=timeout)
            return None, True
        return None, False
    except Exception as e:
        logger.warning("Wikipedia search fallback error: %s", _safe_err(e))
        return None, False


def _detect_contradiction(user_text, wiki_description, wiki_extract, wiki_title):
    """
    Detect if the user's claim CONTRADICTS Wikipedia information.
    Handles cases like:
      - "Lucknow is capital of India" vs Wikipedia: "Capital of Uttar Pradesh, India"
      - "India is capital of London" vs Wikipedia: "India is a country" (not a capital)
      - "Earth is flat" vs Wikipedia: "third planet from the Sun"
    
    Returns (is_contradicted: bool, contradiction_detail: str)
    """
    user_lower = user_text.lower().strip()
    desc_lower = (wiki_description or "").lower()
    extract_lower = (wiki_extract or "").lower()
    wiki_full = f"{desc_lower} {extract_lower}"

    # Pattern 1: "X is capital of Y" claims
    capital_claim_match = re.match(
        r'^(.+?)\s+is\s+(?:the\s+)?capital\s+of\s+(.+?)$',
        user_lower.rstrip('.')
    )
    if capital_claim_match:
        claimed_capital = capital_claim_match.group(1).strip()  # e.g., "india"
        claimed_country = capital_claim_match.group(2).strip()  # e.g., "london"

        # Check 1: Is the subject (claimed capital) actually described as a capital by Wikipedia?
        # If Wikipedia says the subject is a "country", "continent", etc., it can't be a capital
        non_capital_types = ['country', 'continent', 'ocean', 'sea', 'river', 'mountain',
                            'planet', 'star', 'galaxy', 'language', 'religion',
                            'sport', 'game', 'company', 'organization']
        wiki_title_lower = (wiki_title or "").lower()
        
        # Check if the Wikipedia article we found is about the subject (claimed capital)
        if wiki_title_lower and (
            claimed_capital in wiki_title_lower or wiki_title_lower in claimed_capital
        ):
            # We're looking at the subject's Wikipedia page
            for non_type in non_capital_types:
                if non_type in desc_lower:
                    return True, (
                        f"'{claimed_capital.title()}' is described by Wikipedia as a "
                        f"{non_type}, not a capital city. It cannot be the capital of "
                        f"'{claimed_country.title()}'"
                    )
            
            # Check 1b: Subject IS described as a capital, but of a DIFFERENT place
            # e.g., "Lucknow is capital of India" but Wikipedia says "Capital of Uttar Pradesh"
            if 'capital' in desc_lower:
                wiki_capital_of_match = re.search(
                    r'capital\s+(?:of|city\s+of)\s+([\w\s,]+?)(?:\.|$|,)',
                    desc_lower
                )
                if wiki_capital_of_match:
                    wiki_capital_of = wiki_capital_of_match.group(1).strip().rstrip('.')
                    # Check if claimed_country matches what wiki says this is a capital of
                    claimed_words = set(claimed_country.lower().split())
                    wiki_of_words = set(wiki_capital_of.lower().split())
                    # Remove common noise words for comparison
                    noise = {'and', 'the', 'of'}
                    claimed_words -= noise
                    wiki_of_words -= noise
                    if claimed_words and wiki_of_words and not claimed_words.intersection(wiki_of_words):
                        return True, (
                            f"'{claimed_capital.title()}' is the capital of "
                            f"'{wiki_capital_of.title()}', not '{claimed_country.title()}' as claimed"
                        )
            
            # Also check: does Wikipedia say this IS a capital? If not, that's suspicious
            if 'capital' not in desc_lower and 'capital' not in extract_lower[:200]:
                return True, (
                    f"Wikipedia does not describe '{claimed_capital.title()}' as a capital city. "
                    f"Wikipedia says: \"{wiki_description}\""
                )

        # Check 2: If we're looking at the object's (claimed country's) page,
        # check what Wikipedia says the actual capital is
        if wiki_title_lower and (
            claimed_country in wiki_title_lower or wiki_title_lower in claimed_country
        ):
            # Look for "capital of" in wiki description/extract
            wiki_capital_match = re.search(
                r'capital\s+(?:of|city\s+of|and)\s+([\w\s,]+?)(?:\.|$|,)',
                desc_lower
            )
            if not wiki_capital_match:
                wiki_capital_match = re.search(
                    r'capital\s+(?:of|city\s+of|and)\s+([\w\s,]+?)(?:\.|$|,)',
                    extract_lower
                )
            
            if wiki_capital_match:
                wiki_says_capital_of = wiki_capital_match.group(1).strip().rstrip('.')
                user_claim_words = set(claimed_country.lower().split())
                wiki_claim_words = set(wiki_says_capital_of.lower().split())
                if not user_claim_words.issubset(wiki_claim_words) and not wiki_claim_words.issubset(user_claim_words):
                    return True, (
                        f"User claims '{claimed_capital.title()}' is capital of "
                        f"'{claimed_country.title()}' but Wikipedia says capital of "
                        f"'{wiki_says_capital_of.title()}'"
                    )
            
            # Also check: Wikipedia says the actual capital in its extract
            # e.g. "The capital of India is New Delhi" or "its capital is New Delhi"
            actual_capital_match = re.search(
                r'(?:the\s+)?capital\s+(?:is|city\s+is|,)\s+([\w\s]+?)(?:\.|,|\s+and\s)',
                extract_lower
            )
            if actual_capital_match:
                actual_capital = actual_capital_match.group(1).strip()
                if claimed_capital not in actual_capital and actual_capital not in claimed_capital:
                    return True, (
                        f"Wikipedia states the capital is '{actual_capital.title()}', "
                        f"not '{claimed_capital.title()}'"
                    )

    # Pattern 1b: Legacy "X is capital of Y" check (when wiki was found for a different entity)
    capital_match = re.search(r'(?:is|the)\s+(?:the\s+)?capital\s+of\s+([\w\s]+)', user_lower)
    if capital_match and not capital_claim_match:
        user_claims_capital_of = capital_match.group(1).strip().rstrip('.')
        wiki_capital_match = re.search(r'capital\s+(?:of|city\s+of)\s+([\w\s,]+?)(?:\.|$|,)', desc_lower)
        if not wiki_capital_match:
            wiki_capital_match = re.search(r'capital\s+(?:of|city\s+of)\s+([\w\s,]+?)(?:\.|$|,)', extract_lower)
        
        if wiki_capital_match:
            wiki_says_capital_of = wiki_capital_match.group(1).strip().rstrip('.')
            user_claim_words = set(user_claims_capital_of.lower().split())
            wiki_claim_words = set(wiki_says_capital_of.lower().split())
            if not user_claim_words.issubset(wiki_claim_words) and not wiki_claim_words.issubset(user_claim_words):
                return True, f"User claims capital of '{user_claims_capital_of}' but Wikipedia says capital of '{wiki_says_capital_of}'"

    # Pattern 2: "X is Y" simple factual claims — check if Wikipedia description contradicts
    is_match = re.match(r'^(.+?)\s+is\s+(.+?)$', user_lower.rstrip('.'))
    if is_match and desc_lower:
        subject = is_match.group(1).strip()
        user_predicate = is_match.group(2).strip()
        # Check for direct numerical/factual contradictions
        user_numbers = re.findall(r'\d+', user_predicate)
        wiki_numbers = re.findall(r'\d+', desc_lower)
        if user_numbers and wiki_numbers:
            pass  # Too complex for simple check, skip for now

    # Pattern 3: Flat Earth / well-known false claims
    flat_earth_patterns = [
        (r'earth\s+is\s+flat', 'Wikipedia describes Earth as a planet, not flat'),
        (r'sun\s+revolves?\s+around\s+(?:the\s+)?earth', 'Wikipedia states Earth orbits the Sun'),
        (r'moon\s+landing\s+(?:was|is)\s+(?:fake|hoax|faked)', 'Wikipedia documents the Apollo moon landings as real'),
    ]
    for pattern, detail in flat_earth_patterns:
        if re.search(pattern, user_lower):
            return True, detail

    return False, ""


def verify_against_wikipedia(user_text, wiki_data):
    """
    Compare user's claim against Wikipedia data.
    Uses TF-IDF similarity, term overlap, AND contradiction detection.
    
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
    wiki_title = wiki_data.get("title", "")
    wiki_text = f"{description} {extract}"

    if not wiki_text.strip():
        return {
            "verification_score": 0,
            "status": "NOT FOUND",
            "message": "Wikipedia article has no content.",
            "insights": ["Wikipedia article found but contains no useful extract."],
            "wiki_title": wiki_title,
            "wiki_url": wiki_data.get("url", ""),
            "wiki_extract": ""
        }

    # ---- Step 0: Contradiction Detection ----
    is_contradicted, contradiction_detail = _detect_contradiction(
        user_text, description, extract, wiki_title
    )

    if is_contradicted:
        return {
            "verification_score": 5,
            "status": "CONTRADICTED",
            "message": f"Wikipedia CONTRADICTS this claim. {contradiction_detail}.",
            "insights": [
                f"⚠️ Wikipedia contradicts this statement.",
                f"{contradiction_detail}.",
                f"Wikipedia describes \"{wiki_title}\" as: {description}." if description else "",
            ],
            "wiki_title": wiki_title,
            "wiki_url": wiki_data.get("url", ""),
            "wiki_extract": extract[:300] if extract else "",
            "tfidf_score": 0,
            "term_overlap_score": 0,
            "is_contradicted": True
        }

    # ---- Step 1: TF-IDF cosine similarity ----
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

    # ---- Step 2: Key term overlap ----
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

    # ---- Step 3: Description match boost ----
    # If the Wikipedia description closely matches the claim, give a big boost
    desc_boost = 0
    if description:
        desc_lower = description.lower()
        user_lower = user_text.lower()
        # Check if key phrases from description appear in user text
        desc_words = set(w for w in re.sub(r'[^\w\s]', '', desc_lower).split()
                         if w not in stop_words and len(w) > 2)
        user_content = set(w for w in re.sub(r'[^\w\s]', '', user_lower).split()
                           if w not in stop_words and len(w) > 2)
        if desc_words and user_content:
            desc_overlap = len(desc_words & user_content) / len(desc_words)
            if desc_overlap > 0.6:
                desc_boost = 30  # Strong description match
            elif desc_overlap > 0.3:
                desc_boost = 15

    # Combined score: TF-IDF 40% + term overlap 30% + description boost 30%
    verification_score = round(
        tfidf_score * 0.4 + term_score * 0.3 + min(desc_boost, 30) * (100/30) * 0.3, 1
    )
    verification_score = min(100, verification_score)

    # Determine status
    if verification_score > 70:
        status = "VERIFIED FACT"
        message = f"Wikipedia confirms this claim. Article: \"{wiki_title}\"."
    elif verification_score > 45:
        status = "PARTIALLY VERIFIED"
        message = f"Wikipedia contains related information but not a full match. Article: \"{wiki_title}\"."
    elif verification_score > 20:
        status = "WEAK MATCH"
        message = f"Wikipedia has some related content but the claim could not be strongly verified."
    else:
        status = "NOT VERIFIED"
        message = f"The claim does not match Wikipedia content for \"{wiki_title}\"."

    # Generate insights
    insights = []
    if status == "VERIFIED FACT":
        insights.append(f"Wikipedia confirms the statement.")
        insights.append(f"Fact matches trusted knowledge sources.")
    elif status == "PARTIALLY VERIFIED":
        insights.append(f"Wikipedia contains related information in the article \"{wiki_title}\".")
    else:
        insights.append(f"Wikipedia article \"{wiki_title}\" found but claim could not be fully verified.")

    insights.append(f"Wikipedia verification score: {verification_score}%.")

    if description:
        insights.append(f"Wikipedia describes \"{wiki_title}\" as: {description}.")

    return {
        "verification_score": verification_score,
        "status": status,
        "message": message,
        "insights": insights,
        "wiki_title": wiki_title,
        "wiki_url": wiki_data.get("url", ""),
        "wiki_extract": extract[:300] if extract else "",
        "tfidf_score": tfidf_score,
        "term_overlap_score": term_score,
        "is_contradicted": False
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


def _clean_search_query(query):
    """
    Clean a query string for news API search.
    Extracts up to 6 of the most important keywords to ensure we don't over-constrain the search API.
    """
    clean = re.sub(r'[^\w\s]', '', query)
    words = clean.split()
    
    # Filter out stopwords and short words
    content_words = [w for w in words if w.lower() not in stop_words and len(w) > 3]
    
    # If we have too many keywords, try to prioritize capitalized words (likely proper nouns)
    if len(content_words) > 6:
        capitalized = [w for w in content_words if w[0].isupper()]
        others = [w for w in content_words if not w[0].isupper()]
        # Take all capitalized, then fill the rest with other long words up to 6
        selected = capitalized + others
        content_words = selected[:6]
    elif len(content_words) == 0:
        # Fallback if everything was filtered out
        content_words = words[:6]
        
    return ' '.join(content_words)


def search_gnews(query, max_results=10, timeout=2.0):
    """
    Search GNews API for articles matching the query.
    Returns a list of article dicts, or None if the request failed
    (an empty list means "no results", and is distinct from an error).
    """
    if not GNEWS_API_KEY:
        return []

    clean_query = _clean_search_query(query)
    url = "https://gnews.io/api/v4/search"
    params = {
        "q": clean_query,
        "lang": "en",
        "max": max_results,
        "apikey": GNEWS_API_KEY
    }

    try:
        response = _http.get(url, params=params, timeout=timeout)
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
        else:
            logger.warning("GNews API error status %d", response.status_code)
            return None
    except Exception as e:
        logger.warning("GNews error: %s", _safe_err(e))
        return None


def search_newsdata(query, max_results=10, timeout=2.0):
    """
    Search NewsData.io API for articles matching the query.
    Returns a list of article dicts (same format as search_gnews).
    """
    if not NEWSDATA_API_KEY:
        return []

    clean_query = _clean_search_query(query)
    url = "https://newsdata.io/api/1/latest"
    params = {
        "apikey": NEWSDATA_API_KEY,
        "q": clean_query,
        "language": "en",
        "size": max_results
    }

    try:
        response = _http.get(url, params=params, timeout=timeout)
        if response.status_code == 200:
            data = response.json()
            results = data.get("results", [])
            if not results:
                return []
            return [{
                "title": a.get("title", ""),
                "description": a.get("description", "") or "",
                "source": a.get("source_id", "Unknown"),
                "url": a.get("link", ""),
                "published_at": a.get("pubDate", "")
            } for a in results if a.get("title")]
        else:
            logger.warning("NewsData.io API error status %d", response.status_code)
            return None
    except Exception as e:
        logger.warning("NewsData.io error: %s", _safe_err(e))
        return None


def search_newsapi(query, max_results=10, timeout=2.0):
    """
    Search NewsAPI.org for articles matching the query.
    Returns a list of article dicts (same format as search_gnews).
    Note: Free tier only works from localhost (developer use).
    """
    if not NEWSAPI_KEY:
        return []

    clean_query = _clean_search_query(query)
    url = "https://newsapi.org/v2/everything"
    params = {
        "apiKey": NEWSAPI_KEY,
        "q": clean_query,
        "language": "en",
        "sortBy": "relevancy",
        "pageSize": max_results
    }

    try:
        response = _http.get(url, params=params, timeout=timeout)
        if response.status_code == 200:
            data = response.json()
            articles = data.get("articles", [])
            return [{
                "title": a.get("title", ""),
                "description": a.get("description", "") or "",
                "source": a.get("source", {}).get("name", "Unknown"),
                "url": a.get("url", ""),
                "published_at": a.get("publishedAt", "")
            } for a in articles if a.get("title") and a.get("title") != "[Removed]"]
        else:
            logger.warning("NewsAPI.org error status %d", response.status_code)
            return None
    except Exception as e:
        logger.warning("NewsAPI.org error: %s", _safe_err(e))
        return None


def search_serpapi(query, max_results=10, timeout=3.0):
    """
    Search SerpAPI (Google News engine) for articles matching the query.
    Used as a FALLBACK when primary sources return insufficient results.
    Google News provides the most comprehensive real-time coverage.
    
    Returns a list of article dicts (same format as other search functions).
    """
    if not SERPAPI_KEY:
        return []

    clean_query = _clean_search_query(query)
    url = "https://serpapi.com/search"
    params = {
        "engine": "google_news",
        "q": clean_query,
        "gl": "us",
        "hl": "en",
        "api_key": SERPAPI_KEY
    }

    try:
        response = _http.get(url, params=params, timeout=timeout)
        if response.status_code == 200:
            data = response.json()
            articles = []

            # SerpAPI returns news_results as an array
            news_results = data.get("news_results", [])
            for item in news_results[:max_results]:
                # Each item can have "stories" (cluster) or be a single article
                if "stories" in item:
                    # Clustered stories — extract individual articles
                    for story in item["stories"][:3]:
                        articles.append({
                            "title": story.get("title", ""),
                            "description": story.get("snippet", "") or "",
                            "source": story.get("source", {}).get("name", "Unknown") if isinstance(story.get("source"), dict) else story.get("source", "Unknown"),
                            "url": story.get("link", ""),
                            "published_at": story.get("date", "")
                        })
                else:
                    source = item.get("source", {})
                    source_name = source.get("name", "Unknown") if isinstance(source, dict) else str(source)
                    articles.append({
                        "title": item.get("title", ""),
                        "description": item.get("snippet", "") or "",
                        "source": source_name,
                        "url": item.get("link", ""),
                        "published_at": item.get("date", "")
                    })

            logger.info("[SerpAPI] Found %d articles", len(articles))
            return articles
        else:
            logger.warning("SerpAPI error status %d", response.status_code)
            return None
    except Exception as e:
        logger.warning("SerpAPI error: %s", _safe_err(e))
        return None


def _deduplicate_articles(articles):
    """Deduplicate articles by title similarity."""
    seen_titles = set()
    unique = []
    for article in articles:
        title = article.get('title', '')
        if not title:
            continue
        title_key = title.lower().strip()[:60]
        if title_key not in seen_titles:
            seen_titles.add(title_key)
            unique.append(article)
    return unique


def search_all_news_sources(query, max_results=10, timeout=2.0):
    """
    Smart multi-source news search with SerpAPI fallback (cached by normalised query).

    Strategy:
      1. Run GNews, NewsData.io, NewsAPI.org in PARALLEL (fast, 2s timeout)
      2. If combined results < 3 articles → trigger SerpAPI fallback (Google News)
      3. Deduplicate all articles by title

    SerpAPI is the fallback because:
      - It's the most reliable (Google News under the hood)
      - It has the best real-time coverage
      - But it has limited free credits, so we conserve calls

    Results are cached for CACHE_TTL_S unless any source errored (so failures
    are never cached as "no results").
    """
    cache_key = ("news", _clean_search_query(query).lower(), max_results)
    cached = _cache_get(cache_key)
    if cached is not _MISS:
        logger.info("[News] cache hit")
        return list(cached)

    all_articles = []
    sources_used = []
    had_error = False

    # ---- Phase 1: Primary sources in parallel (shared pool, no blocking shutdown) ----
    futures = {
        'gnews': _IO_EXECUTOR.submit(search_gnews, query, max_results, timeout),
        'newsdata': _IO_EXECUTOR.submit(search_newsdata, query, max_results, timeout),
        'newsapi': _IO_EXECUTOR.submit(search_newsapi, query, max_results, timeout),
    }
    futures_wait(list(futures.values()), timeout=timeout + 1)

    for source_name, future in futures.items():
        if not future.done():
            future.cancel()
            had_error = True
            logger.warning("[News] %s timed out", source_name)
            continue
        try:
            articles = future.result()
        except Exception as e:
            had_error = True
            logger.warning("[News] %s failed: %s", source_name, _safe_err(e))
            continue
        if articles is None:
            had_error = True
        elif articles:
            sources_used.append(source_name)
            all_articles.extend(articles)

    # Deduplicate after phase 1
    all_articles = _deduplicate_articles(all_articles)
    logger.info("[News] Phase 1: %d articles from %s", len(all_articles), sources_used)

    # ---- Phase 2: SerpAPI fallback if results are insufficient ----
    if len(all_articles) < 3 and SERPAPI_KEY:
        logger.info("[News] Phase 1 returned only %d articles — triggering SerpAPI fallback", len(all_articles))
        try:
            serp_articles = search_serpapi(query, max_results=max_results, timeout=3.0)
            if serp_articles is None:
                had_error = True
            elif serp_articles:
                sources_used.append('serpapi')
                all_articles.extend(serp_articles)
                all_articles = _deduplicate_articles(all_articles)
                logger.info("[News] Phase 2: SerpAPI added %d articles, total now %d", len(serp_articles), len(all_articles))
        except Exception as e:
            had_error = True
            logger.warning("[News] SerpAPI fallback failed: %s", _safe_err(e))

    logger.info("[News] Final: %d articles from %s", len(all_articles), sources_used)
    if not had_error:
        _cache_set(cache_key, list(all_articles))
    return all_articles


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
        logger.warning("Similarity computation error: %s", _safe_err(e))
        return [0.0] * len(articles)


def is_trusted_source(source_name):
    """Check if a source name contains a trusted source name as a whole phrase."""
    if not isinstance(source_name, str):
        return False
    # "BBC-News", "apnews.com", "www.reuters.com" -> space-separated lowercase words
    normalized = re.sub(r"[^a-z0-9]+", " ", source_name.lower()).strip()
    if not normalized:
        return False
    return any(p.search(normalized) for p in _TRUSTED_SOURCE_PATTERNS)


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

    # Count supporting articles (similarity > 5% — lowered to catch news headlines
    # that use different wording than the original)
    supporting_threshold = 5
    supporting = [(articles[i], similarity_scores[i])
                  for i in range(len(articles))
                  if similarity_scores[i] > supporting_threshold]

    supporting_count = len(supporting)

    # Count trusted sources among ALL articles (not just supporting)
    # A trusted source with even moderate similarity is meaningful
    trusted_all = [(articles[i], similarity_scores[i])
                   for i in range(len(articles))
                   if is_trusted_source(articles[i]['source']) and similarity_scores[i] > 5]
    trusted_count = len(trusted_all)

    # Calculate verification score using WEIGHTED top matches
    # Best match gets highest weight so a single strong match isn't diluted
    if similarity_scores:
        sorted_scores = sorted(similarity_scores, reverse=True)
        
        # Weighted scoring: best match 40%, 2nd 25%, 3rd 15%, 4th 10%, 5th 10%
        weights = [0.40, 0.25, 0.15, 0.10, 0.10]
        weighted_sum = 0
        for i, w in enumerate(weights):
            if i < len(sorted_scores):
                weighted_sum += sorted_scores[i] * w
        
        verification_score = round(weighted_sum, 1)
        
        # Boost if trusted source has a strong match
        best_trusted_score = max([s for _, s in trusted_all], default=0)
        if best_trusted_score > 30:
            # Trusted source with decent match gets a significant boost
            boost = min(best_trusted_score * 0.3, 20)
            verification_score = round(min(100, verification_score + boost), 1)
    else:
        verification_score = 0

    # Determine status — thresholds calibrated for real-world news headlines
    # A BBC article at 52% similarity should be "LIKELY SUPPORTED" at minimum
    if supporting_count == 0:
        status = "UNVERIFIED"
        message = "No sufficient online evidence available."
    elif verification_score > 50 and trusted_count >= 2:
        status = "VERIFIED"
        message = "Multiple trusted news organizations independently confirm this claim."
    elif verification_score > 35 and trusted_count >= 1:
        status = "VERIFIED"
        message = f"Trusted news source confirms this claim (top match: {sorted_scores[0]:.0f}%)."
    elif verification_score > 25 and (trusted_count >= 1 or supporting_count >= 3):
        status = "LIKELY SUPPORTED"
        message = "Evidence from trusted sources supports this claim."
    elif verification_score > 15 and supporting_count >= 2:
        status = "PARTIALLY SUPPORTED"
        message = "Some online sources report similar content."
    elif supporting_count >= 1:
        status = "PARTIALLY SUPPORTED"
        message = "Limited online evidence found."
    else:
        status = "UNVERIFIED"
        message = "Insufficient evidence from trusted sources to verify this claim."

    # Generate insights
    insights = []
    insights.append(f"{supporting_count} supporting article{'s' if supporting_count != 1 else ''} found online.")

    # Add top trusted source matches
    for article, score in sorted(trusted_all, key=lambda x: x[1], reverse=True)[:4]:
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
            t_label = _normalize_label(best_pred['label'])
            t_conf = best_pred['score']
            source = "local"

        # Fallback to HuggingFace API
        if t_label is None:
            raw_label, raw_conf = get_hf_classification(text)
            t_label = _normalize_label(raw_label)
            t_conf = raw_conf
            source = "remote"

        return {"label": t_label, "confidence": round(float(t_conf), 4), "source": source}
    except Exception as e:
        logger.warning("Transformer task error: %s", _safe_err(e))
        return None


def _run_hybrid_ml(text):
    """Run LR + TF-IDF model prediction. Returns dict or None."""
    try:
        if hybrid_clf and tfidf:
            vectorized = tfidf.transform([text])
            h_probs = hybrid_clf.predict_proba(vectorized)[0]
            h_pred_class = np.argmax(h_probs)
            h_conf = float(h_probs[h_pred_class])
            h_label = _normalize_label(hybrid_clf.classes_[h_pred_class])
            return {"label": h_label, "confidence": round(h_conf, 4)}
        return None
    except Exception as e:
        logger.warning("Hybrid ML task error: %s", _safe_err(e))
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
        logger.warning("Linguistic task error: %s", _safe_err(e))
        return None


def _run_wikipedia_verification(text, input_type):
    """Run Wikipedia verification. Always runs for all input types.
    
    For relational claims like 'X is capital of Y', performs TWO Wikipedia lookups:
    1. Look up the subject (X) to check if it can even be a capital
    2. Look up the object (Y) to check what the real capital is
    The strongest contradiction result wins.
    """
    try:
        entity = extract_entity_for_wiki(text)
        logger.info("[Wikipedia] Searching for entity: '%s'", entity)

        wiki_data = search_wikipedia(entity, timeout=1.5)
        
        # For relational claims, also try looking up the object entity
        text_lower = text.lower().strip()
        relational_match = re.match(
            r'^(.+?)\s+is\s+(?:the\s+)?(?:capital|president|prime minister|king|queen|leader|founder|ceo|chairman|head)\s+of\s+(.+?)$',
            text_lower.rstrip('.')
        )
        
        result_subject = None
        result_object = None
        
        if wiki_data:
            result_subject = verify_against_wikipedia(text, wiki_data)
        
        # If this is a relational claim, also look up the object (e.g., "London" in "India is capital of London")
        if relational_match:
            obj_entity = relational_match.group(2).strip().title()
            if obj_entity.lower() != entity.lower():
                logger.info("[Wikipedia] Also searching for object entity: '%s'", obj_entity)
                wiki_data_obj = search_wikipedia(obj_entity, timeout=1.5)
                if wiki_data_obj:
                    result_object = verify_against_wikipedia(text, wiki_data_obj)
        
        # Pick the best result — prioritize contradictions
        if result_subject and result_subject.get('is_contradicted'):
            return result_subject
        if result_object and result_object.get('is_contradicted'):
            return result_object
        
        # If neither found a contradiction, return the one with the higher score
        # (or lower score if both are low — the point is to return the most informative result)
        if result_subject and result_object:
            # If one has a much higher verification score, use it
            # But if both are low, prefer the one that at least found something
            if result_subject.get('status') == 'NOT FOUND' and result_object.get('status') != 'NOT FOUND':
                return result_object
            if result_object.get('status') == 'NOT FOUND' and result_subject.get('status') != 'NOT FOUND':
                return result_subject
            # Return the one with higher verification score
            if (result_object.get('verification_score', 0) > result_subject.get('verification_score', 0)):
                return result_object
            return result_subject
        
        if result_subject:
            return result_subject
        if result_object:
            return result_object

        return {
            "verification_score": 0,
            "status": "NOT FOUND",
            "message": "No Wikipedia article found.",
            "insights": ["No relevant Wikipedia article found for verification."],
            "wiki_title": "",
            "wiki_url": "",
            "wiki_extract": ""
        }
    except Exception as e:
        logger.warning("Wikipedia task error: %s", _safe_err(e))
        return None


def _run_news_verification(text, input_type):
    """Run multi-source news verification. Always runs for all input types."""
    try:
        claim = extract_main_claim(text)
        logger.info("[News] Searching all sources for: '%s'", claim[:80])

        # Search ALL news APIs in parallel
        articles = search_all_news_sources(claim, timeout=2.0)
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
        logger.warning("News verification task error: %s", _safe_err(e))
        return None


# ======================================================================
# REQUEST VALIDATION & ERROR HANDLERS
# ======================================================================

def _get_validated_text():
    """
    Parse and validate the JSON body. Returns (text, None) on success or
    (None, (response, status)) on failure.
    """
    data = request.get_json(silent=True)
    if not isinstance(data, dict) or 'text' not in data:
        return None, (jsonify({"error": "No text provided"}), 400)

    text = data['text']
    if not isinstance(text, str):
        return None, (jsonify({"error": "Text must be a string"}), 400)

    text = text.strip()
    if not text:
        return None, (jsonify({"error": "Empty text provided"}), 400)
    if len(text) > MAX_TEXT_CHARS:
        return None, (jsonify({"error": f"Text too long (max {MAX_TEXT_CHARS} characters)"}), 413)

    return text, None


@app.errorhandler(413)
def handle_request_too_large(e):
    return jsonify({"error": "Request too large"}), 413


@app.errorhandler(429)
def handle_rate_limited(e):
    return jsonify({"error": "Too many requests. Please slow down and try again shortly."}), 429


@app.errorhandler(405)
def handle_method_not_allowed(e):
    return jsonify({"error": "Method not allowed"}), 405


@app.errorhandler(404)
def handle_not_found(e):
    return jsonify({"error": "Not found"}), 404


# ======================================================================
# API ENDPOINTS
# ======================================================================

@app.route('/', methods=['GET'])
@limiter.exempt
def health():
    return jsonify({
        "status": "API is running",
        "models": {
            "lr_tfidf": hybrid_clf is not None,
            "transformer_local": local_transformer is not None,
            "transformer_remote": bool(HF_TOKEN)
        }
    }), 200


def _timed_task(tracker, name, fn, *args):
    """Run fn(*args), recording the task's own start/end times in tracker[name]."""
    entry = tracker[name] = {"start": time.time(), "end": None}
    try:
        return fn(*args)
    finally:
        entry["end"] = time.time()


@app.route('/analyze', methods=['POST'])
@limiter.limit(RATE_LIMIT)
def analyze():
    """
    Unified analysis endpoint — runs ALL tasks in parallel under ONE overall deadline.

    Tasks:
      1. Transformer Model
      2. Hybrid ML Model
      3. Linguistic Analysis (includes clickbait analysis)
      4. Wikipedia Verification
      5. News Verification

    Tasks run on a shared thread pool. Anything not finished within
    ANALYZE_BUDGET_S is reported as "timeout" and does not block the response.
    Returns a unified response with all results.
    """
    text, err = _get_validated_text()
    if err:
        return err

    start_time = time.time()

    try:
        # Step 1: Classify input type
        input_type = classify_input(text)
        logger.info("[Analyze] Input type: %s | Length: %d words", input_type, len(text.split()))

        # Step 2: Run ALL tasks in parallel on the shared pool
        tracker = {}
        task_fns = {
            'transformer': (_run_transformer, text),
            'hybrid': (_run_hybrid_ml, text),
            'linguistic': (_run_linguistic, text),
            'wikipedia': (_run_wikipedia_verification, text, input_type),
            'gnews': (_run_news_verification, text, input_type),
        }
        futures = {
            name: _EXECUTOR.submit(_timed_task, tracker, name, fn_args[0], *fn_args[1:])
            for name, fn_args in task_fns.items()
        }

        # Step 3: ONE overall deadline for all tasks
        remaining = max(0.0, ANALYZE_BUDGET_S - (time.time() - start_time))
        futures_wait(list(futures.values()), timeout=remaining)

        results = {}
        task_status = {}
        for name, future in futures.items():
            if not future.done():
                future.cancel()  # only effective if not yet started
                task_status[name] = "timeout"
                results[name] = None
                logger.warning("[Analyze] Task '%s' timed out after %.1fs budget", name, ANALYZE_BUDGET_S)
                continue
            try:
                results[name] = future.result()
            except Exception as e:
                results[name] = None
                logger.warning("[Analyze] Task '%s' failed: %s", name, _safe_err(e))
            task_status[name] = "ok" if results[name] is not None else "error"

        # timings = each task's own run duration (elapsed-so-far for tasks still running)
        now = time.time()
        timings = {}
        for name in futures:
            entry = tracker.get(name)
            if entry is None:
                timings[name] = 0.0
            else:
                timings[name] = round((entry["end"] or now) - entry["start"], 3)

        transformer = results.get('transformer')
        hybrid = results.get('hybrid')
        linguistic = results.get('linguistic')
        wikipedia = results.get('wikipedia')
        gnews = results.get('gnews')

        # All models AND all verification sources failed -> nothing useful to return
        if transformer is None and hybrid is None and wikipedia is None and gnews is None:
            logger.error("[Analyze] All models and verification sources failed: %s", task_status)
            return jsonify({"error": "Analysis service temporarily unavailable"}), 503

        # Step 4: Fallbacks. A failed model reports UNKNOWN; models are never
        # substituted for each other.
        if transformer is None:
            transformer = {"label": "UNKNOWN", "confidence": 0.5, "source": "none"}

        if hybrid is None:
            hybrid = {"label": "UNKNOWN", "confidence": 0.5}

        if linguistic is None:
            linguistic = {
                "uppercase": 0, "punctuation": 0,
                "clickbait": 0, "complexity": 5, "word_count": len(text.split())
            }

        failed_tasks = [name for name, status in task_status.items() if status != "ok"]

        # Step 5: Build response
        total_time = round(time.time() - start_time, 3)
        logger.info("[Analyze] Total time: %ss | Timings: %s | Status: %s", total_time, timings, task_status)

        payload = {
            "input_type": input_type,
            "transformer": transformer,
            "hybrid": hybrid,
            "raw_features": linguistic,
            "wikipedia": wikipedia,
            "verification": gnews,
            "timings": timings,
            "total_time": total_time,
            "task_status": task_status,
            "degraded": bool(failed_tasks),
            "failed_tasks": failed_tasks
        }

        return jsonify(payload)

    except Exception:
        logger.exception("Analyze error")
        return jsonify({"error": "Internal server error"}), 500


# ---- Legacy Endpoints (backward compatible) ----

@app.route('/predict', methods=['POST'])
@limiter.limit(RATE_LIMIT)
def predict():
    text, err = _get_validated_text()
    if err:
        return err

    try:
        # 1. TRANSFORMER MODEL PREDICTION
        transformer = _run_transformer(text)

        # 2. LR + TF-IDF MODEL PREDICTION
        hybrid = _run_hybrid_ml(text)

        # 3. EXTRACT LINGUISTIC FEATURES
        ling_feats, uppercase_ratio, punct_ratio = extract_features(text)
        clickbait_score = compute_clickbait_score(text, uppercase_ratio, punct_ratio)
        word_count = len(text.split())
        avg_word_length = np.mean([len(w) for w in text.split()]) if word_count > 0 else 0

        # 4. HANDLE MISSING MODELS (no cross-copying between models)
        if transformer is None:
            transformer = {"label": "UNKNOWN", "confidence": 0.5, "source": "none"}
        if hybrid is None:
            hybrid = {"label": "UNKNOWN", "confidence": 0.5}

        # 5. BUILD RESPONSE
        payload = {
            "transformer": transformer,
            "hybrid": hybrid,
            "raw_features": {
                "uppercase": round(float(uppercase_ratio), 4),
                "punctuation": round(float(punct_ratio), 4),
                "clickbait": round(float(clickbait_score), 4),
                "complexity": round(float(avg_word_length), 2),
                "word_count": word_count
            }
        }

        return jsonify(payload)

    except Exception:
        logger.exception("Predict error")
        return jsonify({"error": "Internal server error"}), 500


@app.route('/verify', methods=['POST'])
@limiter.limit(RATE_LIMIT)
def verify():
    """Legacy verify endpoint — kept for backward compatibility."""
    text, err = _get_validated_text()
    if err:
        return err

    try:
        claim = extract_main_claim(text)
        articles = search_gnews(claim, timeout=2.0) or []
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

    except Exception:
        logger.exception("Verify error")
        return jsonify({
            "error": "Verification service unavailable",
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
