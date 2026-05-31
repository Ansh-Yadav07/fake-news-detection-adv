import os
import numpy as np
import string
import re
import joblib
import requests
import time
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
    score += min(trigger_count * 0.12, 0.40)

    # 2. Excessive punctuation — !!! or ??? patterns (up to 0.15)
    exclamation_runs = len(re.findall(r'!{2,}', text))
    question_runs = len(re.findall(r'\?{2,}', text))
    score += min((exclamation_runs + question_runs) * 0.08, 0.15)

    # 3. General punctuation density (up to 0.10)
    if punct_ratio > 0.05:
        score += min((punct_ratio - 0.05) * 2.0, 0.10)

    # 4. Excessive uppercase (up to 0.15)
    if uppercase_ratio > 0.10:
        score += min((uppercase_ratio - 0.10) * 1.5, 0.15)

    # 5. ALL CAPS words pattern (up to 0.10)
    caps_words = sum(1 for w in words if w.isupper() and len(w) > 2)
    caps_ratio = caps_words / total_words
    if caps_ratio > 0.3:
        score += min((caps_ratio - 0.3) * 0.5, 0.10)

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


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    app.run(host='0.0.0.0', port=port, debug=False)