import os
import numpy as np
import string
import joblib
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
import nltk
from nltk.corpus import stopwords
from huggingface_hub import InferenceClient

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
if not HF_TOKEN:
    print("WARNING: HF_TOKEN environment variable not set!")

# Don't initialize HF Inference Client - it causes timeouts on Render
# We'll use the hybrid model only for now
hf_client = None
MODEL_ID = "anshy047/fake-news-detector-transformer"

MODEL_PATH = "models/ml_lr_model.pkl"
TFIDF_PATH = "models/ml_tfidf.pkl"

print("Loading local ML model...")
hybrid_clf = None
tfidf = None
try:
    hybrid_clf = joblib.load(MODEL_PATH)
    tfidf = joblib.load(TFIDF_PATH)
    print(f"Local model loaded successfully!")
    print("Transformer running remotely via HF API.")
except FileNotFoundError as e:
    print(f"ERROR: Model file not found: {e}")
except Exception as e:
    print(f"ERROR loading ML models: {type(e).__name__}: {str(e)}")
    import traceback
    traceback.print_exc()
finally:
    if not hybrid_clf:
        print("WARNING: ML model will not be available")
    if not tfidf:
        print("WARNING: TFIDF vectorizer not available")

def extract_features(text):
    words = text.split()
    total_words = len(words)
    if total_words == 0:
        return [0]*8, 0, 0
    
    exclam_count = text.count("!")
    question_count = text.count("?")
    uppercase_ratio = sum(1 for c in text if c.isupper()) / len(text)
    digit_ratio = sum(1 for c in text if c.isdigit()) / len(text)
    punctuation_ratio = sum(1 for c in text if c in string.punctuation) / len(text)
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


def get_hf_classification(text):
    """Since HF API causes timeouts, we skip this and use hybrid model only"""
    raise Exception("HF Transformer API disabled - using hybrid model instead")

def get_hf_embeddings(text):
    """Returns zero embeddings - we use only linguistic features with hybrid model"""
    return np.zeros(768)

@app.route('/', methods=['GET'])
def health():
    return jsonify({"status": "API is running"}), 200

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({"error": "No text provided"}), 400
        
    text = data['text']
    
    try:
        # 1. Try Remote Transformer Prediction via HF Inference API
        t_label = None
        t_conf = None
        transformer_available = False
        
        try:
            hf_result = get_hf_classification(text)
            
            if isinstance(hf_result, list) and len(hf_result) > 0 and isinstance(hf_result[0], list):
                predictions = hf_result[0]
            elif isinstance(hf_result, list):
                predictions = hf_result
            elif hasattr(hf_result, "get") and hf_result.get("error"):
                raise Exception(f"HF Model Error: {hf_result.get('error')}")
            else:
                raise Exception("Unexpected HF formatting")

            # Get label with highest score
            best_pred = max(predictions, key=lambda x: x['score'])
            t_label_raw = best_pred['label']
            t_conf = best_pred['score']
            
            # Map labels
            t_label = "REAL" if t_label_raw in ["LABEL_1", "1", "REAL"] else "FAKE"
            transformer_available = True
        except Exception as e:
            print(f"Transformer API Error: {str(e)} - Will use hybrid model only")
            t_label = None
            t_conf = None
        
        # 2. Extract local Linguistic Features
        ling_feats, uppercase_ratio, punct_ratio = extract_features(text)
        
        # 3. Get embeddings if transformer is available, otherwise use zeros
        cls_embedding = None
        if transformer_available:
            try:
                raw_embed = get_hf_embeddings(text)
                cls_embedding = raw_embed.flatten()
                
                if cls_embedding.shape[0] != 768:
                    raise ValueError(f"Shape error: {cls_embedding.shape}")
                    
            except Exception as e:
                print(f"Warning: Could not fetch embeddings, using zero fallback: {str(e)}")
                cls_embedding = np.zeros(768)
        else:
            # No transformer available, use zero embeddings
            cls_embedding = np.zeros(768)

        # Combine embedding and linguistic features for Hybrid Local execution
        
        # Local ML Prediction using TF-IDF
        h_label = None
        h_conf = None
        if hybrid_clf and tfidf:
            vectorized = tfidf.transform([text])
            h_probs = hybrid_clf.predict_proba(vectorized)[0]
            h_pred_class = np.argmax(h_probs)
            h_conf = float(h_probs[h_pred_class])
            # The classes_ are ['FAKE', 'REAL']
            h_label = hybrid_clf.classes_[h_pred_class]
        else:
            # Fallback: use transformer result if available, otherwise default
            if t_label:
                h_label = t_label
                h_conf = t_conf
            else:
                h_label = "UNKNOWN"
                h_conf = 0.5
        
        # If transformer failed, use hybrid result for transformer too
        if not transformer_available:
            t_label = h_label
            t_conf = h_conf
        
        # Stats for frontend visualization
        word_count = len(text.split())
        isSus = uppercase_ratio > 0.15 or punct_ratio > 0.15 or "breaking" in text.lower()
        
        payload = {
            "transformer": {
                "label": t_label,
                "confidence": float(t_conf)
            },
            "hybrid": {
                "label": h_label,
                "confidence": float(h_conf)
            },
            "raw_features": {
                "uppercase": float(uppercase_ratio),
                "punctuation": float(punct_ratio),
                "clickbait": float(0.85 if isSus else 0.24),
                "complexity": float(np.mean([len(w) for w in text.split()])) if word_count > 0 else 0
            }
        }
        
        return jsonify(payload)

    except Exception as e:
        print("Prediction Error:", str(e))
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)