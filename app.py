import os
import numpy as np
import string
import joblib
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
import nltk
from nltk.corpus import stopwords
from flask import Flask, request, jsonify
from flask_cors import CORS
import requests

app = Flask(__name__)
CORS(app)

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
HF_TOKEN = os.environ.get("HF_TOKEN", "")  # Read from environment variable on Render
if not HF_TOKEN:
    raise ValueError("HF_TOKEN environment variable not set. Please set it on Render dashboard.")
API_URL = "https://api-inference.huggingface.co/models/anshy047/fake-news-detector-transformer"
FEAT_URL = "https://api-inference.huggingface.co/pipeline/feature-extraction/anshy047/fake-news-detector-transformer"
HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"}

MODEL_PATH = "models/hybrid/hybrid_clf.pkl"
SCALER_PATH = "models/hybrid/scaler.pkl"

print("Loading local Hybrid ML model...")
hybrid_clf = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
print("Local models loaded successfully! Transformer running remotely via HF API.")

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
    """Call HF inference API for sequence classification"""
    response = requests.post(API_URL, headers=HEADERS, json={"inputs": text})
    if response.status_code != 200:
        raise Exception(f"HF Classification API Error: {response.text}")
    return response.json()

def get_hf_embeddings(text):
    """Call HF inference API for feature extraction"""
    response = requests.post(FEAT_URL, headers=HEADERS, json={"inputs": text})
    if response.status_code != 200:
        raise Exception(f"HF Embeddings API Error: {response.text}")
    
    data = response.json()
    
    # Parse flexible JSON responses from feature-extraction pipeline
    if isinstance(data, list) and len(data)>0 and isinstance(data[0], list):
        if len(data[0])>0 and isinstance(data[0][0], list):
            return np.array(data[0][0])
        return np.array(data[0])
        
    return np.array(data)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({"error": "No text provided"}), 400
        
    text = data['text']
    
    try:
        # 1. Remote Transformer Prediction via HF Inference API
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
        
        # 2. Extract local Linguistic Features
        ling_feats, uppercase_ratio, punct_ratio = extract_features(text)
        
        # 3. Get remote Embeddings for Hybrid Model
        cls_embedding = None
        try:
            raw_embed = get_hf_embeddings(text)
            cls_embedding = raw_embed.flatten()
            
            if cls_embedding.shape[0] != 768:
                raise ValueError(f"Shape error: {cls_embedding.shape}")
                
        except Exception as e:
            print(f"Warning: Could not fetch embeddings, using zero fallback: {str(e)}")
            cls_embedding = np.zeros(768)

        # Combine embedding and linguistic features for Hybrid Local execution
        combined_features = np.hstack((cls_embedding, np.array(ling_feats)))
        combined_scaled = scaler.transform(combined_features.reshape(1, -1))
        
        # Local Hybrid Prediction
        h_probs = hybrid_clf.predict_proba(combined_scaled)[0]
        h_pred_class = np.argmax(h_probs)
        h_conf = float(h_probs[h_pred_class])
        h_label = "REAL" if h_pred_class == 1 else "FAKE"
        
        # Stats for frontend visualization
        word_count = len(text.split())
        isSus = uppercase_ratio > 0.15 or punct_ratio > 0.15 or "breaking" in text.lower()
        
        payload = {
            "transformer": {
                "label": t_label,
                "confidence": t_conf
            },
            "hybrid": {
                "label": h_label,
                "confidence": h_conf
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