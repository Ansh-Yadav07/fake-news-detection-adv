import os
import torch
import numpy as np
import string
import joblib
from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import torch.nn.functional as F
import nltk
from nltk.corpus import stopwords

# ---- Initialization and Setup ----
app = Flask(__name__)
# Enable CORS so your Vercel React app can connect to this API
CORS(app)

# Force NLTK download for stopwords
try:
    nltk.download('stopwords', quiet=True)
    stop_words = set(stopwords.words("english"))
except Exception as e:
    stop_words = set()

# ---- Load Models ----
MODEL_PATH = "anshy047/fake-news-detector-transformer"
HYBRID_MODEL_PATH = "models/hybrid/hybrid_clf.pkl"
SCALer_PATH = "models/hybrid/scaler.pkl"

print("Loading Transformer model...")
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu" if not torch.cuda.is_available() else "cuda")
tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_PATH)
transformer_model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH)
transformer_model.eval()
transformer_model.to(device)

print("Loading Hybrid ML model...")
hybrid_clf = joblib.load(HYBRID_MODEL_PATH)
scaler = joblib.load(SCALer_PATH)
print("All models loaded successfully!")

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


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({"error": "No text provided"}), 400
        
    text = data['text']
    
    # 1. Transformer Prediction
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=256)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = transformer_model(**inputs)
        
    probs = F.softmax(outputs.logits, dim=1)
    t_predicted_class = torch.argmax(probs, dim=1).item()
    t_conf = probs[0][t_predicted_class].item()
    t_label = "REAL" if t_predicted_class == 1 else "FAKE"
    
    # 2. Extract Features
    ling_feats, uppercase_ratio, punct_ratio = extract_features(text)
    
    # Needs BERT embeddings for hybrid model
    with torch.no_grad():
        # Get pooled output (embedding) from transformer
        hidden_states = transformer_model.distilbert(**inputs).last_hidden_state
        cls_embedding = hidden_states[:, 0, :].cpu().numpy()
        
    # Combine embedding and linguistic features for Hybrid
    combined_features = np.hstack((cls_embedding, np.array(ling_feats).reshape(1, -1)))
    combined_scaled = scaler.transform(combined_features)
    
    # Hybrid Prediction
    h_probs = hybrid_clf.predict_proba(combined_scaled)[0]
    h_pred_class = np.argmax(h_probs)
    h_conf = float(h_probs[h_pred_class])
    h_label = "REAL" if h_pred_class == 1 else "FAKE"
    
    # Mocking some feature data that the UI expects
    word_count = len(text.split())
    isSus = uppercase_ratio > 0.15 or punct_ratio > 0.15 or "breaking" in text.lower()
    
    # Payload for frontend visualization
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
            "clickbait": float(0.85 if isSus else 0.24), # Synthetic approximation based on thresholds
            "complexity": float(np.mean([len(w) for w in text.split()])) if word_count > 0 else 0
        }
    }
    
    return jsonify(payload)

if __name__ == '__main__':
    # Default port for Render/HuggingFace Spaces
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)