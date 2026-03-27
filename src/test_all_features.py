import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
import pickle
import string
import nltk
from nltk.corpus import stopwords
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, DistilBertModel
import warnings
warnings.filterwarnings('ignore')

# Handle NLTK stopwords download gracefully
try:
    stop_words = set(stopwords.words("english"))
except LookupError:
    import ssl
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context
    nltk.download("stopwords", quiet=True)
    stop_words = set(stopwords.words("english"))


def get_hybrid_features(text: str) -> np.ndarray:
    """Extract 8 handcrafted linguistic features matching the dataset generator."""
    words = text.split()
    total_words = len(words)
    
    if total_words == 0:
        return np.zeros(8)
        
    exclam_count = text.count("!")
    question_count = text.count("?")
    uppercase_ratio = sum(1 for c in text if c.isupper()) / max(len(text), 1)
    digit_ratio = sum(1 for c in text if c.isdigit()) / max(len(text), 1)
    punctuation_ratio = sum(1 for c in text if c in string.punctuation) / max(len(text), 1)
    stopword_ratio = sum(1 for w in words if w.lower() in stop_words) / total_words
    avg_word_length = np.mean([len(w) for w in words])
    text_length = len(text)
    
    return np.array([
        exclam_count,
        question_count,
        uppercase_ratio,
        digit_ratio,
        punctuation_ratio,
        stopword_ratio,
        avg_word_length,
        text_length
    ])


def explain_prediction(features: np.ndarray) -> list:
    """Provide lightweight explainability for the input."""
    # features shape: Nx8 -> [exclam, question, upper, digit, punct, stop, avg_word_len, total_len]
    exclam_count = features[0]
    question_count = features[1]
    uppercase_ratio = features[2]
    text_length = features[7]
    
    reasons = []
    if exclam_count > 2:
        reasons.append("High punctuation (exclamation marks) -> possible fake/clickbait pattern")
    if uppercase_ratio > 0.15:
        reasons.append("High uppercase ratio -> common in sensationalism")
    if question_count >= 2:
        reasons.append("Multiple questions -> often used to provoke uncertainty")
    if text_length < 60:
        reasons.append("Very short text -> might lack sufficient context or sourcing")
        
    if not reasons:
        reasons.append("Linguistically neutral (standard casing and punctuation)")
        
    return reasons


def generate_variations(text: str) -> list:
    """Create perturbed versions of the text to test Transformer robustness."""
    return [
        text,                                  # 1. Original
        text.upper(),                          # 2. Uppercase
        text + " !!!",                         # 3. Add exclamation marks
        text.lower(),                          # 4. Lowercase
        text + " Wait, is this true?"          # 5. Slight rephrased version
    ]


class AdvancedFakeNewsDetector:
    def __init__(self):
        print("\nLoading models... (this may take a few seconds)")
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        
        # Load Transformer and Tokenizer
        transformer_path = "models/transformer"
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(transformer_path)
        self.transformer_clf = DistilBertForSequenceClassification.from_pretrained(transformer_path)
        
        # Load base DistilBERT for embeddings (used by hybrid model)
        # Note: In an ideal world, you'd load the base "distilbert-base-uncased", but since the model 
        # may have learned custom embeddings, we load it directly from sequence clf base.
        self.bert_backbone = self.transformer_clf.distilbert if hasattr(self.transformer_clf, "distilbert") else DistilBertModel.from_pretrained(transformer_path)
        
        self.transformer_clf.to(self.device).eval()
        self.bert_backbone.to(self.device).eval()
        
        # Load Hybrid ML Artifacts
        hybrid_path = "models/hybrid"
        with open(os.path.join(hybrid_path, "hybrid_clf.pkl"), "rb") as f:
            self.hybrid_clf = pickle.load(f)
            
        with open(os.path.join(hybrid_path, "scaler.pkl"), "rb") as f:
            self.scaler = pickle.load(f)
            
        self.label_map = {0: "FAKE", 1: "REAL"}
        print("Initialization complete! Ready.\n")
        
    def predict_transformer(self, text: str):
        """Returns standard label and probability from Transformer head."""
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, padding=True, max_length=256
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.transformer_clf(**inputs)
            
        probs = F.softmax(outputs.logits, dim=1)[0].cpu().numpy()
        pred_idx = np.argmax(probs)
        return self.label_map[pred_idx], probs[pred_idx]
        
    def predict_hybrid(self, text: str, ling_features: np.ndarray):
        """Returns label and probability from Hybrid Logistics Regression pipeline."""
        # 1. Get BERT embedding mirroring original process
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, padding="max_length", max_length=256
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.bert_backbone(**inputs)
            
        # Standard extraction is output.last_hidden_state[:, 0, :]
        if hasattr(outputs, "last_hidden_state"):
            bert_embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy().flatten()
        else:
            # Handle sequenceclassification backbone direct returns
            bert_embedding = outputs[0][:, 0, :].cpu().numpy().flatten()
            
        # 2. Combine and scale
        combined_features = np.hstack((bert_embedding, ling_features)).reshape(1, -1)
        scaled_features = self.scaler.transform(combined_features)
        
        # 3. Predict via LogReg
        probs = self.hybrid_clf.predict_proba(scaled_features)[0]
        pred_idx = np.argmax(probs)
        # Assuming hybrid model output matching final english dataset mapping
        return self.label_map[pred_idx], probs[pred_idx]

    def analyze_text(self, text: str):
        # 1. Base Feature Extraction
        ling_features = get_hybrid_features(text)
        
        # 2. Model Predictions
        t_label, t_conf = self.predict_transformer(text)
        h_label, h_conf = self.predict_hybrid(text, ling_features)
        
        # 3. Disagreement Tracking
        agreement = (t_label == h_label)
        
        # 4. Robustness Testing
        variations = generate_variations(text)
        consistency_count = 0
        for var in variations:
            var_label, _ = self.predict_transformer(var)
            if var_label == t_label:
                consistency_count += 1
                
        robustness_score = (consistency_count / len(variations)) * 100
        
        # 5. Explanability
        explanations = explain_prediction(ling_features)
        
        # ==========================================
        # PRINT FINAL REPORT 
        # ==========================================
        print(f"\nInput: {text}\n")
        print(f"Transformer → {t_label} ({t_conf:.0%})")
        print(f"Hybrid      → {h_label} ({h_conf:.0%})\n")
        
        if agreement:
            print("CONSISTENT")
        else:
            print("⚠️ DISAGREEMENT: Needs verification")
            
        print(f"\nRobustness Score: {robustness_score:.0f}%\n")
        
        print("Explanation:")
        for exp in explanations:
            print(f"- {exp}")
            
        print("\nFinal Verdict: ", end="")
        if not agreement:
            print("UNCERTAIN (Models disagree)")
        elif t_conf < 0.6 or h_conf < 0.6:
            print("UNCERTAIN prediction (Low confidence)")
        else:
            print(t_label)
        print("-" * 50)


def main():
    print("=" * 50)
    print("Advanced Fake News Detection System")
    print("=" * 50)
    
    # Needs to be wrapped in try/except to avoid crashing on missing files
    try:
        detector = AdvancedFakeNewsDetector()
    except Exception as e:
        print(f"Failed to initialize models. Error: {e}")
        print("Make sure you have run both the Transformer and Hybrid training scripts!")
        return

    while True:
        try:
            user_input = input("\nEnter news text (or type 'exit'/'quit'):\n> ")
            if not user_input.strip():
                continue
            if user_input.lower().strip() in ['exit', 'quit']:
                break
                
            detector.analyze_text(user_input.strip())
            
        except KeyboardInterrupt:
            print("\nExiting system...")
            break
        except EOFError:
            break

if __name__ == "__main__":
    main()
