import joblib
import numpy as np
try:
    clf = joblib.load('models/hybrid/hybrid_clf.pkl')
    scaler = joblib.load('models/hybrid/scaler.pkl')
    print("Model loaded successfully!")
    dummy_input = np.zeros((1, 776)) # 768 + 8 features
    scaled = scaler.transform(dummy_input)
    pred_prob = clf.predict_proba(scaled)
    print("Prediction test:", pred_prob)
except Exception as e:
    print("Error:", e)
