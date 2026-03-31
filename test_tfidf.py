import pickle
try:
    with open('models/best_model.pkl', 'rb') as f:
        mod = pickle.load(f)
    print("ml model loaded")
except Exception as e:
    print(e)
