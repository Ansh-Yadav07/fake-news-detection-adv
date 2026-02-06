import numpy as np
from bert_embeddings import get_bert_embedding
from linguistic_features import extract_linguistic_features

def get_hybrid_features(text: str):
    bert_vec = get_bert_embedding(text)            # (768,)
    ling_vec = extract_linguistic_features(text)   # (8,)
    return np.concatenate([bert_vec, ling_vec])    # (776,)
    