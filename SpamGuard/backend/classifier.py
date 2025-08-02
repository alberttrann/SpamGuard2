# backend/classifier.py

import joblib
import numpy as np
import torch
import faiss
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
import pandas as pd
from tqdm import tqdm
import os

from .utils import preprocess_tokenizer

BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BACKEND_DIR, '..'))
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
DATA_CSV_PATH = os.path.join(BACKEND_DIR, 'data', '2cls_spam_text_cls.csv')

class SpamGuardClassifier:
    def __init__(self, model_path_dir=MODELS_DIR):
        self._load_all_components(model_path_dir)
    def _load_all_components(self, model_path_dir):
        print("--- Initializing or Reloading SpamGuard AI Classifier ---")
        try:
            pipeline_path = os.path.join(model_path_dir, 'nb_multinomial_pipeline.joblib')
            encoder_path = os.path.join(model_path_dir, 'label_encoder.joblib')
            self.nb_pipeline = joblib.load(pipeline_path)
            self.label_encoder = joblib.load(encoder_path)
            print(f"✅ Superior MultinomialNB pipeline loaded from '{model_path_dir}'.")
        except Exception as e:
            print(f"🔴 WARNING: Failed to load Naive Bayes model: {e}. The classifier will rely solely on Vector Search.")
            self.nb_pipeline = None; self.label_encoder = None
        if not hasattr(self, 'transformer_model'):
            MODEL_NAME = "intfloat/multilingual-e5-base"
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME); self.transformer_model = AutoModel.from_pretrained(MODEL_NAME).to(self.device).eval()
            print(f"✅ Transformer model loaded on {self.device}.")
        df = pd.read_csv(DATA_CSV_PATH, quotechar='"', on_bad_lines='skip')
        df.dropna(subset=['Message'], inplace=True); self.all_messages = df["Message"].astype(str).tolist(); self.all_labels = df["Category"].tolist()
        embeddings = self._get_embeddings(self.all_messages, "passage"); self.faiss_index = faiss.IndexFlatIP(embeddings.shape[1]); self.faiss_index.add(embeddings.astype('float32'))
        print("✅ FAISS index built successfully."); print("--- SpamGuard AI Classifier is ready. ---")
    def reload(self): self._load_all_components(MODELS_DIR)
    def _average_pool(self, h, m): return (h * m[..., None]).sum(1) / m.sum(-1)[..., None]
    def _get_embeddings(self, texts, prefix, batch_size=32):
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = [f"{prefix}: {text}" for text in texts[i:i + batch_size]]
            tokens = self.tokenizer(batch, max_length=512, padding=True, truncation=True, return_tensors="pt").to(self.device)
            with torch.no_grad(): outputs = self.transformer_model(**tokens)
            embeddings = self._average_pool(outputs.last_hidden_state, tokens['attention_mask'])
            all_embeddings.append(F.normalize(embeddings, p=2, dim=1).cpu().numpy())
        return np.vstack(all_embeddings)
    def classify(self, text: str) -> dict:
        if self.nb_pipeline and self.label_encoder:
            probs = self.nb_pipeline.predict_proba([text])[0]; spam_idx = np.where(self.label_encoder.classes_ == 'spam')[0][0]; spam_prob = probs[spam_idx]
            if spam_prob < 0.15: return {"prediction": "ham", "confidence": 1 - spam_prob, "model": "MultinomialNB", "evidence": None}
            if spam_prob > 0.85: return {"prediction": "spam", "confidence": spam_prob, "model": "MultinomialNB", "evidence": None}
        k=5; q_emb = self._get_embeddings([text], "query", 1); s, i = self.faiss_index.search(q_emb, k); n_l = [self.all_labels[x] for x in i[0]]; p = max(set(n_l),key=n_l.count); c = n_l.count(p)/k; e=[{"similar_message":self.all_messages[idx],"label":self.all_labels[idx],"similarity_score":float(s[0][j])} for j,idx in enumerate(i[0])]

        return {"prediction":p,"confidence":c,"model":"Vector Search (k-NN)","evidence":e}
