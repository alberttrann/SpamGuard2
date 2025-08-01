# backend/train_nb.py (The Final, Corrected Version)

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
import joblib
import os

# Import the shared tokenizer from the utils file
from .utils import preprocess_tokenizer

# Define robust absolute paths
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BACKEND_DIR, '..'))
OUTPUT_MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
INPUT_DATA_PATH = os.path.join(BACKEND_DIR, 'data', '2cls_spam_text_cls.csv')


def retrain_and_save():
    print("--- Starting Production Retraining Process with MultinomialNB ---")
    
    # Load Data
    df = pd.read_csv(INPUT_DATA_PATH, quotechar='"', on_bad_lines='skip')
    df.dropna(subset=['Message', 'Category'], inplace=True)
    df.drop_duplicates(subset=['Message'], inplace=True)
    
    X = df["Message"].astype(str)
    

    le = LabelEncoder()
    y = le.fit_transform(df["Category"])

    # Define Pipeline using the IMPORTED tokenizer
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(tokenizer=preprocess_tokenizer, stop_words=None, ngram_range=(1, 2), max_features=10000)),
        ('smote', SMOTE(random_state=42)),
        ('clf', MultinomialNB(alpha=0.1))
    ])

    print("Training the production MultinomialNB pipeline...")
    pipeline.fit(X, y)
    print("✅ Production model training complete.")

    # Save Artifacts
    os.makedirs(OUTPUT_MODELS_DIR, exist_ok=True)
    joblib.dump(pipeline, os.path.join(OUTPUT_MODELS_DIR, 'nb_multinomial_pipeline.joblib'))
    # Now the `le` variable exists and can be saved correctly.
    joblib.dump(le, os.path.join(OUTPUT_MODELS_DIR, 'label_encoder.joblib'))
    
    print(f"--- Retraining complete. Model saved to '{OUTPUT_MODELS_DIR}'. ---")


if __name__ == "__main__":
    retrain_and_save()