"""
preprocess.py — Data loading, cleaning, and feature engineering
Fake Job Posting Prediction Project
"""

import pandas as pd
import numpy as np
import re
import scipy.sparse as sp
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import pickle

# Download NLTK resources if not already present
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# ── 1. Load dataset ────────────────────────────────────────────────────────────
print("Loading dataset...")
df = pd.read_csv('fake_job_postings.csv')
print(f"  Shape: {df.shape}")
print(f"  Fake postings: {df['fraudulent'].sum()} ({df['fraudulent'].mean()*100:.1f}%)")

# ── 2. Combine text columns ────────────────────────────────────────────────────
text_cols = ['title', 'company_profile', 'description', 'requirements', 'benefits']
for col in text_cols:
    df[col] = df[col].fillna('')

df['combined_text'] = (
    df['title'] + ' ' +
    df['company_profile'] + ' ' +
    df['description'] + ' ' +
    df['requirements'] + ' ' +
    df['benefits']
)

# ── 3. Text cleaning ───────────────────────────────────────────────────────────
lemma = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'<[^>]+>', ' ', text)        # remove HTML tags
    text = re.sub(r'http\S+|www\.\S+', ' ', text)  # remove URLs
    text = re.sub(r'[^a-z\s]', ' ', text)       # keep letters only
    text = re.sub(r'\s+', ' ', text).strip()    # collapse whitespace
    tokens = text.split()
    tokens = [lemma.lemmatize(w) for w in tokens if w not in stop_words and len(w) > 2]
    return ' '.join(tokens)

print("Cleaning text...")
df['clean_text'] = df['combined_text'].apply(clean_text)

# ── 4. Structured feature engineering ─────────────────────────────────────────
print("Engineering features...")

df['desc_len']        = df['description'].apply(len)
df['req_len']         = df['requirements'].apply(len)
df['title_len']       = df['title'].apply(len)
df['has_salary']      = df['salary_range'].notna().astype(int)
df['has_logo']        = df['has_company_logo'].fillna(0).astype(int)
df['has_questions']   = df['has_questions'].fillna(0).astype(int)
df['telecommuting']   = df['telecommuting'].fillna(0).astype(int)
df['profile_present'] = (df['company_profile'] != '').astype(int)

# Encode employment_type
le = LabelEncoder()
df['employment_type_enc'] = le.fit_transform(df['employment_type'].fillna('Unknown'))

# Save label encoder
with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(le, f)

# ── 5. TF-IDF vectorization ────────────────────────────────────────────────────
print("Fitting TF-IDF (max 5000 features, unigrams + bigrams)...")
tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), sublinear_tf=True)
X_text = tfidf.fit_transform(df['clean_text'])

# Save vectorizer for inference
with open('tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(tfidf, f)

# ── 6. Combine sparse TF-IDF + structured features ────────────────────────────
struct_cols = [
    'desc_len', 'req_len', 'title_len',
    'has_salary', 'has_logo', 'has_questions',
    'telecommuting', 'profile_present', 'employment_type_enc'
]
X_struct = sp.csr_matrix(df[struct_cols].values.astype(float))
X = sp.hstack([X_text, X_struct])
y = df['fraudulent'].values

# ── 7. Save processed data ─────────────────────────────────────────────────────
sp.save_npz('X.npz', X)
np.save('y.npy', y)

print(f"\n✓ Preprocessing complete.")
print(f"  Feature matrix shape : {X.shape}")
print(f"  Labels shape         : {y.shape}")
print(f"  Saved: X.npz, y.npy, tfidf_vectorizer.pkl, label_encoder.pkl")
