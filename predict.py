"""
predict.py — Run inference on new job postings
Fake Job Posting Prediction Project

Usage:
    python predict.py                     # runs built-in examples
    python predict.py --interactive       # enter your own posting
"""

import pickle
import re
import argparse
import scipy.sparse as sp
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# ── Load saved artifacts ───────────────────────────────────────────────────────
print("Loading model and vectorizer...")
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('tfidf_vectorizer.pkl', 'rb') as f:
    tfidf = pickle.load(f)
with open('label_encoder.pkl', 'rb') as f:
    le = pickle.load(f)

lemma = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# ── Text cleaning (must match preprocess.py) ───────────────────────────────────
def clean_text(text):
    text = text.lower()
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'http\S+|www\.\S+', ' ', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = text.split()
    tokens = [lemma.lemmatize(w) for w in tokens if w not in stop_words and len(w) > 2]
    return ' '.join(tokens)

# ── Feature builder ────────────────────────────────────────────────────────────
def build_features(title, description, requirements, company_profile='',
                   benefits='', has_salary=0, has_logo=1,
                   has_questions=0, telecommuting=0, employment_type='Full-time'):
    combined = f"{title} {company_profile} {description} {requirements} {benefits}"
    clean = clean_text(combined)

    X_text = tfidf.transform([clean])

    known_types = list(le.classes_)
    emp_enc = le.transform([employment_type])[0] if employment_type in known_types else 0

    struct = [[
        len(description),
        len(requirements),
        len(title),
        int(has_salary),
        int(has_logo),
        int(has_questions),
        int(telecommuting),
        int(bool(company_profile)),
        emp_enc
    ]]
    X_struct = sp.csr_matrix(struct)
    return sp.hstack([X_text, X_struct])

# ── Prediction function ────────────────────────────────────────────────────────
def predict(posting: dict) -> dict:
    X = build_features(
        title=posting.get('title', ''),
        description=posting.get('description', ''),
        requirements=posting.get('requirements', ''),
        company_profile=posting.get('company_profile', ''),
        benefits=posting.get('benefits', ''),
        has_salary=posting.get('has_salary', 0),
        has_logo=posting.get('has_logo', 1),
        has_questions=posting.get('has_questions', 0),
        telecommuting=posting.get('telecommuting', 0),
        employment_type=posting.get('employment_type', 'Full-time'),
    )
    pred = model.predict(X)[0]
    prob = model.predict_proba(X)[0]
    return {
        'prediction': 'FAKE' if pred == 1 else 'REAL',
        'confidence': round(prob[pred] * 100, 1),
        'fake_probability': round(prob[1] * 100, 1),
        'real_probability': round(prob[0] * 100, 1),
    }

def print_result(posting, result):
    label = '🚨 FAKE' if result['prediction'] == 'FAKE' else '✅ REAL'
    print(f"\n{'─'*50}")
    print(f"  Title      : {posting.get('title', 'N/A')}")
    print(f"  Verdict    : {label}")
    print(f"  Fake prob  : {result['fake_probability']}%")
    print(f"  Real prob  : {result['real_probability']}%")
    print(f"  Confidence : {result['confidence']}%")
    print(f"{'─'*50}")

# ── Sample postings ────────────────────────────────────────────────────────────
sample_postings = [
    {
        'title': 'Senior Software Engineer — Python/Django',
        'description': (
            'We are looking for an experienced Python developer to join our '
            'engineering team. You will build scalable REST APIs, mentor junior '
            'developers, and collaborate with product and design teams. '
            '5+ years of experience required.'
        ),
        'requirements': 'Python, Django, PostgreSQL, Docker, AWS. BSc in CS or equivalent.',
        'company_profile': 'Acme Corp is a Series B SaaS company founded in 2015.',
        'has_salary': 1,
        'has_logo': 1,
        'telecommuting': 0,
        'employment_type': 'Full-time',
    },
    {
        'title': 'Work From Home — Earn $500/day, No Experience Needed!',
        'description': (
            'Unlimited earning potential. Work from anywhere. No experience needed. '
            'Click here to apply now. Guaranteed income. Passive income opportunity. '
            'Make money fast. Immediate openings.'
        ),
        'requirements': '',
        'company_profile': '',
        'has_salary': 0,
        'has_logo': 0,
        'telecommuting': 1,
        'employment_type': 'Other',
    },
    {
        'title': 'Data Analyst — Marketing Team',
        'description': (
            'Join our data team to analyse campaign performance, build dashboards '
            'in Tableau, and present insights to stakeholders. We offer competitive '
            'salary, health insurance, and flexible hours.'
        ),
        'requirements': 'SQL, Python or R, Tableau. 2+ years experience.',
        'company_profile': 'GlobalRetail Ltd, 500+ employees, London HQ.',
        'has_salary': 1,
        'has_logo': 1,
        'telecommuting': 0,
        'employment_type': 'Full-time',
    },
]

# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--interactive', action='store_true',
                        help='Enter a custom job posting interactively')
    args = parser.parse_args()

    if args.interactive:
        print("\n── Interactive Mode ──")
        posting = {
            'title':          input("Job title       : "),
            'description':    input("Description     : "),
            'requirements':   input("Requirements    : "),
            'company_profile':input("Company profile : "),
            'has_salary':     int(input("Has salary? (1/0): ") or 0),
            'has_logo':       int(input("Has logo?   (1/0): ") or 1),
            'telecommuting':  int(input("Remote?     (1/0): ") or 0),
            'employment_type':input("Employment type [Full-time]: ") or 'Full-time',
        }
        result = predict(posting)
        print_result(posting, result)
    else:
        print("\n── Running on sample postings ──")
        for posting in sample_postings:
            result = predict(posting)
            print_result(posting, result)

    print("\n✓ Done.")
