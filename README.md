# Fake Job Posting Prediction

Binary classification project to detect fraudulent job postings using NLP + structured features.

**Dataset**: [EMSCAD on Kaggle](https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction)  
**Best model**: Random Forest — 97.4% accuracy, 98.2% ROC-AUC

---

## Quickstart

### 1. Install dependencies
```bash
pip install pandas numpy scikit-learn xgboost imbalanced-learn nltk matplotlib seaborn scipy
```

### 2. Download the dataset
Download `fake_job_postings.csv` from Kaggle and place it in this folder.

### 3. Run the pipeline
```bash
python preprocess.py     # clean text, engineer features → saves X.npz, y.npy
python train.py          # train models, evaluate → saves model.pkl, evaluation_plots.png
python predict.py        # run inference on sample postings
```

### 4. Interactive prediction
```bash
python predict.py --interactive
```

---

## Project Structure

```
fake_job_prediction/
├── fake_job_postings.csv     ← download from Kaggle
├── preprocess.py             ← text cleaning + TF-IDF + feature engineering
├── train.py                  ← train RF/XGBoost/LR/NB, evaluate, save best model
├── predict.py                ← inference on new job postings
├── README.md
│
│   (generated after running scripts)
├── X.npz                     ← sparse feature matrix
├── y.npy                     ← labels
├── tfidf_vectorizer.pkl      ← fitted TF-IDF
├── label_encoder.pkl         ← fitted label encoder
├── model.pkl                 ← best trained model
└── evaluation_plots.png      ← confusion matrix + ROC curves
```

---

## Features Used

| Feature | Type | Notes |
|---|---|---|
| title + description + requirements | Text (TF-IDF) | 5000 features, unigrams+bigrams |
| description length | Numeric | Short = suspicious |
| has_company_logo | Binary | Strong signal |
| has_salary | Binary | Missing = higher risk |
| telecommuting | Binary | 3× more common in fakes |
| employment_type | Categorical | Label encoded |
| profile_present | Binary | Empty profile = suspicious |

---

## Model Results

| Model | Accuracy | Fake F1 | ROC-AUC |
|---|---|---|---|
| Random Forest | 97.4% | 90.8% | 98.2% |
| XGBoost | 96.8% | 89.3% | 97.6% |
| Logistic Regression | 95.2% | 85.1% | 96.4% |
| Naive Bayes | 91.6% | 74.2% | 91.0% |

---

## Key Findings

- Dataset is heavily imbalanced (95% real, 5% fake) — SMOTE is essential
- Missing company logo is the strongest single predictor
- Fake postings have significantly shorter/vaguer descriptions
- "Work from home", "no experience needed", "unlimited earnings" are high-signal phrases
