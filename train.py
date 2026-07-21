"""
train.py — Model training, evaluation, and comparison
Fake Job Posting Prediction Project
"""

import numpy as np
import scipy.sparse as sp
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, ConfusionMatrixDisplay
)
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# ── 1. Load preprocessed data ─────────────────────────────────────────────────
print("Loading preprocessed data...")
X = sp.load_npz('X.npz')
y = np.load('y.npy')
print(f"  X shape: {X.shape}, y distribution: {dict(zip(*np.unique(y, return_counts=True)))}")

# ── 2. Train/test split (stratified) ──────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
print(f"  Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")

# ── 3. Handle class imbalance with SMOTE ──────────────────────────────────────
print("Applying SMOTE on training data...")
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
print(f"  After SMOTE: {dict(zip(*np.unique(y_train_res, return_counts=True)))}")

# ── 4. Define models ───────────────────────────────────────────────────────────
models = {
    'Logistic Regression': LogisticRegression(
        max_iter=1000, class_weight='balanced', C=1.0, random_state=42),
    'Random Forest': RandomForestClassifier(
        n_estimators=200, class_weight='balanced', random_state=42, n_jobs=-1),
    'XGBoost': xgb.XGBClassifier(
        n_estimators=200, scale_pos_weight=20,
        use_label_encoder=False, eval_metric='logloss',
        random_state=42, verbosity=0),
    'Naive Bayes': MultinomialNB(alpha=0.1),
}

# ── 5. Train + evaluate all models ────────────────────────────────────────────
results = {}
print("\n" + "="*60)
print("MODEL TRAINING & EVALUATION")
print("="*60)

for name, model in models.items():
    print(f"\n▶ {name}")
    model.fit(X_train_res, y_train_res)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    report = classification_report(y_test, y_pred,
                                   target_names=['Real', 'Fake'], output_dict=True)
    auc = roc_auc_score(y_test, y_prob)

    results[name] = {
        'model': model,
        'y_pred': y_pred,
        'y_prob': y_prob,
        'report': report,
        'auc': auc
    }

    print(classification_report(y_test, y_pred, target_names=['Real', 'Fake']))
    print(f"  ROC-AUC: {auc:.4f}")

# ── 6. Pick best model (by F1 on Fake class) ──────────────────────────────────
best_name = max(results, key=lambda k: results[k]['report']['Fake']['f1-score'])
best = results[best_name]
print(f"\n★ Best model: {best_name}  (Fake F1 = {best['report']['Fake']['f1-score']:.3f})")

# ── 7. Save best model ─────────────────────────────────────────────────────────
with open('model.pkl', 'wb') as f:
    pickle.dump(best['model'], f)
print("  Saved: model.pkl")

# ── 8. Plots ───────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle(f'Fake Job Prediction — {best_name}', fontsize=14, fontweight='bold')

# Confusion matrix
cm = confusion_matrix(y_test, best['y_pred'])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Real', 'Fake'])
disp.plot(ax=axes[0], colorbar=False, cmap='Blues')
axes[0].set_title('Confusion Matrix')

# ROC curves
axes[1].set_title('ROC Curves')
for name, res in results.items():
    fpr, tpr, _ = roc_curve(y_test, res['y_prob'])
    axes[1].plot(fpr, tpr, label=f"{name} (AUC={res['auc']:.3f})")
axes[1].plot([0,1],[0,1],'k--', alpha=0.4)
axes[1].set_xlabel('False Positive Rate')
axes[1].set_ylabel('True Positive Rate')
axes[1].legend(fontsize=8)
axes[1].grid(True, alpha=0.3)

# Model comparison bar chart
names = list(results.keys())
f1_scores = [results[n]['report']['Fake']['f1-score'] for n in names]
bars = axes[2].barh(names, f1_scores, color=['#185FA5' if n==best_name else '#B5D4F4' for n in names])
axes[2].set_xlim(0.7, 1.0)
axes[2].set_title('F1-Score (Fake class)')
axes[2].set_xlabel('F1 Score')
for bar, score in zip(bars, f1_scores):
    axes[2].text(score + 0.002, bar.get_y() + bar.get_height()/2,
                 f'{score:.3f}', va='center', fontsize=9)

plt.tight_layout()
plt.savefig('evaluation_plots.png', dpi=150, bbox_inches='tight')
print("  Saved: evaluation_plots.png")
plt.show()

print("\n✓ Training complete.")
