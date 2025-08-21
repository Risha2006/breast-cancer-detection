# imports
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             accuracy_score, precision_recall_fscore_support,
                             RocCurveDisplay, PrecisionRecallDisplay)
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import joblib
from pathlib import Path

RANDOM_STATE = 42
OUT_DIR = Path("artifacts"); OUT_DIR.mkdir(exist_ok=True)

# --- Load data (tabular: 30 features) ---
ds = load_breast_cancer(as_frame=True)
X, y = ds.data, ds.target
feature_names = X.columns.tolist()
print(f"Samples: {X.shape[0]}, Features: {X.shape[1]}")
print("Classes:", dict(zip(ds.target_names, np.bincount(y))))

# --- Pipeline (scaler + classifier) ---
clf = Pipeline([
    ("scaler", StandardScaler()),
    ("lr", LogisticRegression(max_iter=10_000, class_weight="balanced",
                              random_state=RANDOM_STATE, n_jobs=None))
])

# --- Robust CV evaluation ---
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
scoring = {"roc_auc": "roc_auc", "ap": "average_precision", "accuracy": "accuracy"}
cv_results = cross_validate(clf, X, y, cv=cv, scoring=scoring, return_estimator=True)
print("\nCV metrics (mean ± std):")
for k in scoring:
    vals = cv_results[f"test_{k}"]
    print(f"  {k:>8}: {vals.mean():.4f} ± {vals.std():.4f}")

# --- Fit on full data (for deployment) ---
clf.fit(X, y)

# --- Global permutation importance (on full fit) ---
perm = permutation_importance(clf, X, y, n_repeats=20, random_state=RANDOM_STATE)
pi_idx = np.argsort(perm.importances_mean)[::-1]
top_k = 10
print("\nTop feature importances (permutation):")
for i in pi_idx[:top_k]:
    print(f"{feature_names[i]:30s}  {perm.importances_mean[i]:.5f}")

# --- Plots (ROC & PR using cross-validated predictions) ---
# To get smooth curves, get out-of-fold predictions:
from sklearn.model_selection import cross_val_predict
oof_proba = cross_val_predict(clf, X, y, cv=cv, method="predict_proba")[:, 1]
roc_auc = roc_auc_score(y, oof_proba)
ap = average_precision_score(y, oof_proba)

plt.figure()
RocCurveDisplay.from_predictions(y, oof_proba)
plt.title(f"ROC curve (AUC = {roc_auc:.3f})")
plt.savefig(OUT_DIR / "roc_curve.png", dpi=160, bbox_inches="tight")

plt.figure()
PrecisionRecallDisplay.from_predictions(y, oof_proba)
plt.title(f"Precision–Recall (AP = {ap:.3f})")
plt.savefig(OUT_DIR / "pr_curve.png", dpi=160, bbox_inches="tight")

# --- Simple threshold analysis ---
thresholds = np.linspace(0.1, 0.9, 9)
records = []
y_hat = (oof_proba >= 0.5).astype(int)
acc = accuracy_score(y, y_hat)
prec, rec, f1, _ = precision_recall_fscore_support(y, y_hat, average="binary")
records.append({"threshold": 0.50, "accuracy": acc, "precision": prec, "recall": rec, "f1": f1})

for t in thresholds:
    yh = (oof_proba >= t).astype(int)
    acc = accuracy_score(y, yh)
    prec, rec, f1, _ = precision_recall_fscore_support(y, yh, average="binary", zero_division=0)
    records.append({"threshold": float(t), "accuracy": acc, "precision": prec, "recall": rec, "f1": f1})
pd.DataFrame(records).to_csv(OUT_DIR / "threshold_sweep.csv", index=False)

# --- Persist model + metadata ---
joblib.dump({
    "model": clf,
    "feature_names": feature_names,
    "target_names": ds.target_names,
    "roc_auc_oof": float(roc_auc),
    "ap_oof": float(ap)
}, OUT_DIR / "breast_cancer_lr.joblib")

print("\nSaved:")
print(f"  Model:        {OUT_DIR/'breast_cancer_lr.joblib'}")
print(f"  ROC curve:    {OUT_DIR/'roc_curve.png'}")
print(f"  PR curve:     {OUT_DIR/'pr_curve.png'}")
print(f"  Thresholds:   {OUT_DIR/'threshold_sweep.csv'}")
