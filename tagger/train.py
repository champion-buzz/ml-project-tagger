"""Pick a model with cross-validation, then score it once on the holdout set.

Run with: python -m tagger.train
"""
import json

import joblib
import numpy as np
from sklearn.base import clone
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from tagger.model import HOLDOUT_CSV, MODEL_PATH, ROOT, TRAIN_CSV, candidates, load

TARGET_PRECISION = 0.90


def pick_threshold(confidence: np.ndarray, correct: np.ndarray) -> float:
    """Lowest confidence cutoff whose accepted predictions reach TARGET_PRECISION."""
    for cutoff in np.arange(0.0, 1.0, 0.01):
        accepted = confidence >= cutoff
        if accepted.any() and correct[accepted].mean() >= TARGET_PRECISION:
            return round(float(cutoff), 2)
    return 1.0


def main() -> None:
    X, y = load(TRAIN_CSV)
    X_holdout, y_holdout = load(HOLDOUT_CSV)
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    cv_scores = {}
    for name, model in candidates().items():
        pred = cross_val_predict(clone(model), X, y, cv=folds)
        cv_scores[name] = round(f1_score(y, pred, average="macro"), 3)
    best_name = max(cv_scores, key=cv_scores.get)
    best = candidates()[best_name]

    # The threshold comes from out-of-fold predictions, so the holdout stays untouched.
    proba = cross_val_predict(clone(best), X, y, cv=folds, method="predict_proba")
    classes = np.unique(y)
    oof_pred = classes[proba.argmax(axis=1)]
    threshold = pick_threshold(proba.max(axis=1), oof_pred == np.array(y))

    best.fit(X, y)
    holdout_proba = best.predict_proba(X_holdout)
    holdout_pred = best.classes_[holdout_proba.argmax(axis=1)]
    accepted = holdout_proba.max(axis=1) >= threshold
    truth = np.array(y_holdout)
    labels = best.classes_.tolist()

    metrics = {
        "cv_macro_f1": cv_scores,
        "selected_model": best_name,
        "review_threshold": threshold,
        "holdout": {
            "n": len(y_holdout),
            "macro_f1": round(f1_score(y_holdout, holdout_pred, average="macro"), 3),
            "accuracy": round(float((holdout_pred == truth).mean()), 3),
            "auto_accepted_share": round(float(accepted.mean()), 3),
            "auto_accepted_accuracy": round(float((holdout_pred[accepted] == truth[accepted]).mean()), 3),
            "per_class": {
                k: {m: round(v, 3) for m, v in row.items()}
                for k, row in classification_report(
                    y_holdout, holdout_pred, output_dict=True, zero_division=0
                ).items()
                if k in labels
            },
            "confusion_matrix": {
                "labels": labels,
                "rows": confusion_matrix(y_holdout, holdout_pred, labels=labels).tolist(),
            },
        },
    }

    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump({"model": best, "threshold": threshold}, MODEL_PATH)
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
