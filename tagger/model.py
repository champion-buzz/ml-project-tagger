"""Data loading and candidate models."""
from pathlib import Path

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import make_pipeline

ROOT = Path(__file__).resolve().parent.parent
TRAIN_CSV = ROOT / "data" / "train.csv"
HOLDOUT_CSV = ROOT / "data" / "holdout.csv"
MODEL_PATH = ROOT / "artifacts" / "model.joblib"


def to_text(title: str, description: str) -> str:
    return f"{title.strip()} {description.strip()}"


def load(path: Path) -> tuple[list[str], list[str]]:
    df = pd.read_csv(path)
    texts = [to_text(t, d) for t, d in zip(df["title"], df["description"])]
    return texts, df["tag"].tolist()


def _tfidf() -> TfidfVectorizer:
    return TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, stop_words="english")


def candidates() -> dict:
    return {
        "majority_baseline": DummyClassifier(strategy="most_frequent"),
        "tfidf_naive_bayes": make_pipeline(_tfidf(), ComplementNB()),
        "tfidf_logreg": make_pipeline(
            _tfidf(), LogisticRegression(C=10, class_weight="balanced", max_iter=2000)
        ),
    }
