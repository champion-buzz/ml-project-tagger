import numpy as np
import pytest
from fastapi.testclient import TestClient
from sklearn.metrics import f1_score

from tagger import api, train
from tagger.model import HOLDOUT_CSV, TRAIN_CSV, candidates, load


@pytest.fixture(scope="module")
def model():
    X, y = load(TRAIN_CSV)
    return candidates()["tfidf_logreg"].fit(X, y)


def test_train_and_holdout_do_not_overlap():
    train_texts, _ = load(TRAIN_CSV)
    holdout_texts, _ = load(HOLDOUT_CSV)
    assert not set(train_texts) & set(holdout_texts)


def test_model_beats_majority_baseline(model):
    X, y = load(TRAIN_CSV)
    X_holdout, y_holdout = load(HOLDOUT_CSV)
    baseline = candidates()["majority_baseline"].fit(X, y)
    model_f1 = f1_score(y_holdout, model.predict(X_holdout), average="macro")
    baseline_f1 = f1_score(y_holdout, baseline.predict(X_holdout), average="macro")
    assert model_f1 > baseline_f1 + 0.3


def test_pick_threshold_finds_lowest_cutoff_reaching_target():
    confidence = np.array([0.2, 0.5, 0.8, 0.9])
    correct = np.array([False, False, True, True])
    assert 0.5 < train.pick_threshold(confidence, correct) <= 0.8


def test_pick_threshold_returns_one_when_target_is_unreachable():
    assert train.pick_threshold(np.array([0.9]), np.array([False])) == 1.0


@pytest.fixture
def client(model, monkeypatch):
    monkeypatch.setattr(api, "_bundle", {"model": model, "threshold": 0.5})
    return TestClient(api.app)


def test_predict_returns_a_known_tag(client, model):
    response = client.post(
        "/predict",
        json={"title": "Object detection with YOLO", "description": "Detect objects in images."},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["tag"] == "computer-vision"
    assert 0 <= body["confidence"] <= 1


def test_needs_review_follows_threshold(model, monkeypatch):
    monkeypatch.setattr(api, "_bundle", {"model": model, "threshold": 1.01})
    response = TestClient(api.app).post("/predict", json={"title": "Object detection"})
    assert response.json()["needs_review"] is True


def test_empty_title_is_rejected(client):
    assert client.post("/predict", json={"title": ""}).status_code == 422
