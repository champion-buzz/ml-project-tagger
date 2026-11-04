"""FastAPI service. Train first, then run: uvicorn tagger.api:app --reload"""
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from tagger.model import MODEL_PATH, to_text

app = FastAPI(title="ML project tagger")
_bundle = None


def get_bundle() -> dict:
    global _bundle
    if _bundle is None:
        if not MODEL_PATH.exists():
            raise HTTPException(503, "No trained model. Run: python -m tagger.train")
        _bundle = joblib.load(MODEL_PATH)
    return _bundle


class Project(BaseModel):
    title: str = Field(min_length=1, max_length=300)
    description: str = Field(default="", max_length=5000)


class Prediction(BaseModel):
    tag: str
    confidence: float
    needs_review: bool


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": MODEL_PATH.exists()}


@app.post("/predict", response_model=Prediction)
def predict(project: Project) -> Prediction:
    bundle = get_bundle()
    proba = bundle["model"].predict_proba([to_text(project.title, project.description)])[0]
    best = int(proba.argmax())
    confidence = float(proba[best])
    return Prediction(
        tag=str(bundle["model"].classes_[best]),
        confidence=round(confidence, 3),
        needs_review=confidence < bundle["threshold"],
    )
