# ml-project-tagger

[![CI](https://github.com/champion-buzz/ml-project-tagger/actions/workflows/ci.yml/badge.svg)](https://github.com/champion-buzz/ml-project-tagger/actions/workflows/ci.yml)

A small end-to-end ML service. It reads a machine learning project's title and description and predicts its topic: `computer-vision`, `natural-language-processing`, `mlops` or `other`. When the model is unsure, it flags the prediction for human review instead of presenting it as settled.

## Results

Models were compared with 5-fold cross-validation on the 764 training rows. The selected model was then scored once on 191 holdout rows.

| Model | CV macro F1 |
| --- | --- |
| Majority-class baseline | 0.144 |
| TF-IDF + Naive Bayes | 0.841 |
| **TF-IDF + logistic regression (selected)** | **0.846** |

| Holdout metric | Value |
| --- | --- |
| Macro F1 | 0.914 |
| Accuracy | 0.927 |
| Predictions accepted without review | 94.8% |
| Accuracy of accepted predictions | 95.6% |

**Read these numbers with care.** The holdout score is higher than every cross-validation fold I ran (0.78 to 0.89 over 25 folds). I checked for leakage and found no shared descriptions, and removing the one holdout row that closely resembled a training row didn't change the score. The holdout set is small, so it probably just came out easy. I'd expect about 0.85 on new data. Naive Bayes and logistic regression are also effectively tied, so the choice between them isn't meaningful. `mlops` is the weakest class (F1 0.85), with only 16 holdout rows.

Full results are in [`results/metrics.json`](results/metrics.json).

## How it works

- **No peeking at the holdout:** model selection and the review threshold both use only out-of-fold predictions from the training data.
- **Review threshold:** the lowest confidence cutoff where accepted cross-validation predictions reach 90% precision.
- **Class imbalance:** logistic regression uses balanced class weights, and models are compared on macro F1, because `mlops` is only about 8% of the data.
- **Tests:** CI checks for train/holdout overlap, that the model beats the baseline, the threshold logic and the API's behavior.

## Run it

```bash
pip install -r requirements.txt
python -m tagger.train        # compares models, saves artifacts/model.joblib
python -m pytest -q
uvicorn tagger.api:app        # docs at http://127.0.0.1:8000/docs
```

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"title": "Object detection with YOLO", "description": "Detect objects in images."}'
```

The response has the shape `{"tag": "...", "confidence": 0.0, "needs_review": false}`.

## Limitations

- The data covers only 2020 projects, so newer topics such as LLM agents may be misclassified.
- Word-based features miss meaning that isn't in the vocabulary. A transformer might do better, but this baseline should be beaten first.
- Confidence comes from uncalibrated logistic regression probabilities, so treat it as a ranking and not as a true probability.

## Credits

The idea and dataset come from [Made With ML](https://github.com/GokuMohandas/Made-With-ML) (MIT, see [`data/NOTICE.md`](data/NOTICE.md)). The code is a separate, simpler implementation, built with AI assistance.
