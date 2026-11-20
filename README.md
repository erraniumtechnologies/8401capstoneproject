# Early-Warning Delinquency Prediction for Digital Lending in Kenya

DSA 8401 Applied Machine Learning – Capstone (Strathmore University, iLabAfrica)
Author: Erra Dennis Ombati (074494)

Predicts, from information available when a loan is issued, the probability that a **currently performing**
digital borrower will fall **one or more days past due**, and serves that probability (plus a collections flag)
through a containerised FastAPI service.

## Architecture

```
CSV (931,800 loans) -> ingest + recompute target -> leakage-free features (52) -> grouped split by borrower
   -> Optuna-tuned gradient-boosted model (+ LR, RF baselines) -> threshold + isotonic/Platt calibration
   -> artifacts/early_delinquency_bundle.joblib -> FastAPI (/score, /score/batch) -> Docker
```

| Path | Purpose |
|---|---|
| `src/config.py` | paths, features, business economics, tuned hyper-parameters |
| `src/features.py` | ingestion, target (`arrears_days = Account_Status_Date - Installment_Due_Date >= 1`), features |
| `src/modeling.py` | grouped splits, preprocessing, model factory, metrics, thresholds, calibration |
| `src/pipeline.py` | scripted end-to-end run (ingest -> preprocess -> train -> evaluate -> save) |
| `api/main.py` | FastAPI service (also serves the dashboard at `/`) |
| `api/static/dashboard.html` | web dashboard: single-borrower scoring and batch queue ranking |
| `Dockerfile`, `docker-compose.yml` | containerised service and optional training job |

## Reproduce from scratch

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
mkdir -p data && cp /path/to/Digital_Banking_MSC_2024_2025_v3.csv data/

python -m src.pipeline                 # full run: Optuna (40 trials), grouped CV, out-of-time, calibration
python -m src.pipeline --no-tune       # reuse the tuned hyper-parameters in src/config.py (much faster)
python -m src.pipeline --sample-clients 6000 --no-tune --skip-cv   # ~20 s smoke test

```

Outputs: `artifacts/early_delinquency_bundle.joblib` and `artifacts/metrics.json`.

## Serve with Docker

```bash
docker compose run --rm train          # optional: train inside the container (needs ./data/*.csv)
docker compose up --build api          # serves http://localhost:8000  (docs at /docs)

curl -s localhost:8000/health
curl -s -X POST localhost:8000/score -H 'content-type: application/json' -d '{
  "features": {"Gender":"MALE","age":29,"age_band":"25-35","delay_risk_class":"Frequent",
               "prior_delinq_rate":0.5,"late_streak":2,"recency_delinq":0.7,"last_arrears":12,
               "share_paid_early":0.1,"utilization_ratio":0.95,"CRB_Grade":"D","grade_ord":4,
               "borrowing_frequency":6,"Estimated_Income":13000}}'
```

Open **http://localhost:8000/** for the dashboard (single borrower form with example profiles, and CSV batch ranking
with a downloadable ranked queue). It calls the same `/score` and `/score/batch` endpoints.

Response: `{"risk_probability": ..., "flag": true|false, "threshold": <solved at training time>, "risk_band": "..."}`.
`/score/batch` takes `{"borrowers":[{...},...]}` and returns them ranked for the collections queue.

## Notes

* LightGBM is the production backend. If it is not installed the code falls back to scikit-learn's
  `HistGradientBoostingClassifier` so the pipeline still runs; the bundle records which backend produced it.
* The raw data is confidential and is **not** included in the repository.

## Threshold note

The API applies the threshold to **calibrated** probabilities, so `src/pipeline.py` solves all thresholds on
calibrated validation probabilities. (Solving them on raw scores and applying them to calibrated scores would
mis-flag borrowers, because class weighting inflates raw scores.) 