"""FastAPI model service for early-delinquency scoring.

Endpoints
  GET  /                web dashboard (single borrower + batch ranking)
  GET  /health          liveness + model metadata
  POST /score           score one borrower (any subset of the 52 features; missing -> imputed)
  POST /score/batch     score many borrowers and return them ranked for the collections queue
"""
import os
import warnings
from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

warnings.filterwarnings("ignore", message="X does not have valid feature names")

BUNDLE_PATH = Path(os.getenv("BUNDLE_PATH", Path(__file__).resolve().parents[1] / "artifacts" / "early_delinquency_bundle.joblib"))

app = FastAPI(title="Early-Delinquency Scoring API", version="1.0.0",
              description="Probability that a performing digital borrower falls >=1 day past due.")
_bundle: Dict[str, Any] = {}


def bundle() -> Dict[str, Any]:
    if not _bundle:
        if not BUNDLE_PATH.exists():
            raise HTTPException(503, f"model bundle not found at {BUNDLE_PATH}; run `python -m src.pipeline`")
        _bundle.update(joblib.load(BUNDLE_PATH))
    return _bundle


class Borrower(BaseModel):
    features: Dict[str, Any] = Field(..., description="Feature name -> value. Unknown keys are rejected; "
                                     "missing known features are imputed by the fitted pipeline.",
                                     examples=[{"Gender": "MALE", "age": 29, "delay_risk_class": "Frequent",
                                                "prior_delinq_rate": 0.5, "late_streak": 2,
                                                "utilization_ratio": 0.95, "CRB_Grade": "D", "grade_ord": 4}])


class BatchRequest(BaseModel):
    borrowers: List[Dict[str, Any]] = Field(..., max_length=10000)


def _score(rows: List[Dict[str, Any]]) -> np.ndarray:
    b = bundle()
    feats = b["features"]
    unknown = {k for r in rows for k in r} - set(feats)
    if unknown:
        raise HTTPException(422, f"unknown features: {sorted(unknown)}")
    df = pd.DataFrame([{f: r.get(f, np.nan) for f in feats} for r in rows])
    num = [c for c in feats if c not in ("Gender", "age_band", "delay_risk_class", "CRB_Grade")]
    df[num] = df[num].apply(pd.to_numeric, errors="coerce")
    p = b["model"].predict_proba(b["prep"].transform(df))[:, 1]
    cal, ct = b["calibrator"], b["cal_type"]
    if cal is not None:
        p = cal.transform(p) if ct == "isotonic" else cal.predict_proba(p.reshape(-1, 1))[:, 1]
    return p


@app.get("/", include_in_schema=False)
def dashboard():
    return FileResponse(Path(__file__).parent / "static" / "dashboard.html")


@app.get("/health")
def health():
    b = bundle()
    return {"status": "ok", "backend": b.get("backend"), "calibrator": b["cal_type"],
            "threshold": b["threshold"], "n_features": len(b["features"]), "features": b["features"],
            "target": b["target"]}


@app.post("/score")
def score(req: Borrower):
    p = float(_score([req.features])[0])
    t = bundle()["threshold"]
    return {"risk_probability": round(p, 4), "flag": bool(p >= t), "threshold": round(t, 4),
            "risk_band": "high" if p >= 0.5 else "medium" if p >= t else "low"}


@app.post("/score/batch")
def score_batch(req: BatchRequest):
    if not req.borrowers:
        raise HTTPException(422, "borrowers must be non-empty")
    p = _score(req.borrowers)
    t = bundle()["threshold"]
    order = np.argsort(-p)
    return {"threshold": round(t, 4), "n": len(p), "n_flagged": int((p >= t).sum()),
            "ranked": [{"index": int(i), "risk_probability": round(float(p[i]), 4), "flag": bool(p[i] >= t)}
                       for i in order]}
