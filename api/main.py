"""Phase 8. FORGE detection API.

Live now: schema, validation, versioning, metrics endpoint. The model is not loaded
because Phase 3 has not run. /v1/detect returns HTTP 503 with the reason rather than
a fabricated score, so nothing downstream can accidentally build on fake output.
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(title="FORGE", version="0.1.0")

MODEL_VERSION: str | None = None  # set once a trained model is registered


class DetectRequest(BaseModel):
    text: str = Field(min_length=1, max_length=200_000)


class Segment(BaseModel):
    start: int
    end: int
    ai_probability: float


class Uncertainty(BaseModel):
    calibrated: bool
    abstained: bool


class DetectResponse(BaseModel):
    prediction: str
    fraction_ai: float
    confidence: float
    model_version: str
    segments: list[Segment]
    uncertainty: Uncertainty


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_version": MODEL_VERSION, "model_loaded": MODEL_VERSION is not None}


@app.post("/v1/detect", response_model=DetectResponse)
def detect(req: DetectRequest) -> DetectResponse:
    if MODEL_VERSION is None:
        raise HTTPException(
            status_code=503,
            detail="No model registered. Phase 3 training has not run. "
            "Returning a score here would be fabricating a result.",
        )
    raise HTTPException(status_code=501, detail="Inference path is Phase 8.")
