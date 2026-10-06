"""FastAPI service for real-time transaction scoring and health checks."""

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, Optional, Union

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator

from fraudguard.data import (
    RAW_PREDICTORS,
    ValidationError,
)
from fraudguard.scoring import FraudScorer

MAX_PAYLOAD_BYTES = 64 * 1024  # 64 KiB

# Global scorer instance
_scorer: Optional[FraudScorer] = None
_scorer_load_error: Optional[str] = None


def get_artifact_dir() -> str:
    env_dir = os.getenv("FRAUDGUARD_ARTIFACT_DIR")
    if env_dir and Path(env_dir).exists():
        return env_dir
    for fallback in [
        "artifacts/champion",
        "artifacts/baselines/provisional",
        "artifacts/baselines/logistic_reference",
    ]:
        if Path(fallback).exists():
            return fallback
    return "artifacts/champion"


def load_global_scorer():
    global _scorer, _scorer_load_error
    art_dir = get_artifact_dir()
    try:
        if Path(art_dir).exists() and (Path(art_dir) / "manifest.json").exists():
            _scorer = FraudScorer.from_directory(art_dir)
            _scorer_load_error = None
        else:
            _scorer = None
            _scorer_load_error = f"Artifact directory missing or incomplete: {art_dir}"
    except Exception as e:
        _scorer = None
        _scorer_load_error = str(e)


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_global_scorer()
    yield


app = FastAPI(
    title="FraudGuard API",
    description="Real-time transaction fraud scoring and decision service",
    version="0.1.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def payload_size_limit_middleware(request: Request, call_next):
    """Enforce strict 64 KiB payload limit."""
    content_length = request.headers.get("content-length")
    if content_length and int(content_length) > MAX_PAYLOAD_BYTES:
        return JSONResponse(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            content={
                "error": "PayloadTooLarge",
                "message": f"Request body exceeds {MAX_PAYLOAD_BYTES} bytes limit.",
            },
        )

    # For chunked or unstated length
    body = await request.body()
    if len(body) > MAX_PAYLOAD_BYTES:
        return JSONResponse(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            content={
                "error": "PayloadTooLarge",
                "message": f"Request body exceeds {MAX_PAYLOAD_BYTES} bytes limit.",
            },
        )

    response = await call_next(request)
    return response


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"error": "ValidationError", "details": exc.errors()},
    )


class TransactionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    TransactionID: int = Field(..., gt=0, description="Unique positive transaction ID")
    TransactionDT: Union[int, float] = Field(..., ge=0, description="Elapsed timestamp seconds")
    TransactionAmt: float = Field(..., ge=0, description="Transaction amount")
    ProductCD: str = Field(..., min_length=1, max_length=128, description="Product code")
    dist1: Optional[float] = Field(default=None, ge=0, description="Distance 1")
    dist2: Optional[float] = Field(default=None, ge=0, description="Distance 2")
    card4: Optional[str] = Field(default=None, max_length=128, description="Card brand")
    card6: Optional[str] = Field(default=None, max_length=128, description="Card category")
    addr1: Optional[Union[int, str]] = Field(default=None, description="Billing address code")
    addr2: Optional[Union[int, str]] = Field(default=None, description="Billing address code 2")
    P_emaildomain: Optional[str] = Field(
        default=None, max_length=128, description="Purchaser email domain"
    )
    R_emaildomain: Optional[str] = Field(
        default=None, max_length=128, description="Recipient email domain"
    )
    M4: Optional[str] = Field(default=None, max_length=128, description="Match status 4")
    M6: Optional[str] = Field(default=None, max_length=128, description="Match status 6")

    @field_validator("TransactionID", mode="before")
    @classmethod
    def validate_id_not_bool(cls, v):
        if isinstance(v, bool):
            raise ValueError("TransactionID cannot be a boolean.")
        return v

    @field_validator("TransactionAmt", mode="before")
    @classmethod
    def validate_amt_not_bool(cls, v):
        if isinstance(v, bool):
            raise ValueError("TransactionAmt cannot be a boolean.")
        return v


class ScoreResponse(BaseModel):
    TransactionID: int
    fraud_score: float
    score_type: str
    review_recommended: bool
    model_version: str
    schema_version: str
    policy_version: str


@app.get("/health", summary="Liveness probe")
def health() -> Dict[str, str]:
    """Liveness probe: returns 200 if API service is running."""
    return {"status": "alive"}


@app.get("/ready", summary="Readiness probe")
def ready() -> Dict[str, Any]:
    """Readiness probe: returns 200 only when trusted model artifacts are loaded."""
    global _scorer, _scorer_load_error
    if _scorer is None:
        load_global_scorer()

    if _scorer is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"status": "not_ready", "error": _scorer_load_error or "Artifacts not loaded"},
        )
    return {
        "status": "ready",
        "model_version": _scorer.model_version,
        "score_type": _scorer.score_type,
        "policy_version": _scorer.policy_version,
    }


@app.get("/model-info", summary="Model metadata")
def model_info() -> Dict[str, Any]:
    """Model information: schema, version, policy parameters, and features."""
    global _scorer
    if _scorer is None:
        load_global_scorer()

    if _scorer is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"status": "not_ready", "error": _scorer_load_error or "Artifacts not loaded"},
        )

    return {
        "model_version": _scorer.model_version,
        "schema_version": _scorer.schema_version,
        "policy_version": _scorer.policy_version,
        "score_type": _scorer.score_type,
        "cutoff": _scorer.cutoff,
        "default_review_fraction": _scorer.default_fraction,
        "raw_predictors": RAW_PREDICTORS,
        "encoded_feature_count": len(
            _scorer.bundle.feature_schema.get("encoded_feature_names", [])
        ),
        "manifest": _scorer.manifest,
    }


@app.post("/v1/score", response_model=ScoreResponse, summary="Score transaction")
def score_transaction(payload: TransactionRequest) -> ScoreResponse:
    """Score a single transaction and return fraud probability and review recommendation."""
    global _scorer
    if _scorer is None:
        load_global_scorer()

    if _scorer is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error": "ModelUnavailable",
                "message": _scorer_load_error or "Artifacts not loaded",
            },
        )

    try:
        data_dict = payload.model_dump()
        result = _scorer.score_single(data_dict)
        return ScoreResponse(**result)
    except ValidationError as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Inference error: {str(e)}"
        )
