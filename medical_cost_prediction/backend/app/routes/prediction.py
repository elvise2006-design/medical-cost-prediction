"""
routes/prediction.py
====================

HTTP layer ONLY.

Each function here does exactly three things:
    1. receive a validated request,
    2. call the service layer,
    3. return a Pydantic response object.

There is no machine learning in this file, and no model knowledge either.
If you swap Polynomial Regression for a Random Forest, nothing here changes.
"""

from fastapi import APIRouter, Depends

from app.config import API_PREFIX, CURRENCY
from app.schemas.prediction import (
    HealthResponse,
    ModelInfoResponse,
    PredictionInput,
    PredictionRequest,
    PredictionResponse,
)
from app.services.dependencies import get_model_service
from app.services.model_service import ModelService

# Every route in this file is mounted under /api/v1 and shows up in Swagger
# under the "medical-cost" tag.
router = APIRouter(prefix=API_PREFIX, tags=["medical-cost"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Liveness check",
)
def health() -> HealthResponse:
    """
    Confirms the API process is up.

    Intentionally does NOT touch the model, so it keeps answering 200 even if
    the dataset is missing. Use /model-info or /predict to check the model.
    """
    return HealthResponse(success=True, status="healthy")


@router.get(
    "/model-info",
    response_model=ModelInfoResponse,
    summary="Which model is loaded and how accurate is it",
)
def model_info(
    service: ModelService = Depends(get_model_service),
) -> ModelInfoResponse:
    """
    Report the model name, its version and its validation metrics.

    These strings come from config.py, so renaming the model never changes
    the shape of this response - only the values.
    """
    return ModelInfoResponse(success=True, **service.model_info())


@router.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Predict medical charges in USD",
)
def predict(
    payload: PredictionRequest,
    service: ModelService = Depends(get_model_service),
) -> PredictionResponse:
    """
    Predict the medical insurance charges for one person.

    `payload` has already been validated by Pydantic by the time we get it
    (age 18-100, bmi 10-70, children 0-10, and the fixed text options).
    """
    # Hand a plain dict to the service: the HTTP layer stays ignorant of how
    # the prediction is actually produced.
    amount = service.predict(payload.model_dump())

    return PredictionResponse(
        success=True,
        predicted_medical_charges_usd=amount,
        currency=CURRENCY,
        model=service.model_name,
        model_version=service.model_version,
        # Echo the normalised input back so the caller can verify it.
        input=PredictionInput(**payload.model_dump()),
    )