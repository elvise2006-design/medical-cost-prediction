"""
schemas/prediction.py
=====================

Pydantic models used to VALIDATE what comes in and to DESCRIBE what goes out.

This file contains no machine-learning code and no HTTP code. Its only jobs:

  * reject bad input early, with friendly messages (HTTP 422)
  * normalise text input (trim + lower-case) so " Male " == "male"
  * document the API in Swagger at http://127.0.0.1:8000/docs
"""

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.config import ALLOWED_VALUES, CURRENCY

# ---------------------------------------------------------------------------
# Small helper reused by the three categorical fields
# ---------------------------------------------------------------------------
def _check_choice(field_name: str, value):
    """
    Trim + lower-case `value`, then confirm it is one of the allowed options.

    Raising ValueError here is what Pydantic turns into a 422 response.
    """
    allowed = ALLOWED_VALUES[field_name]
    if isinstance(value, str):
        value = value.strip().lower()
    if value not in allowed:
        raise ValueError(
            f"'{field_name}' must be one of: {', '.join(allowed)} "
            f"(received {value!r})"
        )
    return value


# ===========================================================================
# REQUEST
# ===========================================================================
class PredictionRequest(BaseModel):
    """
    One person's details.

    Example:
        {"age": 34, "sex": "male", "bmi": 27.8, "children": 2,
         "smoker": "no", "region": "northwest"}
    """

    # extra="forbid" -> an unexpected field is an error, not something we
    # silently ignore.
    model_config = ConfigDict(extra="forbid")

    age: int = Field(
        ..., ge=18, le=100,
        description="Age in years (18-100).",
        examples=[34],
    )
    sex: str = Field(
        ..., description="Either 'male' or 'female' (case-insensitive).",
        examples=["male"],
    )
    bmi: float = Field(
        ..., ge=10, le=70,
        description="Body Mass Index (10-70).",
        examples=[27.8],
    )
    children: int = Field(
        ..., ge=0, le=10,
        description="Number of children (0-10).",
        examples=[2],
    )
    smoker: str = Field(
        ..., description="Either 'yes' or 'no' (case-insensitive).",
        examples=["no"],
    )
    region: str = Field(
        ..., description="One of northeast, northwest, southeast, southwest.",
        examples=["northwest"],
    )

    @field_validator("sex", mode="before")
    @classmethod
    def _validate_sex(cls, value):
        return _check_choice("sex", value)

    @field_validator("smoker", mode="before")
    @classmethod
    def _validate_smoker(cls, value):
        return _check_choice("smoker", value)

    @field_validator("region", mode="before")
    @classmethod
    def _validate_region(cls, value):
        return _check_choice("region", value)


class PredictionInput(BaseModel):
    """
    The normalised request, echoed back in the response so the caller can see
    exactly what the server understood.
    """

    model_config = ConfigDict(extra="forbid")

    age: int = Field(..., examples=[34])
    sex: str = Field(..., examples=["male"])
    bmi: float = Field(..., examples=[27.8])
    children: int = Field(..., examples=[2])
    smoker: str = Field(..., examples=["no"])
    region: str = Field(..., examples=["northwest"])


# ===========================================================================
# RESPONSES
# ===========================================================================
class HealthResponse(BaseModel):
    """GET /api/v1/health"""

    success: bool = True
    status: str = Field(..., examples=["healthy"])


class StatusResponse(BaseModel):
    """GET /  - a friendly index page listing where everything lives."""

    success: bool = True
    message: str
    model: str
    model_version: str
    currency: str = CURRENCY
    docs: str
    endpoints: list[str]


class ModelMetrics(BaseModel):
    """Validation scores on the held-out 20% of the dataset."""

    MAE: float = Field(
        ..., description="Mean Absolute Error, in USD.",
        examples=[4181.5],
    )
    RMSE: float = Field(
        ..., description="Root Mean Squared Error, in USD.",
        examples=[6070.0],
    )
    R2: float = Field(
        ..., description="R-squared, 1.0 is a perfect fit.",
        examples=[0.86],
    )


class ModelInfoResponse(BaseModel):
    """GET /api/v1/model-info"""

    success: bool = True
    model: str = Field(..., examples=["Polynomial Regression"])
    model_version: str = Field(..., examples=["2.0"])
    metrics: ModelMetrics


class PredictionResponse(BaseModel):
    """POST /api/v1/predict"""

    success: bool = True
    predicted_medical_charges_usd: float = Field(
        ..., description="Predicted medical charges in USD.",
        examples=[1234.56],
    )
    currency: str = Field(..., examples=[CURRENCY])
    model: str = Field(..., examples=["Polynomial Regression"])
    model_version: str = Field(..., examples=["2.0"])
    input: PredictionInput


# ===========================================================================
# ERRORS
# ===========================================================================
# Every failure, whatever caused it, looks the same on the wire:
#
#   {"success": false, "error": {"code": "...", "message": "..."}}
#
# That keeps the frontend's error handling to a single code path.
class ErrorDetail(BaseModel):
    code: str = Field(
        ...,
        description="Stable machine-readable code.",
        examples=["VALIDATION_ERROR"],
    )
    message: str = Field(
        ...,
        description="Human-readable explanation.",
        examples=["age must be less than or equal to 100"],
    )


class ErrorResponse(BaseModel):
    """The single error shape used by every failing endpoint."""

    success: bool = False
    error: ErrorDetail