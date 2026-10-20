"""
main.py
=======

The FastAPI application: middleware, routes, error handling and startup.

This file wires the layers together but does not implement any of them.
Everything it needs is imported from config.py, schemas/, routes/ and services/.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.config import (
    API_PREFIX,
    APP_DESCRIPTION,
    APP_TITLE,
    APP_VERSION,
    CORS_ORIGINS,
    CURRENCY,
    MODEL_NAME,
    MODEL_VERSION,
)
from app.routes.prediction import router as prediction_router
from app.schemas.prediction import StatusResponse
from app.services.dependencies import get_model_service
from app.services.model_service import (
    DatasetNotFoundError,
    ModelServiceError,
    PredictionError,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s %(name)s: %(message)s",
)
logger = logging.getLogger("medical_cost")


# ===========================================================================
# Startup / shutdown
# ===========================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Train the model once, when the server starts.

    Calling get_model_service() here populates the lru_cache, so the model is
    already trained by the time the first request arrives.

    If training fails we log the problem but still let the server boot: the
    API stays reachable and /predict returns a proper error code (503 if the
    dataset is missing, 500 otherwise) instead of the whole API disappearing.
    """
    logger.info("Starting up: training %s v%s ...", MODEL_NAME, MODEL_VERSION)
    try:
        service = get_model_service()
    except ModelServiceError as exc:
        logger.error("Model training failed: %s", exc)
        logger.error("The API will stay up, but /predict will return an error.")
    else:
        logger.info("Model ready.")
        logger.info("  rows used        : %s", service.n_rows_)
        logger.info("  duplicates removed: %s", service.n_duplicates_removed_)
        logger.info("  encoded features : %s", len(service.feature_names_))
        logger.info("  metrics          : %s", service.metrics_)
    yield
    logger.info("Shutting down.")


# ===========================================================================
# App
# ===========================================================================
app = FastAPI(
    title=APP_TITLE,
    version=APP_VERSION,
    description=APP_DESCRIPTION,
    lifespan=lifespan,
)

# The frontend runs on a different port (5500) than the API (8000), so the
# browser needs CORS permission to call it.
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


# ===========================================================================
# One helper for the single error shape
# ===========================================================================
def _error(status_code: int, code: str, message: str) -> JSONResponse:
    """
    Build the one and only error format the API ever returns:

        {"success": false, "error": {"code": "...", "message": "..."}}
    """
    return JSONResponse(
        status_code=status_code,
        content={"success": False, "error": {"code": code, "message": message}},
    )


# ===========================================================================
# Error handlers
# ===========================================================================
@app.exception_handler(RequestValidationError)
async def validation_error_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """
    Bad input -> 422.

    Pydantic gives us a list of problems; we surface the first one in plain
    English, because that is what a user reading the page cares about.
    """
    details = exc.errors()
    if details:
        first = details[0]
        location = ".".join(str(part) for part in first.get("loc", []) if part != "body")
        message = first.get("msg", "Invalid input.")
        if location:
            message = f"{location}: {message}"
    else:
        message = "Invalid input."

    if len(details) > 1:
        message = f"{message} (+{len(details) - 1} more problem(s))"

    logger.info("422 validation error on %s: %s", request.url.path, message)
    return _error(422, "VALIDATION_ERROR", message)


@app.exception_handler(DatasetNotFoundError)
async def dataset_missing_handler(
    request: Request, exc: DatasetNotFoundError
) -> JSONResponse:
    """The CSV is missing or unreadable -> 503 (service unavailable)."""
    logger.error("503 dataset unavailable on %s: %s", request.url.path, exc)
    return _error(503, "DATASET_NOT_FOUND", str(exc))


@app.exception_handler(PredictionError)
async def prediction_error_handler(
    request: Request, exc: PredictionError
) -> JSONResponse:
    """Something went wrong while predicting -> 500."""
    logger.exception("500 prediction failure on %s", request.url.path)
    return _error(500, "PREDICTION_FAILED", str(exc))


@app.exception_handler(ModelServiceError)
async def model_service_error_handler(
    request: Request, exc: ModelServiceError
) -> JSONResponse:
    """Any other service-layer failure -> 500."""
    logger.exception("500 model service error on %s", request.url.path)
    return _error(500, "MODEL_ERROR", str(exc))


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(
    request: Request, exc: StarletteHTTPException
) -> JSONResponse:
    """
    Keep built-in HTTP errors (404 unknown route, 405 wrong method, ...)
    in the same envelope as everything else.
    """
    codes = {404: "NOT_FOUND", 405: "METHOD_NOT_ALLOWED"}
    code = codes.get(exc.status_code, "HTTP_ERROR")
    message = exc.detail if isinstance(exc.detail, str) else str(exc.detail)
    return _error(exc.status_code, code, message)


@app.exception_handler(Exception)
async def unhandled_error_handler(
    request: Request, exc: Exception
) -> JSONResponse:
    """Last-resort safety net -> 500, never a stack trace on the wire."""
    logger.exception("500 unhandled error on %s", request.url.path)
    return _error(
        500,
        "INTERNAL_ERROR",
        "An unexpected error occurred. Check the server logs for details.",
    )


# ===========================================================================
# Routes
# ===========================================================================
@app.get("/", response_model=StatusResponse, tags=["meta"], summary="API status")
def root() -> StatusResponse:
    """
    A plain status page, handy for confirming the server is alive and seeing
    where the other endpoints are.
    """
    return StatusResponse(
        message="Medical Cost Prediction API is running.",
        model=MODEL_NAME,
        model_version=MODEL_VERSION,
        currency=CURRENCY,
        docs="/docs",
        endpoints=[
            "GET  /api/v1/health",
            "GET  /api/v1/model-info",
            "POST /api/v1/predict",
        ],
    )


# Health, model-info and predict all live in the router module.
app.include_router(prediction_router)