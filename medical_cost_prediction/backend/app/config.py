"""
config.py
=========

Settings and file paths ONLY.

This module deliberately contains no machine-learning code, no HTTP code and
no business logic. If you want to change where the CSV lives, which columns
are features, or the model name/version that the API reports, this is the
only file you need to edit.

Layering reminder:
    config.py  ->  paths + settings (this file)
    services/  ->  machine learning (imports settings from here)
    routes/    ->  HTTP (imports settings from here)
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# This file lives at:  medical_cost_prediction/backend/app/config.py
#   parents[0] -> backend/app
#   parents[1] -> backend
#   parents[2] -> medical_cost_prediction   <-- project root
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

DATA_DIR: Path = PROJECT_ROOT / "data"

# The training dataset. If this file is missing the API starts but every
# prediction returns HTTP 503 with code "DATASET_NOT_FOUND".
DATASET_PATH: Path = DATA_DIR / "insurance.csv"

# ---------------------------------------------------------------------------
# Dataset schema
# ---------------------------------------------------------------------------
# FEATURE_COLUMNS is the exact column order the model was trained on.
# At prediction time the incoming data is forced back into this order before
# anything else happens, so a caller can never silently change the model's
# input layout.
FEATURE_COLUMNS: list[str] = ["age", "sex", "bmi", "children", "smoker", "region"]

# Numeric columns pass through the encoder untouched.
NUMERIC_COLUMNS: list[str] = ["age", "bmi", "children"]

# Categorical columns get one-hot encoded.
CATEGORICAL_COLUMNS: list[str] = ["sex", "smoker", "region"]

# The column we want to predict (in USD).
TARGET_COLUMN: str = "charges"

# Allowed values per categorical column, stored lower-case.
# The API layer normalises user input (trim + lower-case) before comparing,
# so " Male " and "male" are both accepted.
ALLOWED_VALUES: dict[str, list[str]] = {
    "sex": ["male", "female"],
    "smoker": ["yes", "no"],
    "region": ["northeast", "northwest", "southeast", "southwest"],
}

# ---------------------------------------------------------------------------
# Model identity
# ---------------------------------------------------------------------------
# These two strings are part of the PUBLIC API contract (they are returned by
# /api/v1/model-info and /api/v1/predict). Bump MODEL_VERSION whenever you
# change how the model is built.
MODEL_NAME: str = "Polynomial Regression"
MODEL_VERSION: str = "2.0"

# ---------------------------------------------------------------------------
# Training settings
# ---------------------------------------------------------------------------
TEST_SIZE: float = 0.2        # 80% train / 20% test
RANDOM_STATE: int = 42        # seed, so results are reproducible

# ---------------------------------------------------------------------------
# API settings
# ---------------------------------------------------------------------------
APP_TITLE: str = "Medical Cost Prediction API"
APP_VERSION: str = MODEL_VERSION
APP_DESCRIPTION: str = (
    "Predicts medical insurance charges (USD) from age, sex, BMI, number of "
    "children, smoking status and region."
)

API_PREFIX: str = "/api/v1"
CURRENCY: str = "USD"

# Port the API listens on. Port 8000 was already taken by another local app
# when this project was set up, so 8001 is used instead. If you run
# `uvicorn app.main:app --port 8000` yourself, change API_PORT and the
# API_URL in frontend/app.js to match.
API_PORT: int = 8001

# The frontend is served by `python3 -m http.server 5500 --bind 127.0.0.1`.
CORS_ORIGINS: list[str] = [
    "http://127.0.0.1:5500",
    "http://localhost:5500",
]