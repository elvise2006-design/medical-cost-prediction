"""
services/model_service.py
=========================

ALL machine-learning logic lives in this file.

Nothing in here knows about HTTP, and nothing outside this file should know
about pandas, scikit-learn or the CSV's column names. That separation is what
lets you replace the model later without touching the API or the frontend.

Pipeline of a prediction request:

    dict  ->  DataFrame  ->  _encode()  ->  Pipeline.predict()  ->  float (USD)

The two public entry points are:

    ModelService.train()    # load CSV, clean, encode, fit, score
    ModelService.predict()  # encode one row and predict
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

# Model identity is owned by config.py (settings live in one place) and is
# re-exported here so `model_service.MODEL_NAME` also works.
from app.config import (
    ALLOWED_VALUES,
    CATEGORICAL_COLUMNS,
    DATASET_PATH,
    FEATURE_COLUMNS,
    MODEL_NAME,
    MODEL_VERSION,
    NUMERIC_COLUMNS,
    RANDOM_STATE,
    TARGET_COLUMN,
    TEST_SIZE,
)

# ===========================================================================
# THE ONLY FUNCTION YOU NEED TO EDIT TO SWAP THE MODEL
# ===========================================================================
def _build_model() -> Pipeline:
    """
    Build the machine-learning pipeline.

    Current model: second-degree polynomial expansion of the inputs, then
    standardisation, then ordinary least squares linear regression.

    To try a different model later, change ONLY the body of this function.
    For example, a random forest:

        from sklearn.ensemble import RandomForestRegressor

        return Pipeline(
            steps=[
                ("regressor", RandomForestRegressor(
                    n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1
                )),
            ],
        )

    Note that a tree-based model does not need scaling, so PolynomialFeatures
    and StandardScaler are simply left out.

    Rules to keep the API stable:
      * It must return an object with .fit(X, y) and .predict(X).
      * It must accept a plain DataFrame / numpy array of encoded features.
      * Do not change the input format expected by _encode().
    """
    return Pipeline(
        steps=[
            # Expand features: (x1, x2, x1*x2, x1**2, x2**2, ...)
            # include_bias=False avoids adding a constant column of 1s,
            # because LinearRegression already fits an intercept.
            ("polynomial", PolynomialFeatures(degree=2, include_bias=False)),
            # Put every feature on a similar scale so the fit is stable.
            ("scaler", StandardScaler()),
            # The actual estimator.
            ("regressor", LinearRegression()),
        ],
    )


# ===========================================================================
# Errors
# ===========================================================================
# The HTTP layer catches these and turns them into status codes:
#   DatasetNotFoundError -> 503
#   PredictionError      -> 500
#   ModelServiceError    -> 500
class ModelServiceError(Exception):
    """Base class for every error raised by this service."""


class DatasetNotFoundError(ModelServiceError):
    """The training CSV is missing or unreadable -> HTTP 503."""


class PredictionError(ModelServiceError):
    """A prediction could not be produced -> HTTP 500."""


# ===========================================================================
# The service
# ===========================================================================
class ModelService:
    """
    Loads the dataset, trains the model once, and serves predictions.

    A single instance is shared by every request (see
    services/dependencies.py), so training happens exactly once per process.
    """

    def __init__(
        self,
        dataset_path: Path | str = DATASET_PATH,
        model_name: str = MODEL_NAME,
        model_version: str = MODEL_VERSION,
    ) -> None:
        self.dataset_path = Path(dataset_path)
        self.model_name = model_name
        self.model_version = model_version

        # Filled in by train()
        self.model: Pipeline | None = None
        self.categories_: dict[str, list[str]] = {}   # learned from TRAIN only
        self.feature_names_: list[str] = []
        self.metrics_: dict[str, float] = {}
        self.n_rows_: int = 0
        self.n_duplicates_removed_: int = 0

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------
    def _encode(self, frame: pd.DataFrame, *, fit: bool = False) -> pd.DataFrame:
        """
        Turn raw columns into a numeric feature matrix.

        The SAME function is used for training and for prediction. That is the
        important part: if they ever drifted apart, training would look fine
        and live predictions would silently be wrong.

        Steps:
          1. Enforce the exact training column order (and reject extras).
          2. Normalise categorical text: strip whitespace, lower-case.
          3. On training only: learn the category list from this data.
          4. Reject unknown categories with a clear error.
          5. Build one-hot columns for exactly the learned categories,
             dropping the first one (drop_first=True behaviour).

        Args:
            frame: DataFrame containing FEATURE_COLUMNS.
            fit:   True when training (learn categories), False when predicting.

        Returns:
            A DataFrame of floats with a stable, identical column layout.
        """
        # --- 1. exact training column order -----------------------------
        missing = [c for c in FEATURE_COLUMNS if c not in frame.columns]
        if missing:
            raise PredictionError(
                f"Missing required feature(s): {', '.join(missing)}"
            )
        unexpected = [c for c in frame.columns if c not in FEATURE_COLUMNS]
        if unexpected:
            raise PredictionError(
                f"Unexpected feature(s): {', '.join(unexpected)}"
            )
        # Reindex: guarantees the order regardless of how the caller built it.
        frame = frame.loc[:, FEATURE_COLUMNS].copy()

        # --- 2. normalise categorical text ------------------------------
        for column in CATEGORICAL_COLUMNS:
            frame[column] = frame[column].astype("string").str.strip().str.lower()

        # Numeric values must be present and numeric.
        for column in NUMERIC_COLUMNS:
            if frame[column].isna().any():
                raise PredictionError(f"Feature '{column}' contains a missing value.")
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
            if frame[column].isna().any():
                raise PredictionError(f"Feature '{column}' must be numeric.")

        # --- 3. learn categories from TRAINING data only ---------------
        if fit:
            self.categories_ = {}
            for column in CATEGORICAL_COLUMNS:
                values = sorted(frame[column].dropna().unique().tolist())
                if not values:
                    raise ModelServiceError(
                        f"Column '{column}' in the dataset has no usable values."
                    )
                allowed = ALLOWED_VALUES[column]
                self.categories_[column] = [v for v in values if v in allowed]

        if not self.categories_:
            raise ModelServiceError(
                "Encoder has no categories yet - the model must be trained first."
            )

        # --- 4. reject unknown categories -------------------------------
        # Without this check, an unseen value would quietly become an
        # all-zero row of dummies and produce a plausible-looking but wrong
        # prediction. Failing loudly is much safer.
        for column in CATEGORICAL_COLUMNS:
            known = set(self.categories_[column])
            seen = set(frame[column].dropna().unique().tolist())
            unknown = sorted(seen - known)
            if unknown:
                raise PredictionError(
                    f"Unknown value(s) for '{column}': {', '.join(unknown)}. "
                    f"Allowed values: {', '.join(sorted(known))}."
                )

        # --- 5. build one-hot columns from the learned categories -------
        blocks = [frame.loc[:, NUMERIC_COLUMNS].astype(float)]
        for column in CATEGORICAL_COLUMNS:
            categories = self.categories_[column]
            # categories[1:] == drop_first=True: one category is used as the
            # baseline, which also makes the dummies linearly independent.
            for category in categories[1:]:
                blocks.append(
                    pd.DataFrame(
                        {
                            f"{column}_{category}": (
                                frame[column] == category
                            ).astype(float).to_numpy()
                        },
                        index=frame.index,
                    )
                )

        encoded = pd.concat(blocks, axis=1)

        # Sanity check: every transform must produce the same columns.
        if fit:
            self.feature_names_ = encoded.columns.tolist()
        elif encoded.columns.tolist() != self.feature_names_:
            raise ModelServiceError(
                "Encoded columns do not match the training layout; the model "
                "and the encoder are out of sync."
            )

        return encoded

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------
    def _load_dataset(self) -> pd.DataFrame:
        """
        Read the CSV and drop duplicate rows.

        Raises DatasetNotFoundError when the file is missing, which the HTTP
        layer reports as 503.
        """
        if not self.dataset_path.exists():
            raise DatasetNotFoundError(
                f"Training dataset not found at {self.dataset_path}. "
                "Place insurance.csv in the project's data/ folder and restart "
                "the server."
            )

        try:
            frame = pd.read_csv(self.dataset_path)
        except Exception as exc:  # unreadable file, bad encoding, ...
            raise DatasetNotFoundError(
                f"Could not read the training dataset at {self.dataset_path}: {exc}"
            ) from exc

        required = FEATURE_COLUMNS + [TARGET_COLUMN]
        missing = [c for c in required if c not in frame.columns]
        if missing:
            raise ModelServiceError(
                f"The dataset is missing column(s): {', '.join(missing)}."
            )

        # Drop duplicate rows, then reset the index so it is clean 0..n-1.
        before = len(frame)
        frame = frame.drop_duplicates().reset_index(drop=True)
        self.n_duplicates_removed_ = before - len(frame)
        self.n_rows_ = len(frame)

        return frame

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def train(self) -> dict[str, float]:
        """
        Train the model from scratch and compute validation metrics.

        Returns:
            {"MAE": ..., "RMSE": ..., "R2": ...}
        """
        frame = self._load_dataset()

        features = frame.loc[:, FEATURE_COLUMNS]
        target = frame.loc[:, TARGET_COLUMN].astype(float)

        # 80 / 20 split with a fixed seed so results never change between runs.
        x_train, x_test, y_train, y_test = train_test_split(
            features,
            target,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
        )

        # Categories are learned from the TRAIN portion only, so nothing about
        # the test set leaks into the model.
        x_train_encoded = self._encode(x_train, fit=True)
        x_test_encoded = self._encode(x_test, fit=False)

        # Fresh model every time, in case train() is called more than once.
        self.model = _build_model()
        self.model.fit(x_train_encoded, y_train)

        predictions = self.model.predict(x_test_encoded)

        # RMSE: mean_squared_error no longer has a `squared=False` option in
        # recent scikit-learn versions, so take the square root ourselves.
        self.metrics_ = {
            "MAE": round(float(mean_absolute_error(y_test, predictions)), 2),
            "RMSE": round(
                float(np.sqrt(mean_squared_error(y_test, predictions))), 2
            ),
            "R2": round(float(r2_score(y_test, predictions)), 4),
        }

        return self.metrics_

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------
    def predict(self, payload: dict) -> float:
        """
        Predict medical charges for one person.

        Args:
            payload: {"age": 34, "sex": "male", "bmi": 27.8, ...}

        Returns:
            The predicted charge in USD, rounded to 2 decimals, never negative.
        """
        if self.model is None:
            raise ModelServiceError("The model has not been trained yet.")

        # Only the known feature columns, in the training order.
        absent = [c for c in FEATURE_COLUMNS if c not in payload]
        if absent:
            raise PredictionError(
                f"Missing required feature(s): {', '.join(absent)}"
            )
        frame = pd.DataFrame([{k: payload[k] for k in FEATURE_COLUMNS}])

        encoded = self._encode(frame, fit=False)

        try:
            raw = float(self.model.predict(encoded)[0])
        except Exception as exc:
            raise PredictionError(f"Prediction failed: {exc}") from exc

        if not math.isfinite(raw):
            raise PredictionError("The model produced a non-finite value.")

        # Medical charges can never be negative, so clamp at zero.
        return round(max(raw, 0.0), 2)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------
    def model_info(self) -> dict:
        """Model name, version and metrics, exactly as the API reports them."""
        return {
            "model": self.model_name,
            "model_version": self.model_version,
            "metrics": dict(self.metrics_),
        }