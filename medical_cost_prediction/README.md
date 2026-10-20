# Medical Cost Prediction Using Polynomial Regression

A small full-stack app that estimates medical insurance charges (in USD) from a
person's age, sex, BMI, number of children, smoking status and region.

* **Frontend** — plain HTML / CSS / JavaScript. Collects values, calls the API, shows the answer.
* **Backend** — FastAPI. Validates input, trains a scikit-learn model, returns a prediction.

The whole point of the structure below is that **the model can be replaced
without touching the frontend or the API**.

---

## Table of contents

1. [Project structure](#1-project-structure)
2. [Architecture](#2-architecture)
3. [How the frontend talks to the backend](#3-how-the-frontend-talks-to-the-backend)
4. [Install](#4-install)
5. [Run](#5-run)
6. [URLs](#6-urls)
7. [Swagger / interactive API docs](#7-swagger--interactive-api-docs)
8. [API reference with examples](#8-api-reference-with-examples)
9. [The dataset](#9-the-dataset)
10. [The model](#10-the-model)
11. [How to replace the model later](#11-how-to-replace-the-model-later)
12. [Why the frontend never needs to change](#12-why-the-frontend-never-needs-to-change)
13. [Troubleshooting](#13-troubleshooting)

---

## 1. Project structure

```
medical_cost_prediction/
│
├── backend/
│   ├── requirements.txt
│   └── app/
│       ├── __init__.py
│       ├── main.py                  # FastAPI app, CORS, error handling, startup
│       ├── config.py                # paths and settings ONLY
│       ├── routes/
│       │   └── prediction.py        # HTTP endpoints ONLY
│       ├── schemas/
│       │   └── prediction.py        # Pydantic validation ONLY
│       └── services/
│           ├── model_service.py     # ALL machine-learning logic
│           └── dependencies.py      # one shared trained ModelService
│
├── frontend/
│   ├── index.html                   # the page: hero, form, result, diagnostics
│   ├── style.css                    # looks
│   └── app.js                       # reads form, POSTs JSON, displays result
│
├── data/
│   └── insurance.csv                # the training data
│
├── api-contract.json                # the frozen API contract
└── README.md
```

### Who is allowed to know what

This is the most important idea in the project. Each file has one job:

| File | Allowed to know about | Must **never** know about |
|---|---|---|
| `config.py` | file paths, column names, settings | HTTP, scikit-learn |
| `routes/prediction.py` | HTTP, request/response shapes | pandas, scikit-learn, CSV columns |
| `schemas/prediction.py` | field names and valid values | HTTP, machine learning |
| `services/model_service.py` | pandas, scikit-learn, the CSV | HTTP, FastAPI |
| `frontend/app.js` | the JSON request and response shapes | pandas, scikit-learn, CSV columns, model internals |

If you ever catch yourself importing `sklearn` in a route, or writing the word
`charges` in a JavaScript file, the boundary has been broken.

---

## 2. Architecture

```
┌─────────────────────────────┐
│  Browser (frontend/)        │
│  index.html + style.css     │
│  app.js                     │
│                             │
│  Knows: the JSON shape      │
│  Never knows: the model     │
└──────────────┬──────────────┘
               │  HTTP POST /api/v1/predict
               │  (browser enforces CORS via backend's allow-list)
               ▼
┌─────────────────────────────┐
│  FastAPI (backend/)         │
│                             │
│  main.py                    │  middleware, error -> status codes
│    │                        │
│    ├─ routes/prediction.py  │  HTTP only
│    │     │                  │
│    │     └─ schemas/        │  Pydantic validation only
│    │                        │
│    └─ services/             │
│          ├─ dependencies   │  one shared instance (lru_cache)
│          └─ model_service  │  ALL machine learning
│                │           │
│                ├─ _encode() │  one-hot, drop_first, order enforced
│                └─ Pipeline  │  PolynomialFeatures -> StandardScaler
│                               -> LinearRegression
└──────────────┬──────────────┘
               │  reads at startup
               ▼
        data/insurance.csv
```

### The request path, step by step

1. The browser `POST`s JSON to `/api/v1/predict`.
2. **CORS middleware** checks the `Origin` header against the allow-list in
   `config.py`. A page served from `http://127.0.0.1:5500` is allowed.
3. **Pydantic** (`schemas/prediction.py`) validates the body. Anything out of
   range or misspelled is rejected here with HTTP 422, before the model is
   ever touched. Text values are trimmed and lower-cased first, so `" Male "`
   and `"male"` are the same thing.
4. **The route** (`routes/prediction.py`) calls `service.predict(payload)`.
   That is all it does. It contains no arithmetic and no model knowledge.
5. **The service** (`services/model_service.py`) turns the dict into a
   one-row DataFrame in the exact training column order, runs it through the
   same `_encode()` used during training, and calls `pipeline.predict()`.
6. The result is clamped to `>= 0`, rounded to 2 decimals, and returned.
7. **The frontend** formats the number as currency and shows it. It does not
   calculate anything.

### Startup: why the model is trained once

`main.py` uses FastAPI's `lifespan` handler. On startup it calls
`get_model_service()` once. That function is wrapped in `@lru_cache(maxsize=1)`,
so:

* the first call creates a `ModelService` and trains it,
* every later call returns the *same already-trained object*.

Training 1,337 rows takes well under a second, so it happens during startup and
every request afterwards is instant.

If training fails — most often because `data/insurance.csv` is missing — the
server **still starts**. That way `/health` keeps answering and `/predict` can
return a proper `503` instead of the whole API disappearing.

---

## 3. How the frontend talks to the backend

### The request

`frontend/app.js` sends a single `POST` with a JSON body:

```http
POST /api/v1/predict
Content-Type: application/json

{
  "age": 34,
  "sex": "male",
  "bmi": 27.8,
  "children": 2,
  "smoker": "no",
  "region": "northwest"
}
```

### The response

```json
{
  "success": true,
  "predicted_medical_charges_usd": 7601.87,
  "currency": "USD",
  "model": "Polynomial Regression",
  "model_version": "2.0",
  "input": {
    "age": 34,
    "sex": "male",
    "bmi": 27.8,
    "children": 2,
    "smoker": "no",
    "region": "northwest"
  }
}
```

The frontend reads `predicted_medical_charges_usd`, `currency`, `model`,
`model_version` and the `input` echo, then renders them. The `input` object is
the server's own copy of the request after normalisation, which is handy for
confirming what the server understood. The page also shows a per-month figure
(`annual / 12`) and the model metrics from `GET /api/v1/model-info` — both
pure restatements of numbers the API already returned.

The form can also be pre-filled from the URL, which is handy for sharing a
specific scenario:

```
http://127.0.0.1:5500/?age=45&sex=male&bmi=27.5&children=1&smoker=no&region=southwest
```

Values are clamped to the same ranges the API enforces, and unrecognised ones
are ignored so the form simply keeps its defaults.

### Why two different ports?

The frontend is a *static* file server (`python3 -m http.server 5500`) and the
API is a *dynamic* Python server (`uvicorn ... --port 8001`). They cannot be the
same process, so they are on different ports and the browser treats them as
different origins. `CORSMiddleware` in `main.py` allows exactly the two
frontend origins.

> **Note on ports:** this project normally uses **8001** for the API because
> port 8000 was already occupied by another local application. If you want to
> use 8000, change two things: run uvicorn with `--port 8000`, and update
> `API_URL` in `frontend/app.js`.

### How errors reach the page

Whatever goes wrong, the API answers with the same envelope:

```json
{
  "success": false,
  "error": { "code": "VALIDATION_ERROR", "message": "age: Input should be less than or equal to 100" }
}
```

Because every failure uses that one envelope, the frontend needs exactly one
error path: read `error.message` and show it in the inline banner above the
button. If the API cannot be reached at all, `fetch` rejects and the page
shows its own "Could not reach the service" message instead.

| HTTP | `code` | Meaning |
|---|---|---|
| 422 | `VALIDATION_ERROR` | Input failed validation |
| 500 | `PREDICTION_FAILED` | The model could not produce a number |
| 500 | `MODEL_ERROR` | The model is not in a usable state |
| 503 | `DATASET_NOT_FOUND` | `data/insurance.csv` is missing or unreadable |
| 404 | `NOT_FOUND` | Unknown URL |
| 500 | `INTERNAL_ERROR` | Anything unexpected (details go to the server log) |

---

## 4. Install

Requirements: Python 3.10 or newer, and a browser. Tested on Python 3.14 with
macOS.

```bash
cd medical_cost_prediction/backend

# Create an isolated virtual environment
python3 -m venv .venv

# Activate it
source .venv/bin/activate

# Install the dependencies
pip install -r requirements.txt
```

You should see the prompt change to `(.venv)`. To deactivate later, run
`deactivate`.

The versions in `requirements.txt` are pinned to the exact set tested on this
machine. If you use a different Python version and pip complains, delete the
`==` pins and run `pip install -r requirements.txt` again.

---

## 5. Run

You need **two terminals**.

### Terminal 1 — the backend

```bash
cd medical_cost_prediction/backend
source .venv/bin/activate
uvicorn app.main:app --host 127.0.0.1 --port 8001
```

You should see:

```
INFO:     Starting up: training Polynomial Regression v2.0 ...
INFO:     Model ready.
INFO:       rows used        : 1337
INFO:       duplicates removed: 1
INFO:       encoded features : 8
INFO:       metrics          : {'MAE': 2867.32, 'RMSE': 4646.06, 'R2': 0.8825}
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:8001
```

`Model ready.` means the model trained successfully.

### Terminal 2 — the frontend

```bash
cd medical_cost_prediction/frontend
python3 -m http.server 5500 --bind 127.0.0.1
```

### Open in Safari

```bash
open -a Safari http://127.0.0.1:5500
open -a Safari http://127.0.0.1:8001/docs
```

---

## 6. URLs

| What | URL |
|---|---|
| **App (frontend)** | <http://127.0.0.1:5500> |
| **Swagger docs** | <http://127.0.0.1:8001/docs> |
| Status | <http://127.0.0.1:8001/> |
| Health | <http://127.0.0.1:8001/api/v1/health> |
| Model info | <http://127.0.0.1:8001/api/v1/model-info> |
| Predict | `POST http://127.0.0.1:8001/api/v1/predict` |

### Stopping the servers

Both servers run in the foreground, so **`Ctrl + C`** in the relevant terminal
stops it. If you started them in the background instead:

```bash
# Stop the API
pkill -f "uvicorn app.main:app"

# Stop the static frontend server
pkill -f "http.server 5500"

# Stop the virtual environment (if you are inside one)
deactivate
```

To free port 8001 specifically:

```bash
lsof -ti:8001 | xargs kill
```

---

## 7. Swagger / interactive API docs

FastAPI generates interactive documentation automatically from the Pydantic
schemas:

* **Swagger UI** — <http://127.0.0.1:8001/docs>
* **ReDoc** — <http://127.0.0.1:8001/redoc>
* **Raw schema** — <http://127.0.0.1:8001/openapi.json>

You can test every endpoint from the browser: expand `POST /api/v1/predict`,
click **Try it out**, edit the JSON, and press **Execute**.

---

## 8. API reference with examples

### `GET /` — status

```bash
curl -s http://127.0.0.1:8001/
```

```json
{
  "success": true,
  "message": "Medical Cost Prediction API is running.",
  "model": "Polynomial Regression",
  "model_version": "2.0",
  "currency": "USD",
  "docs": "/docs",
  "endpoints": [
    "GET  /api/v1/health",
    "GET  /api/v1/model-info",
    "POST /api/v1/predict"
  ]
}
```

### `GET /api/v1/health` — liveness

```bash
curl -s http://127.0.0.1:8001/api/v1/health
```

```json
{ "success": true, "status": "healthy" }
```

This endpoint never touches the model, so it returns `200` even when the
dataset is missing.

### `GET /api/v1/model-info` — which model, and how good is it

```bash
curl -s http://127.0.0.1:8001/api/v1/model-info
```

```json
{
  "success": true,
  "model": "Polynomial Regression",
  "model_version": "2.0",
  "metrics": { "MAE": 2867.32, "RMSE": 4646.06, "R2": 0.8825 }
}
```

* **MAE** — Mean Absolute Error. On average the prediction is off by about
  \$2,867.
* **RMSE** — Root Mean Squared Error. Same idea, but large mistakes are
  punished more heavily, so it is always >= MAE.
* **R²** — how much of the variation in charges the model explains.
  `0.8825` means about 88% — a good result for this kind of data.

These numbers come from the held-out 20% of the data, so they describe
performance on people the model has never seen.

### `POST /api/v1/predict` — the prediction

```bash
curl -s -X POST http://127.0.0.1:8001/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{"age":34,"sex":"male","bmi":27.8,"children":2,"smoker":"no","region":"northwest"}'
```

```json
{
  "success": true,
  "predicted_medical_charges_usd": 7601.87,
  "currency": "USD",
  "model": "Polynomial Regression",
  "model_version": "2.0",
  "input": {
    "age": 34,
    "sex": "male",
    "bmi": 27.8,
    "children": 2,
    "smoker": "no",
    "region": "northwest"
  }
}
```

#### Input rules

| Field | Type | Rule | Example |
|---|---|---|---|
| `age` | integer | 18 – 100 | `34` |
| `sex` | string | `male` or `female` | `"male"` |
| `bmi` | number | 10 – 70 | `27.8` |
| `children` | integer | 0 – 10 | `2` |
| `smoker` | string | `yes` or `no` | `"no"` |
| `region` | string | `northeast`, `northwest`, `southeast`, `southwest` | `"northwest"` |

All six fields are required. Unknown extra fields are rejected.

Text values are **trimmed and lower-cased** before checking, so `"  MALE "`,
`"Male"` and `"male"` are all accepted and all produce the same prediction.
Numbers are given to the model exactly as sent.

#### Error examples

**422 — value out of range**

```bash
curl -s -X POST http://127.0.0.1:8001/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{"age":150,"sex":"male","bmi":27.8,"children":2,"smoker":"no","region":"northwest"}'
```

```json
{
  "success": false,
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "age: Input should be less than or equal to 100"
  }
}
```

**422 — value not in the allowed list**

```bash
curl -s -X POST http://127.0.0.1:8001/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{"age":34,"sex":"male","bmi":27.8,"children":2,"smoker":"no","region":"antarctica"}'
```

```json
{
  "success": false,
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "region: Value error, 'region' must be one of: northeast, northwest, southeast, southwest (received 'antarctica')"
  }
}
```

**503 — the dataset is missing**

Move or rename `data/insurance.csv` and restart the backend:

```json
{
  "success": false,
  "error": {
    "code": "DATASET_NOT_FOUND",
    "message": "Training dataset not found at .../data/insurance.csv. Place insurance.csv in the project's data/ folder and restart the server."
  }
}
```

---

## 9. The dataset

`data/insurance.csv` — 1,338 rows plus a header, one row per customer:

| Column | Type | Notes |
|---|---|---|
| `age` | integer | in years |
| `sex` | text | `male` / `female` |
| `bmi` | number | Body Mass Index |
| `children` | integer | number of dependents |
| `smoker` | text | `yes` / `no` |
| `region` | text | one of four US regions |
| `charges` | number | **the target**, in USD |

This is the well-known Kaggle *Medical Insurance Cost* dataset. The project
removes 1 duplicate row, leaving **1,337** rows for training.

The app never invents data. If the file is missing the API returns `503`
instead of training on made-up numbers.

---

## 10. The model

Built in `services/model_service.py`, in `_build_model()`:

```python
Pipeline(steps=[
    ("polynomial", PolynomialFeatures(degree=2, include_bias=False)),
    ("scaler",     StandardScaler()),
    ("regressor",  LinearRegression()),
])
```

What each step does:

1. **`PolynomialFeatures(degree=2)`** — adds interaction and square terms, so
   the model can learn curved relationships. For example, the effect of being
   a smoker can depend on age, and 8 input features become 44.
   `include_bias=False` avoids a column of 1s, because `LinearRegression`
   already fits an intercept.
2. **`StandardScaler()`** — puts every feature on a similar scale, otherwise
   the large numbers (age, charges) would dominate the fit.
3. **`LinearRegression()`** — ordinary least squares.

### Feature encoding

`sex`, `smoker` and `region` are text, so they are one-hot encoded with
`drop_first=True`: one category becomes the baseline and the rest become 0/1
columns. With 3 numeric features that gives 8 columns:

```
age, bmi, children, sex_male, smoker_yes,
region_northwest, region_southeast, region_southwest
```

### `_encode()` — the important function

The **same** `_encode()` function is used for training *and* prediction, in
`ModelService._encode()`. It:

1. **Enforces the exact training column order.** The input is reindexed to
   `FEATURE_COLUMNS`, so the caller cannot accidentally change the layout. If
   the columns already match, this is a no-op.
2. Trims and lower-cases the categorical values.
3. **Learns the category list from the training data only**, so nothing about
   the test set leaks into the model.
4. **Raises an error on an unknown category** instead of quietly producing an
   all-zero row of dummies. This matters: a silently all-zero row still yields
   a confident-looking number that is simply wrong. Failing loudly is safer.
5. Builds one-hot columns from exactly the learned categories, so the column
   layout is guaranteed identical at predict time.

### Why the split happens before encoding

`train_test_split(..., test_size=0.2, random_state=42)` runs **before**
`_encode()`. The categories are therefore learned from the training portion
only, and the test portion is transformed with rules it did not influence. The
fixed `random_state` means the metrics are identical on every run.

### Safety rails

* Predictions are **clamped to `>= 0`** — medical charges cannot be negative.
* Results are rounded to 2 decimal places.
* A non-finite result (`NaN`, `inf`) is treated as an error, not returned as a
  number.

---

## 11. How to replace the model later

Everything you need to change is in **one function**: `_build_model()` in
`backend/app/services/model_service.py`.

### Example: swap in a Random Forest

```python
def _build_model() -> Pipeline:
    from sklearn.ensemble import RandomForestRegressor

    return Pipeline(
        steps=[
            ("regressor", RandomForestRegressor(
                n_estimators=300,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )),
        ],
    )
```

A tree-based model does not need polynomial expansion or scaling, so those two
steps are simply left out.

Then, in `backend/app/config.py`:

```python
MODEL_NAME = "Random Forest"
MODEL_VERSION = "3.0"
```

### The three rules for `_build_model()`

1. It must return an object with `.fit(X, y)` and `.predict(X)`.
2. It must accept the output of `_encode()` as `X` — a DataFrame of floats.
3. Do not change the format `_encode()` produces.

If you do follow those, the change is invisible outside `model_service.py` and
`config.py`.

---

## 12. Why the frontend never needs to change

Consider what happens when the model changes. Four things stay **exactly** the
same:

| Thing | Why it does not change |
|---|---|
| The request body | It describes the *customer*, not the model. Age is still an age. |
| The response body | The keys are fixed by `schemas/prediction.py`. |
| The metrics | `train()` computes MAE/RMSE/R² regardless of the estimator. |
| The error codes | They come from the HTTP layer, not the model. |

The one thing that does change — the *name* — is deliberately a plain string in
the response:

```json
{ "model": "Polynomial Regression", "model_version": "2.0" }
```

The frontend does not compare that string to anything. It just prints it:

```js
document.getElementById("result-model").textContent = data.model;
```

So when you switch to a Random Forest, the page simply reads
`Random Forest (version 3.0)` and the predicted amount changes. **No frontend
edit, no API change, no redeploy of the contract.**

Three design choices make this work:

1. **The API describes the domain, not the algorithm.** Clients send customer
   attributes, not feature vectors. The encoding is a backend implementation
   detail.
2. **The model name is data, not code.** A string in a response can change
   value without changing shape. A hardcoded model name in the frontend could
   not.
3. **The frontend has one job.** Send a request, render a response. It never
   imports a library, reads a file, or computes a prediction — so there is no
   frontend logic that *could* depend on the model.

If you ever need a completely different algorithm (a neural network, an external
API, a lookup table), the same three rules still hold. Only `_build_model()`
changes.

---

## 13. Troubleshooting

**The page says "Could not reach the API"**
The backend is not running, or it is on a different port than `API_URL` in
`frontend/app.js`. Check the port on both sides.

**"Address already in use" when starting uvicorn**
Something else owns that port. Either stop it, or pick another port and update
`API_URL` in `frontend/app.js` (and `CORS_ORIGINS` in `config.py` if you also
change the frontend port).

```bash
lsof -nP -iTCP:8001 -sTCP:LISTEN
```

**`DATASET_NOT_FOUND` (503)**
`data/insurance.csv` is missing or unreadable. Confirm it exists relative to the
project root and restart the backend — the model is only trained at startup.

**The API starts but `metrics` is empty**
Training failed at startup. Read the first lines of the uvicorn output; the
real reason is logged there rather than swallowed.

**`ModuleNotFoundError: No module named 'app'`**
Run uvicorn from inside `backend/`, and make sure the virtual environment is
active (`source .venv/bin/activate`).

**CORS error in the browser console**
The page origin is not in `CORS_ORIGINS` in `config.py`. Open the page via
`http://127.0.0.1:5500` rather than opening `index.html` as a `file://` URL —
browsers treat those differently and `file://` is not allowed.