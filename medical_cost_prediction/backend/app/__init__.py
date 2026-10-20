"""
Medical Cost Prediction API.

A small FastAPI application that serves medical insurance cost predictions
produced by a scikit-learn model.

Layering (each layer only talks to the one below it):

    routes/     -> HTTP only (parse request, call service, shape response)
    schemas/    -> validation only (Pydantic request/response models)
    services/   -> all machine-learning logic lives here
    config.py   -> paths and settings only
"""

__version__ = "2.0"