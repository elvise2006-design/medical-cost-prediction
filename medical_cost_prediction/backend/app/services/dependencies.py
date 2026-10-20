"""
services/dependencies.py
========================

Provides ONE shared, already-trained ModelService to every request.

Why lru_cache?
--------------
Training takes a moment, and the model does not change while the server is
running. `@lru_cache(maxsize=1)` means the service is built and trained on the
first call and every later call returns the very same object.

`main.py` calls this once at startup (in the lifespan handler), which "warms"
the cache. By the time the first real request arrives, the model is ready.
"""

from functools import lru_cache

from app.services.model_service import ModelService


@lru_cache(maxsize=1)
def get_model_service() -> ModelService:
    """
    Return the shared ModelService, training it on first use.

    This is the FastAPI dependency injected into the prediction routes:

        @router.post("/predict")
        def predict(service: ModelService = Depends(get_model_service)):
            ...

    If training fails (for example the CSV is missing) the exception
    propagates and `main.py` converts it into the right HTTP status code.
    """
    service = ModelService()
    service.train()          # <- the single training call for the whole process
    return service