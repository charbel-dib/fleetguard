"""FastAPI factory with explicit startup loading of one immutable release."""

import logging
from contextlib import asynccontextmanager

import anyio
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException
from starlette.responses import JSONResponse

from fleetguard import __version__
from fleetguard.serving.middleware import RequestEnvelope, error_response
from fleetguard.serving.predictor import FLOAT32_MAX, InputError, Predictor
from fleetguard.serving.schemas import (
    BatchResponse,
    ErrorResponse,
    ModelIdentity,
    ModelInfo,
    PredictRequest,
    SingleResponse,
    batch_request_type,
)
from fleetguard.serving.settings import Settings

logger = logging.getLogger("fleetguard.api")


def create_app(settings=None):
    settings = settings or Settings.from_env()

    @asynccontextmanager
    async def lifespan(app):
        # Errors here fail startup; no replacement model, retry, training or hot reload.
        predictor = Predictor(settings)
        ModelIdentity.model_validate(predictor.identity)
        app.state.predictor = predictor
        app.state.inference_limiter = anyio.CapacityLimiter(1)
        logger.info("Ready with frozen release %s", app.state.predictor.identity["release_id"])
        try:
            yield
        finally:
            app.state.predictor = None
            app.state.inference_limiter = None

    app = FastAPI(
        title="FleetGuard API",
        version=__version__,
        lifespan=lifespan,
        telemetry={
            "auto_configure": False,
            "tracing": False,
            "metrics": False,
            "logs": False,
            "operation_spans": False,
        },
        description=(
            "APS diagnostic scores from one frozen release. "
            "Model selection and training are offline."
        ),
    )
    app.state.predictor = None
    app.state.inference_limiter = None
    app.add_middleware(RequestEnvelope, max_bytes=settings.max_request_bytes)
    batch_type = batch_request_type(settings.max_batch_rows)
    prediction_errors = {code: {"model": ErrorResponse} for code in (400, 413, 415, 422, 500, 503)}

    def ready():
        if app.state.predictor is None:
            raise HTTPException(503, "Model is not ready.")
        return app.state.predictor

    @app.exception_handler(RequestValidationError)
    async def invalid_input(request, exc):
        # Never echo the input body/values in validation errors or log them.
        issues = [
            {"loc": list(e["loc"]), "type": e["type"], "message": e["msg"]}
            for e in exc.errors()[:20]
        ]
        return error_response(
            "invalid_request",
            "Request does not satisfy the input contract.",
            status=422,
            request_id=request.state.request_id,
            details=issues,
        )

    @app.exception_handler(InputError)
    async def schema_error(request, exc):
        return error_response(
            "invalid_sensor_schema",
            str(exc),
            status=422,
            request_id=request.state.request_id,
            details=exc.details,
        )

    @app.exception_handler(HTTPException)
    async def http_error(request, exc):
        return error_response(
            "not_ready" if exc.status_code == 503 else "http_error",
            str(exc.detail),
            status=exc.status_code,
            request_id=request.state.request_id,
        )

    @app.get("/health/live", tags=["health"])
    async def live():
        return {"status": "alive", "service_version": __version__}

    @app.get("/health/ready", tags=["health"])
    async def readiness():
        if app.state.predictor is None:
            return JSONResponse({"status": "not_ready"}, status_code=503)
        return {"status": "ready", "release_id": app.state.predictor.identity["release_id"]}

    @app.get(
        "/v1/model",
        response_model=ModelInfo,
        responses={503: {"model": ErrorResponse}},
        tags=["model"],
    )
    async def model_info():
        predictor = ready()
        return {
            "service_version": __version__,
            "contract_version": 1,
            "model": predictor.identity,
            "feature_names": predictor.feature_names,
            "costs": predictor.costs,
            "limits": {
                "max_batch_rows": settings.max_batch_rows,
                "max_request_bytes": settings.max_request_bytes,
                "feature_abs_max": FLOAT32_MAX,
                "missing_value": "null",
                "all_sensor_keys_required": True,
            },
            "label_meanings": {
                "pos": "APS-associated component failure",
                "neg": "failure in components outside APS",
            },
        }

    @app.post(
        "/v1/predict",
        response_model=SingleResponse,
        responses=prediction_errors,
        tags=["predictions"],
    )
    async def predict(payload: PredictRequest, request: Request):
        predictor = ready()
        values = await anyio.to_thread.run_sync(
            predictor.predict,
            [payload.sensors],
            limiter=app.state.inference_limiter,
        )
        return {"model": predictor.identity, "prediction": values[0]}

    @app.post(
        "/v1/predict-batch",
        response_model=BatchResponse,
        responses=prediction_errors,
        tags=["predictions"],
    )
    async def predict_batch(payload: batch_type, request: Request):
        predictor = ready()
        values = await anyio.to_thread.run_sync(
            predictor.predict,
            payload.rows,
            limiter=app.state.inference_limiter,
        )
        return {"model": predictor.identity, "rows": len(values), "predictions": values}

    return app
