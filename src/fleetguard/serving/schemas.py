"""Versioned JSON contract; missing sensors are explicit null values."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, create_model

from fleetguard.serving.predictor import FLOAT32_MAX

SensorValue = Annotated[
    float | None, Field(strict=True, allow_inf_nan=False, ge=-FLOAT32_MAX, le=FLOAT32_MAX)
]
SensorRow = dict[str, SensorValue]


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    sensors: SensorRow


def batch_request_type(max_rows):
    return create_model(
        "BatchRequest",
        __config__=ConfigDict(extra="forbid"),
        rows=(list[SensorRow], Field(min_length=1, max_length=max_rows)),
    )


class ModelIdentity(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    release_id: str
    name: str
    dataset_kind: Literal["official_aps", "synthetic_smoke"]
    pipeline_sha256: str
    freeze_sha256: str
    threshold: float
    calibration: Literal["raw", "sigmoid"]
    policy: Literal["challenge_cost"]
    score_semantics: str


class Prediction(BaseModel):
    row_index: int = Field(ge=0)
    positive_score: float = Field(ge=0, le=1, allow_inf_nan=False)
    predicted_label: Literal["pos", "neg"]
    missing_fraction: float = Field(ge=0, le=1)


class SingleResponse(BaseModel):
    model: ModelIdentity
    prediction: Prediction


class BatchResponse(BaseModel):
    model: ModelIdentity
    rows: int
    predictions: list[Prediction]


class ErrorInfo(BaseModel):
    code: str
    message: str
    details: list[dict]
    request_id: str


class ErrorResponse(BaseModel):
    error: ErrorInfo


class InputLimits(BaseModel):
    max_batch_rows: int
    max_request_bytes: int
    feature_abs_max: float
    missing_value: Literal["null"]
    all_sensor_keys_required: Literal[True]


class ModelInfo(BaseModel):
    service_version: str
    contract_version: Literal[1]
    model: ModelIdentity
    feature_names: list[str]
    costs: dict[str, int]
    limits: InputLimits
    label_meanings: dict[str, str]
