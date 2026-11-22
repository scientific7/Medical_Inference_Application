"""Pydantic wire models; every cross-repository payload is explicitly versioned."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ContractModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    schema_version: Literal["1.0"] = "1.0"


class DatasetReleaseReference(ContractModel):
    dataset_id: str = Field(min_length=1)
    release_id: str = Field(min_length=1)
    receipt_uri: str = Field(min_length=1)
    receipt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    manifest_sha256: dict[str, str]
    status: Literal["published"]

    @field_validator("manifest_sha256")
    @classmethod
    def validate_partitions(cls, value):
        import re
        if set(value) != {"train", "validation", "test"} or any(
            not re.fullmatch(r"[0-9a-f]{64}", digest) for digest in value.values()
        ):
            raise ValueError("Exactly three canonical split digests are required")
        return value

    @classmethod
    def from_verified_portable_v2(cls, receipt: dict, uri: str, digest: str):
        """Map an already authenticated producer v2 receipt; does not authenticate it."""
        if receipt.get("schema_version") != "2.0":
            raise ValueError("Expected Dataset Pipeline portable receipt v2")
        return cls(dataset_id=receipt["dataset_id"], release_id=receipt["release_id"],
                   receipt_uri=uri, receipt_sha256=digest, status=receipt["release_status"],
                   manifest_sha256={key: ref["sha256"] for key, ref in receipt["partitions"].items()})


class DatasetManifestRecord(ContractModel):
    sample_id: str = Field(min_length=1)
    split: Literal["train", "validation", "test"]
    task: Literal["classification", "segmentation", "detection"]
    image_uri: str = Field(min_length=1)
    image_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    label: str | None = None
    patient_group_id: str | None = None
    lesion_group_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ExperimentRecord(ContractModel):
    experiment_id: str = Field(min_length=1)
    dataset: DatasetReleaseReference
    architecture: str
    task: Literal["classification", "segmentation", "detection"]
    framework: Literal["pytorch", "tensorflow"]
    framework_version: str
    git_commit: str
    config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    metrics: dict[str, float] = Field(default_factory=dict)
    created_at: datetime


class ModelArtifact(ContractModel):
    artifact_id: str = Field(min_length=1)
    uri: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    manifest_uri: str = Field(min_length=1)
    manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    format: Literal["torch_state_dict", "torchscript", "tensorflow_savedmodel", "keras"]


class ModelCandidateSubmission(ContractModel):
    model_id: str = Field(min_length=1)
    version: str = Field(min_length=1)
    artifact_path: str = Field(min_length=1, description="Relative path under the configured private development artifact root")
    experiment_id: str = Field(min_length=1)
    dataset_release_id: str = Field(min_length=1)


class ModelRegistration(ContractModel):
    model_id: str = Field(min_length=1)
    version: str = Field(min_length=1)
    experiment: ExperimentRecord
    artifact: ModelArtifact
    model_card: str = Field(min_length=1)
    preprocessing: dict[str, Any]
    class_mapping: dict[str, int] = Field(default_factory=dict)


class ModelRegistryRecord(ContractModel):
    model_id: str = Field(min_length=1)
    version: str = Field(min_length=1)
    state: Literal["candidate_submitted", "technical_review", "approved", "staging",
                   "production", "deprecated", "withdrawn"]
    artifact: ModelArtifact
    experiment_id: str = Field(min_length=1)
    dataset_release_id: str = Field(min_length=1)
    submitted_at: datetime
    approved_at: datetime | None = None


class DeploymentPackage(ContractModel):
    model_id: str = Field(min_length=1)
    version: str = Field(min_length=1)
    artifact: ModelArtifact
    preprocessing: dict[str, Any]
    class_mapping: dict[str, int] = Field(default_factory=dict)
    approved: bool
    approval_reference: str | None = None


class InferenceRequest(ContractModel):
    request_id: str = Field(min_length=1)
    model_id: str = Field(min_length=1)
    model_version: str | None = None
    image_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    content_type: Literal["image/jpeg", "image/png", "image/tiff", "image/webp"] = "image/jpeg"


class PredictionResult(ContractModel):
    prediction_id: str
    model_id: str
    model_version: str
    artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    task: Literal["classification", "segmentation", "detection"]
    output: dict[str, Any]
    latency_ms: float = Field(ge=0)
    created_at: datetime
    status: Literal["completed", "failed"] = "completed"
    input_dimensions: dict[str, int] = Field(default_factory=dict)
    disclaimer: str = "Model-generated research prediction; not a diagnosis."
    lineage: dict[str, Any] = Field(default_factory=dict)


class PlatformEvent(ContractModel):
    event_id: str = Field(min_length=1)
    event_type: Literal["DatasetReleasePublished", "ExperimentStarted", "ExperimentCompleted",
                         "ModelCandidateSubmitted", "ModelRegistered", "ModelApproved",
                         "ModelPromoted", "PredictionCompleted"]
    occurred_at: datetime
    source_repository: str = Field(min_length=1)
    resource_id: str = Field(min_length=1)
    correlation_id: str = Field(min_length=1)
    payload: dict[str, Any] = Field(default_factory=dict)

