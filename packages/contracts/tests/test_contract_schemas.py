import json
from pathlib import Path

from medical_imaging_contracts.models import (
    DatasetManifestRecord, DatasetReleaseReference, DeploymentPackage, ExperimentRecord,
    InferenceRequest, ModelArtifact, ModelCandidateSubmission, ModelRegistryRecord, PlatformEvent,
    PredictionResult,
)


def test_checked_in_schemas_match_wire_models():
    root = Path(__file__).resolve().parents[1] / "schemas" / "v1"
    models = {
        "dataset_release.schema.json": DatasetReleaseReference,
        "dataset_manifest.schema.json": DatasetManifestRecord,
        "experiment.schema.json": ExperimentRecord,
        "model_artifact.schema.json": ModelArtifact,
        "model_candidate_submission.schema.json": ModelCandidateSubmission,
        "model_registry.schema.json": ModelRegistryRecord,
        "deployment_package.schema.json": DeploymentPackage,
        "inference_request.schema.json": InferenceRequest,
        "prediction_result.schema.json": PredictionResult,
        "platform_event.schema.json": PlatformEvent,
    }
    for filename, model in models.items():
        schema = json.loads((root / filename).read_text())
        assert schema["$schema"].endswith("2020-12/schema")
        assert schema["properties"]["schema_version"]["const"] == "1.0"
