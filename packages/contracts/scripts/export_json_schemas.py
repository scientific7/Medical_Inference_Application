"""Export checked-in JSON Schema files from the executable Pydantic wire models."""
import json
from pathlib import Path

from medical_imaging_contracts import models

SCHEMAS = {
    "dataset_release.schema.json": models.DatasetReleaseReference,
    "dataset_manifest.schema.json": models.DatasetManifestRecord,
    "experiment.schema.json": models.ExperimentRecord,
    "model_artifact.schema.json": models.ModelArtifact,
    "model_candidate_submission.schema.json": models.ModelCandidateSubmission,
    "model_registry.schema.json": models.ModelRegistryRecord,
    "deployment_package.schema.json": models.DeploymentPackage,
    "inference_request.schema.json": models.InferenceRequest,
    "prediction_result.schema.json": models.PredictionResult,
    "platform_event.schema.json": models.PlatformEvent,
}

destination = Path(__file__).resolve().parents[1] / "schemas" / "v1"
package_destination = Path(__file__).resolve().parents[1] / "src" / "medical_imaging_contracts" / "schemas" / "v1"
for output_dir in (destination, package_destination):
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, model in SCHEMAS.items():
        schema = model.model_json_schema()
        schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
        schema["$id"] = f"https://github.com/scientific7/medical-imaging-platform-contracts/blob/main/schemas/v1/{name}"
        (output_dir / name).write_text(json.dumps(schema, indent=2, sort_keys=True) + "\n", encoding="utf-8")
