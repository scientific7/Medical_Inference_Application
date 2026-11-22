"""Actual publisher -> reader -> training -> approval -> inference, synthetic only."""
import hashlib
import io
import json
import os
from pathlib import Path

import pytest

if os.getenv("PLATFORM_SYNTHETIC_TEST") != "1":
    pytest.skip("Run scripts/test_platform.py with explicit sibling repositories", allow_module_level=True)

import httpx
from PIL import Image
from fastapi.testclient import TestClient
from medical_imaging_contracts.models import DatasetReleaseReference, ModelCandidateSubmission, PredictionResult
from medical_triage.data_pipeline.portable_release import publish
from medical_imaging_models.data.portable_v2 import open_verified_release

KEY = "synthetic-only-not-a-deployment-key" * 2


def reference(path):
    return {"uri": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def document(path, value):
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    return reference(path)


def published(root, task="classification"):
    root.mkdir(parents=True)
    splits = {}
    for split_index, split in enumerate(("train", "validation", "test")):
        rows = []
        for index in range(2):
            image = root / f"{split}-{index}.png"
            Image.new("RGB", (8, 8), (30 + split_index, 50 + 150 * index, 90)).save(image)
            row = {"sample_id": f"{split}-{index}", "split": split, "task": task,
                   "image_asset_uri": str(image), "image_sha256": reference(image)["sha256"],
                   "patient_group_id": f"synthetic:{split}:{index}"}
            if task == "classification":
                row["label"] = ("benign", "malignant")[index]
            elif task == "segmentation":
                mask = root / f"{split}-{index}-mask.png"
                Image.new("L", (8, 8), 255).save(mask)
                row.update(annotation_uri=str(mask), annotation_sha256=reference(mask)["sha256"])
            else:
                row.update(width=8, height=8, boxes=[{"xyxy": [1, 1, 6, 6], "class_id": 1}] if index else [],
                           detection_status="positive" if index else "verified_negative")
                if not index:
                    row["negative_evidence"] = document(root / f"{split}-negative.json", {
                        "sample_id": row["sample_id"], "status": "verified_negative", "reviewer": "synthetic-fixture"})
            rows.append(row)
        splits[split] = document(root / f"{split}.json", rows)
    spec = {"schema_version": "2.0", "dataset_id": "synthetic", "release_id": f"synthetic-{task}-v1",
            "task": task, "preprocessing_version": "1.0", "annotation_version": "1.0",
            "class_mapping": {"benign": 0, "malignant": 1} if task == "classification" else {"lesion": 1},
            "mask_encoding": "binary-0-255", "box_format": "xyxy-pixels-half-open", "partitions": splits}
    spec["audit_evidence"] = {name: document(root / f"audit-{name}.json", {
        "status": "passed", "release_id": spec["release_id"], "synthetic": True,
        "coverage_complete": True, "partition_digests": {k: v["sha256"] for k, v in splits.items()}})
        for name in ("integrity", "duplicates", "leakage", "license", "annotations")}
    return publish(spec, str(root / "receipt.json"), signing_key=KEY), spec


@pytest.mark.parametrize("task", ["classification", "segmentation", "detection"])
def test_publisher_reader_task_contracts(tmp_path, task):
    ref, spec = published(tmp_path / task, task)
    release = open_verified_release(ref["release_receipt_uri"], ref["release_receipt_digest"],
                                    signing_key=KEY, cache_dir=tmp_path / "cache")
    assert release.report.verified
    assert len(release.records["train"]) == 2
    if task == "detection":
        from medical_imaging_models.frameworks.pytorch import torch_detection_dataset
        _, target, _ = torch_detection_dataset(release, "train")[0]
        assert tuple(target["boxes"].shape) == (0, 4)
    Path(spec["partitions"]["train"]["uri"]).write_bytes(b"[]")
    with pytest.raises(ValueError, match="checksum"):
        open_verified_release(ref["release_receipt_uri"], ref["release_receipt_digest"], signing_key=KEY)


@pytest.mark.parametrize("framework", ["pytorch", "tensorflow"])
def test_train_approve_predict(tmp_path, monkeypatch, framework):
    from medical_imaging_models.core import TrainingConfig
    from medical_imaging_models.training.engine import train
    from medical_imaging_models.packaging import write_model_package, export_torchscript, export_keras_savedmodel
    from imaging_mlops.app import app as governance
    from medical_imaging_inference import app as serving
    from medical_imaging_inference.security import authorize
    from medical_imaging_inference.runtime import ModelCache

    ref, spec = published(tmp_path / "dataset")
    release = open_verified_release(ref["release_receipt_uri"], ref["release_receipt_digest"],
                                    signing_key=KEY, cache_dir=tmp_path / "cache")
    lineage = DatasetReleaseReference.from_verified_portable_v2(
        release.contract, ref["release_receipt_uri"], ref["release_receipt_digest"])
    config = TrainingConfig(framework=framework, architecture="synthetic-linear", image_size=(8, 8),
                            batch_size=2, epochs=1, patience=1, device="cpu", output_dir=str(tmp_path / "training"))
    if framework == "pytorch":
        import torch
        from medical_imaging_models.frameworks.pytorch import torch_dataloader
        torch.manual_seed(17)
        model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(192, 2))
        loaders = [torch_dataloader(release, split, batch_size=2, image_size=(8, 8)) for split in ("train", "validation")]
    else:
        import tensorflow as tf
        from medical_imaging_models.frameworks.tensorflow import tensorflow_dataset
        tf.keras.utils.set_random_seed(17)
        model = tf.keras.Sequential([tf.keras.layers.Input((8, 8, 3)), tf.keras.layers.Flatten(), tf.keras.layers.Dense(2)])
        loaders = [tensorflow_dataset(release, split, batch_size=2, image_size=(8, 8)) for split in ("train", "validation")]
    result = train(model, *loaders, config)
    metadata = {"experiment_id": "synthetic-exp", "dataset_release_id": lineage.release_id,
                "dataset_receipt_digest": lineage.receipt_sha256, "manifest_digests": lineage.manifest_sha256,
                "dataset_receipt_uri": lineage.receipt_uri, "git_commit": "synthetic-test-code",
                "framework": framework, "architecture": config.architecture, "task": "classification",
                "preprocessing": {"image_size": [8, 8], "color": "RGB", "scale": "[0,1]"},
                "class_mapping": release.label_mapping, "config_digest": config.digest(),
                "evaluation_summary": {"validation_loss": result["history"][-1][
                    "validation_loss" if framework == "pytorch" else "val_loss"]}}
    artifacts = tmp_path / "artifacts"
    package = artifacts / "candidate"
    write_model_package(package, metadata, result["best_checkpoint"])
    if framework == "pytorch":
        model.load_state_dict(torch.load(result["best_checkpoint"], weights_only=False, map_location="cpu")["model"])
        export_torchscript(model, torch.zeros(1, 3, 8, 8), package / "inference_model.ts")
    else:
        model = tf.keras.models.load_model(result["best_checkpoint"])
        export_keras_savedmodel(model, tf.zeros((1, 8, 8, 3)), package / "serving_model")
    monkeypatch.setenv("MLOPS_AUTH_MODE", "test")
    monkeypatch.setenv("ENVIRONMENT", "development")
    monkeypatch.setenv("MLOPS_ARTIFACT_ROOT", str(artifacts))
    monkeypatch.setenv("MLOPS_DATABASE_PATH", str(tmp_path / "registry.sqlite3"))
    monkeypatch.setenv("INFERENCE_ARTIFACT_ROOT", str(artifacts))
    monkeypatch.setenv("INFERENCE_DATABASE_PATH", str(tmp_path / "predictions.sqlite3"))
    monkeypatch.setenv("INFERENCE_ENVIRONMENT", "development")
    monkeypatch.delenv("INFERENCE_DATABASE_URL", raising=False)
    monkeypatch.setenv("MLOPS_API_URL", "http://governance")
    monkeypatch.setenv("MLOPS_SERVICE_TOKEN", "synthetic-in-process-token")
    real_async = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: real_async(transport=httpx.ASGITransport(app=governance), **kw))
    monkeypatch.setitem(serving.app.dependency_overrides, authorize,
                        lambda: {"subject": "synthetic-user", "roles": ["inference_user"]})
    monkeypatch.setattr(serving, "models", ModelCache())
    submission = ModelCandidateSubmission(model_id="synthetic-model", version="1", artifact_path="candidate",
                                          experiment_id="synthetic-exp", dataset_release_id=lineage.release_id)
    with TestClient(governance) as registry, TestClient(serving.app) as inference:
        headers = {"Idempotency-Key": "synthetic-candidate-1", "X-Test-Subject": "trainer"}
        registered = registry.post("/api/v1/models/register", json=submission.model_dump(), headers=headers)
        assert registered.status_code == 201, registered.text
        assert registry.post("/api/v1/models/register", json=submission.model_dump(), headers=headers).json() == registered.json()
        sample = release.records["test"][0].image_path.read_bytes()
        route = "/api/v1/predictions/synthetic-model/1"
        files = {"image": ("sample.png", sample, "image/png")}
        assert inference.post(route, files=files).status_code == 404
        review_route = "/api/v1/models/synthetic-model/versions/1"
        assert registry.post(review_route + "/request-review", json={"reason": "Synthetic integration test"}, headers=headers).status_code == 200
        assert registry.post(review_route + "/approve", headers=headers).status_code == 403
        approved = registry.post(review_route + "/approve", headers={"X-Test-Subject": "independent-synthetic-reviewer"})
        assert approved.status_code == 200, approved.text
        response = inference.post(route, files=files)
        assert response.status_code == 200, response.text
        prediction = response.json()
        assert prediction["artifact_sha256"] == approved.json()["artifact_digest"]
        assert approved.json()["metadata"]["dataset_receipt_digest"] == ref["release_receipt_digest"]
        history = inference.get("/api/v1/predictions/" + prediction["prediction_id"])
        assert history.status_code == 200, history.text
        assert history.json()["artifact_sha256"] == prediction["artifact_sha256"]
        assert history.json()["lineage"]["dataset_receipt_digest"] == lineage.receipt_sha256
        assert history.json()["lineage"]["manifest_digests"] == lineage.manifest_sha256
        # Cached models must not bypass a fresh approval lookup.
        import imaging_mlops.registry as registry_module
        with registry_module._connect() as db:
            db.execute("UPDATE model_versions SET state='archived'")
        assert inference.post(route, files=files).status_code == 404
