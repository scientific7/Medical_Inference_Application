import pytest

from medical_imaging_inference.security import verified_package


def test_package_resolution_rejects_path_escape(tmp_path, monkeypatch):
    monkeypatch.setenv("INFERENCE_ARTIFACT_ROOT", str(tmp_path / "approved"))
    with pytest.raises(ValueError, match="private package mount"):
        verified_package({"artifact_path": str(tmp_path / "outside")})


def test_app_has_health_and_prediction_contract():
    from medical_imaging_inference.app import app

    routes = {route.path for route in app.routes}
    assert "/health" in routes
    assert "/api/v1/predictions/{model_id}/{version}" in routes
    assert "/api/v1/predictions/{prediction_id}" in routes


def test_prediction_output_adapters_return_structured_values():
    import numpy as np

    from medical_imaging_inference.app import _interpret

    classification = _interpret("classification", np.array([[1.0, 3.0]]), {"benign": 0, "malignant": 1})
    assert classification["predicted_class"] == "malignant"
    segmentation = _interpret("segmentation", np.array([[[[0.0, 1.0], [0.0, 1.0]]]]), {})
    assert segmentation["mask_shape"] == [2, 2]
    detections = _interpret(
        "detection",
        {"boxes": np.array([[1, 2, 3, 4]]), "labels": np.array([1]), "scores": np.array([0.8])},
        {},
        (100, 50),
        (50, 25),
    )
    assert detections["boxes_xyxy"] == [[2.0, 4.0, 6.0, 8.0]]
