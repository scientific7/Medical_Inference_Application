import numpy as np
import pytest

from medical_imaging_inference.postprocessing import interpret


def test_tensorflow_segmentation_thresholds_scores_and_resizes_to_original():
    raw = np.array([[[[0.2], [0.8]]]])
    result = interpret("segmentation", raw, {"background": 0, "lesion": 1}, (4, 2), framework="tensorflow")
    assert result["mask_shape"] == [2, 4]
    assert result["mask_runs"] == [[0, 2], [1, 2], [0, 2], [1, 2]]
    assert result["class_labels"] == {"0": "background", "1": "lesion"}


def test_multiclass_runs_encode_class_ids_when_a_class_is_absent():
    raw = np.array([[[[0, 5]], [[0, 0]], [[5, 0]]]])
    result = interpret("segmentation", raw, {"background": 0, "healthy": 1, "lesion": 2})
    assert result["mask_values"] == [0, 2]
    assert result["mask_runs"] == [[2, 1], [0, 1]]


def test_classification_rejects_nonfinite_output():
    with pytest.raises(ValueError):
        interpret("classification", np.array([[0, np.nan]]), {"background": 0, "lesion": 1})


def test_multilabel_uses_independent_scores_and_threshold():
    output = interpret(
        "classification",
        np.array([[3.0, -2.0, 2.0]]),
        {"a": 0, "b": 1, "c": 2},
        settings={"classification_mode": "multilabel", "threshold": 0.8, "top_k": 2},
    )
    assert output["predicted_classes"] == ["a", "c"]
    assert len(output["top_k"]) == 2
    assert output["score_type"] == "uncalibrated_sigmoid"


def test_binary_segmentation_uses_package_threshold():
    raw = np.array([[[[0.1, 2.0]]]])
    result = interpret("segmentation", raw, {}, settings={"threshold": 0.75})
    assert result["mask_runs"] == [[0, 1], [1, 1]]


def test_detection_decodes_normalized_coordinates_and_suppresses_by_class():
    raw = {
        "boxes": np.array([[0.1, 0.2, 0.8, 0.9]] * 4),
        "labels": np.array([0, 0, 1, 2]),
        "scores": np.array([0.9, 0.8, 0.7, 0.2]),
    }
    result = interpret(
        "detection",
        raw,
        {"a": 0, "b": 1},
        (100, 50),
        settings={"detection_format": "yxyx_normalized", "threshold": 0.5, "iou_threshold": 0.4},
    )
    assert result["count"] == 2
    assert result["labels"] == ["a", "b"]
    np.testing.assert_allclose(result["boxes_xyxy"], [[20, 5, 90, 40]] * 2)
