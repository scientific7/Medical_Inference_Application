"""Decode supported export outputs into original-image coordinates."""

from collections.abc import Mapping
from typing import Literal

import numpy as np
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field


class OutputSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    version: Literal["1"] = "1"
    classification_mode: Literal["multiclass", "multilabel", "binary"] = "multiclass"
    threshold: float | None = Field(default=None, ge=0, le=1)
    top_k: int = Field(default=5, ge=1, le=100)
    iou_threshold: float | None = Field(default=None, gt=0, le=1)
    detection_format: Literal["xyxy_pixels", "xyxy_normalized", "yxyx_normalized"] = "xyxy_pixels"


def sigmoid(values):
    return np.exp(-np.logaddexp(0.0, -values))


def array(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def interpret(
    task, output, mapping, original_size=None, model_size=None, *, framework="pytorch", settings=None
):
    settings = OutputSettings.model_validate(settings or {})
    if task == "classification":
        return classification(output, mapping, settings)
    if task == "segmentation":
        return segmentation(output, mapping, original_size, framework, settings)
    if task == "detection":
        return detection(output, mapping, original_size, model_size, settings)
    raise ValueError("Unsupported model task")


def classification(output, mapping, settings):
    logits = array(output).astype(np.float64)
    labels = sorted(mapping, key=mapping.get)
    binary = settings.classification_mode == "binary"
    if (
        not labels
        or (binary and len(labels) != 2)
        or logits.shape != (1, 1 if binary else len(labels))
        or not np.isfinite(logits).all()
        or [mapping[label] for label in labels] != list(range(len(labels)))
    ):
        raise ValueError("Classification output does not match the class mapping")
    multilabel = settings.classification_mode == "multilabel"
    if binary:
        positive = float(sigmoid(logits[0, 0]))
        scores = np.array([1 - positive, positive])
    elif multilabel:
        scores = sigmoid(logits[0])
    else:
        weights = np.exp(logits[0] - logits.max())
        scores = weights / weights.sum()
    ranked = np.argsort(-scores, kind="stable")
    threshold = settings.threshold
    if multilabel:
        selected = [index for index in ranked if scores[index] >= (0.5 if threshold is None else threshold)]
    elif binary:
        selected = [int(scores[1] >= (0.5 if threshold is None else threshold))]
    else:
        best = int(ranked[0])
        selected = [best] if threshold is None or scores[best] >= threshold else []
    return {
        "predicted_class": labels[selected[0]] if selected else None,
        "predicted_classes": [labels[index] for index in selected],
        "scores": {label: float(score) for label, score in zip(labels, scores)},
        "top_k": [
            {"label": labels[index], "score": float(scores[index])} for index in ranked[: settings.top_k]
        ],
        "score_type": "uncalibrated_sigmoid" if binary or multilabel else "uncalibrated_softmax",
    }


def segmentation(output, mapping, original_size, framework, settings):
    values = array(output)
    if values.ndim != 4 or values.shape[0] != 1 or not values.size or not np.isfinite(values).all():
        raise ValueError("Segmentation export must return one finite rank-four tensor")
    channels = values[0] if framework == "pytorch" else np.moveaxis(values[0], -1, 0)
    binary = channels.shape[0] == 1
    if binary:
        if framework == "tensorflow" and ((channels < 0).any() or (channels > 1).any()):
            raise ValueError("TensorFlow binary segmentation export must return sigmoid scores")
        scores = sigmoid(channels[0]) if framework == "pytorch" else channels[0]
        threshold = 0.5 if settings.threshold is None else settings.threshold
        mask = (scores >= threshold).astype(np.int32)
    else:
        mask = channels.argmax(axis=0).astype(np.int32)
    if original_size and original_size != (mask.shape[1], mask.shape[0]):
        mask = np.asarray(Image.fromarray(mask).resize(original_size, Image.Resampling.NEAREST))

    flat = mask.ravel()
    starts = np.r_[0, np.flatnonzero(flat[1:] != flat[:-1]) + 1]
    lengths = np.diff(np.r_[starts, flat.size])
    classes = np.unique(flat).tolist()
    labels = {value: label for label, value in mapping.items()}
    result = {
        "mask_shape": list(mask.shape),
        "mask_values": classes,
        "mask_runs": np.column_stack((flat[starts], lengths)).tolist(),
        "class_labels": {str(value): labels[value] for value in classes if value in labels},
    }
    if binary:
        result["mask_row_major_rle"] = ([0] if flat[0] else []) + lengths.tolist()
    return result


def detection(output, mapping, original_size, model_size, settings):
    boxes = output[0] if isinstance(output, (list, tuple)) and len(output) == 1 else output
    if not isinstance(boxes, Mapping):
        raise ValueError("Unsupported detection export output structure")
    coordinates = array(boxes["boxes"]).astype(np.float64)
    class_ids = array(boxes["labels"])
    scores = array(boxes["scores"])
    if (
        coordinates.ndim != 2
        or coordinates.shape[1] != 4
        or class_ids.shape != (len(coordinates),)
        or scores.shape != (len(coordinates),)
        or not np.issubdtype(class_ids.dtype, np.integer)
        or not np.isfinite(coordinates).all()
        or not np.isfinite(scores).all()
    ):
        raise ValueError("Invalid xyxy detection output")
    if len(scores) > 4096 or ((scores < 0) | (scores > 1)).any():
        raise ValueError("Detection scores or candidate count exceed the output contract")
    if settings.detection_format != "xyxy_pixels":
        if original_size is None:
            raise ValueError("Normalized detections require original image dimensions")
        if settings.detection_format == "yxyx_normalized":
            coordinates = coordinates[:, [1, 0, 3, 2]]
        model_size = (1, 1)
    if original_size and model_size:
        scale_x = original_size[0] / model_size[0]
        scale_y = original_size[1] / model_size[1]
        coordinates[:, [0, 2]] = np.clip(coordinates[:, [0, 2]] * scale_x, 0, original_size[0])
        coordinates[:, [1, 3]] = np.clip(coordinates[:, [1, 3]] * scale_y, 0, original_size[1])
    keep = (coordinates[:, 0] < coordinates[:, 2]) & (coordinates[:, 1] < coordinates[:, 3])
    if settings.threshold is not None:
        keep &= scores >= settings.threshold
    indices = np.flatnonzero(keep)
    if settings.iou_threshold is not None:
        indices = suppress_overlaps(coordinates, class_ids, scores, indices, settings.iou_threshold)
    labels = {value: label for label, value in mapping.items()}
    return {
        "boxes_xyxy": coordinates[indices].tolist(),
        "class_ids": class_ids[indices].tolist(),
        "labels": [labels.get(index) for index in class_ids[indices]],
        "scores": scores[indices].tolist(),
        "count": len(indices),
    }


def suppress_overlaps(boxes, classes, scores, indices, threshold):
    remaining = indices[np.argsort(-scores[indices], kind="stable")]
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    selected = []
    while remaining.size:
        current, remaining = int(remaining[0]), remaining[1:]
        selected.append(current)
        top_left = np.maximum(boxes[current, :2], boxes[remaining, :2])
        bottom_right = np.minimum(boxes[current, 2:], boxes[remaining, 2:])
        intersection = np.maximum(0, bottom_right - top_left).prod(axis=1)
        iou = intersection / (area[current] + area[remaining] - intersection)
        remaining = remaining[(classes[remaining] != classes[current]) | (iou <= threshold)]
    return np.asarray(selected, dtype=np.intp)
