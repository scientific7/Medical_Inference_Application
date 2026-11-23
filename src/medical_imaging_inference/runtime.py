"""Process-local cache of framework exports keyed by approved package digest."""

from __future__ import annotations

import os
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Callable

import numpy as np

from .postprocessing import OutputSettings
from .preprocessing import PreprocessingSpec
from .security import verified_package


@dataclass(frozen=True)
class LoadedModel:
    manifest: dict
    preprocessing: PreprocessingSpec
    device: str
    run: Callable[[np.ndarray], object]


def load_export(package: Path, manifest: dict) -> LoadedModel:
    if manifest.get("task") not in {"classification", "segmentation", "detection"}:
        raise ValueError("Unsupported model task")
    if manifest.get("framework") not in {"pytorch", "tensorflow"}:
        raise ValueError("Unsupported model framework")
    if not isinstance(manifest.get("class_mapping"), dict):
        raise ValueError("Model manifest requires a class_mapping object")
    spec = PreprocessingSpec.from_manifest(manifest)
    OutputSettings.model_validate(manifest.get("postprocessing", {}))
    device = os.getenv("INFERENCE_DEVICE", "cpu")
    if device not in {"cpu", "gpu", "cuda", "mps"}:
        raise ValueError("INFERENCE_DEVICE must be cpu, gpu, cuda, or mps")
    if manifest["framework"] == "pytorch":
        import torch

        if device in {"gpu", "cuda"} and not torch.cuda.is_available():
            raise ValueError("PyTorch GPU execution is unavailable")
        if device == "mps" and not torch.backends.mps.is_available():
            raise ValueError("PyTorch MPS execution is unavailable")
        target = "cuda:0" if device in {"gpu", "cuda"} else device
        export = package / "inference_model.ts"
        if not export.is_file():
            raise ValueError("Missing TorchScript export")
        torchvision_output = manifest.get("runtime_output") == "torchvision_detection_tuple"
        if torchvision_output and device == "mps":
            raise ValueError("Torchvision scripted detection is not supported on MPS")
        if torchvision_output:
            # Register Torchvision's custom operators before loading the script.
            import torchvision  # noqa: F401

        model = torch.jit.load(str(export), map_location="cpu" if device == "mps" else target)
        model = model.to(target).eval()

        def run(pixels):
            with torch.inference_mode():
                tensor = torch.from_numpy(pixels).to(target)
                if torchvision_output:
                    _, detections = model([tensor])
                    return detections
                output = model(tensor.unsqueeze(0))
                if manifest.get("runtime_output") == "batched_detection_dict":
                    return {key: output[key][0] for key in ("boxes", "labels", "scores")}
                return output

    else:
        import tensorflow as tf

        if device in {"cuda", "mps"}:
            raise ValueError("TensorFlow requires cpu or gpu device selection")
        if device == "gpu" and not tf.config.list_logical_devices("GPU"):
            raise ValueError("TensorFlow GPU execution is unavailable")
        target = "/GPU:0" if device == "gpu" else "/CPU:0"
        export = package / "serving_model"
        if not export.is_dir():
            raise ValueError("Missing TensorFlow SavedModel export")
        with tf.device(target):
            signature = tf.saved_model.load(str(export)).signatures["serving_default"]
        (input_name,) = signature.structured_input_signature[1]

        def run(pixels):
            with tf.device(target):
                outputs = signature(**{input_name: tf.convert_to_tensor(pixels[None, ...])})
            if manifest["task"] == "detection":
                return outputs
            (output,) = outputs.values()
            return output

    return LoadedModel(manifest, spec, target, run)


class ModelCache:
    def __init__(self, capacity: int = 2):
        if capacity < 1:
            raise ValueError("Model cache capacity must be positive")
        self.capacity = capacity
        self._lock = Lock()
        self._load_ms: dict[tuple[str, str, str], float] = {}
        self._models: OrderedDict[tuple[str, str, str], LoadedModel] = OrderedDict()
        self._hits = 0
        self._misses = 0
        self._evictions = 0

    def get(self, record: dict) -> LoadedModel:
        if record.get("state") != "approved":
            raise ValueError("Model version is not approved")
        key = (record["model_id"], record["version"], record["artifact_digest"])
        with self._lock:
            if key not in self._models:
                self._misses += 1
                started = time.perf_counter()
                package, manifest = verified_package(record)
                self._models[key] = load_export(package, manifest)
                self._load_ms[key] = (time.perf_counter() - started) * 1000
                if len(self._models) > self.capacity:
                    evicted, _ = self._models.popitem(last=False)
                    del self._load_ms[evicted]
                    self._evictions += 1
            else:
                self._hits += 1
            self._models.move_to_end(key)
            return self._models[key]

    def unload(self, model_id: str, version: str) -> bool:
        with self._lock:
            matches = [key for key in self._models if key[:2] == (model_id, version)]
            for key in matches:
                del self._models[key]
                del self._load_ms[key]
            return bool(matches)

    def active(self) -> list[dict]:
        with self._lock:
            return [
                {
                    "model_id": key[0],
                    "version": key[1],
                    "device": model.device,
                    "integrity": "verified",
                    "load_ms": self._load_ms[key],
                }
                for key, model in self._models.items()
            ]

    def stats(self) -> dict:
        with self._lock:
            return {
                "capacity": self.capacity,
                "loaded": len(self._models),
                "hits": self._hits,
                "misses": self._misses,
                "evictions": self._evictions,
            }
