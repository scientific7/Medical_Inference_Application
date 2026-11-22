"""Validate model input contracts and turn decoded images into float32 tensors."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Annotated, Literal

import numpy as np
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field, model_validator

Dimension = Annotated[int, Field(strict=True, gt=0, le=8192)]
ImageFormat = Literal["JPEG", "PNG", "TIFF", "WEBP"]
Scale = Literal["[0,1]", "[0,255]"]
ChannelValue = Annotated[float, Field(allow_inf_nan=False)]

EXTENSIONS = {
    "JPEG": {".jpg", ".jpeg"},
    "PNG": {".png"},
    "TIFF": {".tif", ".tiff"},
    "WEBP": {".webp"},
}
MEDIA_TYPES = {name: f"image/{name.lower()}" for name in EXTENSIONS}
RESAMPLING = {
    "nearest": Image.Resampling.NEAREST,
    "bilinear": Image.Resampling.BILINEAR,
    "bicubic": Image.Resampling.BICUBIC,
    "lanczos": Image.Resampling.LANCZOS,
}
SOURCE_MODES = {"1", "L", "LA", "P", "RGB", "RGBA", "CMYK", "YCbCr"}


class InvalidImage(ValueError):
    """The supplied image does not satisfy the upload contract."""


class PreprocessingSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    version: Literal["1"]
    image_size: tuple[Dimension, Dimension]
    color: Literal["RGB", "BGR"]
    scale: Scale
    resize: Literal["nearest", "bilinear", "bicubic", "lanczos"]
    aspect_ratio: Literal["stretch"]
    orientation: Literal["raw"]
    channel_order: Literal["CHW", "HWC"]
    formats: Annotated[tuple[ImageFormat, ...], Field(min_length=1)]
    mean: tuple[ChannelValue, ChannelValue, ChannelValue] | None = None
    std: tuple[ChannelValue, ChannelValue, ChannelValue] | None = None

    @model_validator(mode="after")
    def validate_normalization(self) -> PreprocessingSpec:
        if (self.mean is None) != (self.std is None):
            raise ValueError("Normalization requires both mean and standard deviation")
        if self.std is not None and any(value <= 0 for value in self.std):
            raise ValueError("Standard deviations must be positive")
        if self.image_size[0] * self.image_size[1] > 16_777_216:
            raise ValueError("Model input size exceeds the tensor limit")
        return self

    @classmethod
    def from_manifest(cls, manifest: dict) -> PreprocessingSpec:
        source = manifest.get("preprocessing")
        order = {"pytorch": "CHW", "tensorflow": "HWC"}.get(manifest.get("framework"))
        if not isinstance(source, dict) or order is None:
            raise ValueError("Model requires a supported framework and preprocessing specification")
        # The existing training manifest contains these three fields. Its
        # classification and segmentation readers supply the rest of this profile.
        if (
            set(source) == {"image_size", "color", "scale"}
            and source["color"] == "RGB"
            and source["scale"] == "[0,1]"
            and manifest.get("task") in {"classification", "segmentation"}
        ):
            source = {
                **source,
                "version": "1",
                "resize": "bilinear",
                "aspect_ratio": "stretch",
                "orientation": "raw",
                "channel_order": order,
                "formats": ["JPEG", "PNG"],
            }
        spec = cls.model_validate(source)
        if spec.channel_order != order:
            raise ValueError("Tensor layout is incompatible with the framework adapter")
        return spec

    def transform(
        self,
        content: bytes,
        *,
        filename: str,
        media_type: str,
        max_pixels: int,
    ) -> tuple[np.ndarray, tuple[int, int]]:
        try:
            with Image.open(io.BytesIO(content), formats=list(self.formats)) as image:
                if image.width * image.height > max_pixels:
                    raise InvalidImage("Image pixel count exceeds the configured limit")
                if (
                    Path(filename).suffix.lower() not in EXTENSIONS[image.format]
                    or media_type.lower() != MEDIA_TYPES[image.format]
                ):
                    raise InvalidImage("Image format, extension and media type must agree")
                if image.mode not in SOURCE_MODES or getattr(image, "n_frames", 1) != 1:
                    raise InvalidImage("Unsupported image channels or frame count")
                original_size = image.size
                image.verify()
            with Image.open(io.BytesIO(content), formats=list(self.formats)) as image:
                height, width = self.image_size
                resized = image.convert("RGB").resize((width, height), RESAMPLING[self.resize])
                values = np.array(resized, dtype=np.float32)
        except (OSError, Image.DecompressionBombError) as exc:
            raise InvalidImage("Image is malformed, unreadable or unsupported by the model") from exc

        if self.color == "BGR":
            values = values[..., ::-1]
        if self.scale == "[0,1]":
            values = values / np.float32(255)
        if self.mean is not None and self.std is not None:
            values = (values - np.asarray(self.mean, dtype=np.float32)) / np.asarray(
                self.std, dtype=np.float32
            )
        if self.channel_order == "CHW":
            values = values.transpose(2, 0, 1)
        return np.ascontiguousarray(values), original_size
