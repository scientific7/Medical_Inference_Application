import io

import numpy as np
import pytest
from PIL import Image

from medical_imaging_inference.preprocessing import InvalidImage, PreprocessingSpec


@pytest.fixture
def png():
    content = io.BytesIO()
    Image.new("RGB", (4, 3), (255, 128, 0)).save(content, format="PNG")
    return content.getvalue()


@pytest.fixture
def manifest():
    return {
        "framework": "pytorch",
        "task": "classification",
        "preprocessing": {"image_size": [2, 5], "color": "RGB", "scale": "[0,1]"},
    }


@pytest.mark.parametrize(
    ("framework", "shape", "pixel_index"),
    [("pytorch", (3, 2, 5), (slice(None), 0, 0)), ("tensorflow", (2, 5, 3), (0, 0))],
)
def test_training_profile_matches_channels_scale_and_layout(manifest, png, framework, shape, pixel_index):
    manifest["framework"] = framework
    spec = PreprocessingSpec.from_manifest(manifest)
    tensor, original = spec.transform(png, filename="input.png", media_type="image/png", max_pixels=12)
    assert original == (4, 3)
    assert tensor.shape == shape
    assert tensor.dtype == np.float32
    np.testing.assert_allclose(tensor[pixel_index], (1, 128 / 255, 0))
    assert tensor.min() == 0
    assert tensor.max() == 1


def test_explicit_bgr_and_normalization(manifest, png):
    explicit = PreprocessingSpec.from_manifest(manifest).model_dump()
    explicit.update(color="BGR", mean=(0, 0.5, 0), std=(1, 0.5, 2))
    spec = PreprocessingSpec.from_manifest({**manifest, "preprocessing": explicit})
    tensor, _ = spec.transform(png, filename="input.png", media_type="image/png", max_pixels=12)
    np.testing.assert_allclose(tensor[:, 0, 0], (0, (128 / 255 - 0.5) / 0.5, 0.5), atol=1e-7)


@pytest.mark.parametrize(
    "source", [{}, {"image_size": [2, 5]}, {"image_size": [0, 5], "color": "RGB", "scale": "[0,1]"}]
)
def test_missing_and_invalid_specifications_are_rejected(manifest, source):
    with pytest.raises(ValueError):
        PreprocessingSpec.from_manifest({**manifest, "preprocessing": source})


@pytest.mark.parametrize(
    ("filename", "media_type", "limit"),
    [
        ("input.jpg", "image/png", 12),
        ("input.png", "image/jpeg", 12),
        ("input.png", "image/png", 11),
    ],
)
def test_upload_format_and_pixel_limit(manifest, png, filename, media_type, limit):
    with pytest.raises(InvalidImage):
        PreprocessingSpec.from_manifest(manifest).transform(
            png,
            filename=filename,
            media_type=media_type,
            max_pixels=limit,
        )


def test_invalid_bytes(manifest):
    with pytest.raises(InvalidImage):
        PreprocessingSpec.from_manifest(manifest).transform(
            b"broken",
            filename="input.png",
            media_type="image/png",
            max_pixels=12,
        )


def test_torch_bilinear_matches_yolo_reader_on_nonuniform_image():
    torch = pytest.importorskip("torch")
    arr = np.arange(5 * 7 * 3, dtype=np.uint8).reshape(5, 7, 3)
    content = io.BytesIO()
    Image.fromarray(arr).save(content, format="PNG")
    spec = PreprocessingSpec.from_manifest(
        {
            "framework": "pytorch",
            "task": "detection",
            "preprocessing": {
                "version": "1",
                "image_size": [2, 3],
                "color": "RGB",
                "scale": "[0,1]",
                "resize": "torch_bilinear",
                "aspect_ratio": "stretch",
                "orientation": "raw",
                "channel_order": "CHW",
                "formats": ["PNG"],
            },
        }
    )
    expected = torch.nn.functional.interpolate(
        torch.tensor(arr).permute(2, 0, 1)[None].float() / 255,
        size=(2, 3),
        mode="bilinear",
        align_corners=False,
    )[0]
    actual, size = spec.transform(
        content.getvalue(),
        filename="sample.png",
        media_type="image/png",
        max_pixels=35,
    )
    np.testing.assert_array_equal(actual, expected.numpy())
    assert size == (7, 5)


def test_faster_rcnn_keeps_native_dimensions():
    manifest = {
        "framework": "pytorch",
        "task": "detection",
        "runtime_output": "torchvision_detection_tuple",
        "preprocessing": {
            "version": "1",
            "image_size": [32, 32],
            "color": "RGB",
            "scale": "[0,1]",
            "resize": "native",
            "aspect_ratio": "preserve",
            "orientation": "raw",
            "channel_order": "CHW",
            "formats": ["PNG"],
        },
    }
    image = io.BytesIO()
    Image.new("RGB", (9, 5), (255, 0, 0)).save(image, format="PNG")
    pixels, size = PreprocessingSpec.from_manifest(manifest).transform(
        image.getvalue(),
        filename="image.png",
        media_type="image/png",
        max_pixels=45,
    )
    assert size == (9, 5)
    assert pixels.shape == (3, 5, 9)
    np.testing.assert_array_equal(pixels[:, 0, 0], [1, 0, 0])
    manifest["preprocessing"]["aspect_ratio"] = "stretch"
    with pytest.raises(ValueError, match="native image dimensions"):
        PreprocessingSpec.from_manifest(manifest)
