from pathlib import Path

from PIL import Image


def test_segmentation_overlay_preserves_background(monkeypatch):
    frontend = Path(__file__).resolve().parents[1] / "apps" / "frontend"
    monkeypatch.syspath_prepend(str(frontend))
    from visualizations import mask_images

    original = Image.new("RGB", (2, 1), (100, 100, 100))
    mask, overlay = mask_images(original, {"mask_shape": [1, 2], "mask_runs": [[0, 1], [1, 1]]})
    assert mask.size == original.size
    assert overlay.getpixel((0, 0)) == original.getpixel((0, 0))
    assert overlay.getpixel((1, 0)) != original.getpixel((1, 0))
