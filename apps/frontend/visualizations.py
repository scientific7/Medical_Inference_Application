"""Render prediction metadata against the image in the current session."""

import numpy as np
from PIL import Image, ImageDraw

PALETTE = ((0, 0, 0), (68, 210, 177), (248, 174, 76), (158, 133, 255))


def mask_images(original: Image.Image, output: dict):
    height, width = output["mask_shape"]
    if (width, height) != original.size:
        raise ValueError("Mask dimensions differ from the source image")
    classes, lengths = np.asarray(output["mask_runs"], dtype=np.int64).T
    if (lengths < 1).any() or (classes < 0).any() or lengths.sum() != height * width:
        raise ValueError("Invalid mask encoding")
    mask = np.repeat(classes, lengths).reshape(height, width)
    pixels = np.asarray(PALETTE, dtype=np.uint8)[mask % len(PALETTE)]
    overlay = np.array(original.convert("RGB"))
    foreground = mask != 0
    overlay[foreground] = overlay[foreground] * 0.5 + pixels[foreground] * 0.5
    return Image.fromarray(pixels), Image.fromarray(overlay)


def annotated_image(original: Image.Image, output: dict):
    image = original.convert("RGB")
    draw = ImageDraw.Draw(image)
    rows = []
    for box, label, class_id, score in zip(
        output["boxes_xyxy"], output["labels"], output["class_ids"], output["scores"]
    ):
        label = label or f"Class {class_id}"
        draw.rectangle(box, outline="#44d2b1", width=3)
        draw.text(
            (box[0], box[1]), f"{label} {score:.2f}", fill="#ffffff", stroke_width=2, stroke_fill="#111827"
        )
        rows.append({"class": label, "score": score, "box": box})
    return image, rows
