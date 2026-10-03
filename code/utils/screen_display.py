from __future__ import annotations

import ast
import base64
import math
import re
from io import BytesIO
from pathlib import Path
from typing import Iterable, Sequence

DEFAULT_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")
DEFAULT_SCREEN_DIR = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "dataset"
    / "cleaned_dataset"
    / "used_dataset_screens"
)


def screen_id_stem(screen_id) -> str:
    """Return the filename stem used by screenshot assets."""
    if isinstance(screen_id, float) and math.isnan(screen_id):
        raise ValueError("screen_id cannot be NaN.")
    try:
        return str(int(screen_id))
    except (TypeError, ValueError):
        return str(screen_id)


def screen_image_path(
    screen_id,
    image_dir: str | Path | None = None,
    *,
    extensions: Sequence[str] = DEFAULT_IMAGE_EXTENSIONS,
) -> Path:
    """Return the screenshot path for a screen_id, or raise if it is missing."""
    image_dir = Path(image_dir) if image_dir is not None else DEFAULT_SCREEN_DIR
    stem = screen_id_stem(screen_id)
    for ext in extensions:
        candidate = image_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No image found for screen_id={screen_id} in {image_dir}")


def find_screen_image(
    screen_id,
    image_dir: str | Path | None = None,
    *,
    extensions: Sequence[str] = DEFAULT_IMAGE_EXTENSIONS,
) -> Path | None:
    """Return the screenshot path for a screen_id, or None if it is missing."""
    try:
        return screen_image_path(screen_id, image_dir, extensions=extensions)
    except FileNotFoundError:
        return None


def image_from_base64(base64_string: str):
    """Decode a base64 image string into a PIL Image."""
    from PIL import Image

    payload = base64_string.strip()
    if payload.lower().startswith("data:image/") and "," in payload:
        payload = payload.split(",", 1)[1]
    return Image.open(BytesIO(base64.b64decode(payload))).convert("RGB")


def load_screen_image(
    *,
    screen_id=None,
    base64_screen: str | None = None,
    image_dir: str | Path | None = None,
    extensions: Sequence[str] = DEFAULT_IMAGE_EXTENSIONS,
):
    from PIL import Image

    if screen_id is not None:
        try:
            return Image.open(screen_image_path(screen_id, image_dir, extensions=extensions)).convert("RGB")
        except FileNotFoundError:
            if base64_screen is None:
                raise
    if base64_screen is not None:
        return image_from_base64(base64_screen)
    raise ValueError("Either screen_id or base64_screen is required.")


def normalize_bbox(bbox) -> tuple[float, float, float, float] | None:
    """Normalize bbox values to (x_min, y_min, x_max, y_max)."""
    if bbox is None or (isinstance(bbox, float) and math.isnan(bbox)):
        return None
    if isinstance(bbox, str):
        bbox = bbox.strip()
        if not bbox or bbox.lower() in {"nan", "none", "null"}:
            return None
        bbox = _parse_bbox_string(bbox)

    if isinstance(bbox, dict):
        if all(key in bbox for key in ("x_min", "y_min", "x_max", "y_max")):
            values = [bbox[key] for key in ("x_min", "y_min", "x_max", "y_max")]
        elif all(key in bbox for key in ("x", "y", "width", "height")):
            x, y = float(bbox["x"]), float(bbox["y"])
            values = [x, y, x + float(bbox["width"]), y + float(bbox["height"])]
        else:
            raise ValueError(f"Expected bbox keys x_min/y_min/x_max/y_max, got {bbox!r}")
    else:
        values = list(bbox)

    if len(values) < 4:
        raise ValueError(f"Expected four bbox coordinates, got {values!r}")
    return tuple(float(value) for value in values[:4])


def _parse_bbox_string(value: str):
    if value.startswith("array(") and value.endswith(")"):
        value = value[len("array(") : -1]
    try:
        return ast.literal_eval(value)
    except (SyntaxError, ValueError):
        numbers = re.findall(r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?", value)
        if len(numbers) >= 4:
            return [float(number) for number in numbers[:4]]
        raise ValueError(f"Could not parse bbox string: {value!r}") from None


def _bbox_items(bboxes) -> list[tuple[int, tuple[float, float, float, float]]]:
    if bboxes is None:
        return []
    try:
        single = normalize_bbox(bboxes)
    except (TypeError, ValueError):
        single = None
    if single is not None:
        return [(0, single)]

    items = []
    for idx, bbox in enumerate(bboxes):
        try:
            normalized = normalize_bbox(bbox)
        except (TypeError, ValueError):
            continue
        if normalized is not None:
            items.append((idx, normalized))
    return items


def draw_bbox(
    ax,
    bbox,
    *,
    image_width: int | float | None = None,
    image_height: int | float | None = None,
    color: str = "tab:red",
    label: str | None = None,
    linewidth: float = 2,
):
    """Draw a normalized bbox on an existing matplotlib axis."""
    from matplotlib.patches import Rectangle

    normalized = normalize_bbox(bbox)
    if normalized is None:
        return None
    if image_width is None or image_height is None:
        if not ax.images:
            raise ValueError("image_width and image_height are required when ax has no image.")
        image_height, image_width = ax.images[0].get_array().shape[:2]

    x1, y1, x2, y2 = normalized
    rect = Rectangle(
        (x1 * image_width, y1 * image_height),
        (x2 - x1) * image_width,
        (y2 - y1) * image_height,
        fill=False,
        edgecolor=color,
        linewidth=linewidth,
    )
    ax.add_patch(rect)
    if label:
        ax.text(
            x1 * image_width,
            max(0, y1 * image_height - 8),
            label,
            color="white",
            fontsize=9,
            weight="bold",
            bbox={"facecolor": color, "edgecolor": color, "pad": 2},
        )
    return rect


def _render_image(
    image,
    *,
    bboxes=None,
    labels: Sequence[str] | None = None,
    colors: Sequence[str] | None = None,
    title: str | None = None,
    width: int = 300,
):
    import matplotlib.pyplot as plt

    image_width, image_height = image.size
    fig, ax = plt.subplots(figsize=(width / 100, (width / image_width * image_height) / 100))
    ax.imshow(image)
    if title:
        ax.set_title(title, loc="left", fontsize=10, pad=12)

    labels = list(labels) if labels is not None else []
    colors = list(colors) if colors is not None else ["tab:red", "tab:cyan", "tab:orange", "tab:green"]
    for display_idx, (source_idx, bbox) in enumerate(_bbox_items(bboxes)):
        draw_bbox(
            ax,
            bbox,
            image_width=image_width,
            image_height=image_height,
            color=colors[display_idx % len(colors)],
            label=labels[source_idx] if source_idx < len(labels) else None,
        )

    ax.axis("off")
    plt.tight_layout()
    plt.show()
    return fig, ax


def show_screen(
    *,
    screen_id=None,
    base64_screen: str | None = None,
    image_dir: str | Path | None = None,
    bboxes=None,
    labels: Sequence[str] | None = None,
    colors: Sequence[str] | None = None,
    title: str | None = None,
    width: int = 300,
    extensions: Sequence[str] = DEFAULT_IMAGE_EXTENSIONS,
):
    """Display a screen from file or base64 using the same optional overlay path."""
    image = load_screen_image(
        screen_id=screen_id,
        base64_screen=base64_screen,
        image_dir=image_dir,
        extensions=extensions,
    )
    return _render_image(image, bboxes=bboxes, labels=labels, colors=colors, title=title, width=width)


def show_screen_grid(
    screen_ids: Iterable,
    image_dir: str | Path | None = None,
    *,
    max_cols: int = 4,
    extensions: Sequence[str] = DEFAULT_IMAGE_EXTENSIONS,
):
    """Display multiple screenshots in a compact matplotlib grid."""
    import matplotlib.pyplot as plt
    import numpy as np

    screen_ids = list(screen_ids)
    if not screen_ids:
        print("No screens to display.")
        return None, None

    cols = min(max_cols, len(screen_ids))
    rows = math.ceil(len(screen_ids) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 5))
    axes = np.array(axes).reshape(-1)

    for ax in axes:
        ax.axis("off")
    for ax, screen_id in zip(axes, screen_ids):
        image_path = find_screen_image(screen_id, image_dir, extensions=extensions)
        if image_path is None:
            ax.set_title(f"{screen_id} missing")
            continue
        ax.imshow(plt.imread(image_path))
        ax.set_title(str(screen_id))

    plt.tight_layout()
    plt.show()
    return fig, axes
