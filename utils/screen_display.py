from __future__ import annotations

import base64
import math
from io import BytesIO
from pathlib import Path
from typing import Iterable, Sequence

from .paths import SCREENS_DIR as DEFAULT_SCREEN_DIR

DEFAULT_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")
DEFAULT_COLORS = ("tab:red", "tab:cyan", "tab:orange", "tab:green")


def find_screen_image(screen_id, image_dir=None, *, extensions: Sequence[str] = DEFAULT_IMAGE_EXTENSIONS) -> Path | None:
    """Return the screenshot path for a screen_id, or None if it is missing."""
    try:
        stem = str(int(screen_id))
    except (TypeError, ValueError):
        stem = str(screen_id)
    image_dir = Path(image_dir or DEFAULT_SCREEN_DIR)
    return next((p for ext in extensions if (p := image_dir / f"{stem}{ext}").exists()), None)


def _bbox(bbox) -> tuple[float, float, float, float] | None:
    """(x_min, y_min, x_max, y_max) from an x_min/y_min/x_max/y_max dict or a 4-sequence; None if unusable."""
    try:
        if isinstance(bbox, dict):
            bbox = [bbox[k] for k in ("x_min", "y_min", "x_max", "y_max")]
        values = tuple(float(v) for v in bbox[:4])
    except (TypeError, ValueError, KeyError, IndexError):
        return None
    return values if len(values) == 4 else None


def _bbox_items(bboxes) -> list[tuple[int, tuple[float, float, float, float]]]:
    """Accept one bbox or a list of them; keep each bbox's source index so labels stay aligned."""
    if bboxes is None:
        return []
    if (single := _bbox(bboxes)) is not None:
        return [(0, single)]
    try:
        return [(i, b) for i, b in enumerate(map(_bbox, bboxes)) if b is not None]
    except TypeError:
        return []


def show_screen(
    *,
    screen_id=None,
    base64_screen: str | None = None,
    image_dir=None,
    bboxes=None,
    labels: Sequence[str] | None = None,
    colors: Sequence[str] | None = None,
    title: str | None = None,
    width: int = 300,
    extensions: Sequence[str] = DEFAULT_IMAGE_EXTENSIONS,
):
    """Display a screen (from file, falling back to base64) with optional labelled bbox overlays."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from PIL import Image

    path = find_screen_image(screen_id, image_dir, extensions=extensions) if screen_id is not None else None
    if path is not None:
        source = path
    elif base64_screen is not None:
        source = BytesIO(base64.b64decode(base64_screen.split(",")[-1]))  # strips an optional data-URL prefix
    elif screen_id is not None:
        raise FileNotFoundError(f"No image found for screen_id={screen_id} in {image_dir or DEFAULT_SCREEN_DIR}")
    else:
        raise ValueError("Either screen_id or base64_screen is required.")
    image = Image.open(source).convert("RGB")

    w, h = image.size
    fig, ax = plt.subplots(figsize=(width / 100, width / w * h / 100))
    ax.imshow(image)
    if title:
        ax.set_title(title, loc="left", fontsize=10, pad=12)

    labels = list(labels or [])
    colors = list(colors or DEFAULT_COLORS)
    for n, (i, (x1, y1, x2, y2)) in enumerate(_bbox_items(bboxes)):
        color = colors[n % len(colors)]
        ax.add_patch(Rectangle((x1 * w, y1 * h), (x2 - x1) * w, (y2 - y1) * h, fill=False, edgecolor=color, linewidth=2))
        if i < len(labels) and labels[i]:
            ax.text(x1 * w, max(0, y1 * h - 8), labels[i], color="white", fontsize=9, weight="bold",
                    bbox={"facecolor": color, "edgecolor": color, "pad": 2})

    ax.axis("off")
    plt.tight_layout()
    plt.show()
    return fig, ax


def show_screen_grid(screen_ids: Iterable, image_dir=None, *, max_cols: int = 4,
                     extensions: Sequence[str] = DEFAULT_IMAGE_EXTENSIONS):
    """Display multiple screenshots in a compact matplotlib grid."""
    import matplotlib.pyplot as plt

    screen_ids = list(screen_ids)
    if not screen_ids:
        print("No screens to display.")
        return None, None

    cols = min(max_cols, len(screen_ids))
    rows = math.ceil(len(screen_ids) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 5), squeeze=False)
    axes = axes.ravel()
    for ax in axes:
        ax.axis("off")
    for ax, screen_id in zip(axes, screen_ids):
        path = find_screen_image(screen_id, image_dir, extensions=extensions)
        if path is None:
            ax.set_title(f"{screen_id} missing")
            continue
        ax.imshow(plt.imread(path))
        ax.set_title(str(screen_id))

    plt.tight_layout()
    plt.show()
    return fig, axes
