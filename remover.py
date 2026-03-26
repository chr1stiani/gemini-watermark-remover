"""
Gemini AI Photo logo removal engine.

Uses pre-extracted alpha maps (bg_48.png, bg_96.png) captured from Gemini's
fixed watermark pattern. The watermark is always white (255) and applied via
alpha compositing:

    watermarked = α × 255 + (1 - α) × original

Reversal:
    original = (watermarked - α × 255) / (1 - α)

No calibration needed – the alpha map is identical across all Gemini images
of the same size class.
"""

from __future__ import annotations

import cv2
import numpy as np
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ASSETS_DIR = Path(__file__).parent / "templates"

LOGO_VALUE = 255.0          # watermark color: pure white
ALPHA_THRESHOLD = 0.002     # ignore noise below this alpha
MAX_ALPHA = 0.99            # cap to prevent division by zero

# Watermark configs:  (alpha_map_file, logo_size, margin)
# - Small: either dimension ≤ 1024  →  48×48, 32px margin
# - Large: both dimensions > 1024   →  96×96, 64px margin
CONFIGS = {
    "small": ("bg_48.png", 48, 32),
    "large": ("bg_96.png", 96, 64),
}

# ---------------------------------------------------------------------------
# Alpha map loading (cached)
# ---------------------------------------------------------------------------

_alpha_cache: dict[str, np.ndarray] = {}


def _load_alpha_map(variant: str) -> np.ndarray:
    """
    Load and normalise an alpha map from the pre-captured background PNG.

    The PNG was captured on a pure black background, so pixel values equal
    α × 255. We take the max across RGB channels and normalise to [0, 1].
    """
    if variant in _alpha_cache:
        return _alpha_cache[variant]

    filename = CONFIGS[variant][0]
    path = ASSETS_DIR / filename
    if not path.exists():
        raise FileNotFoundError(
            f"Alpha map not found: {path}. "
            f"Make sure bg_48.png and bg_96.png are in the templates/ directory."
        )

    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Cannot read alpha map: {path}")

    # Max across BGR channels → single-channel alpha, normalised to [0, 1]
    alpha = np.max(img, axis=2).astype(np.float64) / 255.0
    alpha = np.clip(alpha, 0.0, MAX_ALPHA)
    alpha[alpha < ALPHA_THRESHOLD] = 0.0

    _alpha_cache[variant] = alpha
    return alpha


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pick_variant(width: int, height: int) -> str:
    """Choose 'small' or 'large' watermark variant based on image dimensions."""
    if width > 1024 and height > 1024:
        return "large"
    return "small"


def _watermark_box(
    width: int, height: int, variant: str,
) -> tuple[int, int, int, int]:
    """Return (x1, y1, x2, y2) of the watermark in the image."""
    _, logo_size, margin = CONFIGS[variant]
    x1 = width - margin - logo_size
    y1 = height - margin - logo_size
    x2 = x1 + logo_size
    y2 = y1 + logo_size
    return x1, y1, x2, y2


def _load_image(image: np.ndarray | str | Path) -> np.ndarray:
    if isinstance(image, (str, Path)):
        img = cv2.imread(str(image), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {image}")
        return img
    return image.copy()


# ---------------------------------------------------------------------------
# Core removal
# ---------------------------------------------------------------------------

def remove_watermark(image: np.ndarray | str | Path) -> np.ndarray:
    """
    Remove the Gemini watermark using reverse alpha blending.

    Parameters
    ----------
    image : file path or BGR numpy array

    Returns
    -------
    BGR numpy array with the watermark mathematically removed.
    """
    img = _load_image(image)
    h, w = img.shape[:2]

    variant = _pick_variant(w, h)
    alpha = _load_alpha_map(variant)
    x1, y1, x2, y2 = _watermark_box(w, h, variant)

    ah, aw = alpha.shape
    # Clamp the box to image bounds
    rx1 = max(x1, 0)
    ry1 = max(y1, 0)
    rx2 = min(x2, w)
    ry2 = min(y2, h)

    # Corresponding region in the alpha map
    ax1 = rx1 - x1
    ay1 = ry1 - y1
    ax2 = ax1 + (rx2 - rx1)
    ay2 = ay1 + (ry2 - ry1)

    region = img[ry1:ry2, rx1:rx2].astype(np.float64)
    a = alpha[ay1:ay2, ax1:ax2]

    # Build 3-channel alpha
    a3 = np.stack([a, a, a], axis=-1)

    # Only process pixels where alpha > threshold
    mask = a3 > ALPHA_THRESHOLD

    # Reverse alpha blending:  original = (watermarked - α * 255) / (1 - α)
    restored = region.copy()
    denom = 1.0 - a3
    restored[mask] = (region[mask] - a3[mask] * LOGO_VALUE) / denom[mask]
    restored = np.clip(restored, 0, 255).astype(np.uint8)

    result = img.copy()
    result[ry1:ry2, rx1:rx2] = restored
    return result


# ---------------------------------------------------------------------------
# High-level API
# ---------------------------------------------------------------------------

def process_image(
    input_path: str | Path,
    output_path: str | Path | None = None,
) -> Path:
    """Remove the watermark and save. Returns the output path."""
    input_path = Path(input_path)
    if output_path is None:
        out_dir = input_path.parent / "clean"
        out_dir.mkdir(exist_ok=True)
        output_path = out_dir / input_path.name
    else:
        output_path = Path(output_path)

    result = remove_watermark(input_path)
    cv2.imwrite(str(output_path), result)
    return output_path
