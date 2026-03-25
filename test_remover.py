"""Quick smoke test for the watermark remover."""

import cv2
import numpy as np
from pathlib import Path
from remover import remove_watermark_inpaint, remove_watermark_math, _watermark_region

TEST_DIR = Path(__file__).parent / "test_output"


def _create_test_image(width: int = 800, height: int = 600) -> np.ndarray:
    """Create a gradient test image with a fake watermark burned in."""
    # Blue-to-green gradient background
    img = np.zeros((height, width, 3), dtype=np.uint8)
    for y in range(height):
        for x in range(width):
            img[y, x] = [
                int(255 * x / width),       # B
                int(255 * y / height),       # G
                128,                          # R
            ]

    # Burn a white semi-transparent "star" into the bottom-right corner
    wm_size = 48
    margin = 32
    cx = width - margin - wm_size // 2
    cy = height - margin - wm_size // 2
    alpha = 0.3

    # Draw a simple cross/star shape
    for dy in range(-wm_size // 2, wm_size // 2):
        for dx in range(-wm_size // 2, wm_size // 2):
            dist = abs(dx) + abs(dy)
            if dist < wm_size // 2:
                px, py = cx + dx, cy + dy
                if 0 <= px < width and 0 <= py < height:
                    wm_pixel = np.array([255, 255, 255], dtype=np.float64)
                    orig = img[py, px].astype(np.float64)
                    img[py, px] = np.clip(
                        alpha * wm_pixel + (1 - alpha) * orig, 0, 255
                    ).astype(np.uint8)

    return img


def test_inpaint():
    """Test inpainting removal on a synthetic image."""
    img = _create_test_image()
    result = remove_watermark_inpaint(img, inpaint_radius=5, padding=10)
    assert result.shape == img.shape
    assert result.dtype == np.uint8

    # The watermark region should be different from the input
    h, w = img.shape[:2]
    x1, y1, x2, y2 = _watermark_region(w, h, padding=10)
    roi_before = img[y1:y2, x1:x2]
    roi_after = result[y1:y2, x1:x2]
    diff = np.mean(np.abs(roi_before.astype(float) - roi_after.astype(float)))
    assert diff > 0, "Inpainting should have changed the watermark region"

    TEST_DIR.mkdir(exist_ok=True)
    cv2.imwrite(str(TEST_DIR / "test_input.png"), img)
    cv2.imwrite(str(TEST_DIR / "test_output_inpaint.png"), result)
    print(f"  Inpaint test PASSED (region diff = {diff:.2f})")


def test_math_reversal():
    """Test mathematical reversal with a known template."""
    width, height = 800, 600
    # Create clean background
    img_clean = np.zeros((height, width, 3), dtype=np.uint8)
    for y in range(height):
        for x in range(width):
            img_clean[y, x] = [int(255 * x / width), int(255 * y / height), 128]

    # Create watermark template (BGRA)
    wm_size = 48
    template = np.zeros((wm_size, wm_size, 4), dtype=np.uint8)
    for dy in range(wm_size):
        for dx in range(wm_size):
            dist = abs(dx - wm_size // 2) + abs(dy - wm_size // 2)
            if dist < wm_size // 2:
                template[dy, dx] = [255, 255, 255, 76]  # alpha ≈ 0.3

    # Apply watermark to create the "watermarked" image
    margin = 32
    img_wm = img_clean.copy()
    x1 = width - margin - wm_size
    y1 = height - margin - wm_size
    for dy in range(wm_size):
        for dx in range(wm_size):
            a = template[dy, dx, 3] / 255.0
            wm_rgb = template[dy, dx, :3].astype(np.float64)
            orig = img_clean[y1 + dy, x1 + dx].astype(np.float64)
            img_wm[y1 + dy, x1 + dx] = np.clip(
                a * wm_rgb + (1 - a) * orig, 0, 255
            ).astype(np.uint8)

    # Now reverse it
    result = remove_watermark_math(img_wm, template)

    # Check that result is very close to the original clean image
    max_error = np.max(np.abs(result.astype(float) - img_clean.astype(float)))
    mean_error = np.mean(np.abs(result.astype(float) - img_clean.astype(float)))
    assert max_error <= 2.0, f"Max error too high: {max_error}"
    print(f"  Math reversal test PASSED (max_err={max_error:.1f}, mean_err={mean_error:.4f})")


def test_region_detection():
    """Test watermark region calculation for different image sizes."""
    # Small image
    x1, y1, x2, y2 = _watermark_region(800, 600, padding=0)
    assert x2 == 800 - 32, f"Expected x2={800 - 32}, got {x2}"
    assert y2 == 600 - 32, f"Expected y2={600 - 32}, got {y2}"
    assert x2 - x1 == 48, f"Expected width=48, got {x2 - x1}"
    print(f"  Region test (800×600): ({x1},{y1})-({x2},{y2}) PASSED")

    # Large image
    x1, y1, x2, y2 = _watermark_region(2048, 2048, padding=0)
    print(f"  Region test (2048×2048): ({x1},{y1})-({x2},{y2}) PASSED")


if __name__ == "__main__":
    print("Running tests...")
    test_region_detection()
    test_inpaint()
    test_math_reversal()
    print("\nAll tests passed!")
