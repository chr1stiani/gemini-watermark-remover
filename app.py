"""
Gradio web UI for removing the Gemini AI Photo watermark.
Monochrome dark theme.
"""

import tempfile
import zipfile
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
from PIL import Image

from remover import remove_watermark


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pil_to_cv2(pil_img: Image.Image) -> np.ndarray:
    arr = np.array(pil_img)
    if arr.ndim == 2:
        return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    if arr.shape[2] == 4:
        return cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def _cv2_to_pil(cv2_img: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB))


# ---------------------------------------------------------------------------
# Processing
# ---------------------------------------------------------------------------

def process_images(files, save_webp, output_dir):
    if not files:
        raise gr.Error("Drop some images first.")

    output_dir = output_dir.strip() if output_dir else ""
    use_custom_dir = bool(output_dir)
    out_path = Path(output_dir) if use_custom_dir else Path(tempfile.mkdtemp(prefix="gemini_"))
    out_path.mkdir(parents=True, exist_ok=True)

    results, errors = [], []
    first_pil = None

    for f in files:
        src = Path(f.name) if hasattr(f, "name") else Path(f)
        try:
            img = cv2.imread(str(src), cv2.IMREAD_COLOR)
            if img is None:
                errors.append(f"{src.name}: cannot read")
                continue
            cleaned = remove_watermark(img)
            if save_webp:
                dest = out_path / (src.stem + ".webp")
                cv2.imwrite(str(dest), cleaned, [cv2.IMWRITE_WEBP_QUALITY, 90])
            else:
                dest = out_path / (src.stem + src.suffix)
                cv2.imwrite(str(dest), cleaned)
            results.append(dest.name)
            if first_pil is None:
                first_pil = _cv2_to_pil(cleaned)
        except Exception as e:
            errors.append(f"{src.name}: {e}")

    n = len(files)
    fmt = " as WebP" if save_webp else ""
    status = f"Done — {len(results)}/{n} processed{fmt}"
    if use_custom_dir:
        status += f"  ·  {out_path}"
    if errors:
        status += "\n" + "\n".join(f"✕ {e}" for e in errors)

    zip_path = None
    if not use_custom_dir and results:
        zf_path = out_path / "result.zip"
        with zipfile.ZipFile(zf_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for name in results:
                zf.write(out_path / name, name)
        zip_path = str(zf_path)

    return first_pil, status, zip_path


def convert_to_webp(files, quality, lossless, output_dir):
    if not files:
        raise gr.Error("Drop some images first.")

    output_dir = output_dir.strip() if output_dir else ""
    use_custom_dir = bool(output_dir)
    out_path = Path(output_dir) if use_custom_dir else Path(tempfile.mkdtemp(prefix="webp_"))
    out_path.mkdir(parents=True, exist_ok=True)

    results, errors = [], []
    for f in files:
        src = Path(f.name) if hasattr(f, "name") else Path(f)
        try:
            img = Image.open(src)
            dest = out_path / (src.stem + ".webp")
            img.save(str(dest), "WEBP", quality=quality, lossless=lossless)
            results.append(dest.name)
        except Exception as e:
            errors.append(f"{src.name}: {e}")

    status = f"Done — {len(results)}/{len(files)} converted"
    if use_custom_dir:
        status += f"  ·  {out_path}"
    if errors:
        status += "\n" + "\n".join(f"✕ {e}" for e in errors)

    zip_path = None
    if not use_custom_dir and results:
        zf_path = out_path / "webp.zip"
        with zipfile.ZipFile(zf_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for name in results:
                zf.write(out_path / name, name)
        zip_path = str(zf_path)

    return status, zip_path


# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Reset & Base ─────────────────────────────────────────────── */
.gradio-container {
    background: #0a0a0a !important;
    font-family: 'Inter', -apple-system, system-ui, sans-serif !important;
    color: #e5e5e5 !important;
}
.contain { max-width: 640px !important; margin: 0 auto !important; }

/* ── Hero ─────────────────────────────────────────────────────── */
.hero { text-align: center; padding: 3rem 0 1rem; }
.hero h1 {
    font-size: 1.75rem;
    font-weight: 700;
    color: #fff;
    letter-spacing: -0.04em;
    margin: 0;
}
.hero .sub {
    font-size: 0.8rem;
    color: #555;
    margin-top: 0.3rem;
    font-family: 'JetBrains Mono', monospace;
    letter-spacing: 0.02em;
}

/* ── Card ─────────────────────────────────────────────────────── */
.card {
    background: #111 !important;
    border: 1px solid #1e1e1e !important;
    border-radius: 14px !important;
    padding: 1.5rem !important;
}
.card:hover { border-color: #282828 !important; }

/* ── Tabs ─────────────────────────────────────────────────────── */
button.tab-nav {
    color: #555 !important;
    font-weight: 500 !important;
    font-size: 0.85rem !important;
    border: none !important;
    background: transparent !important;
    padding: 0.6rem 1rem !important;
    transition: color 0.2s !important;
}
button.tab-nav:hover { color: #999 !important; }
button.tab-nav.selected {
    color: #fff !important;
    border-bottom: 2px solid #fff !important;
}

/* ── Upload zone ──────────────────────────────────────────────── */
.drop-area {
    border: 1px dashed #2a2a2a !important;
    border-radius: 12px !important;
    background: #0d0d0d !important;
    transition: border-color 0.2s !important;
}
.drop-area:hover {
    border-color: #444 !important;
}

/* ── Inputs ───────────────────────────────────────────────────── */
input[type="text"], textarea {
    background: #0d0d0d !important;
    border: 1px solid #1e1e1e !important;
    border-radius: 8px !important;
    color: #ccc !important;
    font-size: 0.85rem !important;
}
input[type="text"]:focus, textarea:focus {
    border-color: #333 !important;
    outline: none !important;
}
input[type="text"]::placeholder {
    color: #3a3a3a !important;
}

label, .label-wrap span {
    color: #666 !important;
    font-weight: 500 !important;
    font-size: 0.8rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.06em !important;
}

input[type="checkbox"] { accent-color: #fff !important; }
input[type="range"] { accent-color: #fff !important; }

/* ── Primary button ───────────────────────────────────────────── */
.go-btn button, button.primary {
    background: #fff !important;
    color: #000 !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    padding: 0.7rem 2rem !important;
    letter-spacing: -0.01em;
    transition: all 0.15s ease !important;
    cursor: pointer !important;
}
.go-btn button:hover, button.primary:hover {
    background: #e0e0e0 !important;
    transform: translateY(-1px) !important;
}
.go-btn button:active, button.primary:active {
    background: #ccc !important;
    transform: translateY(0) !important;
}

/* ── Status ───────────────────────────────────────────────────── */
.status textarea {
    background: transparent !important;
    border: none !important;
    color: #444 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.78rem !important;
    padding: 0.5rem 0 !important;
}

/* ── Preview image ────────────────────────────────────────────── */
.preview {
    border-radius: 12px !important;
    overflow: hidden;
    border: 1px solid #1e1e1e !important;
}
.preview img { border-radius: 10px !important; }

/* ── Download ─────────────────────────────────────────────────── */
.dl a, .dl button {
    color: #888 !important;
    font-size: 0.8rem !important;
}

/* ── Divider ──────────────────────────────────────────────────── */
.sep { border-top: 1px solid #1a1a1a; margin: 1rem 0; }

/* ── Footer ───────────────────────────────────────────────────── */
.foot {
    text-align: center;
    padding: 2rem 0 1rem;
}
.foot p {
    color: #333;
    font-size: 0.7rem;
    margin: 0.1rem 0;
    letter-spacing: 0.03em;
}
.foot strong { color: #555; }
"""

# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

with gr.Blocks(title="Gemini Watermark Remover") as demo:

    gr.HTML("""
        <div class="hero">
            <h1>Gemini Watermark Remover</h1>
            <p class="sub">reverse alpha blending &middot; pixel-perfect &middot; no blur</p>
        </div>
    """)

    with gr.Tabs():

        # ── Remove Watermark ──────────────────────────────
        with gr.TabItem("Remove Watermark"):
            with gr.Group(elem_classes="card"):

                inp_files = gr.File(
                    label="Images",
                    file_count="multiple",
                    file_types=["image"],
                    height=120,
                    elem_classes="drop-area",
                )

                gr.HTML('<div class="sep"></div>')

                with gr.Row():
                    inp_webp = gr.Checkbox(label="WebP", value=False)
                    inp_dir = gr.Textbox(
                        label="Output folder",
                        placeholder="optional",
                        scale=3,
                    )

                btn_go = gr.Button(
                    "Remove Watermark",
                    variant="primary",
                    size="lg",
                    elem_classes="go-btn",
                )

                out_status = gr.Textbox(
                    show_label=False,
                    interactive=False,
                    lines=1,
                    elem_classes="status",
                )

                out_preview = gr.Image(
                    show_label=False,
                    type="pil",
                    height=300,
                    elem_classes="preview",
                )

                out_zip = gr.File(
                    label="Download",
                    elem_classes="dl",
                )

            btn_go.click(
                process_images,
                inputs=[inp_files, inp_webp, inp_dir],
                outputs=[out_preview, out_status, out_zip],
            )

        # ── Convert to WebP ───────────────────────────────
        with gr.TabItem("Convert to WebP"):
            with gr.Group(elem_classes="card"):

                webp_files = gr.File(
                    label="Images",
                    file_count="multiple",
                    file_types=["image"],
                    height=120,
                    elem_classes="drop-area",
                )

                gr.HTML('<div class="sep"></div>')

                with gr.Row():
                    webp_quality = gr.Slider(
                        1, 100, value=90, step=1,
                        label="Quality",
                        scale=2,
                    )
                    webp_lossless = gr.Checkbox(label="Lossless", value=False)
                    webp_dir = gr.Textbox(
                        label="Output folder",
                        placeholder="optional",
                        scale=2,
                    )

                btn_webp = gr.Button(
                    "Convert",
                    variant="primary",
                    size="lg",
                    elem_classes="go-btn",
                )

                webp_status = gr.Textbox(
                    show_label=False,
                    interactive=False,
                    lines=1,
                    elem_classes="status",
                )

                webp_zip = gr.File(
                    label="Download",
                    elem_classes="dl",
                )

            btn_webp.click(
                convert_to_webp,
                inputs=[webp_files, webp_quality, webp_lossless, webp_dir],
                outputs=[webp_status, webp_zip],
            )

    gr.HTML("""
        <div class="foot">
            <p>Auto-detects 48&times;48 or 96&times;96 &middot; visible logo only</p>
            <p>Made by <strong>Chr1stiani</strong></p>
        </div>
    """)


if __name__ == "__main__":
    demo.launch(css=CSS)
