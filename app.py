"""
Gradio web UI for removing the Gemini AI Photo watermark.

One-click removal using pre-extracted alpha maps – no calibration needed.
Supports multi-file batch processing and custom output directory.
"""

import shutil
import tempfile
import zipfile
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
from PIL import Image

from remover import remove_watermark, _pick_variant, _watermark_box


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
# Single image (quick preview)
# ---------------------------------------------------------------------------

def process_single(input_image: Image.Image) -> Image.Image:
    if input_image is None:
        raise gr.Error("Nahraj obrázek.")
    bgr = _pil_to_cv2(input_image)
    result = remove_watermark(bgr)
    return _cv2_to_pil(result)


def preview_single(input_image: Image.Image) -> Image.Image | None:
    if input_image is None:
        return None
    bgr = _pil_to_cv2(input_image)
    h, w = bgr.shape[:2]
    variant = _pick_variant(w, h)
    x1, y1, x2, y2 = _watermark_box(w, h, variant)
    prev = bgr.copy()
    cv2.rectangle(prev, (x1, y1), (x2, y2), (0, 0, 255), 2)
    return _cv2_to_pil(prev)


# ---------------------------------------------------------------------------
# Batch processing
# ---------------------------------------------------------------------------

def process_batch(
    files: list[tempfile.NamedTemporaryFile],
    output_dir: str,
) -> tuple[str, str | None]:
    """
    Process multiple files. Save to output_dir if provided, otherwise
    use a temp dir and return a ZIP for download.
    """
    if not files:
        raise gr.Error("Nahraj alespoň jeden obrázek.")

    output_dir = output_dir.strip() if output_dir else ""
    use_custom_dir = bool(output_dir)

    if use_custom_dir:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
    else:
        out_path = Path(tempfile.mkdtemp(prefix="gemini_clean_"))

    results = []
    errors = []

    for f in files:
        src = Path(f.name) if hasattr(f, "name") else Path(f)
        try:
            img = cv2.imread(str(src), cv2.IMREAD_COLOR)
            if img is None:
                errors.append(f"{src.name}: nelze načíst")
                continue

            cleaned = remove_watermark(img)
            dest = out_path / (src.stem + "_clean" + src.suffix)
            cv2.imwrite(str(dest), cleaned)
            results.append(dest.name)
        except Exception as e:
            errors.append(f"{src.name}: {e}")

    # Build status message
    status_lines = [f"Zpracováno: {len(results)}/{len(files)}"]
    if use_custom_dir:
        status_lines.append(f"Uloženo do: {out_path}")
    if errors:
        status_lines.append("Chyby:")
        status_lines.extend(f"  - {e}" for e in errors)
    status = "\n".join(status_lines)

    # If no custom dir, create a ZIP for download
    zip_path = None
    if not use_custom_dir and results:
        zip_file = out_path / "gemini_clean.zip"
        with zipfile.ZipFile(zip_file, "w", zipfile.ZIP_DEFLATED) as zf:
            for name in results:
                zf.write(out_path / name, name)
        zip_path = str(zip_file)

    return status, zip_path


# ---------------------------------------------------------------------------
# WebP conversion
# ---------------------------------------------------------------------------

def convert_to_webp(
    files: list[tempfile.NamedTemporaryFile],
    quality: int,
    lossless: bool,
    output_dir: str,
) -> tuple[str, str | None]:
    if not files:
        raise gr.Error("Nahraj alespoň jeden obrázek.")

    output_dir = output_dir.strip() if output_dir else ""
    use_custom_dir = bool(output_dir)

    if use_custom_dir:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
    else:
        out_path = Path(tempfile.mkdtemp(prefix="webp_"))

    results = []
    errors = []

    for f in files:
        src = Path(f.name) if hasattr(f, "name") else Path(f)
        try:
            img = Image.open(src)
            dest = out_path / (src.stem + ".webp")
            img.save(str(dest), "WEBP", quality=quality, lossless=lossless)
            results.append(dest.name)
        except Exception as e:
            errors.append(f"{src.name}: {e}")

    status_lines = [f"Převedeno: {len(results)}/{len(files)}"]
    if use_custom_dir:
        status_lines.append(f"Uloženo do: {out_path}")
    if errors:
        status_lines.append("Chyby:")
        status_lines.extend(f"  - {e}" for e in errors)
    status = "\n".join(status_lines)

    zip_path = None
    if not use_custom_dir and results:
        zip_file = out_path / "webp_images.zip"
        with zipfile.ZipFile(zip_file, "w", zipfile.ZIP_DEFLATED) as zf:
            for name in results:
                zf.write(out_path / name, name)
        zip_path = str(zip_file)

    return status, zip_path


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

with gr.Blocks(title="Gemini Logo Remover") as demo:
    gr.Markdown(
        """
        # Gemini AI Photo – Odstranění watermarku

        Matematicky přesné odstranění (±1 pixel). Žádné rozmazání, žádná kalibrace.

        `original = (watermarked - α × 255) / (1 - α)`
        """
    )

    with gr.Tabs():
        # ---- Single image ----
        with gr.TabItem("Jeden obrázek"):
            with gr.Row():
                with gr.Column():
                    inp_single = gr.Image(label="Obrázek s watermarkem", type="pil")
                    with gr.Row():
                        btn_preview = gr.Button("Náhled oblasti", variant="secondary")
                        btn_single = gr.Button("Odstranit watermark", variant="primary")
                with gr.Column():
                    out_single = gr.Image(label="Výsledek", type="pil")

            btn_preview.click(preview_single, inputs=inp_single, outputs=out_single)
            btn_single.click(process_single, inputs=inp_single, outputs=out_single)

        # ---- Batch ----
        with gr.TabItem("Více obrázků (batch)"):
            gr.Markdown(
                """
                Nahraj více obrázků najednou. Výsledky se uloží do zvoleného adresáře,
                nebo si je stáhneš jako ZIP.
                """
            )
            with gr.Column():
                inp_files = gr.File(
                    label="Obrázky s watermarkem",
                    file_count="multiple",
                    file_types=["image"],
                )
                inp_output_dir = gr.Textbox(
                    label="Výstupní adresář (volitelné)",
                    placeholder="/home/chris/cleaned_images",
                    info="Pokud nevyplníš, dostaneš ZIP ke stažení.",
                )
                btn_batch = gr.Button("Zpracovat vše", variant="primary")
                out_status = gr.Textbox(label="Stav", interactive=False, lines=5)
                out_zip = gr.File(label="Stáhnout výsledky (ZIP)")

            btn_batch.click(
                process_batch,
                inputs=[inp_files, inp_output_dir],
                outputs=[out_status, out_zip],
            )

        # ---- Convert to WebP ----
        with gr.TabItem("Převod do WebP"):
            gr.Markdown(
                """
                Převeď obrázky (jeden nebo více) do formátu **WebP**.
                Můžeš nastavit kvalitu (1–100) nebo zvolit bezztrátový režim.
                """
            )
            with gr.Column():
                webp_files = gr.File(
                    label="Obrázky k převodu",
                    file_count="multiple",
                    file_types=["image"],
                )
                with gr.Row():
                    webp_quality = gr.Slider(
                        1, 100, value=90, step=1,
                        label="Kvalita",
                        info="1 = nejmenší soubor, 100 = nejlepší kvalita",
                    )
                    webp_lossless = gr.Checkbox(
                        label="Bezztrátový (lossless)",
                        value=False,
                    )
                webp_output_dir = gr.Textbox(
                    label="Výstupní adresář (volitelné)",
                    placeholder="/home/chris/webp_images",
                    info="Pokud nevyplníš, dostaneš ZIP ke stažení.",
                )
                btn_webp = gr.Button("Převést do WebP", variant="primary")
                webp_status = gr.Textbox(label="Stav", interactive=False, lines=5)
                webp_zip = gr.File(label="Stáhnout výsledky (ZIP)")

            btn_webp.click(
                convert_to_webp,
                inputs=[webp_files, webp_quality, webp_lossless, webp_output_dir],
                outputs=[webp_status, webp_zip],
            )

    gr.Markdown(
        """
        ---
        - Automatická detekce: 48×48 (≤1024px) nebo 96×96 (>1024px)
        - Odstraní pouze viditelné logo. SynthID (neviditelný) nelze odstranit.

        Made by **Chr1stiani**
        """
    )


if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft())
