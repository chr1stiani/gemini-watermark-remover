# Gemini Watermark Remover

Mathematically precise removal of the Google Gemini AI Photo watermark from generated images. Zero blur, zero quality loss — pixel-perfect restoration with a maximum error of **±1 per channel** (imperceptible 8-bit quantization).

## How It Works

Google Gemini stamps every generated image with a semi-transparent four-pointed star in the bottom-right corner using alpha compositing:

```
watermarked = α × 255 + (1 - α) × original
```

This tool reverses the formula using a pre-extracted alpha map:

```
original = (watermarked - α × 255) / (1 - α)
```

The alpha map is **fixed** across all Gemini-generated images of the same size class — no calibration needed. Just drop in your image and get a clean result.

### Watermark Variants

| Image size | Logo size | Margin | Alpha map |
|---|---|---|---|
| Either dimension ≤ 1024px | 48×48 | 32px | `bg_48.png` |
| Both dimensions > 1024px | 96×96 | 64px | `bg_96.png` |

> **Note:** This removes only the visible star watermark. Google's invisible SynthID watermark cannot be removed as it is embedded into the pixel generation process itself.

## Features

- **One-click removal** — no setup, no calibration, no template extraction
- **Mathematically exact** — reverse alpha blending, not AI inpainting
- **Batch processing** — process multiple images at once
- **Custom output directory** — choose where to save results, or download as ZIP
- **WebP conversion** — built-in tab for converting images to WebP format
- **Web UI & CLI** — use whichever you prefer

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Web UI
python app.py
# Opens at http://127.0.0.1:7860

# CLI
python cli.py image.png                    # → image_clean.png
python cli.py image.png -o output.png      # custom output path
```

## Requirements

- Python 3.10+
- OpenCV
- NumPy
- Pillow
- Gradio (for web UI only)

```bash
pip install -r requirements.txt
```

## Project Structure

```
├── app.py              # Gradio web UI (single, batch, WebP conversion)
├── cli.py              # Command-line interface
├── remover.py          # Core removal engine
├── requirements.txt    # Python dependencies
├── templates/
│   ├── bg_48.png       # Pre-extracted alpha map (48×48)
│   └── bg_96.png       # Pre-extracted alpha map (96×96)
└── test_remover.py     # Tests
```

## How the Alpha Maps Were Obtained

The alpha maps (`bg_48.png`, `bg_96.png`) are captured by generating a solid black image through Gemini. On a black background (`bg = 0`), the watermarked pixel value directly encodes `α × 255`:

```
watermarked = α × 255 + (1 - α) × 0 = α × 255
```

The max RGB channel is extracted and normalized to `[0, 1]` to produce the alpha map. Since Gemini uses the same watermark pattern for all images of a given size class, this only needs to be done once.

## Credits

Made by **Chr1stiani**

Alpha maps sourced from the open-source community — credit to [GargantuaX/gemini-watermark-remover](https://github.com/GargantuaX/gemini-watermark-remover) and [Pilio](https://pilio.ai/blog/gemini-watermark-how-it-works) for the reverse-engineering research.

## License

MIT
