#!/usr/bin/env python3
"""
CLI for removing the Gemini AI Photo watermark.

Usage:
    python cli.py input.png                # → input_clean.png
    python cli.py input.png -o output.png  # custom output path
"""

import argparse
from pathlib import Path
from remover import process_image


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Remove Gemini AI Photo watermark from images.",
    )
    parser.add_argument("input", type=Path, help="Watermarked image path.")
    parser.add_argument("-o", "--output", type=Path, default=None, help="Output path.")
    args = parser.parse_args()

    if not args.input.exists():
        print(f"Error: {args.input} not found")
        raise SystemExit(1)

    output = process_image(args.input, args.output)
    print(f"Done → {output}")


if __name__ == "__main__":
    main()
