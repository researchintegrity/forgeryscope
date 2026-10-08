"""
Entry point of the Forgeryscope tool image.

Detects duplicated panels, panel regions and western blot lanes in one
figure, and writes into the output directory:

- {name}_result.json: verdict, detected panels and every detection (always);
- {name}_matches.png: the figure with each group of duplicated regions
  filled, outlined and numbered;
- {name}_clusters.png: the duplicated regions on black, one colour per group.

The two images are written only when duplication is found. Progress is
printed on stdout as "[STATUS] ..." lines, errors on stderr as "[ERROR] ...".

Exit codes: 0 done (whether or not duplication was found), 1 invalid input,
2 detection failed, 3 timed out.
"""
import argparse
import json
import signal
import sys
import time
import traceback
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageOps

EXIT_INVALID_INPUT = 1
EXIT_FAILED = 2
EXIT_TIMEOUT = 3

DEFAULT_MAX_PIXELS = 200_000_000

# One colour (RGB) per group of duplicated regions
PALETTE = [
    (230, 25, 75), (0, 130, 200), (60, 180, 75), (245, 130, 48), (145, 30, 180),
    (70, 240, 240), (240, 50, 230), (210, 245, 60), (0, 128, 128), (170, 110, 40),
]
PANEL_COLOUR = (150, 150, 150)
FILL_OPACITY = 0.35


class InputError(Exception):
    """The input image is missing, unreadable or too large."""


class Timeout(Exception):
    pass


def status(message: str) -> None:
    print(f"[STATUS] {message}", flush=True)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect duplicated content in a scientific figure.")
    parser.add_argument("--input", "-i", required=True, help="Image to analyse")
    parser.add_argument("--output", "-o", required=True, help="Directory for the results")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto",
                        help="Where to run the models (auto: a GPU when available)")
    parser.add_argument("--timeout", type=int, default=0,
                        help="Stop after this many seconds (0: no limit)")
    parser.add_argument("--max-pixels", type=int, default=DEFAULT_MAX_PIXELS,
                        help="Largest image accepted (width x height)")
    return parser.parse_args(argv)


def to_rgb(image: Image.Image) -> np.ndarray:
    """An RGB uint8 array of the image, upright as image viewers show it."""
    image = ImageOps.exif_transpose(image)
    if image.mode.startswith("I") or image.mode == "F":
        # More than 8 bits per pixel: convert("RGB") would clip it to white
        values = np.asarray(image, dtype=np.float64)
        peak = float(values.max(initial=0))
        if image.mode.startswith("I;16"):
            scale = 65535.0
        else:
            scale = 255.0 if peak <= 255 else 65535.0 if peak <= 65535 else peak
        gray = np.clip(values * (255.0 / scale), 0, 255).round().astype(np.uint8)
        return np.stack([gray] * 3, axis=-1)
    return np.array(image.convert("RGB"))


def load_image(path: Path, max_pixels: int) -> np.ndarray:
    """The image as an RGB uint8 array."""
    if not path.is_file():
        raise InputError(f"Input image not found: {path}")
    try:
        import pi_heif
        pi_heif.register_heif_opener()
    except ImportError:
        pass
    Image.MAX_IMAGE_PIXELS = None  # checked below, before decoding
    try:
        with Image.open(path) as image:
            if image.width * image.height > max_pixels:
                raise InputError(
                    f"Image too large: {image.width}x{image.height} pixels (limit {max_pixels})"
                )
            return to_rgb(image)
    except InputError:
        raise
    except (OSError, ValueError, SyntaxError) as e:
        raise InputError(f"Cannot read image {path.name}: {e}") from e


def resolve_device(choice: str) -> str:
    import torch

    if choice == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if choice == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested but no GPU is available")
    return choice


def colour(index: int) -> Tuple[int, int, int]:
    return PALETTE[index % len(PALETTE)]


def hex_colour(index: int) -> str:
    return "#%02x%02x%02x" % colour(index)


def _draw_label(img: np.ndarray, text: str, x: int, y: int, fill, thickness: int) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.5 * thickness
    (width, height), baseline = cv2.getTextSize(text, font, scale, thickness)
    pad = 2 * thickness
    x = int(min(max(0, x), img.shape[1] - width - 2 * pad))
    y = int(min(max(0, y), img.shape[0] - height - baseline - 2 * pad))
    cv2.rectangle(img, (x, y), (x + width + 2 * pad, y + height + baseline + 2 * pad), fill, -1)
    cv2.putText(img, text, (x + pad, y + pad + height), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)


def render_matches(img: np.ndarray, panels, detections) -> np.ndarray:
    """The figure with detected panels boxed in grey and each group of
    duplicated regions filled, outlined and numbered in its colour."""
    height, width = img.shape[:2]
    thickness = max(1, round(max(height, width) / 600))
    out = np.ascontiguousarray(img.copy())
    for panel in panels:
        x0, y0, x1, y1 = (int(round(v)) for v in panel.bbox)
        cv2.rectangle(out, (x0, y0), (x1, y1), PANEL_COLOUR, thickness)

    fill = out.copy()
    for k, detection in enumerate(detections):
        fill[detection.mask] = colour(k)
    out = cv2.addWeighted(fill, FILL_OPACITY, out, 1 - FILL_OPACITY, 0)

    for k, detection in enumerate(detections):
        contours, _ = cv2.findContours(detection.mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, colour(k), 2 * thickness)
        for contour in contours:
            x, y, _, _ = cv2.boundingRect(contour)
            _draw_label(out, str(k + 1), x, y, colour(k), thickness)
    return out


def render_clusters(shape, detections) -> np.ndarray:
    """The duplicated regions on black, one colour per group."""
    out = np.zeros((shape[0], shape[1], 3), dtype=np.uint8)
    for k, detection in enumerate(detections):
        out[detection.mask] = colour(k)
    return out


def _detection_entry(k: int, detection) -> dict:
    ys, xs = np.nonzero(detection.mask)
    bbox = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1] if len(xs) else None
    return {
        "id": k + 1,
        "colour": hex_colour(k),
        "level": detection.level,
        "panel_type": detection.panel_type,
        "panel_ids": detection.panel_ids,
        "similarity": round(detection.similarity, 4),
        "bbox": bbox,
        "area": int(detection.mask.sum()),
        "pairs": [
            {
                "panel_ids": list(pair.panel_ids),
                "embedding_score": round(pair.embedding_score, 4),
                "keypoint_match": pair.keypoint_match,
                "inliers": pair.inliers,
                "total_matches": pair.total_matches,
                "match_score": round(pair.match_score, 4),
                "regions": [[list(point) for point in polygon] for polygon in pair.regions],
            }
            for pair in detection.pairs
        ],
        "lane_matches": [
            {
                "panel_ids": list(match.panel_ids),
                "similarity": round(match.similarity, 4),
                "lane_boxes": [list(box) for box in match.lane_boxes],
            }
            for match in detection.lane_matches
        ],
    }


def build_report(image_name: str, shape, result, device: str, config, outputs: dict, timings: dict) -> dict:
    return {
        "tool": "forgeryscope",
        "version": version("forgeryscope"),
        "image": {"filename": image_name, "width": int(shape[1]), "height": int(shape[0])},
        "device": device,
        "verdict": "authentic" if result.authentic else "duplicated",
        "panels": [
            {"id": p.id, "label": p.label, "confidence": round(p.confidence, 3), "bbox": list(p.bbox)}
            for p in result.panels
        ],
        "detections": [_detection_entry(k, d) for k, d in enumerate(result.detections)],
        "outputs": outputs,
        "parameters": asdict(config),
        "timings": timings,
    }


def run(args: argparse.Namespace) -> int:
    input_path = Path(args.input)
    output_dir = Path(args.output)

    img = load_image(input_path, args.max_pixels)
    output_dir.mkdir(parents=True, exist_ok=True)

    status("Loading models")
    started = time.monotonic()
    from forgeryscope.pipeline import DuplicationDetector

    device = resolve_device(args.device)
    detector = DuplicationDetector(device=device)
    loaded = time.monotonic()

    result = detector.detect(img, progress=status)
    detected = time.monotonic()

    status("Writing results")
    base = input_path.stem
    outputs = {"matches_image": None, "clusters_image": None}
    if result.detections:
        outputs["matches_image"] = f"{base}_matches.png"
        outputs["clusters_image"] = f"{base}_clusters.png"
        Image.fromarray(render_matches(img, result.panels, result.detections)).save(output_dir / outputs["matches_image"])
        Image.fromarray(render_clusters(img.shape, result.detections)).save(output_dir / outputs["clusters_image"])

    timings = {"load_models": round(loaded - started, 2), "detection": round(detected - loaded, 2)}
    report = build_report(input_path.name, img.shape, result, device, detector.config, outputs, timings)
    with open(output_dir / f"{base}_result.json", "w") as handle:
        json.dump(report, handle, indent=2)

    if result.authentic:
        status("Done: no duplication found")
    else:
        status(f"Done: {len(result.detections)} group(s) of duplicated regions found")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    def on_timeout(signum, frame):
        raise Timeout()

    if args.timeout > 0:
        signal.signal(signal.SIGALRM, on_timeout)
        signal.alarm(args.timeout)
    try:
        return run(args)
    except InputError as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        return EXIT_INVALID_INPUT
    except Timeout:
        print(f"[ERROR] Timed out after {args.timeout} seconds", file=sys.stderr)
        return EXIT_TIMEOUT
    except Exception as e:
        traceback.print_exc()
        print(f"[ERROR] Detection failed: {e}", file=sys.stderr)
        return EXIT_FAILED
    finally:
        signal.alarm(0)


if __name__ == "__main__":
    sys.exit(main())
