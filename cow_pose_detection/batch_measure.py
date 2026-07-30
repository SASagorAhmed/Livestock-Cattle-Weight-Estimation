#!/usr/bin/env python3
"""Batch cow body measurements for a folder of images.

For each image:
  1. Run pose detection if keypoints JSON is missing (optional --run-pose)
  2. Measure primary cow from pose JSON
  3. Append a row to a CSV summary

Usage:
    python batch_measure.py --images ../uploads --output output
    python batch_measure.py --images ../uploads --run-pose --segment
    python batch_measure.py --pose-dir output --images ../uploads --csv output/batch_measurements.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from measurements import draw_measurements, measure_from_pose_json  # noqa: E402
from segmentation import overlay_segmentation, seg_for_json, segment_cow  # noqa: E402

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}

CSV_COLUMNS = [
    "image",
    "pose_json",
    "cow_id",
    "bbox_confidence",
    "shoulder_width_px",
    "hip_width_px",
    "body_length_px",
    "body_height_px",
    "left_front_leg_length_px",
    "right_front_leg_length_px",
    "left_back_leg_length_px",
    "right_back_leg_length_px",
    "chest_depth_proxy_px",
    "torso_diagonal_px",
    "body_length_over_body_height",
    "shoulder_width_over_body_length",
    "hip_width_over_body_length",
    "chest_depth_over_body_height",
    "scale_cm_per_px",
    "body_length_cm",
    "body_height_cm",
    "shoulder_width_cm",
    "hip_width_cm",
    "body_pixel_area",
    "torso_pixel_area",
    "body_perimeter_px",
]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Batch cow body measurements.")
    p.add_argument("--images", type=Path, required=True,
                   help="Folder containing cow images")
    p.add_argument("--pose-dir", type=Path, default=None,
                   help="Folder with existing *_keypoints.json (default: --output)")
    p.add_argument("--output", type=Path, default=SCRIPT_DIR / "output",
                   help="Output folder for JSON/PNG/CSV")
    p.add_argument("--csv", type=Path, default=None,
                   help="CSV path (default: <output>/batch_measurements.csv)")
    p.add_argument("--run-pose", action="store_true",
                   help="Run detect_cow_pose.py for images missing keypoints JSON")
    p.add_argument("--segment", action="store_true",
                   help="Optional local YOLO segmentation")
    p.add_argument("--seg-model", type=Path, default=None)
    p.add_argument("--min-conf", type=float, default=0.2)
    p.add_argument("--scale-cm-per-px", type=float, default=None)
    p.add_argument("--ref-px", type=float, default=None)
    p.add_argument("--ref-cm", type=float, default=None)
    return p


def find_pose_json(pose_dir: Path, stem: str) -> Path | None:
    candidate = pose_dir / f"{stem}_keypoints.json"
    return candidate if candidate.is_file() else None


def run_pose_for_image(image_path: Path, output_dir: Path) -> Path | None:
    """Invoke pose detector for one image; return keypoints JSON path."""
    from detector import CowPoseDetector

    detector = CowPoseDetector()
    detector.process_image(
        image_path=image_path,
        output_dir=output_dir,
        save_img=True,
        save_json=True,
        show=False,
        draw_labels=True,
    )
    return find_pose_json(output_dir, image_path.stem)


def row_from_measurement(
    image_path: Path,
    pose_path: Path,
    m,
) -> dict:
    px = m.pixels
    ratios = m.ratios
    cm = m.centimeters or {}
    seg = m.segmentation or {}
    return {
        "image": str(image_path),
        "pose_json": str(pose_path),
        "cow_id": m.cow_id,
        "bbox_confidence": m.bbox_confidence,
        "shoulder_width_px": px.get("shoulder_width"),
        "hip_width_px": px.get("hip_width"),
        "body_length_px": px.get("body_length"),
        "body_height_px": px.get("body_height"),
        "left_front_leg_length_px": px.get("left_front_leg_length"),
        "right_front_leg_length_px": px.get("right_front_leg_length"),
        "left_back_leg_length_px": px.get("left_back_leg_length"),
        "right_back_leg_length_px": px.get("right_back_leg_length"),
        "chest_depth_proxy_px": px.get("chest_depth_proxy"),
        "torso_diagonal_px": px.get("torso_diagonal"),
        "body_length_over_body_height": ratios.get("body_length_over_body_height"),
        "shoulder_width_over_body_length": ratios.get("shoulder_width_over_body_length"),
        "hip_width_over_body_length": ratios.get("hip_width_over_body_length"),
        "chest_depth_over_body_height": ratios.get("chest_depth_over_body_height"),
        "scale_cm_per_px": m.scale_cm_per_px,
        "body_length_cm": cm.get("body_length"),
        "body_height_cm": cm.get("body_height"),
        "shoulder_width_cm": cm.get("shoulder_width"),
        "hip_width_cm": cm.get("hip_width"),
        "body_pixel_area": seg.get("body_pixel_area"),
        "torso_pixel_area": seg.get("torso_pixel_area"),
        "body_perimeter_px": seg.get("body_perimeter_px"),
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    images_dir = args.images
    if not images_dir.is_dir():
        print(f"Images folder not found: {images_dir}", file=sys.stderr)
        return 1

    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)
    pose_dir = args.pose_dir or output_dir
    csv_path = args.csv or (output_dir / "batch_measurements.csv")

    images = sorted(
        p for p in images_dir.iterdir()
        if p.suffix.lower() in IMAGE_EXTS and p.is_file()
    )
    if not images:
        print(f"No images found in {images_dir}", file=sys.stderr)
        return 1

    rows: list[dict] = []
    pose_detector = None

    for img_path in images:
        stem = img_path.stem
        print(f"\n=== {img_path.name} ===")
        pose_path = find_pose_json(pose_dir, stem)
        if pose_path is None and args.run_pose:
            print("  Running pose detection...")
            if pose_detector is None:
                from detector import CowPoseDetector
                pose_detector = CowPoseDetector()
            pose_detector.process_image(
                image_path=img_path,
                output_dir=pose_dir,
                save_img=True,
                save_json=True,
                show=False,
            )
            pose_path = find_pose_json(pose_dir, stem)

        if pose_path is None:
            print(f"  SKIP: no keypoints JSON (use --run-pose)")
            continue

        try:
            m = measure_from_pose_json(
                pose_path,
                min_keypoint_conf=args.min_conf,
                scale_cm_per_px=args.scale_cm_per_px,
                reference_length_px=args.ref_px,
                reference_length_cm=args.ref_cm,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"  ERROR measuring: {exc}")
            continue

        img = cv2.imread(str(img_path))
        if img is not None:
            if args.segment:
                seg = segment_cow(img, m.bbox, model_path=args.seg_model)
                if seg is not None:
                    m.segmentation = seg_for_json(seg)
                    img = overlay_segmentation(img, seg)
            unit = "cm" if m.centimeters else "px"
            annotated = draw_measurements(img, m, unit=unit)
            out_png = output_dir / f"{stem}_measurements.png"
            cv2.imwrite(str(out_png), annotated)

        out_json = output_dir / f"{stem}_measurements.json"
        payload = {
            "source_pose": str(pose_path),
            "source_image": str(img_path),
            "primary_cow": m.to_dict(),
        }
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        rows.append(row_from_measurement(img_path, pose_path, m))
        bl = m.pixels.get("body_length")
        print(f"  body_length={bl:.1f}px" if bl else "  body_length=N/A")

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"\nWrote {len(rows)} rows -> {csv_path}")
    return 0 if rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
