#!/usr/bin/env python3
"""Measure cow body dimensions from pose JSON + optional segmentation.

Usage:
    python measure_cow.py --pose output/img_9_keypoints.json --image ../uploads/img_9.jpg
    python measure_cow.py --pose output/img_9_keypoints.json --image ../uploads/img_9.jpg --segment
    python measure_cow.py --pose output/img_9_keypoints.json --image ../uploads/img_9.jpg \\
        --ref-px 100 --ref-cm 20
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from measurements import (  # noqa: E402
    draw_measurements,
    load_pose_json,
    measure_from_pose_json,
)
from segmentation import overlay_segmentation, seg_for_json, segment_cow  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Cow body measurements from pose JSON.")
    p.add_argument("--pose", type=Path, required=True,
                   help="Path to *_keypoints.json from detect_cow_pose.py")
    p.add_argument("--image", type=Path, default=None,
                   help="Source image (for annotation / segmentation). "
                        "If omitted, tries pose JSON 'source' field.")
    p.add_argument("--output", type=Path, default=SCRIPT_DIR / "output",
                   help="Output directory")
    p.add_argument("--min-conf", type=float, default=0.2,
                   help="Ignore keypoints below this confidence")
    p.add_argument("--segment", action="store_true",
                   help="Run optional local YOLO segmentation")
    p.add_argument("--seg-model", type=Path, default=None,
                   help="Path to local YOLO-seg .pt/.onnx (optional)")
    p.add_argument("--scale-cm-per-px", type=float, default=None,
                   help="Direct scale factor (cm per pixel)")
    p.add_argument("--ref-px", type=float, default=None,
                   help="Reference marker length in pixels")
    p.add_argument("--ref-cm", type=float, default=None,
                   help="Reference marker known length in centimeters")
    p.add_argument("--no-annotate", action="store_true",
                   help="Skip saving annotated measurement image")
    return p


def resolve_image(args_image: Path | None, pose_data: dict) -> Path | None:
    if args_image is not None:
        return args_image
    src = pose_data.get("source")
    if not src:
        return None
    p = Path(src)
    if p.is_file():
        return p
    # Try relative to SCRIPT_DIR
    alt = (SCRIPT_DIR / src).resolve()
    if alt.is_file():
        return alt
    return None


def stem_from_pose(pose_path: Path) -> str:
    name = pose_path.stem
    if name.endswith("_keypoints"):
        name = name[: -len("_keypoints")]
    return name


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.pose.is_file():
        print(f"Pose JSON not found: {args.pose}", file=sys.stderr)
        return 1

    pose_data = load_pose_json(args.pose)
    image_path = resolve_image(args.image, pose_data)

    m = measure_from_pose_json(
        args.pose,
        min_keypoint_conf=args.min_conf,
        scale_cm_per_px=args.scale_cm_per_px,
        reference_length_px=args.ref_px,
        reference_length_cm=args.ref_cm,
    )

    seg_info = None
    annotated = None
    if image_path is not None and image_path.is_file():
        img = cv2.imread(str(image_path))
        if img is None:
            print(f"Warning: cannot read image {image_path}", file=sys.stderr)
        else:
            if args.segment:
                seg_info = segment_cow(img, m.bbox, model_path=args.seg_model)
                if seg_info is not None:
                    m.segmentation = seg_for_json(seg_info)
                    img = overlay_segmentation(img, seg_info)
                else:
                    print("Warning: no cow segmentation mask found", file=sys.stderr)

            unit = "cm" if m.centimeters else "px"
            annotated = draw_measurements(img, m, unit=unit)

    args.output.mkdir(parents=True, exist_ok=True)
    stem = stem_from_pose(args.pose)

    out_json = args.output / f"{stem}_measurements.json"
    payload = {
        "source_pose": str(args.pose),
        "source_image": str(image_path) if image_path else None,
        "primary_cow": m.to_dict(),
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved measurements: {out_json}")

    if annotated is not None and not args.no_annotate:
        out_img = args.output / f"{stem}_measurements.png"
        cv2.imwrite(str(out_img), annotated)
        print(f"Saved annotation: {out_img}")

    # Console summary
    print(f"\nPrimary cow #{m.cow_id}  bbox_conf={m.bbox_confidence:.3f}")
    for k, v in m.pixels.items():
        print(f"  {k}: {v:.2f} px" if v is not None else f"  {k}: N/A")
    print("Ratios:")
    for k, v in m.ratios.items():
        print(f"  {k}: {v:.4f}" if v is not None else f"  {k}: N/A")
    if m.centimeters:
        print("Centimeters:")
        for k, v in m.centimeters.items():
            print(f"  {k}: {v:.2f} cm" if v is not None else f"  {k}: N/A")
    if m.segmentation:
        print("Segmentation:")
        for k in ("body_pixel_area", "torso_pixel_area", "body_perimeter_px"):
            print(f"  {k}: {m.segmentation.get(k)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
