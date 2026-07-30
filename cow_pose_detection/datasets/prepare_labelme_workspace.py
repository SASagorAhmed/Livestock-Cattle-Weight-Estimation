#!/usr/bin/env python3
"""Create a LabelMe workspace for the 4 target keypoints.

Copies side-view images and writes LabelMe JSON templates.
If existing LabelMe files have mappable aliases, those points are pre-filled.

Usage:
  python prepare_labelme_workspace.py
  python prepare_labelme_workspace.py --root PATH
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from four_point_schema import KEYPOINT_DESCRIPTIONS, KEYPOINT_NAMES, canonicalize_label

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def is_side(path: Path) -> bool:
    name = path.name.lower()
    parts = "/".join(path.parts).lower()
    return any(k in name or k in parts for k in ("side", "lateral", "侧面"))


def load_existing_points(json_path: Path | None) -> dict[str, list[float]]:
    if json_path is None or not json_path.exists():
        return {}
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    out: dict[str, list[float]] = {}
    for shape in data.get("shapes") or []:
        if shape.get("shape_type") not in (None, "point"):
            continue
        canon = canonicalize_label(str(shape.get("label") or ""))
        pts = shape.get("points") or []
        if canon and pts and len(pts[0]) >= 2:
            out[canon] = [float(pts[0][0]), float(pts[0][1])]
    return out


def make_labelme(image_name: str, width: int, height: int, points: dict[str, list[float]]) -> dict:
    shapes = []
    for name in KEYPOINT_NAMES:
        if name not in points:
            continue
        shapes.append({
            "label": name,
            "points": [points[name]],
            "group_id": None,
            "description": KEYPOINT_DESCRIPTIONS[name],
            "shape_type": "point",
            "flags": {},
        })
    return {
        "version": "5.0.1",
        "flags": {},
        "shapes": shapes,
        "imagePath": image_name,
        "imageData": None,
        "imageHeight": height,
        "imageWidth": width,
        "_target_keypoints": KEYPOINT_NAMES,
        "_instructions": KEYPOINT_DESCRIPTIONS,
    }


def image_size(path: Path) -> tuple[int, int]:
    try:
        import cv2
        img = cv2.imread(str(path))
        if img is None:
            return 0, 0
        h, w = img.shape[:2]
        return int(w), int(h)
    except Exception:
        return 0, 0


def main() -> int:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=script_dir / "horqin_h2s22wr5py" / "raw")
    parser.add_argument("--out", type=Path, default=script_dir / "horqin_h2s22wr5py" / "labelme_workspace")
    parser.add_argument("--prefer-side", action="store_true", default=True)
    parser.add_argument("--include-back", action="store_true", default=False)
    args = parser.parse_args()

    if not args.root.exists():
        print("Raw dataset not found at", args.root)
        print("Download DOI 10.17632/h2s22wr5py.2 and extract into that folder.")
        return 1

    images = [p for p in args.root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    if args.prefer_side and not args.include_back:
        side = [p for p in images if is_side(p)]
        if side:
            images = side
        else:
            print("WARNING: could not detect side-view filenames; including all images.")
            print("Prefer side views only for the diagonal formula keypoints.")

    json_by_stem: dict[str, Path] = {}
    for p in args.root.rglob("*.json"):
        json_by_stem[p.stem] = p

    args.out.mkdir(parents=True, exist_ok=True)
    labels_dir = args.out / "labels"
    images_dir = args.out / "images"
    labels_dir.mkdir(exist_ok=True)
    images_dir.mkdir(exist_ok=True)

    (args.out / "labels.txt").write_text("\n".join(KEYPOINT_NAMES) + "\n", encoding="utf-8")

    n = 0
    for src in sorted(images):
        dst_img = images_dir / src.name
        if not dst_img.exists():
            shutil.copy2(src, dst_img)
        w, h = image_size(dst_img)
        existing = load_existing_points(json_by_stem.get(src.stem))
        sibling = src.with_suffix(".json")
        if sibling.exists():
            existing.update(load_existing_points(sibling))

        payload = make_labelme(src.name, w, h, existing)
        meta = {
            "image": src.name,
            "placed": sorted(existing.keys()),
            "still_needed": [k for k in KEYPOINT_NAMES if k not in existing],
            "descriptions": KEYPOINT_DESCRIPTIONS,
        }
        (labels_dir / f"{src.stem}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        (labels_dir / f"{src.stem}.meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        # Also place json beside image for LabelMe convenience
        (images_dir / f"{src.stem}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        n += 1

    (args.out / "HOW_TO_RELABEL.md").write_text(
        "\n".join([
            "# Relabel 4 keypoints in LabelMe",
            "",
            "1. Install: `pip install labelme`",
            "2. Run:",
            f"   `labelme \"{images_dir}\" --labels \"{args.out / 'labels.txt'}\" --nodata`",
            "3. Place exactly these points on each side-view image:",
            *[f"   - `{k}`: {KEYPOINT_DESCRIPTIONS[k]}" for k in KEYPOINT_NAMES],
            "4. Save each JSON (overwrite the template beside the image).",
            "5. Convert: `python labelme_to_yolo_pose.py`",
            "",
            "Horqin original OBL/WH/HG/HL labels usually do NOT match these four points.",
            "Correct them manually before training.",
        ]) + "\n",
        encoding="utf-8",
    )

    print(f"Prepared {n} images in {args.out}")
    print("Open HOW_TO_RELABEL.md and launch LabelMe.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
