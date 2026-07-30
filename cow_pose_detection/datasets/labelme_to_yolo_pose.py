#!/usr/bin/env python3
"""Convert LabelMe 4-keypoint annotations to YOLO-pose dataset layout.

Requires all 4 keypoints present and visible on each training image.

Usage:
  python labelme_to_yolo_pose.py
  python labelme_to_yolo_pose.py --workspace PATH --val-ratio 0.2
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from pathlib import Path

from four_point_schema import (
    CLASS_NAME,
    FLIP_IDX,
    KEYPOINT_NAMES,
    NUM_KEYPOINTS,
    SKELETON,
    canonicalize_label,
)

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def parse_labelme(path: Path) -> dict[str, tuple[float, float]] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    pts: dict[str, tuple[float, float]] = {}
    for shape in data.get("shapes") or []:
        if shape.get("shape_type") not in (None, "point"):
            continue
        canon = canonicalize_label(str(shape.get("label") or ""))
        arr = shape.get("points") or []
        if not canon or not arr or len(arr[0]) < 2:
            continue
        pts[canon] = (float(arr[0][0]), float(arr[0][1]))
    if len(pts) < NUM_KEYPOINTS or any(k not in pts for k in KEYPOINT_NAMES):
        return None
    return pts


def bbox_from_kpts(kpts: list[tuple[float, float]], w: int, h: int, pad: float = 0.05):
    xs = [p[0] for p in kpts]
    ys = [p[1] for p in kpts]
    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)
    bw = max(x2 - x1, 1.0)
    bh = max(y2 - y1, 1.0)
    x1 -= bw * pad
    x2 += bw * pad
    y1 -= bh * pad
    y2 += bh * pad
    x1 = max(0.0, x1)
    y1 = max(0.0, y1)
    x2 = min(float(w), x2)
    y2 = min(float(h), y2)
    cx = ((x1 + x2) / 2.0) / w
    cy = ((y1 + y2) / 2.0) / h
    nw = (x2 - x1) / w
    nh = (y2 - y1) / h
    return cx, cy, nw, nh


def image_size(path: Path) -> tuple[int, int]:
    import cv2
    img = cv2.imread(str(path))
    if img is None:
        raise RuntimeError(f"Cannot read image: {path}")
    h, w = img.shape[:2]
    return w, h


def yolo_line(w: int, h: int, pts: dict[str, tuple[float, float]]) -> str:
    ordered = [pts[n] for n in KEYPOINT_NAMES]
    cx, cy, bw, bh = bbox_from_kpts(ordered, w, h)
    parts = ["0", f"{cx:.6f}", f"{cy:.6f}", f"{bw:.6f}", f"{bh:.6f}"]
    for x, y in ordered:
        parts.append(f"{x / w:.6f}")
        parts.append(f"{y / h:.6f}")
        parts.append("2")  # visible
    return " ".join(parts)


def main() -> int:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workspace",
        type=Path,
        default=script_dir / "horqin_h2s22wr5py" / "labelme_workspace",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=script_dir / "horqin_h2s22wr5py" / "yolo_pose",
    )
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    images_dir = args.workspace / "images"
    if not images_dir.exists():
        print("Workspace images missing. Run prepare_labelme_workspace.py first.")
        return 1

    samples: list[tuple[Path, Path]] = []
    for img in sorted(images_dir.iterdir()):
        if img.suffix.lower() not in IMAGE_EXTS:
            continue
        js = img.with_suffix(".json")
        if not js.exists():
            alt = args.workspace / "labels" / f"{img.stem}.json"
            js = alt if alt.exists() else js
        pts = parse_labelme(js) if js.exists() else None
        if pts is None:
            print(f"SKIP (need all 4 keypoints): {img.name}")
            continue
        samples.append((img, js))

    if len(samples) < 2:
        print(f"Only {len(samples)} fully labelled image(s). Relabel more side views first.")
        return 1

    random.seed(args.seed)
    random.shuffle(samples)
    n_val = max(1, int(round(len(samples) * args.val_ratio)))
    if n_val >= len(samples):
        n_val = 1
    val_set = samples[:n_val]
    train_set = samples[n_val:]

    for split in ("train", "val"):
        (args.out / "images" / split).mkdir(parents=True, exist_ok=True)
        (args.out / "labels" / split).mkdir(parents=True, exist_ok=True)

    def write_split(items: list[tuple[Path, Path]], split: str) -> None:
        for img, js in items:
            pts = parse_labelme(js)
            assert pts is not None
            w, h = image_size(img)
            dst_img = args.out / "images" / split / img.name
            shutil.copy2(img, dst_img)
            line = yolo_line(w, h, pts)
            (args.out / "labels" / split / f"{img.stem}.txt").write_text(line + "\n", encoding="utf-8")

    write_split(train_set, "train")
    write_split(val_set, "val")

    data_yaml = args.out / "data.yaml"
    data_yaml.write_text(
        "\n".join([
            f"path: {args.out.resolve().as_posix()}",
            "train: images/train",
            "val: images/val",
            f"names:",
            f"  0: {CLASS_NAME}",
            f"kpt_shape: [{NUM_KEYPOINTS}, 3]",
            f"flip_idx: {FLIP_IDX}",
            f"# skeleton (for plots): {SKELETON}",
            f"# keypoints: {KEYPOINT_NAMES}",
            "",
        ]),
        encoding="utf-8",
    )

    print(f"Wrote YOLO-pose dataset: {args.out}")
    print(f"  train={len(train_set)}  val={len(val_set)}")
    print(f"  data.yaml → {data_yaml}")
    print("Next: python train_yolo_pose.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
