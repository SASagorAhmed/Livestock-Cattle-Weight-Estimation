#!/usr/bin/env python3
"""Train YOLOv8/YOLO11 pose model with exactly 4 keypoints.

Usage:
  python train_yolo_pose.py
  python train_yolo_pose.py --model yolo11n-pose.pt --epochs 100
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path


def main() -> int:
    script_dir = Path(__file__).resolve().parent
    pose_dir = script_dir.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=script_dir / "horqin_h2s22wr5py" / "yolo_pose" / "data.yaml",
    )
    parser.add_argument("--model", type=str, default="yolo11n-pose.pt")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", type=str, default="")
    parser.add_argument(
        "--project",
        type=Path,
        default=script_dir / "horqin_h2s22wr5py" / "runs",
    )
    parser.add_argument("--name", type=str, default="four_point_pose")
    args = parser.parse_args()

    if not args.data.exists():
        print("Missing data.yaml. Run labelme_to_yolo_pose.py after relabeling.")
        return 1

    try:
        from ultralytics import YOLO
    except ImportError:
        print("ultralytics not installed. pip install ultralytics")
        return 1

    model = YOLO(args.model)
    kwargs = dict(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=str(args.project),
        name=args.name,
        exist_ok=True,
    )
    if args.device:
        kwargs["device"] = args.device

    results = model.train(**kwargs)
    best = Path(results.save_dir) / "weights" / "best.pt"
    export_dir = pose_dir / "models"
    export_dir.mkdir(parents=True, exist_ok=True)
    dest = export_dir / "four_point_pose.pt"
    if best.exists():
        shutil.copy2(best, dest)
        print("Copied best weights →", dest)
    else:
        print("WARNING: best.pt not found at", best)
        return 1

    # Optional ONNX for runtime parity with other pose models
    try:
        exported = YOLO(str(dest)).export(format="onnx", imgsz=args.imgsz)
        print("ONNX export:", exported)
    except Exception as exc:  # noqa: BLE001
        print("ONNX export skipped:", exc)

    print("Training complete. App will auto-suggest using:", dest)
    return 0


if __name__ == "__main__":
    sys.exit(main())
