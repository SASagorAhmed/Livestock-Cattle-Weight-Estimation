#!/usr/bin/env python3
"""CLI for cow body keypoint detection using ViTPose (AP-10K) + YOLO.

Fully offline inference. Accepts images, video files, or webcam input.
Saves annotated images/videos and keypoint JSON files.

Usage:
    python detect_cow_pose.py --input path/to/cow.jpg
    python detect_cow_pose.py --input path/to/video.mp4 --save-video
    python detect_cow_pose.py --webcam 0
    python detect_cow_pose.py --input path/to/cow.jpg --no-labels --conf 0.4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from detector import CowPoseDetector  # noqa: E402


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv"}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Detect cow body keypoints using ViTPose (AP-10K) + YOLO."
    )
    p.add_argument("--input", type=Path,
                    help="Path to an image or video file.")
    p.add_argument("--webcam", type=int, default=None,
                    help="Webcam device ID (e.g. 0).")
    p.add_argument("--output", type=Path, default=SCRIPT_DIR / "output",
                    help="Output directory for annotated files and JSON.")
    p.add_argument("--vitpose", type=str,
                    default=str(SCRIPT_DIR / "models" / "vitpose-b-ap10k.onnx"),
                    help="Path to ViTPose ONNX model.")
    p.add_argument("--yolo", type=str,
                    default=str(SCRIPT_DIR / "models" / "yolov8s.onnx"),
                    help="Path to YOLOv8 ONNX model.")
    p.add_argument("--conf", type=float, default=0.3,
                    help="Minimum keypoint confidence threshold.")
    p.add_argument("--yolo-size", type=int, default=640,
                    help="YOLO input image size.")
    p.add_argument("--show", action="store_true",
                    help="Display results in a window.")
    p.add_argument("--no-labels", action="store_true",
                    help="Don't draw keypoint labels on the image.")
    p.add_argument("--no-save-img", action="store_true",
                    help="Don't save annotated image/video.")
    p.add_argument("--no-save-json", action="store_true",
                    help="Don't save keypoints JSON.")
    p.add_argument("--save-video", action="store_true",
                    help="Save annotated video (for video/webcam input).")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.input is None and args.webcam is None:
        print("Error: provide --input <file> or --webcam <id>", file=sys.stderr)
        return 1

    detector = CowPoseDetector(
        vitpose_path=args.vitpose,
        yolo_path=args.yolo,
        conf_threshold=args.conf,
        yolo_size=args.yolo_size,
    )
    print(f"Models loaded (ViTPose: {Path(args.vitpose).name}, "
          f"YOLO: {Path(args.yolo).name})")

    if args.webcam is not None:
        detector.process_webcam(
            camera_id=args.webcam,
            output_dir=args.output,
            save_video=args.save_video,
            save_json=not args.no_save_json,
            draw_labels=not args.no_labels,
        )
        return 0

    input_path = args.input
    if not input_path.is_file():
        print(f"Error: file not found: {input_path}", file=sys.stderr)
        return 1

    ext = input_path.suffix.lower()

    if ext in IMAGE_EXTS:
        results, annotated = detector.process_image(
            image_path=input_path,
            output_dir=args.output,
            save_img=not args.no_save_img,
            save_json=not args.no_save_json,
            show=args.show,
            draw_labels=not args.no_labels,
        )
        print(f"\nDetected {len(results)} cow(s)")
        for r in results:
            above = sum(1 for k in r.keypoints if k[2] > args.conf)
            print(f"  Cow #{r.cow_id}: {above}/17 keypoints above {args.conf} confidence")

    elif ext in VIDEO_EXTS:
        all_results = detector.process_video(
            video_path=input_path,
            output_dir=args.output,
            save_video=args.save_video or not args.no_save_img,
            save_json=not args.no_save_json,
            show=args.show,
            draw_labels=not args.no_labels,
        )
        total_cows = sum(len(fr) for fr in all_results)
        print(f"\nProcessed {len(all_results)} frames, {total_cows} total cow detections")

    else:
        print(f"Error: unsupported file type: {ext}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
