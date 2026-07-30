"""Offline cow body keypoint detector using ViTPose (AP-10K) + YOLO.

Uses easy_ViTPose's VitInference with ONNX models for fully offline inference.
Detects 17 AP-10K keypoints on cows: nose, eyes, neck, shoulders, elbows,
front hooves, hips, knees, back hooves, and tail root.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from cow_keypoints import (
    AP10K_KEYPOINT_NAMES,
    CowPoseResult,
    draw_cow_pose,
)

SCRIPT_DIR = Path(__file__).resolve().parent
EASY_VITPOSE_DIR = SCRIPT_DIR / "easy_ViTPose"
MODELS_DIR = SCRIPT_DIR / "models"
OUTPUT_DIR = SCRIPT_DIR / "output"

sys.path.insert(0, str(EASY_VITPOSE_DIR))
from easy_ViTPose.inference import VitInference  # noqa: E402
from easy_ViTPose.vit_utils.inference import NumpyEncoder, VideoReader  # noqa: E402


DEFAULT_VITPOSE_ONNX = str(MODELS_DIR / "vitpose-b-ap10k.onnx")
DEFAULT_YOLO_ONNX = str(MODELS_DIR / "yolov8s.onnx")


class CowPoseDetector:
    """End-to-end offline cow pose detector.

    Args:
        vitpose_path: Path to ViTPose ONNX model (AP-10K trained).
        yolo_path: Path to YOLOv8 ONNX model.
        conf_threshold: Minimum keypoint confidence for drawing/reporting.
        yolo_size: YOLO input image size.
        device: Inference device ('cpu' or 'cuda').
    """

    def __init__(
        self,
        vitpose_path: str = DEFAULT_VITPOSE_ONNX,
        yolo_path: str = DEFAULT_YOLO_ONNX,
        conf_threshold: float = 0.3,
        yolo_size: int = 640,
        device: str | None = None,
    ) -> None:
        self.conf_threshold = conf_threshold
        self.model = VitInference(
            model=vitpose_path,
            yolo=yolo_path,
            model_name=None,  # ONNX doesn't need model_name
            det_class="cow",
            dataset="ap10k",
            yolo_size=yolo_size,
            device=device or "cpu",
            is_video=False,
            single_pose=False,
            yolo_step=1,
        )

    def detect_image(self, image_path: str | Path) -> list[CowPoseResult]:
        """Run detection on a single image file. Returns list of CowPoseResult."""
        img = cv2.imread(str(image_path))
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return self._detect_frame(img_rgb)

    def detect_frame(self, frame_bgr: np.ndarray) -> list[CowPoseResult]:
        """Run detection on a BGR frame (e.g. from webcam/video)."""
        img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        return self._detect_frame(img_rgb)

    def _detect_frame(self, img_rgb: np.ndarray) -> list[CowPoseResult]:
        frame_kpts = self.model.inference(img_rgb)
        bboxes, ids, scores = self.model._tracker_res

        results = []
        for cow_id, bbox, score in zip(ids, bboxes, scores):
            kpts = frame_kpts.get(cow_id)
            if kpts is None:
                continue
            results.append(CowPoseResult(
                cow_id=int(cow_id),
                bbox=[int(b) for b in bbox],
                bbox_confidence=float(score),
                keypoints=np.array(kpts, dtype=np.float64),
            ))
        return results

    def process_image(
        self,
        image_path: str | Path,
        output_dir: str | Path | None = None,
        save_img: bool = True,
        save_json: bool = True,
        show: bool = False,
        draw_labels: bool = True,
    ) -> tuple[list[CowPoseResult], np.ndarray]:
        """Process a single image: detect, annotate, save."""
        image_path = Path(image_path)
        output_dir = Path(output_dir or OUTPUT_DIR)
        output_dir.mkdir(parents=True, exist_ok=True)

        img = cv2.imread(str(image_path))
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self._detect_frame(img_rgb)

        annotated = img.copy()
        for r in results:
            annotated = draw_cow_pose(annotated, r, self.conf_threshold,
                                      draw_labels=draw_labels)

        stem = image_path.stem
        if save_img:
            out_img = output_dir / f"{stem}_pose.png"
            cv2.imwrite(str(out_img), annotated)
            print(f"Saved annotated image: {out_img}")

        if save_json:
            out_json = output_dir / f"{stem}_keypoints.json"
            payload = {
                "source": str(image_path),
                "num_cows_detected": len(results),
                "detections": [r.to_dict() for r in results],
                "keypoint_schema": dict(AP10K_KEYPOINT_NAMES),
            }
            with open(out_json, "w") as f:
                json.dump(payload, f, indent=2, cls=NumpyEncoder)
            print(f"Saved keypoints JSON: {out_json}")

        if show:
            cv2.imshow("Cow Pose Detection", annotated)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return results, annotated

    def process_video(
        self,
        video_path: str | Path,
        output_dir: str | Path | None = None,
        save_video: bool = True,
        save_json: bool = True,
        show: bool = False,
        draw_labels: bool = False,
    ) -> list[list[CowPoseResult]]:
        """Process a video file, frame by frame."""
        video_path = Path(video_path)
        output_dir = Path(output_dir or OUTPUT_DIR)
        output_dir.mkdir(parents=True, exist_ok=True)

        self.model.is_video = True
        self.model.reset()

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        writer = None
        if save_video:
            out_path = output_dir / f"{video_path.stem}_pose.mp4"
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))

        reader = VideoReader(str(video_path))
        all_results: list[list[CowPoseResult]] = []

        for i, frame_rgb in enumerate(reader):
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            results = self._detect_frame(frame_rgb)
            all_results.append(results)

            annotated = frame_bgr.copy()
            for r in results:
                annotated = draw_cow_pose(annotated, r, self.conf_threshold,
                                          draw_labels=draw_labels)

            if writer:
                writer.write(annotated)
            if show:
                cv2.imshow("Cow Pose Detection", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            if (i + 1) % 30 == 0 or i + 1 == total_frames:
                print(f"  Frame {i+1}/{total_frames}")

        if writer:
            writer.release()
            print(f"Saved annotated video: {out_path}")

        if save_json:
            json_path = output_dir / f"{video_path.stem}_keypoints.json"
            payload = {
                "source": str(video_path),
                "total_frames": len(all_results),
                "frames": [
                    {"frame": i, "detections": [r.to_dict() for r in fr]}
                    for i, fr in enumerate(all_results)
                ],
                "keypoint_schema": dict(AP10K_KEYPOINT_NAMES),
            }
            with open(json_path, "w") as f:
                json.dump(payload, f, indent=2, cls=NumpyEncoder)
            print(f"Saved keypoints JSON: {json_path}")

        cv2.destroyAllWindows()
        self.model.is_video = False
        return all_results

    def process_webcam(
        self,
        camera_id: int = 0,
        output_dir: str | Path | None = None,
        save_video: bool = False,
        save_json: bool = False,
        draw_labels: bool = False,
    ) -> None:
        """Run live detection on webcam. Press 'q' to quit, 's' to save a snapshot."""
        output_dir = Path(output_dir or OUTPUT_DIR)
        output_dir.mkdir(parents=True, exist_ok=True)

        self.model.is_video = True
        self.model.reset()

        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open webcam {camera_id}")

        writer = None
        if save_video:
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            out_path = output_dir / "webcam_pose.mp4"
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(out_path), fourcc, 15.0, (w, h))

        all_results: list[list[CowPoseResult]] = []
        snap_count = 0
        print("Webcam started. Press 'q' to quit, 's' to save snapshot.")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self._detect_frame(frame_rgb)
            all_results.append(results)

            annotated = frame.copy()
            for r in results:
                annotated = draw_cow_pose(annotated, r, self.conf_threshold,
                                          draw_labels=draw_labels)

            if writer:
                writer.write(annotated)

            cv2.imshow("Cow Pose Detection - Webcam", annotated)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("s"):
                snap_count += 1
                snap_path = output_dir / f"webcam_snap_{snap_count:04d}.png"
                cv2.imwrite(str(snap_path), annotated)
                json_path = output_dir / f"webcam_snap_{snap_count:04d}_keypoints.json"
                payload = {
                    "source": f"webcam:{camera_id}",
                    "num_cows_detected": len(results),
                    "detections": [r.to_dict() for r in results],
                    "keypoint_schema": dict(AP10K_KEYPOINT_NAMES),
                }
                with open(json_path, "w") as f:
                    json.dump(payload, f, indent=2, cls=NumpyEncoder)
                print(f"Snapshot saved: {snap_path}")

        cap.release()
        if writer:
            writer.release()
            print(f"Saved webcam video: {out_path}")

        if save_json and all_results:
            json_path = output_dir / "webcam_keypoints.json"
            payload = {
                "source": f"webcam:{camera_id}",
                "total_frames": len(all_results),
                "frames": [
                    {"frame": i, "detections": [r.to_dict() for r in fr]}
                    for i, fr in enumerate(all_results)
                ],
                "keypoint_schema": dict(AP10K_KEYPOINT_NAMES),
            }
            with open(json_path, "w") as f:
                json.dump(payload, f, indent=2, cls=NumpyEncoder)
            print(f"Saved webcam JSON: {json_path}")

        cv2.destroyAllWindows()
        self.model.is_video = False
