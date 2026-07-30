"""Infer 4 Smartphone-Diagonal keypoints with a local YOLO-pose checkpoint.

Does not call cloud APIs. Returns None suggestions if weights are missing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from config import POSE_DIR

KEYPOINT_NAMES = [
    "A_start_lower_chest",
    "A_end_withers",
    "B_start_tail_head",
    "B_end_shoulder_region",
]

WEIGHTS_CANDIDATES = [
    POSE_DIR / "models" / "four_point_pose.pt",
    POSE_DIR / "models" / "four_point_pose.onnx",
    POSE_DIR / "datasets" / "horqin_h2s22wr5py" / "runs" / "four_point_pose" / "weights" / "best.pt",
]

_model = None
_model_path: Path | None = None


def four_point_model_path() -> Path | None:
    for p in WEIGHTS_CANDIDATES:
        if p.exists():
            return p
    return None


def four_point_model_available() -> bool:
    return four_point_model_path() is not None


def _load_model():
    global _model, _model_path
    path = four_point_model_path()
    if path is None:
        return None
    if _model is not None and _model_path == path:
        return _model
    from ultralytics import YOLO
    _model = YOLO(str(path))
    _model_path = path
    return _model


def suggest_four_points(
    image_path: Path,
    bbox: list[float] | tuple[float, ...] | None = None,
) -> dict[str, Any]:
    """Run YOLO-pose and return named keypoints in image coordinates.

    No mock coordinates — if inference fails or model missing, available=False.
    """
    path = four_point_model_path()
    if path is None:
        return {
            "available": False,
            "reason": (
                "four_point_pose.pt not found. Train with "
                "cow_pose_detection/datasets/train_yolo_pose.py after relabeling."
            ),
            "keypoints": None,
            "model_path": None,
        }

    model = _load_model()
    if model is None:
        return {
            "available": False,
            "reason": "Failed to load four-point pose model.",
            "keypoints": None,
            "model_path": str(path),
        }

    try:
        kwargs: dict[str, Any] = {"verbose": False}
        # Ultralytics accepts path; optional crop via bbox not always needed
        results = model.predict(str(image_path), **kwargs)
    except Exception as exc:  # noqa: BLE001
        return {
            "available": False,
            "reason": f"Inference failed: {exc}",
            "keypoints": None,
            "model_path": str(path),
        }

    if not results:
        return {
            "available": False,
            "reason": "No pose predictions returned.",
            "keypoints": None,
            "model_path": str(path),
        }

    r0 = results[0]
    if r0.keypoints is None or r0.keypoints.xy is None or len(r0.keypoints.xy) == 0:
        return {
            "available": False,
            "reason": "Model returned no keypoints.",
            "keypoints": None,
            "model_path": str(path),
        }

    # Pick detection with largest box area if multiple
    best_i = 0
    if r0.boxes is not None and len(r0.boxes) > 1:
        areas = []
        for i in range(len(r0.boxes)):
            xyxy = r0.boxes.xyxy[i].tolist()
            areas.append(max(0.0, (xyxy[2] - xyxy[0]) * (xyxy[3] - xyxy[1])))
        best_i = int(max(range(len(areas)), key=lambda i: areas[i]))

    xy = r0.keypoints.xy[best_i].tolist()
    conf = None
    if r0.keypoints.conf is not None:
        conf = r0.keypoints.conf[best_i].tolist()

    if len(xy) < 4:
        return {
            "available": False,
            "reason": f"Expected 4 keypoints, got {len(xy)}.",
            "keypoints": None,
            "model_path": str(path),
        }

    keypoints: dict[str, dict[str, float]] = {}
    for i, name in enumerate(KEYPOINT_NAMES):
        x, y = float(xy[i][0]), float(xy[i][1])
        if not (x == x and y == y):  # NaN check
            return {
                "available": False,
                "reason": f"Keypoint {name} is invalid (NaN).",
                "keypoints": None,
                "model_path": str(path),
            }
        entry: dict[str, float] = {"x": round(x, 2), "y": round(y, 2)}
        if conf is not None and i < len(conf):
            entry["confidence"] = round(float(conf[i]), 4)
        keypoints[name] = entry

    # Optional: if bbox provided, prefer points inside bbox (soft check)
    _ = bbox

    return {
        "available": True,
        "reason": None,
        "keypoints": keypoints,
        "model_path": str(path),
    }
