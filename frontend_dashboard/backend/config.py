"""Shared paths and constants for the dashboard backend."""

from __future__ import annotations

from pathlib import Path

DASHBOARD_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = DASHBOARD_ROOT.parent
POSE_DIR = PROJECT_ROOT / "cow_pose_detection"
RESULTS_DIR = DASHBOARD_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MAX_UPLOAD_BYTES = 15 * 1024 * 1024  # 15 MB
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
LOW_CONF_THRESHOLD = 0.3
MIN_KPT_CONF = 0.2

BEST_MODEL_PATH = PROJECT_ROOT / "best_weight_model.pkl"
FEATURE_COLUMNS_PATH = PROJECT_ROOT / "feature_columns.json"

ALLOWED_RESULT_FILES = {
    "original_image.jpg",
    "detection_image.jpg",
    "pose_image.jpg",
    "measurements_image.jpg",
    "measure_outline.jpg",
    "body_outline.jpg",
    "segmentation_mask.png",
    "segmentation_overlay.png",
    "four_point_morpho_debug.jpg",
    "keypoints.json",
    "measurements.json",
    "weight_prediction.json",
    "complete_report.json",
    "measurements.csv",
}
