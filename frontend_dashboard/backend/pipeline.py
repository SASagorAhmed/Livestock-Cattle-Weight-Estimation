"""Pipeline orchestration: reuses existing cattle + cow_pose_detection modules."""

from __future__ import annotations

import csv
import json
import math
import sys
import time
import uuid
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from werkzeug.utils import secure_filename

from config import (
    ALLOWED_EXTENSIONS,
    BEST_MODEL_PATH,
    FEATURE_COLUMNS_PATH,
    LOW_CONF_THRESHOLD,
    MAX_UPLOAD_BYTES,
    MIN_KPT_CONF,
    POSE_DIR,
    PROJECT_ROOT,
    RESULTS_DIR,
)

# Make project modules importable
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(POSE_DIR) not in sys.path:
    sys.path.insert(0, str(POSE_DIR))

from cattle_model import CLASS_PARAMS, CattleWeightEstimator, estimate_weight_from_probs  # noqa: E402
from cow_keypoints import AP10K_KEYPOINT_NAMES, AP10K_SKELETON, draw_cow_pose  # noqa: E402
from detector import CowPoseDetector  # noqa: E402
from measurements import (  # noqa: E402
    apply_scale,
    bbox_area,
    compute_measurements_from_keypoints,
    draw_measurements,
    estimate_scale_from_reference,
)
from segmentation import overlay_segmentation, seg_for_json, segment_cow  # noqa: E402


# Cached heavy models (loaded once)
_pose_detector: CowPoseDetector | None = None
_weight_estimator: CattleWeightEstimator | None = None


def get_pose_detector() -> CowPoseDetector:
    global _pose_detector
    if _pose_detector is None:
        _pose_detector = CowPoseDetector(conf_threshold=LOW_CONF_THRESHOLD, device="cpu")
    return _pose_detector


def get_weight_estimator() -> CattleWeightEstimator:
    global _weight_estimator
    if _weight_estimator is None:
        _weight_estimator = CattleWeightEstimator(backend="auto")
    return _weight_estimator


def measurement_model_available() -> bool:
    return BEST_MODEL_PATH.is_file() and FEATURE_COLUMNS_PATH.is_file()


def new_run_id() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"{stamp}_{uuid.uuid4().hex[:8]}"


def validate_upload(filename: str, size: int) -> None:
    if size <= 0:
        raise ValueError("Empty upload")
    if size > MAX_UPLOAD_BYTES:
        raise ValueError(f"File too large (max {MAX_UPLOAD_BYTES // (1024*1024)} MB)")
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise ValueError(f"Unsupported format '{ext}'. Allowed: {sorted(ALLOWED_EXTENSIONS)}")


def _file_url(run_id: str, name: str) -> str:
    return f"/api/run/{run_id}/file/{name}"


def _detection_to_dict(r) -> dict[str, Any]:
    kpts = {}
    for idx, name in AP10K_KEYPOINT_NAMES.items():
        y, x, score = r.keypoints[idx]
        kpts[name] = {
            "x": round(float(x), 2),
            "y": round(float(y), 2),
            "confidence": round(float(score), 4),
            "status": (
                "ok" if score >= LOW_CONF_THRESHOLD
                else ("low" if score > 0 else "missing")
            ),
        }
    box = [int(v) for v in r.bbox]
    return {
        "cow_id": int(r.cow_id),
        "bbox": box,
        "bbox_confidence": round(float(r.bbox_confidence), 4),
        "bbox_area": round(bbox_area(box), 2),
        "keypoints": kpts,
    }


def _draw_detection_image(img_bgr: np.ndarray, detections: list[dict], selected_id: int) -> np.ndarray:
    out = img_bgr.copy()
    for d in detections:
        x1, y1, x2, y2 = d["bbox"]
        selected = d["cow_id"] == selected_id
        color = (0, 220, 0) if selected else (180, 180, 80)
        thickness = 3 if selected else 2
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
        label = f"Cow #{d['cow_id']} ({d['bbox_confidence']:.2f})"
        if selected:
            label += " [PRIMARY]"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(out, (x1, max(0, y1 - th - 8)), (x1 + tw + 4, y1), color, -1)
        cv2.putText(out, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)
    return out


def _draw_pose_image(img_bgr: np.ndarray, pose_results) -> np.ndarray:
    out = img_bgr.copy()
    for r in pose_results:
        out = draw_cow_pose(out, r, conf_threshold=LOW_CONF_THRESHOLD, draw_labels=True)
    return out


def _euclid_detail(
    name: str,
    start_name: str,
    end_name: str,
    p1: list[float] | None,
    p2: list[float] | None,
    result_px: float | None,
) -> dict[str, Any]:
    if p1 is None or p2 is None or result_px is None:
        return {
            "name": name,
            "start_point": start_name,
            "end_point": end_name,
            "x1": None, "y1": None, "x2": None, "y2": None,
            "formula": "distance_px = sqrt((x2 - x1)^2 + (y2 - y1)^2)",
            "substituted": "Insufficient keypoints (missing or low confidence)",
            "result_px": None,
            "available": False,
        }
    x1, y1 = p1[0], p1[1]
    x2, y2 = p2[0], p2[1]
    dx, dy = x2 - x1, y2 - y1
    return {
        "name": name,
        "start_point": start_name,
        "end_point": end_name,
        "x1": x1, "y1": y1, "x2": x2, "y2": y2,
        "formula": "distance_px = sqrt((x2 - x1)^2 + (y2 - y1)^2)",
        "substituted": (
            f"distance = sqrt(({x2} - {x1})^2 + ({y2} - {y1})^2) "
            f"= sqrt({dx:.2f}^2 + {dy:.2f}^2) = {result_px:.2f} px"
        ),
        "result_px": round(result_px, 2),
        "available": True,
    }


def _chain_detail(
    name: str,
    points: list[tuple[str, list[float] | None]],
    result_px: float | None,
) -> dict[str, Any]:
    valid = [(n, p) for n, p in points if p is not None]
    if len(valid) < 2 or result_px is None:
        return {
            "name": name,
            "start_point": points[0][0] if points else None,
            "end_point": points[-1][0] if points else None,
            "x1": None, "y1": None, "x2": None, "y2": None,
            "formula": "sum of consecutive Euclidean segments",
            "substituted": "Insufficient keypoints for chain length",
            "result_px": None,
            "available": False,
            "segments": [],
        }
    segs = []
    total = 0.0
    for i in range(len(valid) - 1):
        a_name, a = valid[i]
        b_name, b = valid[i + 1]
        d = math.hypot(b[0] - a[0], b[1] - a[1])
        total += d
        segs.append({
            "from": a_name, "to": b_name,
            "x1": a[0], "y1": a[1], "x2": b[0], "y2": b[1],
            "distance_px": round(d, 2),
        })
    return {
        "name": name,
        "start_point": valid[0][0],
        "end_point": valid[-1][0],
        "x1": valid[0][1][0], "y1": valid[0][1][1],
        "x2": valid[-1][1][0], "y2": valid[-1][1][1],
        "formula": "chain = sum_i sqrt((x_{i+1}-x_i)^2 + (y_{i+1}-y_i)^2)",
        "substituted": " + ".join(f"{s['distance_px']}" for s in segs) + f" = {result_px:.2f} px",
        "result_px": round(result_px, 2),
        "available": True,
        "segments": segs,
    }


def build_pixel_calculations(detection: dict, measurements) -> list[dict[str, Any]]:
    k = detection["keypoints"]
    lm = measurements.landmarks
    px = measurements.pixels

    def kp(name: str) -> list[float] | None:
        p = k.get(name)
        if not p or p.get("confidence", 0) < MIN_KPT_CONF:
            return None
        return [p["x"], p["y"]]

    details = [
        _euclid_detail("shoulder_width", "left_shoulder", "right_shoulder",
                       kp("left_shoulder"), kp("right_shoulder"), px.get("shoulder_width")),
        _euclid_detail("hip_width", "left_hip", "right_hip",
                       kp("left_hip"), kp("right_hip"), px.get("hip_width")),
        _euclid_detail("body_length", "shoulder_center", "hip_center",
                       lm.get("shoulder_center"), lm.get("hip_center"), px.get("body_length")),
        _euclid_detail("lower_chest", "shoulder_center", "hip_center",
                       lm.get("shoulder_center"), lm.get("hip_center"), px.get("lower_chest")),
        _euclid_detail("body_height", "back_top", "ground",
                       lm.get("a_end") or lm.get("back_top"), lm.get("ground"),
                       px.get("body_height")),
        _chain_detail("left_front_leg_length", [
            ("left_shoulder", kp("left_shoulder")),
            ("left_elbow", kp("left_elbow")),
            ("left_front_hoof", kp("left_front_hoof")),
        ], px.get("left_front_leg_length")),
        _chain_detail("right_front_leg_length", [
            ("right_shoulder", kp("right_shoulder")),
            ("right_elbow", kp("right_elbow")),
            ("right_front_hoof", kp("right_front_hoof")),
        ], px.get("right_front_leg_length")),
        _chain_detail("left_back_leg_length", [
            ("left_hip", kp("left_hip")),
            ("left_knee", kp("left_knee")),
            ("left_back_hoof", kp("left_back_hoof")),
        ], px.get("left_back_leg_length")),
        _chain_detail("right_back_leg_length", [
            ("right_hip", kp("right_hip")),
            ("right_knee", kp("right_knee")),
            ("right_back_hoof", kp("right_back_hoof")),
        ], px.get("right_back_leg_length")),
        _euclid_detail("chest_depth_proxy", "shoulder_center", "elbow_center",
                       lm.get("shoulder_center"), lm.get("elbow_center"),
                       px.get("chest_depth_proxy")),
        _euclid_detail("torso_diagonal", "cross_shoulders_hips", "cross_shoulders_hips",
                       lm.get("shoulder_center"), lm.get("hip_center"),
                       px.get("torso_diagonal")),
    ]
    return details


def _heuristic_weight(image_path: Path) -> dict[str, Any]:
    est = get_weight_estimator()
    result = est.predict(image_path)
    class_index = result.class_index
    mult, offset = CLASS_PARAMS[class_index]
    conf = result.confidence
    clamped = conf if not (conf < 0.3 or conf > 1.7) else 0.9
    age_proxy = clamped * mult
    return {
        "mode": "heuristic",
        "model_name": "h5model.h5 + model_regression",
        "model_type": "MobileNetV2 classifier + LinearRegression heuristic",
        "backend": result.backend,
        "cnn_class": class_index,
        "cnn_confidence": round(conf, 6),
        "confidence_clamped": round(clamped, 6),
        "age_multiplier": mult,
        "age_proxy": round(result.age_proxy, 6),
        "regression_equation": "weight_lb = 31.4188 * age_proxy + 706.6833  (fitted coef/intercept)",
        "regression_lb": round(result.regression_lb, 4),
        "lb_to_kg_factor": 2.205,
        "class_offset_kg": offset,
        "weight_kg": round(result.weight_kg, 2),
        "hardcoded_pm_kg_note": (
            "The old UI appended ±15 KG as a hardcoded string. "
            "It is NOT a validated confidence interval."
        ),
        "class_probs": [round(float(p), 6) for p in result.class_probs],
        "stages": [
            f"1. CNN softmax class = {class_index} (confidence={conf:.4f})",
            f"2. age_proxy = clamp(confidence) * {mult} = {result.age_proxy:.4f}",
            f"3. regression_lb = LinearRegression(age_proxy) = {result.regression_lb:.2f} lb",
            f"4. weight_kg = regression_lb / 2.205 + ({offset}) = {result.weight_kg:.2f} kg",
        ],
        "available": True,
    }


def _measurement_model_weight(measurements: dict) -> dict[str, Any]:
    if not measurement_model_available():
        return {
            "mode": "measurement",
            "available": False,
            "reason": (
                "best_weight_model.pkl and/or feature_columns.json not found. "
                "Measurement-based prediction is disabled."
            ),
        }
    import pickle

    with open(FEATURE_COLUMNS_PATH, encoding="utf-8") as f:
        feature_columns = json.load(f)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with open(BEST_MODEL_PATH, "rb") as f:
            model = pickle.load(f)

    px = measurements.get("measurements_px", {})
    ratios = measurements.get("ratios", {})
    seg = measurements.get("segmentation") or {}
    feature_map = {
        **{f"{k}_px": v for k, v in px.items()},
        **px,
        **ratios,
        "body_pixel_area": seg.get("body_pixel_area"),
        "torso_pixel_area": seg.get("torso_pixel_area"),
        "body_perimeter_px": seg.get("body_perimeter_px"),
    }
    values = []
    missing = []
    used = []
    for col in feature_columns:
        val = feature_map.get(col)
        used.append({"name": col, "value": val})
        if val is None:
            missing.append(col)
            values.append(0.0)
        else:
            values.append(float(val))
    arr = np.array([values], dtype=np.float64)
    raw = float(model.predict(arr)[0])
    return {
        "mode": "measurement",
        "available": True,
        "model_name": "best_weight_model.pkl",
        "model_type": type(model).__name__,
        "feature_names": list(feature_columns),
        "feature_values": used,
        "missing_features": missing,
        "raw_prediction": round(raw, 4),
        "weight_kg": round(raw, 2),
    }


def build_normalized_features(m_dict: dict) -> list[dict[str, Any]]:
    px = m_dict.get("measurements_px", {})
    ratios = m_dict.get("ratios", {})
    seg = m_dict.get("segmentation") or {}
    body_a = seg.get("body_pixel_area")
    torso_a = seg.get("torso_pixel_area")
    peri = seg.get("body_perimeter_px")
    bh = px.get("body_height")

    def r(a, b):
        if a is None or b is None or b == 0:
            return None
        return round(a / b, 4)

    return [
        {
            "name": "body_length / body_height",
            "formula": "body_length_px / body_height_px",
            "value": ratios.get("body_length_over_body_height"),
            "numerator": px.get("body_length"),
            "denominator": px.get("body_height"),
        },
        {
            "name": "shoulder_width / body_length",
            "formula": "shoulder_width_px / body_length_px",
            "value": ratios.get("shoulder_width_over_body_length"),
            "numerator": px.get("shoulder_width"),
            "denominator": px.get("body_length"),
        },
        {
            "name": "hip_width / body_length",
            "formula": "hip_width_px / body_length_px",
            "value": ratios.get("hip_width_over_body_length"),
            "numerator": px.get("hip_width"),
            "denominator": px.get("body_length"),
        },
        {
            "name": "chest_depth / body_height",
            "formula": "chest_depth_proxy_px / body_height_px",
            "value": ratios.get("chest_depth_over_body_height"),
            "numerator": px.get("chest_depth_proxy"),
            "denominator": px.get("body_height"),
        },
        {
            "name": "torso_area / full_body_area",
            "formula": "torso_pixel_area / body_pixel_area",
            "value": r(torso_a, body_a),
            "numerator": torso_a,
            "denominator": body_a,
        },
        {
            "name": "perimeter / body_height",
            "formula": "body_perimeter_px / body_height_px",
            "value": r(peri, bh),
            "numerator": peri,
            "denominator": bh,
        },
    ]


def process_image(
    file_storage,
    enable_segmentation: bool = False,
    prediction_mode: str = "heuristic",
    reference_px: float | None = None,
    reference_cm: float | None = None,
) -> dict[str, Any]:
    t0 = time.time()
    filename = file_storage.filename or "upload.jpg"
    ext = Path(secure_filename(filename)).suffix.lower() or ".jpg"
    if ext not in ALLOWED_EXTENSIONS:
        raise ValueError(f"Unsupported format '{ext}'. Allowed: {sorted(ALLOWED_EXTENSIONS)}")

    run_id = new_run_id()
    run_dir = RESULTS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save original as jpg for consistency when possible
    original_path = run_dir / "original_image.jpg"
    tmp_path = run_dir / f"_upload{ext}"
    file_storage.save(tmp_path)
    size = tmp_path.stat().st_size
    if size <= 0:
        tmp_path.unlink(missing_ok=True)
        raise ValueError("Empty upload")
    if size > MAX_UPLOAD_BYTES:
        tmp_path.unlink(missing_ok=True)
        raise ValueError(f"File too large (max {MAX_UPLOAD_BYTES // (1024 * 1024)} MB)")

    img = cv2.imread(str(tmp_path))
    if img is None:
        raise ValueError("Could not decode image")
    cv2.imwrite(str(original_path), img)
    if tmp_path != original_path:
        tmp_path.unlink(missing_ok=True)

    warnings_list: list[str] = []
    steps: dict[str, Any] = {
        "upload": {"status": "completed"},
        "detection": {"status": "processing"},
        "pose": {"status": "pending"},
        "measurements": {"status": "pending"},
        "pixel_calculations": {"status": "pending"},
        "segmentation": {"status": "pending" if enable_segmentation else "skipped"},
        "scale": {"status": "pending"},
        "features": {"status": "pending"},
        "weight": {"status": "pending"},
        "report": {"status": "pending"},
    }

    # --- Pose + detection ---
    try:
        detector = get_pose_detector()
        pose_results = detector.detect_image(original_path)
        if not pose_results:
            steps["detection"] = {"status": "failed", "error": "No cows detected"}
            raise RuntimeError("No cows detected in the image")
        detections = [_detection_to_dict(r) for r in pose_results]
        primary = max(detections, key=lambda d: d["bbox_area"])
        selected_id = primary["cow_id"]

        det_img = _draw_detection_image(img, detections, selected_id)
        cv2.imwrite(str(run_dir / "detection_image.jpg"), det_img)
        pose_img = _draw_pose_image(img, pose_results)
        cv2.imwrite(str(run_dir / "pose_image.jpg"), pose_img)

        keypoints_payload = {
            "source": str(original_path),
            "num_cows_detected": len(detections),
            "selected_cow_id": selected_id,
            "detections": detections,
            "keypoint_schema": {str(k): v for k, v in AP10K_KEYPOINT_NAMES.items()},
            "skeleton": AP10K_SKELETON,
        }
        with open(run_dir / "keypoints.json", "w", encoding="utf-8") as f:
            json.dump(keypoints_payload, f, indent=2)

        steps["detection"] = {
            "status": "completed",
            "num_cows": len(detections),
            "selected_cow_id": selected_id,
        }
        steps["pose"] = {"status": "completed", "num_keypoints": 17}
    except Exception as exc:
        steps["detection"]["status"] = "failed"
        steps["detection"]["error"] = str(exc)
        raise

    # --- Measurements for selected cow ---
    try:
        steps["measurements"]["status"] = "processing"
        selected_det = next(d for d in detections if d["cow_id"] == selected_id)

        # Segmentation first so body_height display can snap A End to mask
        seg_info = None
        mask_for_meas = None
        if enable_segmentation:
            steps["segmentation"]["status"] = "processing"
            seg_info = segment_cow(img, selected_det.get("bbox"))
            if seg_info is None:
                warnings_list.append("Segmentation enabled but no cow mask found")
                steps["segmentation"] = {"status": "failed", "error": "No mask found"}
            else:
                mask_for_meas = seg_info["mask"]
                cv2.imwrite(str(run_dir / "segmentation_mask.png"), (mask_for_meas * 255).astype(np.uint8))
                overlay = overlay_segmentation(img, seg_info)
                cv2.imwrite(str(run_dir / "segmentation_overlay.png"), overlay)
                steps["segmentation"] = {
                    "status": "completed",
                    "body_pixel_area": None,
                    "torso_pixel_area": None,
                    "body_perimeter_px": None,
                }
        else:
            steps["segmentation"] = {"status": "skipped"}

        m = compute_measurements_from_keypoints(
            selected_det, min_keypoint_conf=MIN_KPT_CONF, mask=mask_for_meas,
        )
        if seg_info is not None:
            m.segmentation = seg_for_json(seg_info)
            steps["segmentation"] = {
                "status": "completed",
                "body_pixel_area": m.segmentation.get("body_pixel_area"),
                "torso_pixel_area": m.segmentation.get("torso_pixel_area"),
                "body_perimeter_px": m.segmentation.get("body_perimeter_px"),
            }

        # Scale
        scale_status = {
            "provided": False,
            "cm_per_px": None,
            "message": "No reference scale was provided. Pixel measurements are not real-world centimeters.",
        }
        if reference_px and reference_cm and reference_px > 0 and reference_cm > 0:
            cm_per_px = estimate_scale_from_reference(reference_px, reference_cm)
            apply_scale(m, cm_per_px)
            scale_status = {
                "provided": True,
                "reference_px": reference_px,
                "reference_cm": reference_cm,
                "cm_per_px": round(cm_per_px, 6),
                "message": f"Scale = {reference_cm} cm / {reference_px} px = {cm_per_px:.6f} cm/px",
            }
            steps["scale"] = {"status": "completed", **scale_status}
        else:
            steps["scale"] = {"status": "skipped", **scale_status}

        meas_img = draw_measurements(img, m, unit="cm" if m.centimeters else "px")
        if seg_info is not None:
            meas_img = overlay_segmentation(meas_img, seg_info, alpha=0.25)
        cv2.imwrite(str(run_dir / "measurements_image.jpg"), meas_img)

        m_dict = m.to_dict()
        m_dict["measurement_lines"] = [
            {"p1": list(a), "p2": list(b), "name": n} for a, b, n in m.lines.segments
        ]
        pixel_calcs = build_pixel_calculations(selected_det, m)
        m_dict["pixel_calculations"] = pixel_calcs
        normalized = build_normalized_features(m_dict)
        m_dict["normalized_features"] = normalized

        with open(run_dir / "measurements.json", "w", encoding="utf-8") as f:
            json.dump(m_dict, f, indent=2)

        # CSV
        with open(run_dir / "measurements.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["measurement", "value_px", "value_cm"])
            for k, v in (m_dict.get("measurements_px") or {}).items():
                cm_v = (m_dict.get("measurements_cm") or {}).get(k) if m_dict.get("measurements_cm") else None
                writer.writerow([k, v, cm_v])

        steps["measurements"] = {"status": "completed"}
        steps["pixel_calculations"] = {"status": "completed"}
        steps["features"] = {"status": "completed"}
    except Exception as exc:
        steps["measurements"] = {"status": "failed", "error": str(exc)}
        raise

    # --- Weight ---
    try:
        steps["weight"]["status"] = "processing"
        heuristic = _heuristic_weight(original_path)
        measurement_pred = _measurement_model_weight(m_dict)

        if prediction_mode == "measurement":
            if not measurement_pred.get("available"):
                warnings_list.append(measurement_pred.get("reason", "Measurement model unavailable"))
                selected_weight = heuristic
                prediction_mode = "heuristic"
            else:
                selected_weight = measurement_pred
        else:
            selected_weight = heuristic

        weight_payload = {
            "selected_mode": prediction_mode,
            "heuristic": heuristic,
            "measurement_model": measurement_pred,
            "selected": selected_weight,
            "measurement_model_files_exist": measurement_model_available(),
        }
        with open(run_dir / "weight_prediction.json", "w", encoding="utf-8") as f:
            json.dump(weight_payload, f, indent=2)
        steps["weight"] = {"status": "completed", "mode": prediction_mode}
    except Exception as exc:
        steps["weight"] = {"status": "failed", "error": str(exc)}
        raise

    # Low-confidence keypoints
    low_kpts = [
        name for name, p in selected_det["keypoints"].items()
        if p["confidence"] < LOW_CONF_THRESHOLD
    ]
    if low_kpts:
        warnings_list.append(f"Low-confidence keypoints: {', '.join(low_kpts)}")

    elapsed = time.time() - t0
    report = {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "processing_time_sec": round(elapsed, 3),
        "steps": steps,
        "warnings": warnings_list,
        "disclaimer": "Estimated result only — not a replacement for a calibrated livestock weighing scale.",
        "num_cows_detected": len(detections),
        "selected_cow_id": selected_id,
        "detections": detections,
        "selected_detection": selected_det,
        "measurements": m_dict,
        "scale": scale_status,
        "normalized_features": normalized,
        "weight": weight_payload,
        "files": {
            name: _file_url(run_id, name)
            for name in (
                "original_image.jpg", "detection_image.jpg", "pose_image.jpg",
                "measurements_image.jpg", "keypoints.json", "measurements.json",
                "weight_prediction.json", "complete_report.json", "measurements.csv",
            )
        },
        "final": {
            "weight_kg": selected_weight.get("weight_kg"),
            "selected_cow_id": selected_id,
            "selected_model": prediction_mode,
            "num_cows": len(detections),
            "pose_status": steps["pose"]["status"],
            "segmentation_status": steps["segmentation"]["status"],
            "scale_status": "provided" if scale_status.get("provided") else "not_provided",
            "low_confidence_keypoints": low_kpts,
            "warnings": warnings_list,
            "processing_time_sec": round(elapsed, 3),
        },
    }
    if (run_dir / "segmentation_mask.png").is_file():
        report["files"]["segmentation_mask.png"] = _file_url(run_id, "segmentation_mask.png")
        report["files"]["segmentation_overlay.png"] = _file_url(run_id, "segmentation_overlay.png")

    with open(run_dir / "complete_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    steps["report"] = {"status": "completed"}
    report["steps"] = steps
    return report


def load_run(run_id: str) -> dict[str, Any]:
    path = RESULTS_DIR / run_id / "complete_report.json"
    if not path.is_file():
        raise FileNotFoundError(f"Run not found: {run_id}")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def reselect_cow(run_id: str, cow_id: int, enable_segmentation: bool = False,
                 prediction_mode: str = "heuristic",
                 reference_px: float | None = None,
                 reference_cm: float | None = None) -> dict[str, Any]:
    """Recompute measurements/weight for a different detected cow in an existing run."""
    t0 = time.time()
    run_dir = RESULTS_DIR / run_id
    report = load_run(run_id)
    detections = report["detections"]
    selected = next((d for d in detections if d["cow_id"] == cow_id), None)
    if selected is None:
        raise ValueError(f"Cow id {cow_id} not in run")

    img = cv2.imread(str(run_dir / "original_image.jpg"))
    if img is None:
        raise RuntimeError("Original image missing from run folder")

    det_img = _draw_detection_image(img, detections, cow_id)
    cv2.imwrite(str(run_dir / "detection_image.jpg"), det_img)

    # Rebuild pose image highlighting all (same as before)
    # Keep existing pose_image

    warnings_list: list[str] = []
    seg_info = None
    mask_for_meas = None
    if enable_segmentation:
        seg_info = segment_cow(img, selected.get("bbox"))
        if seg_info:
            mask_for_meas = seg_info["mask"]
            cv2.imwrite(str(run_dir / "segmentation_mask.png"), (mask_for_meas * 255).astype(np.uint8))
            overlay = overlay_segmentation(img, seg_info)
            cv2.imwrite(str(run_dir / "segmentation_overlay.png"), overlay)
        else:
            warnings_list.append("Segmentation enabled but no cow mask found")
    else:
        # Reuse existing run mask when present
        mask_path = run_dir / "segmentation_mask.png"
        if mask_path.is_file():
            raw = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if raw is not None:
                mask_for_meas = (np.squeeze(raw) > 127).astype(np.uint8)

    m = compute_measurements_from_keypoints(
        selected, min_keypoint_conf=MIN_KPT_CONF, mask=mask_for_meas,
    )
    if seg_info is not None:
        m.segmentation = seg_for_json(seg_info)

    scale_status = {
        "provided": False,
        "cm_per_px": None,
        "message": "No reference scale was provided. Pixel measurements are not real-world centimeters.",
    }
    if reference_px and reference_cm and reference_px > 0 and reference_cm > 0:
        cm_per_px = estimate_scale_from_reference(reference_px, reference_cm)
        apply_scale(m, cm_per_px)
        scale_status = {
            "provided": True,
            "reference_px": reference_px,
            "reference_cm": reference_cm,
            "cm_per_px": round(cm_per_px, 6),
            "message": f"Scale = {reference_cm} cm / {reference_px} px = {cm_per_px:.6f} cm/px",
        }

    meas_img = draw_measurements(img, m, unit="cm" if m.centimeters else "px")
    if seg_info is not None:
        meas_img = overlay_segmentation(meas_img, seg_info, alpha=0.25)
    cv2.imwrite(str(run_dir / "measurements_image.jpg"), meas_img)

    m_dict = m.to_dict()
    m_dict["measurement_lines"] = [
        {"p1": list(a), "p2": list(b), "name": n} for a, b, n in m.lines.segments
    ]
    m_dict["pixel_calculations"] = build_pixel_calculations(selected, m)
    normalized = build_normalized_features(m_dict)
    m_dict["normalized_features"] = normalized

    with open(run_dir / "measurements.json", "w", encoding="utf-8") as f:
        json.dump(m_dict, f, indent=2)

    heuristic = _heuristic_weight(run_dir / "original_image.jpg")
    measurement_pred = _measurement_model_weight(m_dict)
    mode = prediction_mode
    if mode == "measurement" and not measurement_pred.get("available"):
        warnings_list.append(measurement_pred.get("reason", "Measurement model unavailable"))
        mode = "heuristic"
        selected_weight = heuristic
    else:
        selected_weight = measurement_pred if mode == "measurement" else heuristic

    weight_payload = {
        "selected_mode": mode,
        "heuristic": heuristic,
        "measurement_model": measurement_pred,
        "selected": selected_weight,
        "measurement_model_files_exist": measurement_model_available(),
    }
    with open(run_dir / "weight_prediction.json", "w", encoding="utf-8") as f:
        json.dump(weight_payload, f, indent=2)

    low_kpts = [n for n, p in selected["keypoints"].items() if p["confidence"] < LOW_CONF_THRESHOLD]
    if low_kpts:
        warnings_list.append(f"Low-confidence keypoints: {', '.join(low_kpts)}")

    elapsed = time.time() - t0
    report["selected_cow_id"] = cow_id
    report["selected_detection"] = selected
    report["measurements"] = m_dict
    report["scale"] = scale_status
    report["normalized_features"] = normalized
    report["weight"] = weight_payload
    report["warnings"] = warnings_list
    report["processing_time_sec"] = round(report.get("processing_time_sec", 0) + elapsed, 3)
    report["final"] = {
        "weight_kg": selected_weight.get("weight_kg"),
        "selected_cow_id": cow_id,
        "selected_model": mode,
        "num_cows": report["num_cows_detected"],
        "pose_status": "completed",
        "segmentation_status": "completed" if seg_info else ("skipped" if not enable_segmentation else "failed"),
        "scale_status": "provided" if scale_status.get("provided") else "not_provided",
        "low_confidence_keypoints": low_kpts,
        "warnings": warnings_list,
        "processing_time_sec": report["processing_time_sec"],
    }
    with open(run_dir / "complete_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    return report


def apply_scale_to_run(run_id: str, reference_px: float, reference_cm: float) -> dict[str, Any]:
    report = load_run(run_id)
    run_dir = RESULTS_DIR / run_id
    m_dict = report["measurements"]
    cm_per_px = estimate_scale_from_reference(reference_px, reference_cm)
    px = m_dict.get("measurements_px") or {}
    m_dict["scale_cm_per_px"] = round(cm_per_px, 6)
    m_dict["measurements_cm"] = {
        k: (round(v * cm_per_px, 2) if v is not None else None) for k, v in px.items()
    }
    scale_status = {
        "provided": True,
        "reference_px": reference_px,
        "reference_cm": reference_cm,
        "cm_per_px": round(cm_per_px, 6),
        "message": f"Scale = {reference_cm} cm / {reference_px} px = {cm_per_px:.6f} cm/px",
    }
    report["measurements"] = m_dict
    report["scale"] = scale_status
    report["final"]["scale_status"] = "provided"
    with open(run_dir / "measurements.json", "w", encoding="utf-8") as f:
        json.dump(m_dict, f, indent=2)
    with open(run_dir / "complete_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    return report
