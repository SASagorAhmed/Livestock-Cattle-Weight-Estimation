"""Stage-based run pipeline for guided live UX.

Does not modify pose/weight model files. Reuses existing project modules.
"""

from __future__ import annotations

import csv
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
from werkzeug.utils import secure_filename

from config import (
    ALLOWED_EXTENSIONS,
    LOW_CONF_THRESHOLD,
    MAX_UPLOAD_BYTES,
    MIN_KPT_CONF,
    RESULTS_DIR,
)
from pipeline import (
    _detection_to_dict,
    _draw_detection_image,
    _draw_pose_image,
    _file_url,
    _heuristic_weight,
    _measurement_model_weight,
    apply_scale,
    build_normalized_features,
    build_pixel_calculations,
    estimate_scale_from_reference,
    get_pose_detector,
    get_weight_estimator,
    load_run,
    measurement_model_available,
    new_run_id,
)

# Import pose/measurement helpers via pipeline's sys.path setup
from cow_keypoints import (  # noqa: E402
    AP10K_KEYPOINT_NAMES,
    AP10K_SKELETON,
    CowPoseResult,
    draw_cow_pose,
)
from measurements import (  # noqa: E402
    compute_measurements_from_keypoints,
    draw_measurements,
)
from segmentation import overlay_segmentation, seg_for_json, segment_cow, compute_anatomical_regions  # noqa: E402
from body_contour_guide import bake_red_outline_image, body_contour_payload  # noqa: E402
import numpy as np


def _load_mask_u8(mask_path: Path) -> np.ndarray | None:
    if not mask_path.is_file():
        return None
    raw = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if raw is None or int(cv2.countNonZero(raw)) == 0:
        return None
    mask = (raw > 127).astype(np.uint8)
    # Some PNG loads / writes leave a trailing channel dim (H, W, 1)
    mask = np.squeeze(mask)
    if mask.ndim != 2:
        mask = np.atleast_2d(mask)
    return mask


def _seg_info_from_mask(mask_bin: np.ndarray, cow_bbox: list, model_name: str = "cached_mask") -> dict[str, Any]:
    """Build segment_cow-like metrics from an existing binary mask (no YOLO)."""
    mask_bin = np.squeeze(np.asarray(mask_bin))
    if mask_bin.ndim != 2:
        raise ValueError(f"Expected 2D mask, got shape {getattr(mask_bin, 'shape', None)}")
    if mask_bin.dtype != np.uint8:
        mask_bin = (mask_bin > 0).astype(np.uint8)
    elif int(mask_bin.max()) > 1:
        mask_bin = (mask_bin > 127).astype(np.uint8)
    x1, y1, x2, y2 = [int(v) for v in cow_bbox[:4]]
    h, w = mask_bin.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w - 1, x2), min(h - 1, y2)
    body_area = int(mask_bin.sum())
    contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    perimeter = float(cv2.arcLength(contours[0], True)) if contours else 0.0
    bw, bh = max(1, x2 - x1), max(1, y2 - y1)
    tx1 = int(x1 + 0.15 * bw)
    tx2 = int(x2 - 0.15 * bw)
    ty1 = int(y1 + 0.20 * bh)
    ty2 = int(y1 + 0.70 * bh)
    torso_mask = np.zeros_like(mask_bin)
    torso_mask[ty1:ty2, tx1:tx2] = mask_bin[ty1:ty2, tx1:tx2]
    torso_area = int(torso_mask.sum())
    belly_points: list[list[int]] = []
    if contours:
        pts = contours[0].reshape(-1, 2)
        xs = pts[:, 0]
        step = max(1, (tx2 - tx1) // 40)
        for x in range(tx1, tx2, step):
            col = pts[(xs >= x) & (xs < x + step)]
            if len(col):
                belly_points.append([int(x), int(col[:, 1].max())])
    return {
        "model": model_name,
        "bbox_iou": 1.0,
        "bbox_confidence": 1.0,
        "body_pixel_area": body_area,
        "torso_pixel_area": torso_area,
        "body_perimeter_px": round(perimeter, 2),
        "belly_boundary_points": belly_points,
        "torso_roi": [tx1, ty1, tx2, ty2],
        "mask": mask_bin,
    }


def _ensure_body_outline(
    run_id: str,
    run_dir: Path,
    img: np.ndarray,
    selected: dict[str, Any],
    report: dict[str, Any],
    *,
    log_prefix: str = "body-outline",
) -> tuple[dict[str, Any] | None, str | None, dict[str, str]]:
    """Display-only: YOLO-seg once → mask + body_outline.jpg (+ measure_outline alias).

    Does not update normalized_features / weight-model keys.
    Returns (body_contour, error, extra_files).
    """
    extra: dict[str, str] = {}
    mask_path = run_dir / "segmentation_mask.png"
    body_outline_path = run_dir / "body_outline.jpg"
    measure_outline_path = run_dir / "measure_outline.jpg"
    try:
        mask = _load_mask_u8(mask_path)
        if mask is None:
            bbox = selected.get("bbox")
            print(f"[{log_prefix}] running YOLO-seg bbox={bbox}", flush=True)
            seg_info = segment_cow(img, bbox)
            if seg_info is None or seg_info.get("mask") is None:
                err = "No cow mask from YOLO-seg"
                print(f"[{log_prefix}] FAIL: {err}", flush=True)
                return None, err, extra
            mask = seg_info["mask"]
            mask_u8 = (mask.astype(np.uint8) * 255) if np.max(mask) <= 1 else mask.astype(np.uint8)
            cv2.imwrite(str(mask_path), mask_u8)
            print(f"[{log_prefix}] saved mask {mask_path}", flush=True)
        else:
            print(f"[{log_prefix}] reused mask {mask_path}", flush=True)

        contour = body_contour_payload(mask)
        extra["segmentation_mask.png"] = _file_url(run_id, "segmentation_mask.png")
        report["files"]["segmentation_mask.png"] = extra["segmentation_mask.png"]

        ok = bake_red_outline_image(img, mask, body_outline_path, thickness=3)
        if ok and body_outline_path.is_file():
            extra["body_outline.jpg"] = _file_url(run_id, "body_outline.jpg")
            report["files"]["body_outline.jpg"] = extra["body_outline.jpg"]
            # Compatibility alias for Measure UI
            try:
                import shutil
                shutil.copy2(body_outline_path, measure_outline_path)
                extra["measure_outline.jpg"] = _file_url(run_id, "measure_outline.jpg")
                report["files"]["measure_outline.jpg"] = extra["measure_outline.jpg"]
            except OSError:
                pass
            print(f"[{log_prefix}] baked {body_outline_path}", flush=True)
            _save_report(run_id, report)
            return contour, None, extra

        err = "Failed to bake body_outline.jpg"
        print(f"[{log_prefix}] FAIL: {err}", flush=True)
        _save_report(run_id, report)
        return contour, err, extra
    except Exception as exc:  # noqa: BLE001
        err = f"{type(exc).__name__}: {exc}"
        print(f"[{log_prefix}] EXCEPTION: {err}", flush=True)
        return None, err, extra


def _run_dir(run_id: str) -> Path:
    d = RESULTS_DIR / run_id
    if not d.is_dir():
        raise FileNotFoundError(f"Run not found: {run_id}")
    return d


def _save_report(run_id: str, report: dict[str, Any]) -> None:
    path = RESULTS_DIR / run_id / "complete_report.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)


def _empty_report(run_id: str, enable_segmentation: bool, prediction_mode: str) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "enable_segmentation": enable_segmentation,
        "prediction_mode": prediction_mode,
        "processing_time_sec": 0.0,
        "steps": {
            "upload": {"status": "completed"},
            "detection": {"status": "pending"},
            "pose": {"status": "pending"},
            "measurements": {"status": "pending"},
            "pixel_calculations": {"status": "pending"},
            "segmentation": {"status": "pending" if enable_segmentation else "skipped"},
            "scale": {"status": "pending"},
            "features": {"status": "pending"},
            "weight": {"status": "pending"},
            "report": {"status": "pending"},
        },
        "warnings": [],
        "disclaimer": (
            "Estimated result only — not a replacement for a calibrated livestock weighing scale."
        ),
        "num_cows_detected": 0,
        "selected_cow_id": None,
        "detections": [],
        "selected_detection": None,
        "measurements": None,
        "scale": {
            "provided": False,
            "cm_per_px": None,
            "message": (
                "No reference scale was provided. "
                "Pixel measurements are not real-world centimeters."
            ),
        },
        "normalized_features": [],
        "weight": None,
        "files": {
            "original_image.jpg": _file_url(run_id, "original_image.jpg"),
            "complete_report.json": _file_url(run_id, "complete_report.json"),
        },
        "final": None,
        "pose_results_cache": None,  # not persisted as numpy; keypoints in detections
    }


def create_run(
    file_storage,
    enable_segmentation: bool = False,
    prediction_mode: str = "smartphone_diagonal",
) -> dict[str, Any]:
    filename = file_storage.filename or "upload.jpg"
    ext = Path(secure_filename(filename)).suffix.lower() or ".jpg"
    if ext not in ALLOWED_EXTENSIONS:
        raise ValueError(f"Unsupported format '{ext}'. Allowed: {sorted(ALLOWED_EXTENSIONS)}")
    if prediction_mode not in ("heuristic", "measurement", "smartphone_diagonal"):
        raise ValueError(
            "prediction_mode must be heuristic, measurement, or smartphone_diagonal"
        )

    run_id = new_run_id()
    run_dir = RESULTS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

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
    if tmp_path.resolve() != original_path.resolve():
        tmp_path.unlink(missing_ok=True)

    report = _empty_report(run_id, enable_segmentation, prediction_mode)
    h, w = img.shape[:2]
    report["image_size"] = {"width": w, "height": h}
    _save_report(run_id, report)

    return {
        "run_id": run_id,
        "status": "created",
        "enable_segmentation": enable_segmentation,
        "prediction_mode": prediction_mode,
        "image_size": report["image_size"],
        "files": {"original_image.jpg": _file_url(run_id, "original_image.jpg")},
    }


def stage_detect(run_id: str) -> dict[str, Any]:
    t0 = time.time()
    run_dir = _run_dir(run_id)
    report = load_run(run_id)
    report["steps"]["detection"] = {"status": "processing"}
    _save_report(run_id, report)

    img = cv2.imread(str(run_dir / "original_image.jpg"))
    if img is None:
        raise RuntimeError("Original image missing")

    detector = get_pose_detector()
    pose_results = detector.detect_image(run_dir / "original_image.jpg")
    if not pose_results:
        report["steps"]["detection"] = {"status": "failed", "error": "No cows detected"}
        _save_report(run_id, report)
        raise RuntimeError("No cows detected in the image")

    detections = [_detection_to_dict(r) for r in pose_results]
    primary = max(detections, key=lambda d: d["bbox_area"])
    selected_id = primary["cow_id"]

    # Cache raw keypoints arrays as lists inside detections (already there)
    # Also store pose_results as serializable for pose stage redraw
    det_img = _draw_detection_image(img, detections, selected_id)
    cv2.imwrite(str(run_dir / "detection_image.jpg"), det_img)

    # Save full keypoints payload early
    keypoints_payload = {
        "source": str(run_dir / "original_image.jpg"),
        "num_cows_detected": len(detections),
        "selected_cow_id": selected_id,
        "detections": detections,
        "keypoint_schema": {str(k): v for k, v in AP10K_KEYPOINT_NAMES.items()},
        "skeleton": AP10K_SKELETON,
    }
    with open(run_dir / "keypoints.json", "w", encoding="utf-8") as f:
        json.dump(keypoints_payload, f, indent=2)

    # Persist raw keypoints for pose image redraw (y,x,score arrays)
    raw_cache = {}
    for r in pose_results:
        raw_cache[str(r.cow_id)] = r.keypoints.tolist()
    with open(run_dir / "_pose_cache.json", "w", encoding="utf-8") as f:
        json.dump(raw_cache, f)

    report["num_cows_detected"] = len(detections)
    report["detections"] = detections
    report["selected_cow_id"] = selected_id
    report["selected_detection"] = primary
    report["needs_cow_selection"] = len(detections) > 1
    report["steps"]["detection"] = {
        "status": "completed",
        "num_cows": len(detections),
        "selected_cow_id": selected_id,
        "elapsed_sec": round(time.time() - t0, 3),
    }
    report["files"]["detection_image.jpg"] = _file_url(run_id, "detection_image.jpg")
    report["files"]["keypoints.json"] = _file_url(run_id, "keypoints.json")
    report["processing_time_sec"] = round(
        report.get("processing_time_sec", 0) + (time.time() - t0), 3
    )
    _save_report(run_id, report)

    return {
        "run_id": run_id,
        "stage": "detect",
        "status": "completed",
        "num_cows_detected": len(detections),
        "needs_cow_selection": len(detections) > 1,
        "selected_cow_id": selected_id,
        "detections": detections,
        "files": {
            "original_image.jpg": _file_url(run_id, "original_image.jpg"),
            "detection_image.jpg": _file_url(run_id, "detection_image.jpg"),
        },
        "image_size": report.get("image_size"),
    }


def stage_select_cow(run_id: str, cow_id: int) -> dict[str, Any]:
    run_dir = _run_dir(run_id)
    report = load_run(run_id)
    detections = report.get("detections") or []
    selected = next((d for d in detections if d["cow_id"] == cow_id), None)
    if selected is None:
        raise ValueError(f"Cow id {cow_id} not found in run")

    img = cv2.imread(str(run_dir / "original_image.jpg"))
    if img is None:
        raise RuntimeError("Original image missing")
    det_img = _draw_detection_image(img, detections, cow_id)
    cv2.imwrite(str(run_dir / "detection_image.jpg"), det_img)

    report["selected_cow_id"] = cow_id
    report["selected_detection"] = selected
    with open(run_dir / "keypoints.json", encoding="utf-8") as f:
        kp = json.load(f)
    kp["selected_cow_id"] = cow_id
    with open(run_dir / "keypoints.json", "w", encoding="utf-8") as f:
        json.dump(kp, f, indent=2)

    _save_report(run_id, report)
    return {
        "run_id": run_id,
        "stage": "select_cow",
        "status": "completed",
        "selected_cow_id": cow_id,
        "selected_detection": selected,
        "files": {"detection_image.jpg": _file_url(run_id, "detection_image.jpg")},
    }


def stage_pose(run_id: str) -> dict[str, Any]:
    t0 = time.time()
    run_dir = _run_dir(run_id)
    report = load_run(run_id)
    report["steps"]["pose"] = {"status": "processing"}
    _save_report(run_id, report)

    img = cv2.imread(str(run_dir / "original_image.jpg"))
    if img is None:
        raise RuntimeError("Original image missing")

    selected_id = report.get("selected_cow_id")
    if selected_id is None:
        raise RuntimeError("No cow selected")

    cache_path = run_dir / "_pose_cache.json"
    if not cache_path.is_file():
        raise RuntimeError("Pose cache missing; re-run detect")

    with open(cache_path, encoding="utf-8") as f:
        raw_cache = json.load(f)

    # Draw all cows, emphasize selected
    pose_results = []
    for d in report["detections"]:
        cid = str(d["cow_id"])
        if cid not in raw_cache:
            continue
        kpts = np.array(raw_cache[cid], dtype=np.float64)
        pose_results.append(CowPoseResult(
            cow_id=int(d["cow_id"]),
            bbox=d["bbox"],
            bbox_confidence=d["bbox_confidence"],
            keypoints=kpts,
        ))

    selected = report["selected_detection"]
    kpts = selected["keypoints"]

    # Display-only unified red outline FIRST so A End can snap to mask top
    body_contour, body_contour_error, outline_files = _ensure_body_outline(
        run_id, run_dir, img, selected, report, log_prefix="pose-outline",
    )
    mask = _load_mask_u8(run_dir / "segmentation_mask.png")

    pose_img = img.copy()
    for r in pose_results:
        pose_img = draw_cow_pose(
            pose_img, r,
            conf_threshold=LOW_CONF_THRESHOLD,
            draw_labels=(r.cow_id == selected_id),
            draw_bbox=True,
        )

    # Display-only Lower chest (same geometry as body_length; separate label)
    from lower_chest_guide import draw_lower_chest_line
    draw_lower_chest_line(pose_img, kpts)
    # Body height vertical + A End on red silhouette upper border
    from a_end_guide import body_height_axis, draw_a_end_vertical
    draw_a_end_vertical(pose_img, kpts, bbox=selected.get("bbox"), mask=mask)
    a_end_line = body_height_axis(kpts, bbox=selected.get("bbox"), mask=mask)
    cv2.imwrite(str(run_dir / "pose_image.jpg"), pose_img)

    ok_count = sum(1 for p in kpts.values() if p.get("confidence", 0) >= LOW_CONF_THRESHOLD)
    low = [n for n, p in kpts.items() if p.get("confidence", 0) < LOW_CONF_THRESHOLD]

    from cow_morpho_heuristic import infer_head_direction
    head_detected = any(
        kpts.get(n, {}).get("confidence", 0) >= LOW_CONF_THRESHOLD
        for n in ("nose", "left_eye", "right_eye")
    )
    head_direction = infer_head_direction(kpts)

    report["steps"]["pose"] = {
        "status": "completed",
        "detected_points": ok_count,
        "total_points": 17,
        "low_confidence": low,
        "elapsed_sec": round(time.time() - t0, 3),
    }
    report["files"]["pose_image.jpg"] = _file_url(run_id, "pose_image.jpg")
    report["processing_time_sec"] = round(
        report.get("processing_time_sec", 0) + (time.time() - t0), 3
    )
    _save_report(run_id, report)

    # Keypoint groups for frontend reveal animation
    groups = {
        "head": ["nose", "left_eye", "right_eye"],
        "upper_body": ["neck", "left_shoulder", "right_shoulder"],
        "front_legs": ["left_elbow", "right_elbow", "left_front_hoof", "right_front_hoof"],
        "rear_body": [
            "left_hip", "right_hip", "left_knee", "right_knee",
            "left_back_hoof", "right_back_hoof", "tail_root",
        ],
    }

    files_out = {
        "pose_image.jpg": _file_url(run_id, "pose_image.jpg"),
        "original_image.jpg": _file_url(run_id, "original_image.jpg"),
        **outline_files,
    }

    return {
        "run_id": run_id,
        "stage": "pose",
        "status": "completed",
        "selected_cow_id": selected_id,
        "selected_detection": selected,
        "keypoint_groups": groups,
        "skeleton": AP10K_SKELETON,
        "detected_points": ok_count,
        "total_points": 17,
        "low_confidence_keypoints": low,
        "head_direction": head_direction,
        "head_detected": head_detected,
        "body_contour": body_contour,
        "body_contour_error": body_contour_error,
        "a_end_line": a_end_line if a_end_line.get("detected") else None,
        "files": files_out,
        "image_size": report.get("image_size"),
    }


def stage_measure(run_id: str) -> dict[str, Any]:
    t0 = time.time()
    run_dir = _run_dir(run_id)
    report = load_run(run_id)
    report["steps"]["measurements"] = {"status": "processing"}
    _save_report(run_id, report)

    img = cv2.imread(str(run_dir / "original_image.jpg"))
    selected = report.get("selected_detection")
    if img is None or not selected:
        raise RuntimeError("Missing image or selected detection")

    # Shared YOLO mask first so Back top / A End snap to red silhouette
    body_contour, body_contour_error, outline_files = _ensure_body_outline(
        run_id, run_dir, img, selected, report, log_prefix="measure-outline",
    )
    mask = _load_mask_u8(run_dir / "segmentation_mask.png")

    m = compute_measurements_from_keypoints(
        selected, min_keypoint_conf=MIN_KPT_CONF, mask=mask,
    )
    meas_img = draw_measurements(img, m, unit="px")
    cv2.imwrite(str(run_dir / "measurements_image.jpg"), meas_img)

    m_dict = m.to_dict()
    m_dict["measurement_lines"] = [
        {"p1": list(a), "p2": list(b), "name": n} for a, b, n in m.lines.segments
    ]
    pixel_calcs = build_pixel_calculations(selected, m)
    m_dict["pixel_calculations"] = pixel_calcs
    # Features without seg yet
    m_dict["normalized_features"] = build_normalized_features(m_dict)

    with open(run_dir / "measurements.json", "w", encoding="utf-8") as f:
        json.dump(m_dict, f, indent=2)
    with open(run_dir / "measurements.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["measurement", "value_px", "value_cm"])
        for k, v in (m_dict.get("measurements_px") or {}).items():
            writer.writerow([k, v, None])

    report["measurements"] = m_dict
    report["normalized_features"] = m_dict["normalized_features"]
    report["steps"]["measurements"] = {"status": "completed", "elapsed_sec": round(time.time() - t0, 3)}
    report["steps"]["pixel_calculations"] = {"status": "completed"}
    report["files"]["measurements_image.jpg"] = _file_url(run_id, "measurements_image.jpg")
    report["files"]["measurements.json"] = _file_url(run_id, "measurements.json")
    report["files"]["measurements.csv"] = _file_url(run_id, "measurements.csv")
    report["processing_time_sec"] = round(
        report.get("processing_time_sec", 0) + (time.time() - t0), 3
    )
    _save_report(run_id, report)

    # Ordered measurement sequence for UI autoplay
    order = [
        "body_length", "lower_chest", "body_height", "chest_depth_proxy",
        "left_front_leg_length", "right_front_leg_length",
        "left_back_leg_length", "right_back_leg_length",
        "torso_diagonal", "shoulder_width", "hip_width",
    ]
    calc_map = {c["name"]: c for c in pixel_calcs}
    sequence = [calc_map[n] for n in order if n in calc_map]

    files_out: dict[str, str] = {
        "original_image.jpg": _file_url(run_id, "original_image.jpg"),
        "measurements_image.jpg": _file_url(run_id, "measurements_image.jpg"),
        **outline_files,
    }

    return {
        "run_id": run_id,
        "stage": "measure",
        "status": "completed",
        "measurements": m_dict,
        "measurement_sequence": sequence,
        "body_contour": body_contour,
        "body_contour_error": body_contour_error,
        "files": files_out,
        "image_size": report.get("image_size"),
    }


def stage_segment(run_id: str) -> dict[str, Any]:
    t0 = time.time()
    run_dir = _run_dir(run_id)
    report = load_run(run_id)

    if not report.get("enable_segmentation"):
        report["steps"]["segmentation"] = {"status": "skipped"}
        _save_report(run_id, report)
        return {
            "run_id": run_id,
            "stage": "segment",
            "status": "skipped",
            "message": "Segmentation was disabled for this run.",
            "segmentation": None,
        }

    report["steps"]["segmentation"] = {"status": "processing"}
    _save_report(run_id, report)

    img = cv2.imread(str(run_dir / "original_image.jpg"))
    selected = report.get("selected_detection")
    if img is None or not selected:
        raise RuntimeError("Missing image or detection")

    bbox = selected["bbox"]
    mask_path = run_dir / "segmentation_mask.png"
    cached = _load_mask_u8(mask_path)
    if cached is not None:
        print(f"[segment] reusing mask {mask_path}", flush=True)
        seg_info = _seg_info_from_mask(cached, bbox)
    else:
        seg_info = segment_cow(img, bbox)
    if seg_info is None:
        report["steps"]["segmentation"] = {"status": "failed", "error": "No cow mask found"}
        report["warnings"] = list(set(report.get("warnings", []) + [
            "Segmentation enabled but no cow mask found"
        ]))
        _save_report(run_id, report)
        return {
            "run_id": run_id,
            "stage": "segment",
            "status": "failed",
            "error": "No cow mask found",
            "segmentation": None,
        }

    mask = seg_info["mask"]
    mask = np.squeeze(np.asarray(mask))
    if mask.ndim != 2:
        raise RuntimeError(f"Invalid segmentation mask shape: {mask.shape}")
    seg_info["mask"] = mask
    from cow_morpho_heuristic import infer_head_direction
    kpts = selected.get("keypoints") or {}
    hd = infer_head_direction(kpts)
    seg_info.update(compute_anatomical_regions(mask, kpts, hd, bbox=bbox))
    cv2.imwrite(str(run_dir / "segmentation_mask.png"), (mask * 255).astype(np.uint8) if np.max(mask) <= 1 else mask.astype(np.uint8))
    overlay = overlay_segmentation(img, seg_info, keypoints=kpts)
    # Same Morpho-style red silhouette on segmentation overlay
    mask_draw = (mask * 255).astype(np.uint8) if np.max(mask) <= 1 else mask.astype(np.uint8)
    contours, _ = cv2.findContours(mask_draw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cv2.drawContours(overlay, contours, -1, (0, 0, 255), 3)
    cv2.imwrite(str(run_dir / "segmentation_overlay.png"), overlay)
    seg_json = seg_for_json(seg_info)

    # Keep unified outline assets fresh
    body_contour, body_contour_error, outline_files = _ensure_body_outline(
        run_id, run_dir, img, selected, report, log_prefix="segment-outline",
    )

    m_dict = report.get("measurements") or {}
    m_dict["segmentation"] = seg_json
    m_dict["normalized_features"] = build_normalized_features(m_dict)
    report["measurements"] = m_dict
    report["normalized_features"] = m_dict["normalized_features"]
    with open(run_dir / "measurements.json", "w", encoding="utf-8") as f:
        json.dump(m_dict, f, indent=2)

    report["steps"]["segmentation"] = {
        "status": "completed",
        "body_pixel_area": seg_json.get("body_pixel_area"),
        "torso_pixel_area": seg_json.get("torso_pixel_area"),
        "body_perimeter_px": seg_json.get("body_perimeter_px"),
        "elapsed_sec": round(time.time() - t0, 3),
    }
    report["files"]["segmentation_mask.png"] = _file_url(run_id, "segmentation_mask.png")
    report["files"]["segmentation_overlay.png"] = _file_url(run_id, "segmentation_overlay.png")
    report["processing_time_sec"] = round(
        report.get("processing_time_sec", 0) + (time.time() - t0), 3
    )
    _save_report(run_id, report)

    files_out = {
        "original_image.jpg": _file_url(run_id, "original_image.jpg"),
        "segmentation_mask.png": _file_url(run_id, "segmentation_mask.png"),
        "segmentation_overlay.png": _file_url(run_id, "segmentation_overlay.png"),
        **outline_files,
    }

    return {
        "run_id": run_id,
        "stage": "segment",
        "status": "completed",
        "segmentation": seg_json,
        "body_contour": body_contour,
        "body_contour_error": body_contour_error,
        "image_size": report.get("image_size"),
        "files": files_out,
    }


def stage_scale(
    run_id: str,
    reference_px: float | None = None,
    reference_cm: float | None = None,
    skip: bool = False,
    point_a: dict | None = None,
    point_b: dict | None = None,
    four_points: dict | None = None,
) -> dict[str, Any]:
    from diagonal_formula import compute_diagonal_formula

    run_dir = _run_dir(run_id)
    report = load_run(run_id)
    m_dict = report.get("measurements") or {}
    mode = report.get("prediction_mode", "smartphone_diagonal")

    if mode == "smartphone_diagonal" and (skip or not reference_px or not reference_cm):
        # Prefer real image-pixel distance from the two clicked reference endpoints
        if (
            isinstance(point_a, dict) and isinstance(point_b, dict)
            and "x" in point_a and "y" in point_a
            and "x" in point_b and "y" in point_b
            and reference_cm
        ):
            dx = float(point_b["x"]) - float(point_a["x"])
            dy = float(point_b["y"]) - float(point_a["y"])
            reference_px = float((dx * dx + dy * dy) ** 0.5)
        if skip or not reference_px or not reference_cm:
            raise ValueError(
                "Smartphone Diagonal Formula requires a valid reference scale "
                "(cannot skip). Provide reference_px and reference_cm."
            )

    # Recompute reference_px from image-space endpoints when available
    if (
        not skip
        and isinstance(point_a, dict)
        and isinstance(point_b, dict)
        and "x" in point_a and "y" in point_a
        and "x" in point_b and "y" in point_b
    ):
        dx = float(point_b["x"]) - float(point_a["x"])
        dy = float(point_b["y"]) - float(point_a["y"])
        measured = float((dx * dx + dy * dy) ** 0.5)
        if measured > 0:
            reference_px = measured

    if skip or not reference_px or not reference_cm:
        scale_status = {
            "provided": False,
            "reference_scale_used": False,
            "cm_per_px": None,
            "reference_px": None,
            "reference_cm": None,
            "point_a": None,
            "point_b": None,
            "reference_pixels": None,
            "reference_length_cm": None,
            "converted_measurements": None,
            "message": (
                "No reference scale was provided. "
                "Pixel measurements and normalised features will be used."
            ),
        }
        report["scale"] = scale_status
        report["steps"]["scale"] = {"status": "skipped", **scale_status}
        report["steps"]["features"] = {"status": "completed"}
        _save_report(run_id, report)
        return {
            "run_id": run_id,
            "stage": "scale",
            "status": "skipped",
            "scale": scale_status,
            "measurements": m_dict,
            "normalized_features": report.get("normalized_features") or [],
        }

    cm_per_px = estimate_scale_from_reference(float(reference_px), float(reference_cm))
    px = m_dict.get("measurements_px") or {}
    measurements_cm = {
        k: (round(v * cm_per_px, 2) if v is not None else None) for k, v in px.items()
    }

    # Area uses squared scale; perimeter is linear
    seg = m_dict.get("segmentation") or {}
    converted: dict[str, Any] = dict(measurements_cm)
    if seg.get("body_perimeter_px") is not None:
        converted["body_perimeter_cm"] = round(float(seg["body_perimeter_px"]) * cm_per_px, 2)
    if seg.get("body_pixel_area") is not None:
        converted["body_area_cm2"] = round(float(seg["body_pixel_area"]) * (cm_per_px ** 2), 2)
    if seg.get("torso_pixel_area") is not None:
        converted["torso_area_cm2"] = round(float(seg["torso_pixel_area"]) * (cm_per_px ** 2), 2)

    m_dict["scale_cm_per_px"] = round(cm_per_px, 6)
    m_dict["measurements_cm"] = measurements_cm
    m_dict["converted_measurements"] = converted

    scale_status = {
        "provided": True,
        "reference_scale_used": True,
        "reference_px": float(reference_px),
        "reference_cm": float(reference_cm),
        "reference_pixels": float(reference_px),
        "reference_length_cm": float(reference_cm),
        "cm_per_px": round(cm_per_px, 6),
        "point_a": point_a,
        "point_b": point_b,
        "converted_measurements": converted,
        "message": (
            f"Scale = {reference_cm} cm / {reference_px} px = {cm_per_px:.6f} cm/px"
        ),
    }
    report["measurements"] = m_dict
    report["scale"] = scale_status
    # Four-point diagonal inputs (optional; required later for smartphone_diagonal predict)
    if four_points is not None:
        report["four_points"] = four_points
        report["diagonal_formula"] = compute_diagonal_formula(
            four_points if isinstance(four_points, dict) else {},
            scale_status.get("cm_per_px"),
        )
    elif "four_points" in report and scale_status.get("cm_per_px"):
        # Recompute when scale changes even if four_points not resent
        report["diagonal_formula"] = compute_diagonal_formula(
            report.get("four_points") or {},
            scale_status.get("cm_per_px"),
        )
    else:
        report["diagonal_formula"] = None

    report["steps"]["scale"] = {"status": "completed", **scale_status}
    report["steps"]["features"] = {"status": "completed"}

    with open(run_dir / "measurements.json", "w", encoding="utf-8") as f:
        json.dump(m_dict, f, indent=2)
    with open(run_dir / "measurements.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["measurement", "value_px", "value_cm"])
        for k, v in px.items():
            writer.writerow([k, v, measurements_cm.get(k)])
        if "body_perimeter_cm" in converted:
            writer.writerow(["body_perimeter", seg.get("body_perimeter_px"), converted["body_perimeter_cm"]])
        if "body_area_cm2" in converted:
            writer.writerow(["body_area_px2", seg.get("body_pixel_area"), converted["body_area_cm2"]])
        if "torso_area_cm2" in converted:
            writer.writerow(["torso_area_px2", seg.get("torso_pixel_area"), converted["torso_area_cm2"]])

    # Refresh measurement image with cm labels (display only; model features unchanged)
    img = cv2.imread(str(run_dir / "original_image.jpg"))
    if img is not None and report.get("selected_detection"):
        selected = report["selected_detection"]
        mask = _load_mask_u8(run_dir / "segmentation_mask.png")
        m = compute_measurements_from_keypoints(
            selected, min_keypoint_conf=MIN_KPT_CONF, mask=mask,
        )
        apply_scale(m, cm_per_px)
        meas_img = draw_measurements(img, m, unit="cm")
        cv2.imwrite(str(run_dir / "measurements_image.jpg"), meas_img)

    _save_report(run_id, report)
    return {
        "run_id": run_id,
        "stage": "scale",
        "status": "completed",
        "scale": scale_status,
        "measurements": m_dict,
        "four_points": report.get("four_points"),
        "diagonal_formula": report.get("diagonal_formula"),
        "normalized_features": report.get("normalized_features") or m_dict.get("normalized_features") or [],
    }


def stage_suggest_four_points(
    run_id: str,
    head_direction: str | None = None,
) -> dict[str, Any]:
    """Auto-suggest 4 diagonal keypoints via CowMorphoHeuristic.

    Does not auto-commit into report['four_points'] — UI applies after review.
    """
    from cow_morpho_heuristic import suggest_four_points_morpho

    run_dir = _run_dir(run_id)
    report = load_run(run_id)
    image_path = run_dir / "original_image.jpg"
    mask_path = run_dir / "segmentation_mask.png"
    selected = report.get("selected_detection") or {}
    bbox = selected.get("bbox")
    kpts = selected.get("keypoints") or {}
    seg = (report.get("measurements") or {}).get("segmentation") or {}
    belly_hints = seg.get("belly_boundary_points")

    # Reuse Pose/Measure YOLO mask; regenerate if missing so Morpho shares one mask
    if not mask_path.is_file():
        img = cv2.imread(str(image_path))
        if img is not None and selected:
            _ensure_body_outline(
                run_id, run_dir, img, selected, report, log_prefix="morpho-outline",
            )
            _save_report(run_id, report)

    debug_path = run_dir / "four_point_morpho_debug.jpg"
    suggestion = suggest_four_points_morpho(
        mask_path=mask_path,
        keypoints=kpts,
        bbox=bbox,
        head_direction=head_direction,
        image_path=image_path,
        debug_out_path=debug_path if mask_path.is_file() else None,
        belly_boundary_points=belly_hints,
    )

    if suggestion.get("available") and suggestion.get("debug_image"):
        report["files"] = report.get("files") or {}
        report["files"]["four_point_morpho_debug.jpg"] = _file_url(
            run_id, "four_point_morpho_debug.jpg"
        )

    report["four_point_suggestion"] = suggestion
    _save_report(run_id, report)

    files = {}
    if (run_dir / "four_point_morpho_debug.jpg").is_file():
        files["four_point_morpho_debug.jpg"] = _file_url(run_id, "four_point_morpho_debug.jpg")

    return {
        "run_id": run_id,
        "stage": "four_point_suggest",
        "status": "completed" if suggestion.get("available") else "unavailable",
        "model_available": True,
        "files": files,
        **suggestion,
    }


def stage_redraw_four_point_debug(
    run_id: str,
    keypoints: dict[str, Any],
    *,
    a_end_line: dict | None = None,
    lower_chest_guide: dict | None = None,
    head_anchor: dict | None = None,
    head_direction: str | None = None,
) -> dict[str, Any]:
    """Redraw Morpho debug JPEG from current (possibly manual) keypoints.

    Does not re-run the heuristic or overwrite the user's point coordinates.
    """
    from cow_morpho_heuristic import draw_debug_image
    from a_end_guide import synthesize_a_end_line

    run_dir = _run_dir(run_id)
    report = load_run(run_id)
    image_path = run_dir / "original_image.jpg"
    mask_path = run_dir / "segmentation_mask.png"
    selected = report.get("selected_detection") or {}
    bbox = selected.get("bbox")
    pose_kpts = selected.get("keypoints") or {}

    if not isinstance(keypoints, dict) or not keypoints:
        raise ValueError("keypoints required to redraw debug image")

    required = (
        "A_start_lower_chest",
        "A_end_withers",
        "B_start_tail_head",
        "B_end_shoulder_region",
    )
    keypoints_out: dict[str, dict[str, Any]] = {}
    for name in required:
        p = keypoints.get(name)
        if not isinstance(p, dict) or "x" not in p or "y" not in p:
            raise ValueError(f"Missing keypoint {name}")
        keypoints_out[name] = {
            "x": float(p["x"]),
            "y": float(p["y"]),
            "name": name,
            "status": p.get("status") or "manual_corrected",
            "method": p.get("method") or "manual",
            "confidence": float(p.get("confidence") or 1.0),
            "anatomy_label": p.get("anatomy_label"),
        }

    img = cv2.imread(str(image_path))
    if img is None:
        raise RuntimeError("original_image.jpg missing")

    if not mask_path.is_file():
        _ensure_body_outline(
            run_id, run_dir, img, selected, report, log_prefix="morpho-debug-outline",
        )
        _save_report(run_id, report)

    mask = _load_mask_u8(mask_path)
    if mask is None:
        raise RuntimeError("segmentation_mask.png unavailable for debug redraw")

    ae = a_end_line if isinstance(a_end_line, dict) else None
    a_end_pt = keypoints_out["A_end_withers"]
    if not (ae and ae.get("detected") and ae.get("a_end") and ae.get("ground")):
        ae = synthesize_a_end_line(
            (a_end_pt["x"], a_end_pt["y"]),
            keypoints=pose_kpts,
            bbox=bbox,
            mask=mask,
        )
    else:
        # Keep vertical under current A End tip
        ae = {
            **ae,
            "detected": True,
            "a_end": [float(a_end_pt["x"]), float(a_end_pt["y"])],
            "p1": [float(a_end_pt["x"]), float(a_end_pt["y"])],
            "ground": [float(a_end_pt["x"]), float(ae["ground"][1])],
            "p2": [float(a_end_pt["x"]), float(ae["ground"][1])],
            "label": ae.get("label") or "A End",
            "line_label": ae.get("line_label") or "Body height",
        }

    guide = lower_chest_guide if isinstance(lower_chest_guide, dict) else None
    debug_path = run_dir / "four_point_morpho_debug.jpg"
    draw_debug_image(
        img,
        mask,
        keypoints_out,
        debug_path,
        head_anchor=head_anchor if isinstance(head_anchor, dict) else None,
        pose_keypoints=pose_kpts,
        head_dir=head_direction,
        lower_chest_guide=guide,
        a_end_line=ae,
        bbox=list(bbox[:4]) if bbox and len(bbox) >= 4 else None,
    )

    report["files"] = report.get("files") or {}
    report["files"]["four_point_morpho_debug.jpg"] = _file_url(
        run_id, "four_point_morpho_debug.jpg"
    )
    _save_report(run_id, report)

    return {
        "run_id": run_id,
        "stage": "four_point_debug",
        "status": "completed",
        "a_end_line": ae,
        "files": {
            "four_point_morpho_debug.jpg": report["files"]["four_point_morpho_debug.jpg"],
        },
    }


def stage_predict(run_id: str) -> dict[str, Any]:
    t0 = time.time()
    run_dir = _run_dir(run_id)
    report = load_run(run_id)
    report["steps"]["weight"] = {"status": "processing"}
    _save_report(run_id, report)

    # Warm cache
    get_weight_estimator()

    stages_progress = [
        "Analysing body proportions",
        "Preparing model input",
        "Running local weight model",
        "Finalising estimate",
    ]

    m_dict = report.get("measurements") or {}
    mode = report.get("prediction_mode", "smartphone_diagonal")
    warnings_list = list(report.get("warnings") or [])

    heuristic = _heuristic_weight(run_dir / "original_image.jpg")
    measurement_pred = _measurement_model_weight(m_dict)

    from diagonal_formula import compute_diagonal_formula
    scale_info = report.get("scale") or {}
    four_points = report.get("four_points") or {}
    # Always recompute from current points+scale — never reuse stale weight
    diagonal_pred = compute_diagonal_formula(
        four_points if isinstance(four_points, dict) else {},
        scale_info.get("cm_per_px"),
    )
    report["diagonal_formula"] = diagonal_pred

    if mode == "smartphone_diagonal":
        if not diagonal_pred.get("available"):
            raise ValueError(
                diagonal_pred.get("reason")
                or "Smartphone Diagonal Formula is not ready (need scale + 4 points)."
            )
        selected_weight = {
            "available": True,
            "weight_kg": diagonal_pred["weight_kg"],
            "weight_lb": diagonal_pred["weight_lb"],
            "source": "smartphone_diagonal_formula",
            "method": diagonal_pred["method"],
            "status": diagonal_pred["status"],
            "details": diagonal_pred,
        }
        # Keep heuristic for comparison only — do not feed 4 points into h5
    elif mode == "measurement":
        if not measurement_pred.get("available"):
            warnings_list.append(measurement_pred.get("reason", "Measurement model unavailable"))
            selected_weight = heuristic
            mode = "heuristic"
        else:
            selected_weight = measurement_pred
    else:
        selected_weight = heuristic

    weight_payload = {
        "selected_mode": mode,
        "heuristic": heuristic,
        "measurement_model": measurement_pred,
        "smartphone_diagonal": diagonal_pred,
        "selected": selected_weight,
        "measurement_model_files_exist": measurement_model_available(),
        "progress_stages": stages_progress,
    }
    with open(run_dir / "weight_prediction.json", "w", encoding="utf-8") as f:
        json.dump(weight_payload, f, indent=2)

    selected = report.get("selected_detection") or {}
    low_kpts = [
        n for n, p in (selected.get("keypoints") or {}).items()
        if p.get("confidence", 0) < LOW_CONF_THRESHOLD
    ]
    if low_kpts:
        warnings_list.append(f"Low-confidence keypoints: {', '.join(low_kpts)}")

    elapsed = round(report.get("processing_time_sec", 0) + (time.time() - t0), 3)
    report["weight"] = weight_payload
    report["warnings"] = warnings_list
    report["processing_time_sec"] = elapsed
    report["steps"]["weight"] = {"status": "completed", "mode": mode}
    report["steps"]["report"] = {"status": "completed"}
    report["files"]["weight_prediction.json"] = _file_url(run_id, "weight_prediction.json")
    scale_info = report.get("scale") or {}
    cm_vals = m_dict.get("measurements_cm") or {}
    diag = report.get("diagonal_formula") or {}
    report["final"] = {
        "weight_kg": selected_weight.get("weight_kg"),
        "weight_lb": selected_weight.get("weight_lb") or diag.get("weight_lb"),
        "selected_cow_id": report.get("selected_cow_id"),
        "selected_model": "Cow Morpho Heuristic" if mode == "smartphone_diagonal" else mode,
        "num_cows": report.get("num_cows_detected"),
        "pose_status": report["steps"]["pose"].get("status"),
        "segmentation_status": report["steps"]["segmentation"].get("status"),
        "scale_status": "provided" if scale_info.get("provided") else "not_provided",
        "reference_scale_used": bool(scale_info.get("reference_scale_used") or scale_info.get("provided")),
        "detected_points": report["steps"]["pose"].get("detected_points"),
        "body_length_px": (m_dict.get("measurements_px") or {}).get("body_length"),
        "body_height_px": (m_dict.get("measurements_px") or {}).get("body_height"),
        "chest_depth_proxy_px": (m_dict.get("measurements_px") or {}).get("chest_depth_proxy"),
        "body_length_cm": cm_vals.get("body_length"),
        "body_height_cm": cm_vals.get("body_height"),
        "chest_depth_proxy_cm": cm_vals.get("chest_depth_proxy"),
        "cm_per_px": scale_info.get("cm_per_px"),
        "low_confidence_keypoints": low_kpts,
        "warnings": warnings_list,
        "processing_time_sec": elapsed,
        "smartphone_diagonal": diag if diag.get("available") else None,
        "A_px": diag.get("A_px"),
        "B_px": diag.get("B_px"),
        "A_cm": diag.get("A_cm"),
        "B_cm": diag.get("B_cm"),
        "estimated_heart_girth_C_cm": diag.get("estimated_heart_girth_C_cm"),
        "estimated_heart_girth_C_in": diag.get("estimated_heart_girth_C_in"),
        "diagonal_method": diag.get("method"),
        "diagonal_status": diag.get("status"),
        "point_detector": diag.get("point_detector") or (report.get("four_point_suggestion") or {}).get("point_detector"),
    }
    _save_report(run_id, report)

    return {
        "run_id": run_id,
        "stage": "predict",
        "status": "completed",
        "progress_stages": stages_progress,
        "weight": weight_payload,
        "final": report["final"],
        "report": report,
        "files": report["files"],
    }
