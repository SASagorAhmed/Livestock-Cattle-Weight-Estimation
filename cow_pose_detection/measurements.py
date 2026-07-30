"""Cow body measurements from pose JSON (pixel-based + normalized ratios).

Reads cow_pose_detection keypoints JSON, selects the primary cow by largest
bounding box, and computes morphometric distances without modifying pose models.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def _dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a[:2] - b[:2]))


def _mid(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a[:2] + b[:2]) / 2.0


def _pt(kpts: dict[str, dict], name: str) -> np.ndarray | None:
    """Return (x, y, conf) or None if missing/invalid."""
    k = kpts.get(name)
    if k is None:
        return None
    return np.array([float(k["x"]), float(k["y"]), float(k["confidence"])], dtype=np.float64)


def _chain_length(points: list[np.ndarray | None]) -> float | None:
    """Sum consecutive segment lengths; None if fewer than 2 valid points."""
    valid = [p for p in points if p is not None]
    if len(valid) < 2:
        return None
    total = 0.0
    for i in range(len(valid) - 1):
        total += _dist(valid[i], valid[i + 1])
    return total


def bbox_area(bbox: list[int] | list[float]) -> float:
    x1, y1, x2, y2 = bbox
    return max(0.0, float(x2 - x1)) * max(0.0, float(y2 - y1))


def select_primary_cow(detections: list[dict[str, Any]]) -> dict[str, Any]:
    """Pick the detection with the largest bounding box area."""
    if not detections:
        raise ValueError("No cow detections in pose JSON")
    return max(detections, key=lambda d: bbox_area(d["bbox"]))


def load_pose_json(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@dataclass
class MeasurementLines:
    """Endpoints used for drawing annotated measurement overlays."""
    segments: list[tuple[tuple[float, float], tuple[float, float], str]] = field(default_factory=list)


@dataclass
class CowMeasurements:
    cow_id: int
    bbox: list[int]
    bbox_confidence: float
    pixels: dict[str, float | None]
    ratios: dict[str, float | None]
    landmarks: dict[str, list[float] | None]
    lines: MeasurementLines
    scale_cm_per_px: float | None = None
    centimeters: dict[str, float | None] | None = None
    segmentation: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "cow_id": self.cow_id,
            "bbox": self.bbox,
            "bbox_confidence": round(self.bbox_confidence, 4),
            "measurements_px": {
                k: (round(v, 2) if v is not None else None) for k, v in self.pixels.items()
            },
            "ratios": {
                k: (round(v, 4) if v is not None else None) for k, v in self.ratios.items()
            },
            "landmarks": self.landmarks,
        }
        if self.scale_cm_per_px is not None:
            out["scale_cm_per_px"] = round(self.scale_cm_per_px, 6)
            out["measurements_cm"] = {
                k: (round(v, 2) if v is not None else None)
                for k, v in (self.centimeters or {}).items()
            }
        if self.segmentation is not None:
            out["segmentation"] = self.segmentation
        return out


def compute_measurements_from_keypoints(
    detection: dict[str, Any],
    min_keypoint_conf: float = 0.2,
    mask: Any | None = None,
) -> CowMeasurements:
    """Compute pixel measurements and ratios from one cow detection dict.

    Optional ``mask`` (YOLO-seg binary) snaps display A End / body_h line to the
    silhouette top. Model ``body_height`` px still uses keypoint back_top.
    """
    kpts_raw = detection["keypoints"]
    # Filter low-confidence keypoints to None
    kpts: dict[str, np.ndarray | None] = {}
    for name in (
        "nose", "left_eye", "right_eye", "neck", "tail_root",
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_front_hoof", "right_front_hoof",
        "left_hip", "right_hip", "left_knee", "right_knee",
        "left_back_hoof", "right_back_hoof",
    ):
        p = _pt(kpts_raw, name)
        if p is None or p[2] < min_keypoint_conf:
            kpts[name] = None
        else:
            kpts[name] = p

    ls, rs = kpts["left_shoulder"], kpts["right_shoulder"]
    lh, rh = kpts["left_hip"], kpts["right_hip"]
    le, re_ = kpts["left_elbow"], kpts["right_elbow"]
    lf, rf = kpts["left_front_hoof"], kpts["right_front_hoof"]
    lk, rk = kpts["left_knee"], kpts["right_knee"]
    lb, rb = kpts["left_back_hoof"], kpts["right_back_hoof"]
    neck, tail = kpts["neck"], kpts["tail_root"]

    lines = MeasurementLines()
    landmarks: dict[str, list[float] | None] = {}

    # --- Shoulder / hip width ---
    shoulder_width = _dist(ls, rs) if ls is not None and rs is not None else None
    if shoulder_width is not None:
        lines.segments.append(((ls[0], ls[1]), (rs[0], rs[1]), "shoulder_w"))

    hip_width = _dist(lh, rh) if lh is not None and rh is not None else None
    if hip_width is not None:
        lines.segments.append(((lh[0], lh[1]), (rh[0], rh[1]), "hip_w"))

    # --- Centers ---
    shoulder_center = _mid(ls, rs) if ls is not None and rs is not None else None
    hip_center = _mid(lh, rh) if lh is not None and rh is not None else None
    elbow_center = _mid(le, re_) if le is not None and re_ is not None else None

    if shoulder_center is not None:
        landmarks["shoulder_center"] = [round(float(shoulder_center[0]), 2),
                                        round(float(shoulder_center[1]), 2)]
    else:
        landmarks["shoulder_center"] = None
    if hip_center is not None:
        landmarks["hip_center"] = [round(float(hip_center[0]), 2),
                                   round(float(hip_center[1]), 2)]
    else:
        landmarks["hip_center"] = None

    # --- Body length (shoulder center -> hip center) ---
    body_length = (
        _dist(shoulder_center, hip_center)
        if shoulder_center is not None and hip_center is not None
        else None
    )
    if body_length is not None:
        lines.segments.append((
            (float(shoulder_center[0]), float(shoulder_center[1])),
            (float(hip_center[0]), float(hip_center[1])),
            "body_len",
        ))
        # NEW display-only parallel line (same geometry; separate label)
        lines.segments.append((
            (float(shoulder_center[0]), float(shoulder_center[1])),
            (float(hip_center[0]), float(hip_center[1])),
            "lower_chest",
        ))

    # --- Body height: neck/back to ground (mean hoof y), vertical component ---
    # Model body_height uses keypoint back_top; display vertical extends to A End (upper border).
    hoof_ys = [p[1] for p in (lf, rf, lb, rb) if p is not None]
    top_ys = [p[1] for p in (neck, tail, ls, rs) if p is not None]
    body_height = None
    if hoof_ys and top_ys:
        top_y = float(min(top_ys))
        ground_y = float(np.mean(hoof_ys))
        body_height = abs(ground_y - top_y)
        hx = float(shoulder_center[0]) if shoulder_center is not None else float(
            (detection["bbox"][0] + detection["bbox"][2]) / 2
        )
        landmarks["back_top"] = [round(hx, 2), round(top_y, 2)]
        landmarks["ground"] = [round(hx, 2), round(ground_y, 2)]
        try:
            from a_end_guide import body_height_axis
            axis = body_height_axis(
                detection.get("keypoints"),
                bbox=detection.get("bbox"),
                mask=mask,
            )
            if axis.get("detected") and axis.get("a_end"):
                landmarks["a_end"] = axis["a_end"]
                ae = axis["a_end"]
                lines.segments.append((
                    (float(ae[0]), float(ae[1])),
                    (hx, ground_y),
                    "body_h",
                ))
            else:
                landmarks["a_end"] = [round(hx, 2), round(top_y, 2)]
                lines.segments.append(((hx, top_y), (hx, ground_y), "body_h"))
        except Exception:
            landmarks["a_end"] = [round(hx, 2), round(top_y, 2)]
            lines.segments.append(((hx, top_y), (hx, ground_y), "body_h"))

    # --- Leg lengths (shoulder/hip -> elbow/knee -> hoof) ---
    left_front_leg = _chain_length([ls, le, lf])
    right_front_leg = _chain_length([rs, re_, rf])
    left_back_leg = _chain_length([lh, lk, lb])
    right_back_leg = _chain_length([rh, rk, rb])

    if ls is not None and le is not None:
        lines.segments.append(((ls[0], ls[1]), (le[0], le[1]), "LF_leg"))
    if le is not None and lf is not None:
        lines.segments.append(((le[0], le[1]), (lf[0], lf[1]), "LF_leg"))
    if rs is not None and re_ is not None:
        lines.segments.append(((rs[0], rs[1]), (re_[0], re_[1]), "RF_leg"))
    if re_ is not None and rf is not None:
        lines.segments.append(((re_[0], re_[1]), (rf[0], rf[1]), "RF_leg"))
    if lh is not None and lk is not None:
        lines.segments.append(((lh[0], lh[1]), (lk[0], lk[1]), "LB_leg"))
    if lk is not None and lb is not None:
        lines.segments.append(((lk[0], lk[1]), (lb[0], lb[1]), "LB_leg"))
    if rh is not None and rk is not None:
        lines.segments.append(((rh[0], rh[1]), (rk[0], rk[1]), "RB_leg"))
    if rk is not None and rb is not None:
        lines.segments.append(((rk[0], rk[1]), (rb[0], rb[1]), "RB_leg"))

    # --- Chest depth proxy: shoulder_center -> elbow_center ---
    chest_depth = None
    if shoulder_center is not None and elbow_center is not None:
        chest_depth = _dist(shoulder_center, elbow_center)
        lines.segments.append((
            (float(shoulder_center[0]), float(shoulder_center[1])),
            (float(elbow_center[0]), float(elbow_center[1])),
            "chest",
        ))
        landmarks["elbow_center"] = [round(float(elbow_center[0]), 2),
                                     round(float(elbow_center[1]), 2)]
    elif shoulder_center is not None and (le is not None or re_ is not None):
        # Fallback: average of available elbows
        elbows = [p for p in (le, re_) if p is not None]
        ec = np.mean([e[:2] for e in elbows], axis=0)
        chest_depth = _dist(shoulder_center, ec)
        lines.segments.append((
            (float(shoulder_center[0]), float(shoulder_center[1])),
            (float(ec[0]), float(ec[1])),
            "chest",
        ))

    # --- Torso diagonal: avg of L_shoulder->R_hip and R_shoulder->L_hip ---
    diags = []
    if ls is not None and rh is not None:
        d = _dist(ls, rh)
        diags.append(d)
        lines.segments.append(((ls[0], ls[1]), (rh[0], rh[1]), "torso_diag"))
    if rs is not None and lh is not None:
        d = _dist(rs, lh)
        diags.append(d)
        lines.segments.append(((rs[0], rs[1]), (lh[0], lh[1]), "torso_diag"))
    torso_diagonal = float(np.mean(diags)) if diags else None

    pixels = {
        "shoulder_width": shoulder_width,
        "hip_width": hip_width,
        "body_length": body_length,
        "lower_chest": body_length,  # display-only; same geometry as body_length
        "body_height": body_height,
        "left_front_leg_length": left_front_leg,
        "right_front_leg_length": right_front_leg,
        "left_back_leg_length": left_back_leg,
        "right_back_leg_length": right_back_leg,
        "chest_depth_proxy": chest_depth,
        "torso_diagonal": torso_diagonal,
    }

    def _ratio(a: float | None, b: float | None) -> float | None:
        if a is None or b is None or b == 0:
            return None
        return a / b

    ratios = {
        "body_length_over_body_height": _ratio(body_length, body_height),
        "shoulder_width_over_body_length": _ratio(shoulder_width, body_length),
        "hip_width_over_body_length": _ratio(hip_width, body_length),
        "chest_depth_over_body_height": _ratio(chest_depth, body_height),
    }

    return CowMeasurements(
        cow_id=int(detection.get("cow_id", 0)),
        bbox=[int(v) for v in detection["bbox"]],
        bbox_confidence=float(detection.get("bbox_confidence", 0.0)),
        pixels=pixels,
        ratios=ratios,
        landmarks=landmarks,
        lines=lines,
    )


def apply_scale(m: CowMeasurements, cm_per_px: float) -> CowMeasurements:
    """Convert pixel measurements to centimeters using a scale factor."""
    m.scale_cm_per_px = float(cm_per_px)
    m.centimeters = {
        k: (v * cm_per_px if v is not None else None) for k, v in m.pixels.items()
    }
    return m


def estimate_scale_from_reference(
    reference_length_px: float,
    reference_length_cm: float,
) -> float:
    """cm_per_px = known_cm / measured_px."""
    if reference_length_px <= 0:
        raise ValueError("reference_length_px must be > 0")
    if reference_length_cm <= 0:
        raise ValueError("reference_length_cm must be > 0")
    return reference_length_cm / reference_length_px


# Colors for measurement overlay (BGR)
_LINE_COLORS: dict[str, tuple[int, int, int]] = {
    "shoulder_w": (0, 200, 255),
    "hip_w": (255, 0, 128),
    "body_len": (0, 255, 128),
    "lower_chest": (255, 200, 0),
    "body_h": (255, 200, 50),
    "LF_leg": (0, 165, 255),
    "RF_leg": (0, 140, 255),
    "LB_leg": (200, 0, 255),
    "RB_leg": (180, 0, 200),
    "chest": (50, 255, 255),
    "torso_diag": (180, 180, 0),
}


def draw_measurements(
    image_bgr: np.ndarray,
    measurements: CowMeasurements,
    show_labels: bool = True,
    unit: str = "px",
) -> np.ndarray:
    """Draw measurement lines and labels on a BGR image."""
    img = image_bgr.copy()
    x1, y1, x2, y2 = measurements.bbox
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    values = measurements.pixels
    if unit == "cm" and measurements.centimeters:
        values = measurements.centimeters

    # Deduplicate labels: show one label per measurement type near first segment
    labeled: set[str] = set()
    for (p1, p2, name) in measurements.lines.segments:
        color = _LINE_COLORS.get(name, (255, 255, 255))
        pt1 = (int(p1[0]), int(p1[1]))
        pt2 = (int(p2[0]), int(p2[1]))
        cv2.line(img, pt1, pt2, color, 2, cv2.LINE_AA)
        if name == "body_h":
            # Upper tip = A End (red border); lower = Ground
            upper = pt1 if pt1[1] <= pt2[1] else pt2
            lower = pt2 if pt1[1] <= pt2[1] else pt1
            cv2.circle(img, lower, 5, color, -1, cv2.LINE_AA)
            cv2.circle(img, upper, 8, (0, 0, 255), -1, cv2.LINE_AA)
            cv2.circle(img, upper, 8, (0, 0, 200), 2, cv2.LINE_AA)
            cv2.putText(
                img, "A End", (upper[0] + 10, upper[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 200), 2, cv2.LINE_AA,
            )
            cv2.putText(
                img, "Ground", (lower[0] + 10, lower[1] + 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA,
            )
        else:
            cv2.circle(img, pt1, 4, color, -1, cv2.LINE_AA)
            cv2.circle(img, pt2, 4, color, -1, cv2.LINE_AA)

        if show_labels and name not in labeled:
            labeled.add(name)
            key_map = {
                "shoulder_w": "shoulder_width",
                "hip_w": "hip_width",
                "body_len": "body_length",
                "lower_chest": "lower_chest",
                "body_h": "body_height",
                "LF_leg": "left_front_leg_length",
                "RF_leg": "right_front_leg_length",
                "LB_leg": "left_back_leg_length",
                "RB_leg": "right_back_leg_length",
                "chest": "chest_depth_proxy",
                "torso_diag": "torso_diagonal",
            }
            val_key = key_map.get(name, name)
            val = values.get(val_key)
            if val is not None:
                mid = ((pt1[0] + pt2[0]) // 2, (pt1[1] + pt2[1]) // 2)
                if name == "lower_chest":
                    label_name = "Lower chest"
                elif name == "body_h":
                    label_name = "Body height"
                else:
                    label_name = name
                text = f"{label_name}:{val:.0f}{unit}"
                cv2.putText(img, text, mid, cv2.FONT_HERSHEY_SIMPLEX,
                            0.4, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(img, text, mid, cv2.FONT_HERSHEY_SIMPLEX,
                            0.4, color, 1, cv2.LINE_AA)

    # Summary panel
    panel_lines = [
        f"Cow #{measurements.cow_id}",
        f"body_len={_fmt(values.get('body_length'))}{unit}",
        f"body_h={_fmt(values.get('body_height'))}{unit}",
        f"shoulder_w={_fmt(values.get('shoulder_width'))}{unit}",
        f"hip_w={_fmt(values.get('hip_width'))}{unit}",
    ]
    if measurements.ratios.get("body_length_over_body_height") is not None:
        panel_lines.append(
            f"L/H={measurements.ratios['body_length_over_body_height']:.3f}"
        )

    y0 = 20
    for line in panel_lines:
        cv2.putText(img, line, (10, y0), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, line, (10, y0), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1, cv2.LINE_AA)
        y0 += 18

    return img


def _fmt(v: float | None) -> str:
    return f"{v:.1f}" if v is not None else "NA"


def measure_from_pose_json(
    pose_json_path: str | Path,
    min_keypoint_conf: float = 0.2,
    scale_cm_per_px: float | None = None,
    reference_length_px: float | None = None,
    reference_length_cm: float | None = None,
) -> CowMeasurements:
    """Load pose JSON, select primary cow, compute measurements."""
    data = load_pose_json(pose_json_path)
    primary = select_primary_cow(data["detections"])
    m = compute_measurements_from_keypoints(primary, min_keypoint_conf)

    if scale_cm_per_px is not None:
        apply_scale(m, scale_cm_per_px)
    elif reference_length_px is not None and reference_length_cm is not None:
        apply_scale(m, estimate_scale_from_reference(reference_length_px, reference_length_cm))

    return m
