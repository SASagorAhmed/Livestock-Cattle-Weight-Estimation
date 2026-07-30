"""Smartphone Diagonal Formula — experimental weight estimate.

weight_lb = (C * C * B) / 300
where A, B are in inches and C = 2 * A.

Never uses h5model.h5 or measurement pickle features.
"""

from __future__ import annotations

import math
from typing import Any

CM_PER_INCH = 2.54
LB_PER_KG = 2.20462

FOUR_POINT_KEYS = (
    "A_start_lower_chest",
    "A_end_withers",
    "B_start_tail_head",
    "B_end_shoulder_region",
)


def _normalize_points(points: dict | None) -> dict:
    """Accept legacy B_end_forward_shoulder_lower alias."""
    if not points:
        return {}
    out = dict(points)
    if "B_end_shoulder_region" not in out and "B_end_forward_shoulder_lower" in out:
        out["B_end_shoulder_region"] = out["B_end_forward_shoulder_lower"]
    return out


def _dist(a: dict[str, float], b: dict[str, float]) -> float:
    return float(math.hypot(float(b["x"]) - float(a["x"]), float(b["y"]) - float(a["y"])))


def compute_diagonal_formula(
    points: dict[str, dict[str, float]],
    cm_per_px: float | None,
) -> dict[str, Any]:
    """Compute A/B/C and weight from 4 image points + scale.

    Returns status incomplete if scale or points are missing/invalid.
    No mock or fallback numeric weights.
    """
    base: dict[str, Any] = {
        "method": "Smartphone Diagonal Formula",
        "status": "Experimental",
        "available": False,
        "weight_kg": None,
        "weight_lb": None,
        "point_detector": None,
    }

    if cm_per_px is None or not math.isfinite(float(cm_per_px)) or float(cm_per_px) <= 0:
        return {
            **base,
            "reason": "Valid reference scale (cm/px) is required before real-world weight calculation.",
        }

    points = _normalize_points(points)
    missing = [k for k in FOUR_POINT_KEYS if k not in (points or {})]
    if missing:
        return {**base, "reason": f"Missing keypoints: {', '.join(missing)}"}

    try:
        p = {k: {"x": float(points[k]["x"]), "y": float(points[k]["y"])} for k in FOUR_POINT_KEYS}
    except (KeyError, TypeError, ValueError):
        return {**base, "reason": "Invalid keypoint coordinates."}

    a_px = _dist(p["A_start_lower_chest"], p["A_end_withers"])
    b_px = _dist(p["B_start_tail_head"], p["B_end_shoulder_region"])
    if a_px < 1.0 or b_px < 1.0:
        return {**base, "reason": "A or B pixel length is too small."}

    scale = float(cm_per_px)
    a_cm = a_px * scale
    b_cm = b_px * scale
    a_in = a_cm / CM_PER_INCH
    b_in = b_cm / CM_PER_INCH
    c_in = 2.0 * a_in
    c_cm = c_in * CM_PER_INCH

    weight_lb = (c_in * c_in * b_in) / 300.0
    if not math.isfinite(weight_lb) or weight_lb <= 0:
        return {**base, "reason": "Formula produced a non-finite or non-positive weight."}

    weight_kg = weight_lb / LB_PER_KG

    # Propagate detector method from point metadata if present
    detector = None
    for k in FOUR_POINT_KEYS:
        meta = points.get(k) or {}
        if isinstance(meta, dict) and meta.get("method"):
            detector = meta.get("method")
            break

    return {
        "method": "Smartphone Diagonal Formula",
        "status": "Experimental",
        "available": True,
        "reason": None,
        "point_detector": detector,
        "A_px": round(a_px, 4),
        "B_px": round(b_px, 4),
        "A_cm": round(a_cm, 4),
        "B_cm": round(b_cm, 4),
        "C_in": round(c_in, 4),
        "C_cm": round(c_cm, 4),
        "estimated_heart_girth_C_cm": round(c_cm, 4),
        "estimated_heart_girth_C_in": round(c_in, 4),
        "weight_lb": round(weight_lb, 4),
        "weight_kg": round(weight_kg, 4),
        "cm_per_px": round(scale, 8),
        "points": p,
        "formula": "weight_lb = (C^2 * B) / 300 with C = 2*A; A,B in inches",
    }
