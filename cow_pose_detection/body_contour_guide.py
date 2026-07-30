"""Display-only cow silhouette outline (Morpho-style red border source).

Extracts an external contour polyline from a binary segmentation mask.
Not used by weight-model features or measurement formulas.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np


def largest_component_mask(binary: np.ndarray) -> np.ndarray:
    """Keep the largest connected component (8-connectivity)."""
    if binary is None or binary.size == 0:
        return binary
    if binary.dtype != np.uint8:
        work = (binary > 0).astype(np.uint8) * 255
    else:
        work = binary.copy()
        if work.max() <= 1:
            work = work * 255
    num, labels, stats, _ = cv2.connectedComponentsWithStats(work, connectivity=8)
    if num <= 1:
        return work
    areas = stats[1:, cv2.CC_STAT_AREA]
    best = 1 + int(np.argmax(areas))
    out = np.zeros_like(work)
    out[labels == best] = 255
    return out


def contour_polyline_from_mask(
    mask: np.ndarray,
    *,
    max_points: int = 400,
) -> list[list[float]]:
    """Return closed external contour as [[x, y], ...] for SVG display.

    Uses the largest connected component, then findContours (same idea as
    Morpho debug red border). Downsamples dense contours for JSON/SVG size.
    """
    if mask is None or mask.size == 0:
        return []
    clean = largest_component_mask(mask)
    if clean is None or int(cv2.countNonZero(clean)) == 0:
        return []

    contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []

    # Prefer the longest contour (usually the body silhouette)
    contour = max(contours, key=lambda c: int(cv2.contourArea(c)))
    pts = contour.reshape(-1, 2)
    if pts.shape[0] == 0:
        return []

    n = int(pts.shape[0])
    if n > max_points:
        idx = np.linspace(0, n - 1, max_points, dtype=np.int32)
        pts = pts[idx]

    return [[float(x), float(y)] for x, y in pts]


def body_contour_payload(mask: np.ndarray, *, max_points: int = 400) -> dict[str, Any] | None:
    """JSON-friendly contour payload, or None if empty."""
    pts = contour_polyline_from_mask(mask, max_points=max_points)
    if len(pts) < 3:
        return None
    return {
        "points": pts,
        "closed": True,
        "line_label": "Body outline",
    }


def bake_red_outline_image(
    image_bgr: np.ndarray,
    mask: np.ndarray,
    out_path: str | Path,
    *,
    thickness: int = 2,
) -> bool:
    """Bake Morpho-style red silhouette border onto a JPEG (display only).

    Same technique as Morpho draw_debug_image:
    findContours + drawContours(..., (0, 0, 255), 2).
    Does not modify Morpho code.
    """
    if image_bgr is None or mask is None:
        return False
    clean = largest_component_mask(mask)
    if clean is None or int(cv2.countNonZero(clean)) == 0:
        return False
    # OpenCV findContours expects 0/255 binary
    if clean.max() <= 1:
        clean = (clean * 255).astype(np.uint8)
    vis = image_bgr.copy()
    contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return False
    cv2.drawContours(vis, contours, -1, (0, 0, 255), thickness)
    return bool(cv2.imwrite(str(out_path), vis))

