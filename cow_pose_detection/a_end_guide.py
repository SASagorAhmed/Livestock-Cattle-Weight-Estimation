"""A End / Back top tip of Body height vertical (display + Morpho).

Sits on the cow upper body border (mask silhouette or keypoint back top).
Never uses detection bbox top (that floats above the cow).
Model body_height px stays the keypoint formula in measurements.py.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _ok_kpt(keypoints: dict | None, name: str, min_conf: float = 0.15) -> dict | None:
    if not keypoints:
        return None
    p = keypoints.get(name)
    if not p or p.get("status") == "missing":
        return None
    try:
        x, y = float(p["x"]), float(p["y"])
        conf = float(p.get("confidence", 0) or 0)
    except (KeyError, TypeError, ValueError):
        return None
    if conf < min_conf and p.get("status") != "ok":
        return None
    return {"x": x, "y": y, "confidence": conf}


def _shoulder_center_x(keypoints: dict | None, bbox: list | None) -> float | None:
    ls = _ok_kpt(keypoints, "left_shoulder")
    rs = _ok_kpt(keypoints, "right_shoulder")
    if ls and rs:
        hx = 0.5 * (ls["x"] + rs["x"])
    elif ls:
        hx = ls["x"]
    elif rs:
        hx = rs["x"]
    elif bbox and len(bbox) >= 4:
        hx = 0.5 * (float(bbox[0]) + float(bbox[2]))
    else:
        return None
    if bbox and len(bbox) >= 4:
        hx = max(float(bbox[0]), min(float(bbox[2]), hx))
    return hx


def _keypoint_back_top_y(keypoints: dict | None) -> float | None:
    names = ("neck", "tail_root", "left_shoulder", "right_shoulder")
    ys = []
    for n in names:
        p = _ok_kpt(keypoints, n)
        if p:
            ys.append(p["y"])
    if not ys:
        return None
    return float(min(ys))


def _ground_y(keypoints: dict | None) -> float | None:
    names = (
        "left_front_hoof", "right_front_hoof",
        "left_back_hoof", "right_back_hoof",
    )
    ys = []
    for n in names:
        p = _ok_kpt(keypoints, n)
        if p:
            ys.append(p["y"])
    if not ys:
        return None
    return float(np.mean(ys))


def _mask_top_y(mask: np.ndarray | None, hx: float, window: int = 6) -> float | None:
    if mask is None:
        return None
    h, w = mask.shape[:2]
    xi = int(round(hx))
    if xi < 0 or xi >= w:
        return None
    x0 = max(0, xi - window)
    x1 = min(w - 1, xi + window)
    ys = []
    for x in range(x0, x1 + 1):
        col = np.where(mask[:, x] > 0)[0]
        if col.size:
            ys.append(int(col.min()))
    if not ys:
        return None
    return float(min(ys))


def body_height_axis(
    keypoints: dict | None,
    bbox: list | None = None,
    mask: np.ndarray | None = None,
) -> dict[str, Any]:
    """Return A End (cow upper border) and Ground for the body-height vertical.

    Priority: mask silhouette top at hx, else keypoint back top. Never bbox top.
    """
    hx = _shoulder_center_x(keypoints, bbox)
    ground = _ground_y(keypoints)
    kpt_top = _keypoint_back_top_y(keypoints)
    if hx is None or ground is None:
        return {
            "detected": False,
            "hx": None,
            "a_end": None,
            "ground": None,
            "back_top_keypoint": None,
            "label": "A End",
        }

    mask_top = _mask_top_y(mask, hx)
    # Prefer true cow surface (YOLO silhouette); never use detection-box top.
    if mask_top is not None:
        upper_y = mask_top
    elif kpt_top is not None:
        upper_y = kpt_top
    else:
        return {
            "detected": False,
            "hx": round(hx, 2),
            "a_end": None,
            "ground": [round(hx, 2), round(ground, 2)],
            "back_top_keypoint": None,
            "label": "A End",
        }

    # Clamp inside bbox Y if provided (floor only — never use bbox top as target)
    if bbox and len(bbox) >= 4:
        by1, by2 = float(bbox[1]), float(bbox[3])
        upper_y = max(by1, min(by2, upper_y))
        ground = max(by1, min(by2, ground))

    a_end = [round(hx, 2), round(float(upper_y), 2)]
    ground_pt = [round(hx, 2), round(float(ground), 2)]
    back_kpt = [round(hx, 2), round(kpt_top, 2)] if kpt_top is not None else None
    return {
        "detected": True,
        "hx": round(hx, 2),
        "a_end": a_end,
        "ground": ground_pt,
        "back_top_keypoint": back_kpt,
        "p1": a_end,
        "p2": ground_pt,
        "label": "A End",
        "line_label": "Body height",
    }


def synthesize_a_end_line(
    a_end_xy: tuple[float, float] | list[float],
    keypoints: dict | None = None,
    bbox: list | None = None,
    mask: np.ndarray | None = None,
) -> dict[str, Any]:
    """Build Body height vertical with given A End tip and Ground below it."""
    ax, ay = float(a_end_xy[0]), float(a_end_xy[1])
    axis = body_height_axis(keypoints, bbox=bbox, mask=mask)
    if axis.get("detected") and axis.get("ground"):
        gx, gy = float(axis["ground"][0]), float(axis["ground"][1])
        # Keep vertical at A End x
        return {
            **axis,
            "detected": True,
            "hx": round(ax, 2),
            "a_end": [round(ax, 2), round(ay, 2)],
            "ground": [round(ax, 2), round(gy, 2)],
            "p1": [round(ax, 2), round(ay, 2)],
            "p2": [round(ax, 2), round(gy, 2)],
            "label": "A End",
            "line_label": "Body height",
        }
    gy = _ground_y(keypoints)
    if gy is None and mask is not None:
        h_m, w_m = mask.shape[:2]
        xi = max(0, min(w_m - 1, int(round(ax))))
        col = np.where(mask[:, xi] > 0)[0]
        if col.size:
            gy = float(col.max())
    if gy is None:
        gy = ay + 120.0
    return {
        "detected": True,
        "hx": round(ax, 2),
        "a_end": [round(ax, 2), round(ay, 2)],
        "ground": [round(ax, 2), round(float(gy), 2)],
        "p1": [round(ax, 2), round(ay, 2)],
        "p2": [round(ax, 2), round(float(gy), 2)],
        "back_top_keypoint": None,
        "label": "A End",
        "line_label": "Body height",
    }


def draw_a_end_vertical(
    img,
    keypoints: dict | None,
    bbox: list | None = None,
    mask: np.ndarray | None = None,
    *,
    line_color: tuple[int, int, int] = (0, 200, 80),
    a_end_fill: tuple[int, int, int] = (0, 0, 255),
    thickness: int = 2,
) -> dict[str, Any]:
    """Draw Ground→A End vertical + red-bordered A End marker. Returns axis dict."""
    import cv2

    axis = body_height_axis(keypoints, bbox=bbox, mask=mask)
    if not axis.get("detected") or not axis.get("a_end") or not axis.get("ground"):
        return axis
    p_top = (int(axis["a_end"][0]), int(axis["a_end"][1]))
    p_bot = (int(axis["ground"][0]), int(axis["ground"][1]))
    cv2.line(img, p_top, p_bot, line_color, thickness, cv2.LINE_AA)
    cv2.circle(img, p_bot, 5, line_color, -1, cv2.LINE_AA)
    cv2.circle(img, p_top, 8, a_end_fill, -1, cv2.LINE_AA)
    cv2.circle(img, p_top, 8, (0, 0, 220), 2, cv2.LINE_AA)
    cv2.putText(
        img, "A End",
        (p_top[0] + 10, p_top[1] - 8),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 220), 2, cv2.LINE_AA,
    )
    cv2.putText(
        img, "Ground",
        (p_bot[0] + 10, p_bot[1] + 4),
        cv2.FONT_HERSHEY_SIMPLEX, 0.45, line_color, 1, cv2.LINE_AA,
    )
    return axis
