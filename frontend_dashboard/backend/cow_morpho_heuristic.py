"""CowMorphoHeuristic — auto-suggest 4 Smartphone Diagonal points from mask + pose.

Uses largest cow mask component, per-column top/bottom silhouette, and AP-10K
keypoints. Does not use h5model.h5 or fixed/mock coordinates.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from config import POSE_DIR

if str(POSE_DIR) not in sys.path:
    sys.path.insert(0, str(POSE_DIR))

from anatomy_tail import point_inside_mask, snap_to_mask_upper, tail_head_on_body  # noqa: E402
from lower_chest_guide import lower_chest_line_dict  # noqa: E402
from a_end_guide import body_height_axis, draw_a_end_vertical, synthesize_a_end_line  # noqa: E402

METHOD = "CowMorphoHeuristic"

KEYPOINT_NAMES = [
    "A_start_lower_chest",
    "A_end_withers",
    "B_start_tail_head",
    "B_end_shoulder_region",
]


def _ok_kpt(keypoints: dict | None, name: str, min_conf: float = 0.15) -> dict | None:
    if not keypoints:
        return None
    p = keypoints.get(name)
    if not p:
        return None
    if p.get("status") == "missing":
        return None
    try:
        x, y = float(p["x"]), float(p["y"])
        conf = float(p.get("confidence", 0) or 0)
    except (KeyError, TypeError, ValueError):
        return None
    if not (np.isfinite(x) and np.isfinite(y)):
        return None
    if conf < min_conf and p.get("status") != "ok":
        return None
    return {"x": x, "y": y, "confidence": conf}


def _mean_xy(points: list[dict]) -> dict | None:
    if not points:
        return None
    return {
        "x": sum(p["x"] for p in points) / len(points),
        "y": sum(p["y"] for p in points) / len(points),
    }


def _build_head_anchor(
    keypoints: dict | None,
    head_dir: str,
    x_min: int,
    x_max: int,
) -> dict[str, Any] | None:
    """Head reference from pose — used to bound searches, not a formula point."""
    nose = _ok_kpt(keypoints, "nose")
    le = _ok_kpt(keypoints, "left_eye")
    re = _ok_kpt(keypoints, "right_eye")
    neck = _ok_kpt(keypoints, "neck")
    head_pts = [p for p in (nose, le, re) if p]
    if not head_pts and not neck:
        return None

    head_center = nose or _mean_xy(head_pts)
    if not head_center and neck:
        head_center = neck

    span = max(1, x_max - x_min)
    # Head-side extreme x (toward facing direction)
    if head_dir == "right":
        head_x = max(p["x"] for p in head_pts) if head_pts else float(head_center["x"])
        # Exclude columns on the head side of neck
        exclude_hi = int(neck["x"] + 0.08 * span) if neck else int(head_x - 0.05 * span)
        exclude_lo = x_min
    else:
        head_x = min(p["x"] for p in head_pts) if head_pts else float(head_center["x"])
        exclude_lo = int(neck["x"] - 0.08 * span) if neck else int(head_x + 0.05 * span)
        exclude_hi = x_max

    return {
        "head_x": float(head_x),
        "head_y": float(head_center["y"]) if head_center else None,
        "neck_x": float(neck["x"]) if neck else None,
        "exclude_lo": exclude_lo,
        "exclude_hi": exclude_hi,
        "detected": True,
    }


def infer_head_direction(keypoints: dict | None) -> str | None:
    """Return 'left' or 'right' (head side in image), or None if unknown."""
    nose = _ok_kpt(keypoints, "nose")
    le = _ok_kpt(keypoints, "left_eye")
    re = _ok_kpt(keypoints, "right_eye")
    head = nose or _mean_xy([p for p in (le, re) if p])
    tail = _ok_kpt(keypoints, "tail_root")
    if not tail:
        hips = [_ok_kpt(keypoints, n) for n in ("left_hip", "right_hip")]
        hips = [h for h in hips if h]
        tail = _mean_xy(hips)
    if not head or not tail:
        return None
    dx = head["x"] - tail["x"]
    if abs(dx) < 8.0:
        return None
    return "right" if dx > 0 else "left"


def _largest_component_mask(binary: np.ndarray) -> np.ndarray:
    num, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if num <= 1:
        return binary
    areas = stats[1:, cv2.CC_STAT_AREA]
    best = 1 + int(np.argmax(areas))
    out = np.zeros_like(binary)
    out[labels == best] = 255
    return out


def _tight_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def _silhouette(
    mask: np.ndarray, x1: int, x2: int, y1: int, y2: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    width = x2 - x1 + 1
    top = np.full(width, -1, dtype=np.int32)
    bottom = np.full(width, -1, dtype=np.int32)
    xs = np.arange(width, dtype=np.int32)
    for i, x in enumerate(range(x1, x2 + 1)):
        col = mask[y1 : y2 + 1, x]
        nz = np.flatnonzero(col > 0)
        if nz.size == 0:
            continue
        top[i] = y1 + int(nz[0])
        bottom[i] = y1 + int(nz[-1])
    return xs + x1, top, bottom


def _point_entry(
    name: str,
    x: float,
    y: float,
    conf: float = 0.7,
    *,
    anatomy_label: str | None = None,
    source_keypoint: str | None = None,
) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "x": round(float(x), 2),
        "y": round(float(y), 2),
        "name": name,
        "confidence": round(float(conf), 4),
        "status": "auto_suggested",
        "method": METHOD,
    }
    if anatomy_label:
        entry["anatomy_label"] = anatomy_label
    if source_keypoint:
        entry["source_keypoint"] = source_keypoint
    return entry


def _robust_low_belly_xy(cx: np.ndarray, cy: np.ndarray) -> tuple[float, float]:
    """Pick robust lowest belly point (90th pct capped by IQR) in lower-chest band."""
    ys = np.asarray(cy, dtype=float)
    xs = np.asarray(cx, dtype=float)
    if ys.size == 0:
        return 0.0, 0.0
    med = float(np.median(ys))
    iqr = float(np.percentile(ys, 75) - np.percentile(ys, 25))
    p90 = float(np.percentile(ys, 90))
    target_y = min(p90, med + 1.5 * max(iqr, 1.0))
    tol = max(3.0, 0.02 * (float(ys.max()) - float(ys.min()) + 1.0))
    near = ys >= (target_y - tol)
    if not np.any(near):
        near = ys >= float(np.percentile(ys, 85))
    sel_x = xs[near]
    sel_y = ys[near]
    return float(np.median(sel_x)), float(np.median(sel_y))


def _infer_forward_shoulder_name(
    forward_shoulder: str | None,
    head_dir: str,
) -> str:
    if forward_shoulder in ("left_shoulder", "right_shoulder"):
        return forward_shoulder
    return "right_shoulder" if head_dir == "right" else "left_shoulder"


def _snap_b_end_to_shoulder(
    keypoints: dict | None,
    forward_shoulder: str,
) -> tuple[float, float, float] | None:
    sh = _ok_kpt(keypoints, forward_shoulder)
    if not sh:
        return None
    conf = min(0.92, 0.75 + 0.25 * sh.get("confidence", 0.5))
    return float(sh["x"]), float(sh["y"]), conf


def _shoulder_anatomy_label(
    source_keypoint: str | None,
    head_dir: str | None = None,
) -> str:
    if source_keypoint == "right_shoulder":
        return "Right shoulder"
    if source_keypoint == "left_shoulder":
        return "Left shoulder"
    if head_dir == "right":
        return "Right shoulder"
    if head_dir == "left":
        return "Left shoulder"
    return "Forward shoulder"


def _point_display_label(name: str, pt: dict[str, Any]) -> str:
    base = {
        "A_start_lower_chest": "A Start",
        "A_end_withers": "A End",
        "B_start_tail_head": "B Start",
        "B_end_shoulder_region": "B End",
    }.get(name, name)
    anatomy = pt.get("anatomy_label")
    if anatomy:
        return f"{base} - {anatomy}"
    return base


def _chest_band(
    head_dir: str,
    x_min: int,
    x_max: int,
    keypoints: dict | None,
    head_anchor: dict | None,
) -> tuple[int, int]:
    """X range for lower-chest band just behind front legs."""
    span = max(1, x_max - x_min)
    front_refs = [
        _ok_kpt(keypoints, n)
        for n in (
            "left_front_hoof", "right_front_hoof",
            "left_elbow", "right_elbow",
            "left_shoulder", "right_shoulder",
        )
    ]
    front_refs = [p for p in front_refs if p]

    if front_refs:
        if head_dir == "right":
            front_x = max(p["x"] for p in front_refs)
            lo = int(front_x - 0.14 * span)
            hi = int(front_x - 0.03 * span)
        else:
            front_x = min(p["x"] for p in front_refs)
            lo = int(front_x + 0.03 * span)
            hi = int(front_x + 0.14 * span)
    else:
        if head_dir == "right":
            lo = x_min + int(0.58 * span)
            hi = x_min + int(0.72 * span)
        else:
            lo = x_min + int(0.28 * span)
            hi = x_min + int(0.42 * span)

    lo = max(x_min, lo)
    hi = min(x_max, hi)
    if lo > hi:
        lo, hi = min(lo, hi), max(lo, hi)

    # Never include head columns
    if head_anchor:
        if head_dir == "right":
            hi = min(hi, head_anchor["exclude_hi"])
        else:
            lo = max(lo, head_anchor["exclude_lo"])

    return lo, hi


def _detect_tail_head(
    mask: np.ndarray,
    xs: np.ndarray,
    top: np.ndarray,
    bottom: np.ndarray,
    head_dir: str,
    keypoints: dict | None,
) -> tuple[float, float, float, str] | None:
    result = tail_head_on_body(mask, xs, top, bottom, head_dir, keypoints)
    if result is None:
        return None
    x_best, y_best, conf, source = result
    return x_best, y_best, conf, source


def _tail_from_segment_anchor(
    mask: np.ndarray,
    xs: np.ndarray,
    top: np.ndarray,
    tail_anchor: dict | None,
) -> tuple[float, float, float, str] | None:
    """Prefer Segment step tail_anchor; snap onto mask upper border."""
    if not tail_anchor or not tail_anchor.get("detected"):
        return None
    try:
        tx = float(tail_anchor["x"])
        ty = float(tail_anchor["y"])
    except (KeyError, TypeError, ValueError):
        return None
    if not (np.isfinite(tx) and np.isfinite(ty)):
        return None
    snapped = snap_to_mask_upper(mask, xs, top, tx, ty)
    if snapped is None:
        if point_inside_mask(mask, tx, ty):
            return tx, ty, float(tail_anchor.get("confidence") or 0.85), "segmentation_tail_anchor"
        return None
    conf = float(tail_anchor.get("confidence") or 0.9)
    return snapped[0], snapped[1], conf, "segmentation_tail_anchor"

def _find_hump_peak(
    x_vals: np.ndarray,
    t_vals: np.ndarray,
    window: int = 5,
) -> int:
    """Index of local highest point (withers hump) using smoothed upper contour."""
    if len(t_vals) < 3:
        return int(np.argmin(t_vals))
    # Smooth top contour
    kernel = np.ones(window) / window
    smooth = np.convolve(t_vals.astype(float), kernel, mode="same")
    # Local minima in y = peaks on back
    best_i = int(np.argmin(smooth))
    # Prefer a local minimum surrounded by higher y (true peak)
    for i in range(1, len(smooth) - 1):
        if smooth[i] <= smooth[i - 1] and smooth[i] <= smooth[i + 1]:
            if smooth[i] < smooth[best_i]:
                best_i = i
    return best_i


def _detect_withers(
    xs: np.ndarray,
    top: np.ndarray,
    head_dir: str,
    keypoints: dict | None,
    head_anchor: dict | None,
) -> tuple[float, float, float] | None:
    """P2: withers hump peak on upper contour behind neck, head excluded."""
    valid = top >= 0
    if not np.any(valid):
        return None
    x_vals = xs[valid].astype(float)
    t_vals = top[valid].astype(float)
    x_min, x_max = int(x_vals.min()), int(x_vals.max())
    span = max(1, x_max - x_min)

    neck = _ok_kpt(keypoints, "neck")
    shoulders = [_ok_kpt(keypoints, n) for n in ("left_shoulder", "right_shoulder")]
    shoulders = [s for s in shoulders if s]

    if head_dir == "right":
        # Withers: between mid-body and neck, excluding head
        front_lo = x_min + int(0.50 * span)
        front_hi = x_max - int(0.08 * span)
        if neck:
            front_lo = max(front_lo, int(neck["x"]) - int(0.02 * span))
            front_hi = min(front_hi, int(neck["x"]) + int(0.28 * span))
        if head_anchor:
            front_hi = min(front_hi, head_anchor["exclude_hi"])
    else:
        front_lo = x_min + int(0.08 * span)
        front_hi = x_max - int(0.50 * span)
        if neck:
            front_lo = max(front_lo, int(neck["x"]) - int(0.28 * span))
            front_hi = min(front_hi, int(neck["x"]) + int(0.02 * span))
        if head_anchor:
            front_lo = max(front_lo, head_anchor["exclude_lo"])

    zone = (x_vals >= front_lo) & (x_vals <= front_hi)
    if not np.any(zone):
        if head_dir == "right":
            zone = (x_vals >= x_min + 0.45 * span) & (x_vals <= x_max - 0.10 * span)
        else:
            zone = (x_vals <= x_max - 0.45 * span) & (x_vals >= x_min + 0.10 * span)
    if not np.any(zone):
        return None

    cx = x_vals[zone]
    cy = t_vals[zone]
    best_i = _find_hump_peak(cx, cy)
    wx, wy = float(cx[best_i]), float(cy[best_i])

    # Light bias toward neck–shoulder midpoint when pose available
    if neck and shoulders:
        if head_dir == "right":
            sh = max(shoulders, key=lambda p: p["x"])
        else:
            sh = min(shoulders, key=lambda p: p["x"])
        mid_x = 0.5 * neck["x"] + 0.5 * sh["x"]
        mid_y = 0.5 * neck["y"] + 0.5 * sh["y"]
        if front_lo <= mid_x <= front_hi:
            wx = 0.75 * wx + 0.25 * mid_x
            # Re-sample top at blended x
            xi = int(round(wx))
            idx = np.where(xs == xi)[0]
            if idx.size and top[idx[0]] >= 0:
                wy = float(top[idx[0]])
            else:
                wy = 0.75 * wy + 0.25 * mid_y

    return wx, wy, 0.85


def _pose_center(keypoints: dict | None, left_name: str, right_name: str) -> dict | None:
    left = _ok_kpt(keypoints, left_name)
    right = _ok_kpt(keypoints, right_name)
    if left and right:
        return {
            "x": 0.5 * (left["x"] + right["x"]),
            "y": 0.5 * (left["y"] + right["y"]),
        }
    return left or right


def _lower_chest_guide_line(keypoints: dict | None) -> dict[str, Any]:
    """Independent guide (not a model feature): shoulder_center → hip_center."""
    return lower_chest_line_dict(keypoints)


def _y_on_segment(x: float, p1: list[float], p2: list[float]) -> float:
    x1, y1 = float(p1[0]), float(p1[1])
    x2, y2 = float(p2[0]), float(p2[1])
    if abs(x2 - x1) < 1e-6:
        return 0.5 * (y1 + y2)
    t = (x - x1) / (x2 - x1)
    t = max(0.0, min(1.0, t))
    return y1 + t * (y2 - y1)


def _detect_lower_chest(
    xs: np.ndarray,
    bottom: np.ndarray,
    head_dir: str,
    keypoints: dict | None,
    head_anchor: dict | None,
    belly_hints: list | None,
) -> tuple[float, float, float] | None:
    """P1: midpoint of shoulder_center ↔ hip_center (center of Lower chest line)."""
    del belly_hints  # guide line replaces belly-contour A Start
    guide = _lower_chest_guide_line(keypoints)
    if guide.get("detected"):
        mid = guide.get("mid")
        if mid and len(mid) >= 2:
            return float(mid[0]), float(mid[1]), 0.92
        p1, p2 = guide.get("p1"), guide.get("p2")
        if p1 and p2:
            return (
                0.5 * (float(p1[0]) + float(p2[0])),
                0.5 * (float(p1[1]) + float(p2[1])),
                0.92,
            )

    # Fallback: previous robust belly contour if guide unavailable
    valid = bottom >= 0
    if not np.any(valid):
        return None
    x_vals = xs[valid].astype(float)
    b_vals = bottom[valid].astype(float)
    x_min, x_max = int(x_vals.min()), int(x_vals.max())
    lo, hi = _chest_band(head_dir, x_min, x_max, keypoints, head_anchor)
    zone = (x_vals >= lo) & (x_vals <= hi)
    if not np.any(zone):
        return None
    mid_x, mid_y = _robust_low_belly_xy(x_vals[zone], b_vals[zone])
    return mid_x, mid_y, 0.80


def _detect_shoulder_region(
    xs: np.ndarray,
    top: np.ndarray,
    bottom: np.ndarray,
    withers: tuple[float, float, float],
    head_dir: str,
    keypoints: dict | None,
    head_anchor: dict | None,
) -> tuple[float, float, float, str | None] | None:
    """P4: forward shoulder at detected pose keypoint, contour fallback if missing."""
    wx, wy, wconf = withers
    valid = top >= 0
    if not np.any(valid):
        return None
    x_vals = xs[valid]
    t_vals = top[valid]
    bot_map = {int(xs[i]): int(bottom[i]) for i in range(len(xs)) if bottom[i] >= 0}

    span = max(1, int(x_vals.max()) - int(x_vals.min()))
    neck = _ok_kpt(keypoints, "neck")

    left_sh = _ok_kpt(keypoints, "left_shoulder")
    right_sh = _ok_kpt(keypoints, "right_shoulder")
    forward_name: str | None = None
    forward_sh = None
    if head_dir == "right":
        if right_sh:
            forward_sh, forward_name = right_sh, "right_shoulder"
        elif left_sh:
            forward_sh, forward_name = left_sh, "left_shoulder"
    else:
        if left_sh:
            forward_sh, forward_name = left_sh, "left_shoulder"
        elif right_sh:
            forward_sh, forward_name = right_sh, "right_shoulder"

    if forward_sh:
        conf = min(0.92, 0.75 + 0.25 * forward_sh.get("confidence", 0.5))
        return float(forward_sh["x"]), float(forward_sh["y"]), conf, forward_name

    if head_dir == "right":
        tx = wx + max(12.0, 0.05 * span)
        if neck:
            tx = min(tx, neck["x"] + 0.06 * span)
        if head_anchor:
            tx = min(tx, head_anchor["exclude_hi"] - 2)
    else:
        tx = wx - max(12.0, 0.05 * span)
        if neck:
            tx = max(tx, neck["x"] - 0.06 * span)
        if head_anchor:
            tx = max(tx, head_anchor["exclude_lo"] + 2)

    xi = int(round(tx))
    idx = np.where(xs == xi)[0]
    if idx.size == 0:
        diffs = np.abs(x_vals - xi)
        j = int(np.argmin(diffs))
        xi = int(x_vals[j])
        ty = float(t_vals[j])
    else:
        ty = float(top[idx[0]])

    by = bot_map.get(xi)
    if by is None:
        by = int(ty + 40)

    body_depth = max(20.0, float(by - ty))
    y = max(ty + 4.0, min(wy + 0.18 * body_depth, ty + 0.35 * body_depth))
    y = max(y, wy + 4.0)
    conf = min(0.85, wconf * 0.95)
    return float(xi), float(y), conf, forward_name


def draw_debug_image(
    image_bgr: np.ndarray,
    mask: np.ndarray,
    keypoints_out: dict[str, dict[str, Any]],
    out_path: Path,
    head_anchor: dict | None = None,
    chest_band: tuple[int, int] | None = None,
    pose_keypoints: dict | None = None,
    head_dir: str | None = None,
    lower_chest_guide: dict | None = None,
    a_end_line: dict | None = None,
    bbox: list | None = None,
) -> None:
    vis = image_bgr.copy()
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis, contours, -1, (0, 0, 255), 2)

    if chest_band:
        lo, hi = chest_band
        h_img = vis.shape[0]
        cv2.line(vis, (lo, 0), (lo, h_img - 1), (255, 200, 0), 1)
        cv2.line(vis, (hi, 0), (hi, h_img - 1), (255, 200, 0), 1)
        cv2.putText(vis, "chest band", (lo + 4, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1)

    if head_anchor and head_anchor.get("detected"):
        hx_raw = head_anchor.get("head_x", head_anchor.get("x"))
        hy_raw = head_anchor.get("head_y", head_anchor.get("y"))
        if hx_raw is not None and hy_raw is not None:
            hx = int(float(hx_raw))
            hy = int(float(hy_raw))
            cv2.circle(vis, (hx, hy), 10, (255, 0, 255), 2)
            cv2.putText(vis, "head", (hx + 12, hy - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 255), 2)

    # Body height vertical (A End → Ground) on cow photo
    ae = a_end_line or {}
    if ae.get("detected") and ae.get("a_end") and ae.get("ground"):
        p_top = (int(ae["a_end"][0]), int(ae["a_end"][1]))
        p_bot = (int(ae["ground"][0]), int(ae["ground"][1]))
        cv2.line(vis, p_top, p_bot, (0, 200, 80), 2, cv2.LINE_AA)
        cv2.circle(vis, p_bot, 5, (0, 200, 80), -1, cv2.LINE_AA)
        cv2.circle(vis, p_top, 8, (0, 0, 255), -1, cv2.LINE_AA)
        cv2.circle(vis, p_top, 8, (0, 0, 200), 2, cv2.LINE_AA)
        mid = ((p_top[0] + p_bot[0]) // 2 + 8, (p_top[1] + p_bot[1]) // 2)
        cv2.putText(
            vis, ae.get("line_label") or "Body height",
            mid, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 160, 60), 2, cv2.LINE_AA,
        )
        cv2.putText(
            vis, ae.get("label") or "A End",
            (p_top[0] + 10, p_top[1] - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 200), 2, cv2.LINE_AA,
        )
        cv2.putText(
            vis, "Ground",
            (p_bot[0] + 10, p_bot[1] + 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 80), 1, cv2.LINE_AA,
        )
    elif pose_keypoints is not None:
        draw_a_end_vertical(vis, pose_keypoints, bbox=bbox, mask=mask)

    guide = lower_chest_guide or {}
    if guide.get("detected") and guide.get("p1") and guide.get("p2"):
        gp1 = (int(guide["p1"][0]), int(guide["p1"][1]))
        gp2 = (int(guide["p2"][0]), int(guide["p2"][1]))
        cyan = (255, 220, 0)
        cv2.line(vis, gp1, gp2, cyan, 2, cv2.LINE_AA)
        cv2.circle(vis, gp1, 5, cyan, -1)
        cv2.circle(vis, gp2, 5, cyan, -1)
        mx = (gp1[0] + gp2[0]) // 2
        my = (gp1[1] + gp2[1]) // 2
        cv2.putText(
            vis, guide.get("label") or "Lower chest",
            (mx - 60, my - 12),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, cyan, 2, cv2.LINE_AA,
        )

    a1 = keypoints_out.get("A_start_lower_chest")
    a2 = keypoints_out.get("A_end_withers")
    b1 = keypoints_out.get("B_start_tail_head")
    b2 = keypoints_out.get("B_end_shoulder_region")
    if a1 and a2:
        cv2.line(vis, (int(a1["x"]), int(a1["y"])), (int(a2["x"]), int(a2["y"])), (0, 180, 0), 2)
    if b1 and b2:
        cv2.line(vis, (int(b1["x"]), int(b1["y"])), (int(b2["x"]), int(b2["y"])), (0, 140, 255), 2)

    labels = {
        "A_start_lower_chest": "A Start",
        "A_end_withers": "A End",
        "B_start_tail_head": "B Start",
        "B_end_shoulder_region": "B End",
    }
    colors = {
        "A_start_lower_chest": (0, 200, 0),
        "A_end_withers": (0, 200, 0),
        "B_start_tail_head": (0, 140, 255),
        "B_end_shoulder_region": (0, 140, 255),
    }
    for name in labels:
        p = keypoints_out.get(name)
        if not p:
            continue
        lab = _point_display_label(name, p)
        pt = (int(p["x"]), int(p["y"]))
        cv2.circle(vis, pt, 8, colors[name], -1)
        cv2.circle(vis, pt, 8, (255, 255, 255), 2)
        cv2.putText(vis, lab, (pt[0] + 10, pt[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, colors[name], 2, cv2.LINE_AA)

    if pose_keypoints and head_dir in ("left", "right"):
        ref_name = "right_shoulder" if head_dir == "right" else "left_shoulder"
        sh = _ok_kpt(pose_keypoints, ref_name)
        if sh:
            pt = (int(sh["x"]), int(sh["y"]))
            label = _shoulder_anatomy_label(ref_name)
            color = (0, 255, 255)
            cv2.circle(vis, pt, 8, color, -1)
            cv2.circle(vis, pt, 8, (0, 0, 0), 2)
            cv2.putText(
                vis, label, (pt[0] + 10, pt[1] + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2,
            )

    cv2.imwrite(str(out_path), vis)


def suggest_four_points_morpho(
    mask_path: Path | str,
    keypoints: dict | None,
    bbox: list | tuple | None = None,
    head_direction: str | None = None,
    image_path: Path | str | None = None,
    debug_out_path: Path | str | None = None,
    belly_boundary_points: list | None = None,
    tail_anchor: dict | None = None,
) -> dict[str, Any]:
    """Suggest 4 points. No mock coords — fail cleanly if inputs insufficient."""
    mask_path = Path(mask_path)
    if not mask_path.is_file():
        return {
            "available": False,
            "reason": "segmentation_mask.png not found. Run Pose (or Segment) so YOLO-seg can create the shared mask.",
            "head_direction_required": False,
            "inferred_head_direction": None,
            "keypoints": None,
            "point_detector": METHOD,
        }

    raw = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if raw is None:
        return {
            "available": False,
            "reason": "Could not read segmentation mask.",
            "head_direction_required": False,
            "inferred_head_direction": None,
            "keypoints": None,
            "point_detector": METHOD,
        }
    raw = np.squeeze(raw)
    if raw.ndim != 2:
        return {
            "available": False,
            "reason": f"Unexpected mask shape {getattr(raw, 'shape', None)}.",
            "head_direction_required": False,
            "inferred_head_direction": None,
            "keypoints": None,
            "point_detector": METHOD,
        }

    _, binary = cv2.threshold(raw, 127, 255, cv2.THRESH_BINARY)
    mask = _largest_component_mask(binary)
    mask = np.squeeze(mask)
    tight = _tight_bbox(mask)
    if tight is None:
        return {
            "available": False,
            "reason": "Empty cow mask after largest-component filter.",
            "head_direction_required": False,
            "inferred_head_direction": None,
            "keypoints": None,
            "point_detector": METHOD,
        }

    x1, y1, x2, y2 = tight
    if bbox and len(bbox) >= 4:
        bx1, by1, bx2, by2 = [int(v) for v in bbox[:4]]
        x1 = max(x1, bx1)
        y1 = max(y1, by1)
        x2 = min(x2, bx2)
        y2 = min(y2, by2)
        if x2 <= x1 or y2 <= y1:
            x1, y1, x2, y2 = tight

    inferred = infer_head_direction(keypoints)
    hd = (head_direction or "").strip().lower() or None
    if hd not in ("left", "right"):
        hd = inferred
    if hd not in ("left", "right"):
        return {
            "available": False,
            "reason": "Head direction could not be determined. Select Left or Right, then retry.",
            "head_direction_required": True,
            "inferred_head_direction": None,
            "keypoints": None,
            "point_detector": METHOD,
        }

    head_anchor = _build_head_anchor(keypoints, hd, x1, x2)
    xs, top, bottom = _silhouette(mask, x1, x2, y1, y2)
    chest_band = _chest_band(hd, x1, x2, keypoints, head_anchor)

    # Prefer Segment tail_anchor; fall back to mask contour heuristic
    p3 = _tail_from_segment_anchor(mask, xs, top, tail_anchor)
    if p3 is None:
        p3 = _detect_tail_head(mask, xs, top, bottom, hd, keypoints)
    if p3 is None:
        return {
            "available": False,
            "reason": "Could not detect tail head (P3) on rear upper contour.",
            "head_direction_required": False,
            "inferred_head_direction": hd,
            "keypoints": None,
            "point_detector": METHOD,
        }
    # Final safety: never emit B Start outside the mask
    snapped_p3 = snap_to_mask_upper(mask, xs, top, p3[0], p3[1])
    if snapped_p3 is not None:
        p3 = (snapped_p3[0], snapped_p3[1], p3[2], p3[3] if len(p3) > 3 else "snapped")
    elif not point_inside_mask(mask, p3[0], p3[1]):
        return {
            "available": False,
            "reason": "Tail head (P3) could not be snapped inside the segmentation mask.",
            "head_direction_required": False,
            "inferred_head_direction": hd,
            "keypoints": None,
            "point_detector": METHOD,
        }

    p2_axis = body_height_axis(
        keypoints,
        bbox=list(bbox[:4]) if bbox and len(bbox) >= 4 else [x1, y1, x2, y2],
        mask=mask,
    )
    if p2_axis.get("detected") and p2_axis.get("a_end"):
        p2 = (float(p2_axis["a_end"][0]), float(p2_axis["a_end"][1]), 0.92)
    else:
        p2 = _detect_withers(xs, top, hd, keypoints, head_anchor)
    if p2 is None:
        return {
            "available": False,
            "reason": "Could not detect withers / A End (P2) on upper body-height tip.",
            "head_direction_required": False,
            "inferred_head_direction": hd,
            "keypoints": None,
            "point_detector": METHOD,
        }

    p1 = _detect_lower_chest(xs, bottom, hd, keypoints, head_anchor, belly_boundary_points)
    if p1 is None:
        return {
            "available": False,
            "reason": "Could not detect lower chest (P1) on belly contour.",
            "head_direction_required": False,
            "inferred_head_direction": hd,
            "keypoints": None,
            "point_detector": METHOD,
        }

    p4 = _detect_shoulder_region(xs, top, bottom, p2, hd, keypoints, head_anchor)
    if p4 is None:
        return {
            "available": False,
            "reason": "Could not detect forward shoulder region (P4).",
            "head_direction_required": False,
            "inferred_head_direction": hd,
            "keypoints": None,
            "point_detector": METHOD,
        }

    p4_x, p4_y, p4_c, forward_shoulder = p4
    forward_shoulder = _infer_forward_shoulder_name(forward_shoulder, hd)
    snapped = _snap_b_end_to_shoulder(keypoints, forward_shoulder)
    if snapped:
        p4_x, p4_y, p4_c = snapped

    guide = _lower_chest_guide_line(keypoints)
    bb = list(bbox[:4]) if bbox and len(bbox) >= 4 else [x1, y1, x2, y2]
    a_end_axis = synthesize_a_end_line(
        (float(p2[0]), float(p2[1])),
        keypoints=keypoints,
        bbox=bb,
        mask=mask,
    )
    out_kpts = {
        "A_start_lower_chest": _point_entry(
            "A_start_lower_chest", p1[0], p1[1], p1[2], anatomy_label="Lower chest",
        ),
        "A_end_withers": _point_entry(
            "A_end_withers", p2[0], p2[1], p2[2], anatomy_label="Back top",
        ),
        "B_start_tail_head": _point_entry(
            "B_start_tail_head",
            p3[0],
            p3[1],
            p3[2],
            anatomy_label="Tail head",
        ),
        "B_end_shoulder_region": _point_entry(
            "B_end_shoulder_region",
            p4_x,
            p4_y,
            p4_c,
            anatomy_label="Forward shoulder region",
        ),
    }

    debug_file = None
    if debug_out_path is not None:
        img = None
        if image_path and Path(image_path).is_file():
            img = cv2.imread(str(image_path))
        if img is None:
            img = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        draw_debug_image(
            img, mask, out_kpts, Path(debug_out_path),
            head_anchor, chest_band,
            pose_keypoints=keypoints,
            head_dir=hd,
            lower_chest_guide=guide,
            a_end_line=a_end_axis,
            bbox=bb,
        )
        debug_file = str(Path(debug_out_path).name)

    head_anchor_out = None
    if head_anchor and head_anchor.get("detected"):
        head_anchor_out = {
            "x": head_anchor.get("head_x"),
            "y": head_anchor.get("head_y"),
            "detected": True,
        }

    return {
        "available": True,
        "reason": None,
        "head_direction_required": False,
        "inferred_head_direction": inferred,
        "head_direction_used": hd,
        "head_detected": bool(head_anchor and head_anchor.get("detected")),
        "head_anchor": head_anchor_out,
        "keypoints": out_kpts,
        "lower_chest_guide_line": guide,
        "a_end_line": a_end_axis,
        "point_detector": METHOD,
        "method": METHOD,
        "debug_image": debug_file,
    }
