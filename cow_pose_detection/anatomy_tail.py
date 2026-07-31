"""Tail-head join on cow body mask — shared by morpho heuristic and segmentation."""

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
    if not (np.isfinite(x) and np.isfinite(y)):
        return None
    if conf < min_conf and p.get("status") != "ok":
        return None
    return {"x": x, "y": y, "confidence": conf}


def point_inside_mask(mask: np.ndarray, x: float, y: float) -> bool:
    h, w = mask.shape[:2]
    xi, yi = int(round(x)), int(round(y))
    if xi < 0 or yi < 0 or xi >= w or yi >= h:
        return False
    return bool(mask[yi, xi] > 0)


def snap_to_mask_upper(
    mask: np.ndarray,
    xs: np.ndarray,
    top: np.ndarray,
    x: float,
    y: float | None = None,
) -> tuple[float, float] | None:
    """Snap (x,y) onto the upper silhouette so the point lies on the mask border.

    Prefer the same column's top; if that column is empty, search nearest valid column.
    """
    if xs.size == 0 or top.size == 0:
        return None
    valid = top >= 0
    if not np.any(valid):
        return None
    x_vals = xs[valid].astype(float)
    t_vals = top[valid].astype(float)

    # Prefer exact / nearest column by x
    idx = int(np.argmin(np.abs(x_vals - float(x))))
    sx = float(x_vals[idx])
    sy = float(t_vals[idx])

    # If caller y is inside mask and near top, keep x but snap y to top of that column
    top_y = _top_at_x(xs, top, sx)
    if top_y is not None:
        sy = top_y

    if not point_inside_mask(mask, sx, sy):
        # Walk to nearest in-mask silhouette sample
        order = np.argsort(np.abs(x_vals - float(x)))
        found = False
        for i in order:
            cx, cy = float(x_vals[i]), float(t_vals[i])
            if point_inside_mask(mask, cx, cy):
                sx, sy = cx, cy
                found = True
                break
        if not found:
            return None

    return sx, sy


def _top_at_x(xs: np.ndarray, top: np.ndarray, x: float) -> float | None:
    xi = int(round(x))
    idx = np.where(xs == xi)[0]
    if idx.size and top[idx[0]] >= 0:
        return float(top[idx[0]])
    return None


def silhouette_from_mask(
    mask: np.ndarray, x_min: int | None = None, x_max: int | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-column top/bottom silhouette for mask columns with foreground pixels."""
    ys, xs_fg = np.where(mask > 0)
    if len(xs_fg) == 0:
        return (
            np.array([], dtype=np.int32),
            np.array([], dtype=np.int32),
            np.array([], dtype=np.int32),
        )
    if x_min is None:
        x_min = int(xs_fg.min())
    if x_max is None:
        x_max = int(xs_fg.max())
    width = x_max - x_min + 1
    xs = np.arange(x_min, x_max + 1, dtype=np.int32)
    top = np.full(width, -1, dtype=np.int32)
    bottom = np.full(width, -1, dtype=np.int32)
    for i, x in enumerate(xs):
        col_ys = np.where(mask[:, x] > 0)[0]
        if col_ys.size:
            top[i] = int(col_ys.min())
            bottom[i] = int(col_ys.max())
    return xs, top, bottom


def tail_head_on_body(
    mask: np.ndarray,
    xs: np.ndarray,
    top: np.ndarray,
    bottom: np.ndarray,
    head_dir: str,
    keypoints: dict | None,
) -> tuple[float, float, float, str] | None:
    """Tail-head join on upper rear silhouette, inside mask (not tail tip).

    Returns (x, y, confidence, source) or None if undetectable.
    """
    valid = top >= 0
    if not np.any(valid):
        return None

    x_vals = xs[valid].astype(float)
    t_vals = top[valid].astype(float)
    x_min, x_max = int(x_vals.min()), int(x_vals.max())
    span = max(1, x_max - x_min)

    if head_dir == "right":
        rear_lo = x_min
        rear_hi = x_min + int(0.35 * span)
    else:
        rear_lo = x_max - int(0.35 * span)
        rear_hi = x_max

    zone = (x_vals >= rear_lo) & (x_vals <= rear_hi)
    if not np.any(zone):
        return None

    zx = x_vals[zone]
    zt = t_vals[zone]
    rear_span = max(1, int(zx.max()) - int(zx.min()))
    tail_col_frac = max(1, int(0.15 * rear_span))

    if head_dir == "right":
        col_cut = int(zx.min()) + tail_col_frac
        col_mask = zx <= col_cut
    else:
        col_cut = int(zx.max()) - tail_col_frac
        col_mask = zx >= col_cut

    if not np.any(col_mask):
        col_mask = np.ones_like(zx, dtype=bool)

    cx = zx[col_mask]
    cy = zt[col_mask]
    best_i = int(np.argmin(cy))
    x_best, y_best = float(cx[best_i]), float(cy[best_i])
    conf = 0.75
    source = "rear_contour"

    tr = _ok_kpt(keypoints, "tail_root")
    if tr:
        tx, ty = tr["x"], tr["y"]
        top_y = _top_at_x(xs, top, tx)
        on_body = (
            point_inside_mask(mask, tx, ty)
            and top_y is not None
            and abs(ty - top_y) <= 40.0
        )
        if on_body:
            x_best = float(tx)
            y_best = top_y
            conf = min(0.95, 0.7 + 0.25 * tr.get("confidence", 0.5))
            source = "tail_root_on_body"
        elif rear_lo - 5 <= tx <= rear_hi + 5 and top_y is not None:
            # tail_root x in rear zone but y is tail tip — keep contour, slight x blend only
            x_best = 0.7 * x_best + 0.3 * tx
            xi_top = _top_at_x(xs, top, x_best)
            if xi_top is not None:
                y_best = xi_top

    snapped = snap_to_mask_upper(mask, xs, top, x_best, y_best)
    if snapped is None:
        if not point_inside_mask(mask, x_best, y_best):
            return None
        return x_best, y_best, conf, source
    x_best, y_best = snapped
    return x_best, y_best, conf, source
