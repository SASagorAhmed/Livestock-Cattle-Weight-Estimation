"""Lower chest guide line: shoulder_center → hip_center (display only).

Same geometry as body_length, but a separate label so model features stay unchanged.
"""

from __future__ import annotations

from typing import Any


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


def _midpoint(a: dict, b: dict) -> tuple[float, float]:
    return 0.5 * (a["x"] + b["x"]), 0.5 * (a["y"] + b["y"])


def lower_chest_endpoints(
    keypoints: dict | None,
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    """Return ((sx, sy), (hx, hy)) for shoulder_center → hip_center, or None."""
    ls = _ok_kpt(keypoints, "left_shoulder")
    rs = _ok_kpt(keypoints, "right_shoulder")
    lh = _ok_kpt(keypoints, "left_hip")
    rh = _ok_kpt(keypoints, "right_hip")

    if ls and rs:
        sc = _midpoint(ls, rs)
    elif ls:
        sc = (ls["x"], ls["y"])
    elif rs:
        sc = (rs["x"], rs["y"])
    else:
        return None

    if lh and rh:
        hc = _midpoint(lh, rh)
    elif lh:
        hc = (lh["x"], lh["y"])
    elif rh:
        hc = (rh["x"], rh["y"])
    else:
        return None

    return sc, hc


def lower_chest_line_dict(keypoints: dict | None) -> dict[str, Any]:
    """JSON-friendly lower chest guide (not a weight-model feature)."""
    ends = lower_chest_endpoints(keypoints)
    if ends is None:
        return {
            "detected": False,
            "p1": None,
            "p2": None,
            "mid": None,
            "label": "Lower chest",
        }
    (sx, sy), (hx, hy) = ends
    mx, my = 0.5 * (sx + hx), 0.5 * (sy + hy)
    return {
        "detected": True,
        "p1": [round(sx, 2), round(sy, 2)],
        "p2": [round(hx, 2), round(hy, 2)],
        "mid": [round(mx, 2), round(my, 2)],
        "label": "Lower chest",
    }


def draw_lower_chest_line(
    img,
    keypoints: dict | None,
    *,
    color: tuple[int, int, int] = (255, 200, 0),
    thickness: int = 2,
    draw_mid: bool = True,
) -> None:
    """Draw Lower chest line + midpoint + label on a BGR image (in-place)."""
    import cv2

    line = lower_chest_line_dict(keypoints)
    if not line.get("detected") or not line.get("p1") or not line.get("p2"):
        return
    p1 = (int(line["p1"][0]), int(line["p1"][1]))
    p2 = (int(line["p2"][0]), int(line["p2"][1]))
    cv2.line(img, p1, p2, color, thickness, cv2.LINE_AA)
    cv2.circle(img, p1, 5, color, -1, cv2.LINE_AA)
    cv2.circle(img, p2, 5, color, -1, cv2.LINE_AA)
    mid = line.get("mid")
    if draw_mid and mid and len(mid) >= 2:
        mc = (int(mid[0]), int(mid[1]))
        cv2.circle(img, mc, 7, (0, 220, 255), -1, cv2.LINE_AA)
        cv2.circle(img, mc, 7, (0, 0, 0), 2, cv2.LINE_AA)
        mx, my = mc
    else:
        mx = (p1[0] + p2[0]) // 2
        my = (p1[1] + p2[1]) // 2
    cv2.putText(
        img, line.get("label") or "Lower chest",
        (mx - 50, my - 14),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA,
    )
