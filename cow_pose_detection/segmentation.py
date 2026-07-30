"""Optional local YOLO segmentation helpers for cow body area metrics.

Uses ultralytics YOLO-seg ONNX/PT locally. Does not call any cloud API.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from anatomy_tail import silhouette_from_mask, tail_head_on_body
from lower_chest_guide import lower_chest_line_dict
from a_end_guide import body_height_axis


# COCO class id for cow
COW_CLASS_ID = 19


def _iou(box_a: list[int], box_b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def ensure_seg_model(model_path: str | Path | None = None) -> Path:
    """Return path to a local YOLO-seg model, downloading yolov8n-seg.pt once if needed."""
    script_dir = Path(__file__).resolve().parent
    models_dir = script_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    if model_path is not None:
        p = Path(model_path)
        if not p.is_file():
            raise FileNotFoundError(f"Segmentation model not found: {p}")
        return p

    # Prefer local ONNX, then PT
    for name in ("yolov8n-seg.onnx", "yolov8n-seg.pt"):
        candidate = models_dir / name
        if candidate.is_file():
            return candidate

    # Download lightweight PT weights via ultralytics (local cache after first run)
    import shutil
    from ultralytics import YOLO

    pt_path = models_dir / "yolov8n-seg.pt"
    model = YOLO("yolov8n-seg.pt")
    # Prefer copying the downloaded checkpoint into models/
    src = None
    if hasattr(model, "ckpt_path") and model.ckpt_path:
        src = Path(str(model.ckpt_path))
    for cand in (Path("yolov8n-seg.pt"), Path.cwd() / "yolov8n-seg.pt"):
        if cand.is_file():
            src = cand
            break
    if src is not None and src.is_file():
        if src.resolve() != pt_path.resolve():
            shutil.copy2(src, pt_path)
        return pt_path

    raise RuntimeError("Could not obtain a local YOLO segmentation model")


def segment_cow(
    image_bgr: np.ndarray,
    cow_bbox: list[int],
    model_path: str | Path | None = None,
    conf: float = 0.25,
) -> dict[str, Any] | None:
    """Run local YOLO-seg and return mask metrics for the cow matching cow_bbox.

    Returns None if no cow mask is found.
    """
    from ultralytics import YOLO

    path = ensure_seg_model(model_path)
    model = YOLO(str(path), task="segment")
    results = model.predict(
        source=image_bgr,
        conf=conf,
        classes=[COW_CLASS_ID],
        verbose=False,
        device="cpu",
    )
    if not results:
        return None
    r0 = results[0]
    if r0.masks is None or r0.boxes is None or len(r0.boxes) == 0:
        return None

    boxes = r0.boxes.xyxy.cpu().numpy()
    masks = r0.masks.data.cpu().numpy()  # (N, Hm, Wm)
    scores = r0.boxes.conf.cpu().numpy()

    # Pick mask with highest IoU vs pose bbox
    best_i, best_iou = -1, 0.0
    for i, box in enumerate(boxes):
        iou = _iou(cow_bbox, box.tolist())
        if iou > best_iou:
            best_iou, best_i = iou, i
    if best_i < 0 or best_iou < 0.1:
        return None

    h, w = image_bgr.shape[:2]
    mask_small = masks[best_i]
    mask = cv2.resize(mask_small, (w, h), interpolation=cv2.INTER_LINEAR)
    mask_bin = (mask > 0.5).astype(np.uint8)

    # Keep mask inside detection bbox so region tint cannot spill outside the cow box
    x1, y1, x2, y2 = [int(v) for v in cow_bbox[:4]]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w - 1, x2), min(h - 1, y2)
    if x2 > x1 and y2 > y1:
        clipped = np.zeros_like(mask_bin)
        clipped[y1 : y2 + 1, x1 : x2 + 1] = mask_bin[y1 : y2 + 1, x1 : x2 + 1]
        mask_bin = clipped

    # Largest component only (drop stray blobs outside the cow body)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_bin, connectivity=8)
    if n_labels > 1:
        # label 0 is background
        areas = stats[1:, cv2.CC_STAT_AREA]
        best_label = 1 + int(np.argmax(areas))
        mask_bin = (labels == best_label).astype(np.uint8)

    body_area = int(mask_bin.sum())
    contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    perimeter = float(cv2.arcLength(contours[0], True)) if contours else 0.0

    # Torso ROI: middle band of bbox (exclude head/legs roughly)
    bw, bh = x2 - x1, y2 - y1
    tx1 = int(x1 + 0.15 * bw)
    tx2 = int(x2 - 0.15 * bw)
    ty1 = int(y1 + 0.20 * bh)
    ty2 = int(y1 + 0.70 * bh)
    torso_mask = np.zeros_like(mask_bin)
    torso_mask[ty1:ty2, tx1:tx2] = mask_bin[ty1:ty2, tx1:tx2]
    torso_area = int(torso_mask.sum())

    # Belly boundary: lowest contour points in torso x-range
    belly_points: list[list[int]] = []
    if contours:
        pts = contours[0].reshape(-1, 2)
        xs = pts[:, 0]
        in_torso = (xs >= tx1) & (xs <= tx2)
        if in_torso.any():
            # For each x-bin, take max y (belly lower edge in image coords)
            for x in range(tx1, tx2, max(1, (tx2 - tx1) // 40)):
                col = pts[(xs >= x) & (xs < x + max(1, (tx2 - tx1) // 40))]
                if len(col):
                    belly_points.append([int(x), int(col[:, 1].max())])

    return {
        "model": str(path.name),
        "bbox_iou": round(float(best_iou), 4),
        "bbox_confidence": round(float(scores[best_i]), 4),
        "body_pixel_area": body_area,
        "torso_pixel_area": torso_area,
        "body_perimeter_px": round(perimeter, 2),
        "belly_boundary_points": belly_points,
        "torso_roi": [tx1, ty1, tx2, ty2],
        "mask": mask_bin,  # callers may strip before JSON save
    }


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


def _mask_x_bounds(mask: np.ndarray) -> tuple[int, int]:
    m = np.squeeze(np.asarray(mask))
    if m.ndim != 2:
        m = np.atleast_2d(m)
    ys, xs = np.where(m > 0)
    if len(xs) == 0:
        return 0, max(0, m.shape[1] - 1)
    return int(xs.min()), int(xs.max())


def _silhouette_top(mask: np.ndarray, x_min: int, x_max: int) -> tuple[np.ndarray, np.ndarray]:
    width = x_max - x_min + 1
    xs = np.arange(x_min, x_max + 1, dtype=np.int32)
    top = np.full(width, -1, dtype=np.int32)
    for i, x in enumerate(xs):
        col_ys = np.where(mask[:, x] > 0)[0]
        if col_ys.size:
            top[i] = int(col_ys.min())
    return xs, top


def _head_anchor_simple(
    keypoints: dict | None,
    head_dir: str,
    x_min: int,
    x_max: int,
) -> dict[str, Any] | None:
    nose = _ok_kpt(keypoints, "nose")
    le = _ok_kpt(keypoints, "left_eye")
    re = _ok_kpt(keypoints, "right_eye")
    neck = _ok_kpt(keypoints, "neck")
    head_pts = [p for p in (nose, le, re) if p]
    if not head_pts and not neck:
        return None
    span = max(1, x_max - x_min)
    if head_dir == "right":
        exclude_hi = int(neck["x"] + 0.08 * span) if neck else x_max
        exclude_lo = x_min
    else:
        exclude_lo = int(neck["x"] - 0.08 * span) if neck else x_min
        exclude_hi = x_max
    return {"exclude_lo": exclude_lo, "exclude_hi": exclude_hi, "detected": True}


def _lower_chest_band(
    head_dir: str,
    x_min: int,
    x_max: int,
    keypoints: dict | None,
    head_anchor: dict | None,
) -> tuple[int, int]:
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
    if head_anchor:
        if head_dir == "right":
            hi = min(hi, head_anchor["exclude_hi"])
        else:
            lo = max(lo, head_anchor["exclude_lo"])
    return lo, hi


def _upper_chest_band(
    head_dir: str,
    x_min: int,
    x_max: int,
    keypoints: dict | None,
    head_anchor: dict | None,
) -> tuple[int, int]:
    span = max(1, x_max - x_min)
    neck = _ok_kpt(keypoints, "neck")
    if head_dir == "right":
        lo = x_min + int(0.50 * span)
        hi = x_max - int(0.08 * span)
        if neck:
            lo = max(lo, int(neck["x"]) - int(0.02 * span))
            hi = min(hi, int(neck["x"]) + int(0.28 * span))
        if head_anchor:
            hi = min(hi, head_anchor["exclude_hi"])
    else:
        lo = x_min + int(0.08 * span)
        hi = x_max - int(0.50 * span)
        if neck:
            lo = max(lo, int(neck["x"]) - int(0.28 * span))
            hi = min(hi, int(neck["x"]) + int(0.02 * span))
        if head_anchor:
            lo = max(lo, head_anchor["exclude_lo"])
    lo = max(x_min, min(lo, x_max))
    hi = max(x_min, min(hi, x_max))
    if lo > hi:
        lo, hi = min(lo, hi), max(lo, hi)
    return lo, hi


def _band_pixel_area(mask: np.ndarray, x_lo: int, x_hi: int) -> int:
    h, w = mask.shape[:2]
    x_lo = max(0, min(w - 1, x_lo))
    x_hi = max(0, min(w - 1, x_hi))
    if x_lo > x_hi:
        x_lo, x_hi = x_hi, x_lo
    region = np.zeros_like(mask)
    region[:, x_lo : x_hi + 1] = mask[:, x_lo : x_hi + 1]
    return int(region.sum())


def _detect_tail_anchor(
    mask: np.ndarray,
    keypoints: dict | None,
    head_dir: str,
) -> dict[str, Any]:
    x_min, x_max = _mask_x_bounds(mask)
    xs, top, bottom = silhouette_from_mask(mask, x_min, x_max)
    result = tail_head_on_body(mask, xs, top, bottom, head_dir, keypoints)
    if result is None:
        return {"x": None, "y": None, "detected": False, "source": "contour"}
    x_best, y_best, conf, source = result
    return {
        "x": round(x_best, 2),
        "y": round(y_best, 2),
        "detected": True,
        "source": source,
        "confidence": round(conf, 4),
    }


def _shoulder_markers(keypoints: dict | None) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for name in ("left_shoulder", "right_shoulder"):
        p = _ok_kpt(keypoints, name)
        if p:
            out.append({
                "name": name,
                "x": round(p["x"], 2),
                "y": round(p["y"], 2),
                "confidence": round(p["confidence"], 4),
            })
    return out


def compute_anatomical_regions(
    mask: np.ndarray,
    keypoints: dict | None,
    head_direction: str | None,
    bbox: list | None = None,
) -> dict[str, Any]:
    """Derive upper/lower chest bands, tail anchor, and shoulder markers from mask + pose."""
    hd = (head_direction or "left").strip().lower()
    if hd not in ("left", "right"):
        hd = "left"
    x_min, x_max = _mask_x_bounds(mask)
    head_anchor = _head_anchor_simple(keypoints, hd, x_min, x_max)
    upper_lo, upper_hi = _upper_chest_band(hd, x_min, x_max, keypoints, head_anchor)
    lower_lo, lower_hi = _lower_chest_band(hd, x_min, x_max, keypoints, head_anchor)
    a_end_axis = body_height_axis(keypoints, bbox=bbox, mask=mask)
    return {
        "upper_chest_band": [upper_lo, upper_hi],
        "lower_chest_band": [lower_lo, lower_hi],
        "upper_chest_pixel_area": _band_pixel_area(mask, upper_lo, upper_hi),
        "lower_chest_pixel_area": _band_pixel_area(mask, lower_lo, lower_hi),
        "tail_anchor": _detect_tail_anchor(mask, keypoints, hd),
        "shoulder_markers": _shoulder_markers(keypoints),
        "lower_chest_line": lower_chest_line_dict(keypoints),
        "a_end_line": a_end_axis,
        "head_direction_used": hd,
    }


def _tint_band(
    img: np.ndarray,
    mask: np.ndarray,
    x_lo: int,
    x_hi: int,
    color_bgr: tuple[int, int, int],
    alpha: float,
) -> None:
    h, w = mask.shape[:2]
    x_lo = max(0, min(w - 1, x_lo))
    x_hi = max(0, min(w - 1, x_hi))
    if x_lo > x_hi:
        return
    band = np.zeros_like(mask)
    band[:, x_lo : x_hi + 1] = mask[:, x_lo : x_hi + 1]
    m3 = band.astype(bool)
    color = np.array(color_bgr, dtype=np.float64)
    img[m3] = (img[m3].astype(np.float64) * (1 - alpha) + color * alpha).astype(np.uint8)


def _mask_y_extent_at_x(mask: np.ndarray, x: int, window: int = 2) -> tuple[int, int] | None:
    """Return (y_top, y_bot) of mask near column x, or None if empty."""
    h, w = mask.shape[:2]
    x0 = max(0, x - window)
    x1 = min(w - 1, x + window)
    ys = []
    for xi in range(x0, x1 + 1):
        col = np.where(mask[:, xi] > 0)[0]
        if col.size:
            ys.append(int(col.min()))
            ys.append(int(col.max()))
    if not ys:
        return None
    return min(ys), max(ys)


def _draw_band_edge(
    img: np.ndarray,
    mask: np.ndarray | None,
    x: int,
    color: tuple[int, int, int],
    label: str | None = None,
    label_y: int | None = None,
) -> None:
    """Draw a vertical band edge only within the cow mask Y extent (not full frame)."""
    h_img = img.shape[0]
    if mask is not None:
        ext = _mask_y_extent_at_x(mask, x)
        if ext is None:
            return
        y0, y1 = ext
    else:
        y0, y1 = 0, h_img - 1
    cv2.line(img, (x, y0), (x, y1), color, 1, cv2.LINE_AA)
    if label:
        ty = label_y if label_y is not None else max(12, y0 + 14)
        cv2.putText(img, label, (x + 4, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)


def overlay_segmentation(
    image_bgr: np.ndarray,
    seg: dict[str, Any],
    alpha: float = 0.35,
    keypoints: dict | None = None,
) -> np.ndarray:
    """Overlay segmentation mask, chest bands, belly boundary, shoulders, and tail."""
    del keypoints  # markers stored in seg dict
    img = image_bgr.copy()
    mask = seg.get("mask")

    if mask is not None:
        color = np.zeros_like(img)
        color[:, :] = (0, 180, 255)
        m3 = mask.astype(bool)
        img[m3] = (img[m3] * (1 - alpha) + color[m3] * alpha).astype(np.uint8)

        upper = seg.get("upper_chest_band")
        if upper and len(upper) == 2:
            _tint_band(img, mask, int(upper[0]), int(upper[1]), (0, 180, 0), 0.28)
        lower = seg.get("lower_chest_band")
        if lower and len(lower) == 2:
            _tint_band(img, mask, int(lower[0]), int(lower[1]), (255, 120, 0), 0.28)

    upper = seg.get("upper_chest_band")
    if upper and len(upper) == 2:
        lo, hi = int(upper[0]), int(upper[1])
        _draw_band_edge(img, mask, lo, (0, 200, 0), "Upper chest")
        _draw_band_edge(img, mask, hi, (0, 200, 0))

    lower = seg.get("lower_chest_band")
    if lower and len(lower) == 2:
        lo, hi = int(lower[0]), int(lower[1])
        if mask is not None:
            ext = _mask_y_extent_at_x(mask, lo)
            ly = (ext[0] + 34) if ext else 42
        else:
            ly = 42
        _draw_band_edge(img, mask, lo, (255, 160, 0), "Lower chest band", label_y=ly)
        _draw_band_edge(img, mask, hi, (255, 160, 0))

    # Pose-based Lower chest line (shoulder center → hip center); display only
    lc = seg.get("lower_chest_line") or {}
    if lc.get("detected") and lc.get("p1") and lc.get("p2"):
        p1 = (int(lc["p1"][0]), int(lc["p1"][1]))
        p2 = (int(lc["p2"][0]), int(lc["p2"][1]))
        color = (255, 200, 0)
        cv2.line(img, p1, p2, color, 2, cv2.LINE_AA)
        cv2.circle(img, p1, 5, color, -1, cv2.LINE_AA)
        cv2.circle(img, p2, 5, color, -1, cv2.LINE_AA)
        mid = lc.get("mid")
        if mid and len(mid) >= 2:
            mc = (int(mid[0]), int(mid[1]))
        else:
            mc = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)
        cv2.circle(img, mc, 7, (0, 220, 255), -1, cv2.LINE_AA)
        cv2.circle(img, mc, 7, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(
            img, lc.get("label") or "Lower chest",
            (mc[0] - 50, mc[1] - 14),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA,
        )

    # Body height vertical + A End on cow upper border
    ae = seg.get("a_end_line") or {}
    if ae.get("detected") and ae.get("a_end") and ae.get("ground"):
        p_top = (int(ae["a_end"][0]), int(ae["a_end"][1]))
        p_bot = (int(ae["ground"][0]), int(ae["ground"][1]))
        cv2.line(img, p_top, p_bot, (0, 200, 80), 2, cv2.LINE_AA)
        cv2.circle(img, p_bot, 5, (0, 200, 80), -1, cv2.LINE_AA)
        cv2.circle(img, p_top, 8, (0, 0, 255), -1, cv2.LINE_AA)
        cv2.circle(img, p_top, 8, (0, 0, 200), 2, cv2.LINE_AA)
        mid = ((p_top[0] + p_bot[0]) // 2 + 8, (p_top[1] + p_bot[1]) // 2)
        cv2.putText(
            img, ae.get("line_label") or "Body height",
            mid, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 160, 60), 2, cv2.LINE_AA,
        )
        cv2.putText(
            img, ae.get("label") or "A End", (p_top[0] + 10, p_top[1] - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 200), 2, cv2.LINE_AA,
        )
        cv2.putText(
            img, "Ground", (p_bot[0] + 10, p_bot[1] + 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 80), 1, cv2.LINE_AA,
        )

    for pt in seg.get("belly_boundary_points") or []:
        cv2.circle(img, (int(pt[0]), int(pt[1])), 2, (0, 0, 255), -1)

    roi = seg.get("torso_roi")
    if roi:
        cv2.rectangle(img, (roi[0], roi[1]), (roi[2], roi[3]), (255, 128, 0), 1)

    _SHOULDER_LABELS = {
        "left_shoulder": "Left shoulder",
        "right_shoulder": "Right shoulder",
    }
    hd = (seg.get("head_direction_used") or "left").strip().lower()
    forward_shoulder = "right_shoulder" if hd == "right" else "left_shoulder"
    for m in seg.get("shoulder_markers") or []:
        name = m.get("name", "")
        if name and name != forward_shoulder:
            continue
        x, y = int(m["x"]), int(m["y"])
        cv2.circle(img, (x, y), 8, (0, 255, 255), -1)
        cv2.circle(img, (x, y), 8, (0, 0, 0), 2)
        label = _SHOULDER_LABELS.get(name, name or "Shoulder")
        cv2.putText(img, label, (x + 10, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    tail = seg.get("tail_anchor") or {}
    if tail.get("detected") and tail.get("x") is not None and tail.get("y") is not None:
        tx, ty = int(tail["x"]), int(tail["y"])
        cv2.circle(img, (tx, ty), 10, (255, 0, 255), 2)
        cv2.putText(img, "Tail", (tx + 12, ty - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 255), 2)

    return img


def seg_for_json(seg: dict[str, Any] | None) -> dict[str, Any] | None:
    """Drop heavy mask array before writing JSON."""
    if seg is None:
        return None
    return {k: v for k, v in seg.items() if k != "mask"}
