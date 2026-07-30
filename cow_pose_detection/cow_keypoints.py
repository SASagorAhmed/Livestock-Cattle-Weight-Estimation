"""AP-10K cow body keypoint definitions and drawing utilities.

All 17 AP-10K keypoints mapped to descriptive cow anatomy names.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

AP10K_KEYPOINT_NAMES: dict[int, str] = {
    0: "left_eye",
    1: "right_eye",
    2: "nose",
    3: "neck",
    4: "tail_root",
    5: "left_shoulder",
    6: "left_elbow",
    7: "left_front_hoof",
    8: "right_shoulder",
    9: "right_elbow",
    10: "right_front_hoof",
    11: "left_hip",
    12: "left_knee",
    13: "left_back_hoof",
    14: "right_hip",
    15: "right_knee",
    16: "right_back_hoof",
}

AP10K_SKELETON: list[tuple[int, int]] = [
    (0, 1), (0, 2), (1, 2), (2, 3), (3, 4),
    (3, 5), (5, 6), (6, 7),
    (3, 8), (8, 9), (9, 10),
    (4, 11), (11, 12), (12, 13),
    (4, 14), (14, 15), (15, 16),
]

SKELETON_COLORS: list[tuple[int, int, int]] = [
    (255, 200, 50),   # eye-eye
    (255, 200, 50),   # leye-nose
    (255, 200, 50),   # reye-nose
    (0, 255, 128),    # nose-neck
    (180, 180, 0),    # neck-tail
    (0, 200, 255),    # neck-lshoulder
    (0, 200, 255),    # lshoulder-lelbow
    (0, 200, 255),    # lelbow-lfhoof
    (255, 128, 0),    # neck-rshoulder
    (255, 128, 0),    # rshoulder-relbow
    (255, 128, 0),    # relbow-rfhoof
    (128, 0, 255),    # tail-lhip
    (128, 0, 255),    # lhip-lknee
    (128, 0, 255),    # lknee-lbhoof
    (255, 0, 128),    # tail-rhip
    (255, 0, 128),    # rhip-rknee
    (255, 0, 128),    # rknee-rbhoof
]

KEYPOINT_COLORS: list[tuple[int, int, int]] = [
    (255, 255, 0),   # left_eye
    (255, 255, 0),   # right_eye
    (255, 200, 50),  # nose
    (0, 255, 128),   # neck
    (180, 180, 0),   # tail_root
    (0, 200, 255),   # left_shoulder
    (0, 200, 255),   # left_elbow
    (0, 200, 255),   # left_front_hoof
    (255, 128, 0),   # right_shoulder
    (255, 128, 0),   # right_elbow
    (255, 128, 0),   # right_front_hoof
    (128, 0, 255),   # left_hip
    (128, 0, 255),   # left_knee
    (128, 0, 255),   # left_back_hoof
    (255, 0, 128),   # right_hip
    (255, 0, 128),   # right_knee
    (255, 0, 128),   # right_back_hoof
]


@dataclass
class CowPoseResult:
    """Result for a single detected cow."""
    cow_id: int
    bbox: list[int]  # [x1, y1, x2, y2]
    bbox_confidence: float
    keypoints: np.ndarray  # (17, 3) -> (y, x, score)
    keypoint_names: dict[int, str] = field(default_factory=lambda: dict(AP10K_KEYPOINT_NAMES))

    def to_dict(self) -> dict[str, Any]:
        kpts = {}
        for idx, name in self.keypoint_names.items():
            y, x, score = self.keypoints[idx]
            kpts[name] = {
                "x": round(float(x), 2),
                "y": round(float(y), 2),
                "confidence": round(float(score), 4),
            }
        return {
            "cow_id": self.cow_id,
            "bbox": self.bbox,
            "bbox_confidence": round(self.bbox_confidence, 4),
            "keypoints": kpts,
        }


def draw_cow_pose(
    image: np.ndarray,
    result: CowPoseResult,
    conf_threshold: float = 0.3,
    draw_bbox: bool = True,
    draw_labels: bool = True,
) -> np.ndarray:
    """Draw keypoints, skeleton, and bbox on an image (BGR)."""
    img = image.copy()
    kpts = result.keypoints  # (17, 3): y, x, score

    # Bbox
    if draw_bbox:
        x1, y1, x2, y2 = result.bbox
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"Cow #{result.cow_id} ({result.bbox_confidence:.2f})"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw, y1), (0, 255, 0), -1)
        cv2.putText(img, label, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    # Skeleton
    for i, (a, b) in enumerate(AP10K_SKELETON):
        if kpts[a][2] > conf_threshold and kpts[b][2] > conf_threshold:
            pt1 = (int(kpts[a][1]), int(kpts[a][0]))
            pt2 = (int(kpts[b][1]), int(kpts[b][0]))
            color = SKELETON_COLORS[i % len(SKELETON_COLORS)]
            cv2.line(img, pt1, pt2, color, 2, cv2.LINE_AA)

    # Keypoints
    radius = max(3, min(img.shape[:2]) // 150)
    for idx in range(17):
        y, x, score = kpts[idx]
        if score > conf_threshold:
            color = KEYPOINT_COLORS[idx]
            center = (int(x), int(y))
            cv2.circle(img, center, radius, color, -1, cv2.LINE_AA)
            cv2.circle(img, center, radius, (0, 0, 0), 1, cv2.LINE_AA)
            if draw_labels:
                name = AP10K_KEYPOINT_NAMES[idx]
                cv2.putText(img, name, (int(x) + radius + 2, int(y) + 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)

    return img
