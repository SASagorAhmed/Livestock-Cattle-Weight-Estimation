"""Canonical 4 keypoints for Smartphone Diagonal Formula (YOLO-pose)."""

from __future__ import annotations

KEYPOINT_NAMES: list[str] = [
    "A_start_lower_chest",
    "A_end_withers",
    "B_start_tail_head",
    "B_end_shoulder_region",
]

KEYPOINT_DESCRIPTIONS: dict[str, str] = {
    "A_start_lower_chest": (
        "Lower chest / brisket / lowest center belly point just behind the front legs"
    ),
    "A_end_withers": (
        "Withers / high shoulder hump / highest shoulder point"
    ),
    "B_start_tail_head": (
        "Top of tail / tail head where the tail joins the body"
    ),
    "B_end_shoulder_region": (
        "Forward shoulder region, slightly lower than the withers "
        "(not the top withers point and not the head)"
    ),
}

# YOLO-pose class name
CLASS_NAME = "cow"
NUM_KEYPOINTS = 4
# Skeleton edges for visualization: A-line and B-line
SKELETON = [[0, 1], [2, 3]]
# Left-right flip index (identity — side-view landmarks are not L/R pairs)
FLIP_IDX = [0, 1, 2, 3]

# Possible LabelMe label aliases → canonical name (for importing older Horqin labels)
LABEL_ALIASES: dict[str, str] = {
    "a_start_lower_chest": "A_start_lower_chest",
    "lower_chest": "A_start_lower_chest",
    "brisket": "A_start_lower_chest",
    "belly": "A_start_lower_chest",
    "chest_base": "A_start_lower_chest",
    "a_end_withers": "A_end_withers",
    "withers": "A_end_withers",
    "withers_height": "A_end_withers",
    "wh": "A_end_withers",
    "b_start_tail_head": "B_start_tail_head",
    "tail_head": "B_start_tail_head",
    "tail_root": "B_start_tail_head",
    "tail": "B_start_tail_head",
    "b_end_shoulder_region": "B_end_shoulder_region",
    "b_end_forward_shoulder_lower": "B_end_shoulder_region",
    "forward_shoulder": "B_end_shoulder_region",
    "shoulder": "B_end_shoulder_region",
    "scapula": "B_end_shoulder_region",
}


def canonicalize_label(raw: str) -> str | None:
    s = (raw or "").strip()
    if s in KEYPOINT_NAMES:
        return s
    return LABEL_ALIASES.get(s.lower().replace(" ", "_").replace("-", "_"))
