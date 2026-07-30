#!/usr/bin/env python3
"""Inspect a downloaded Horqin / Mendeley cattle biometric dataset folder.

Expected source: https://data.mendeley.com/datasets/h2s22wr5py/2
DOI: 10.17632/h2s22wr5py.2

Usage:
  python inspect_dataset.py
  python inspect_dataset.py --root "D:/path/to/extracted/dataset"
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
TABLE_EXTS = {".xlsx", ".xls", ".csv"}


def _default_roots(script_dir: Path) -> list[Path]:
    return [
        script_dir / "horqin_h2s22wr5py" / "raw",
        script_dir / "horqin_h2s22wr5py",
        script_dir / "horqin",
        script_dir,
    ]


def find_files(root: Path) -> dict[str, list[Path]]:
    images: list[Path] = []
    labelme: list[Path] = []
    tables: list[Path] = []
    other_json: list[Path] = []
    if not root.exists():
        return {"images": [], "labelme": [], "tables": [], "other_json": []}

    for p in root.rglob("*"):
        if not p.is_file():
            continue
        ext = p.suffix.lower()
        if ext in IMAGE_EXTS:
            images.append(p)
        elif ext in TABLE_EXTS:
            tables.append(p)
        elif ext == ".json":
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                other_json.append(p)
                continue
            if isinstance(data, dict) and "shapes" in data and "imagePath" in data:
                labelme.append(p)
            else:
                other_json.append(p)
    return {
        "images": sorted(images),
        "labelme": sorted(labelme),
        "tables": sorted(tables),
        "other_json": sorted(other_json),
    }


def classify_view(path: Path) -> str:
    name = path.name.lower()
    parts = "/".join(path.parts).lower()
    if any(k in name or k in parts for k in ("side", "lateral", "ce", "侧面")):
        return "side"
    if any(k in name or k in parts for k in ("back", "dorsal", "top", "背面", "俯")):
        return "back"
    return "unknown"


def labelme_labels(files: list[Path]) -> Counter:
    counts: Counter = Counter()
    for p in files[:200]:
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        for shape in data.get("shapes") or []:
            lab = str(shape.get("label") or "").strip()
            if lab:
                counts[lab] += 1
    return counts


def main() -> int:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Extracted dataset root (default: search under datasets/)",
    )
    args = parser.parse_args()

    roots = [args.root] if args.root else _default_roots(script_dir)
    root = next((r for r in roots if r and r.exists() and any(r.iterdir())), None)

    report: dict = {
        "dataset_expected": {
            "name": "Cattle side view and back view dataset",
            "doi": "10.17632/h2s22wr5py.2",
            "url": "https://data.mendeley.com/datasets/h2s22wr5py/2",
            "paper_notes": [
                "72 side-view + 72 back-view PNG images",
                "Excel/XLSX manual measurements (OBL, WH, HG, HL, weight)",
                "LabelMe annotations for biometric measurement points",
            ],
        },
        "status": "missing",
        "root": None,
    }

    if root is None:
        print("DATASET NOT FOUND")
        print("Download from:", report["dataset_expected"]["url"])
        print("Extract into:", script_dir / "horqin_h2s22wr5py" / "raw")
        out = script_dir / "horqin_h2s22wr5py" / "inspect_report.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print("Wrote", out)
        return 1

    found = find_files(root)
    views = Counter(classify_view(p) for p in found["images"])
    labels = labelme_labels(found["labelme"])

    from four_point_schema import KEYPOINT_NAMES

    exact = [n for n in KEYPOINT_NAMES if n in labels]
    report.update({
        "status": "found",
        "root": str(root.resolve()),
        "counts": {
            "images": len(found["images"]),
            "side_view_guess": views.get("side", 0),
            "back_view_guess": views.get("back", 0),
            "unknown_view": views.get("unknown", 0),
            "labelme_json": len(found["labelme"]),
            "excel_or_csv": len(found["tables"]),
            "other_json": len(found["other_json"]),
        },
        "tables": [str(p.relative_to(root)) for p in found["tables"][:20]],
        "sample_images": [str(p.relative_to(root)) for p in found["images"][:12]],
        "labelme_label_histogram": dict(labels.most_common(40)),
        "four_keypoint_exact_match": exact,
        "four_keypoint_ready": len(exact) == 4,
        "recommendation": (
            "Existing LabelMe labels match the 4 target keypoints. Convert with labelme_to_yolo_pose.py"
            if len(exact) == 4
            else (
                "Labels do NOT exactly match the 4 target keypoints. "
                "Run prepare_labelme_workspace.py then relabel side-view images in LabelMe."
            )
        ),
    })

    print("=== Dataset inspect ===")
    print("Root:", report["root"])
    for k, v in report["counts"].items():
        print(f"  {k}: {v}")
    print("Tables:", report["tables"] or "(none)")
    print("LabelMe labels (top):", report["labelme_label_histogram"] or "(none)")
    print("4-keypoint exact match:", exact or "(none)")
    print("Recommendation:", report["recommendation"])

    out = script_dir / "horqin_h2s22wr5py" / "inspect_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("Wrote", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
