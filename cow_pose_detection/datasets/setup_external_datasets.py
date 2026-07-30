#!/usr/bin/env python3
"""Prepare folders and optionally clone the public livestock keypoint repo.

Offline only — no inference APIs. Dataset authors may still require a browser
download for packed images (see datasets/README.md).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LIVESTOCK_URL = "https://github.com/yww0411/Livestock-keypoint-detection.git"
LIVESTOCK_DIR = ROOT / "livestock-keypoint-detection"
HORQIN_DIR = ROOT / "horqin"
CUSTOM_DIR = ROOT / "custom_four_points"


def ensure_dirs() -> None:
    for d in (LIVESTOCK_DIR, HORQIN_DIR, CUSTOM_DIR):
        d.mkdir(parents=True, exist_ok=True)
    print(f"Ready folders under: {ROOT}")
    print(f"  - {LIVESTOCK_DIR.name}/")
    print(f"  - {HORQIN_DIR.name}/  (drop Mendeley download here)")
    print(f"  - {CUSTOM_DIR.name}/ (your Labelme export later)")


def clone_livestock() -> int:
    if (LIVESTOCK_DIR / ".git").exists():
        print(f"Already cloned: {LIVESTOCK_DIR}")
        return 0
    LIVESTOCK_DIR.mkdir(parents=True, exist_ok=True)
    # Clone into existing empty dir
    cmd = [
        "git",
        "clone",
        "--depth",
        "1",
        "--branch",
        "master",
        LIVESTOCK_URL,
        str(LIVESTOCK_DIR),
    ]
    print("Running:", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError:
        print("git not found. Install Git, or clone manually:", LIVESTOCK_URL)
        return 1
    except subprocess.CalledProcessError as exc:
        print("Clone failed:", exc)
        return exc.returncode or 1
    print("Clone complete.")
    print("Next: read livestock-keypoint-detection/README.md for data_process layout.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--clone-livestock",
        action="store_true",
        help="Shallow-clone yww0411/Livestock-keypoint-detection into datasets/",
    )
    args = parser.parse_args()
    ensure_dirs()
    if args.clone_livestock:
        return clone_livestock()
    print("Folders created. Re-run with --clone-livestock to fetch the public repo.")
    print("For Horqin cattle images, download from Mendeley into datasets/horqin/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
