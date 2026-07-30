"""Exercise staged API on sample uploads; write stage screenshots into results."""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[2]
UPLOADS = ROOT / "uploads"
API = "http://127.0.0.1:5001"
SHOTS = ROOT / "frontend_dashboard" / "results" / "_stage_screenshots"
SHOTS.mkdir(parents=True, exist_ok=True)

MODELS = {
    "h5model.h5": ROOT / "h5model.h5",
    "model_regression": ROOT / "model_regression",
    "vitpose-b-ap10k.onnx": ROOT / "cow_pose_detection" / "models" / "vitpose-b-ap10k.onnx",
    "yolov8s.onnx": ROOT / "cow_pose_detection" / "models" / "yolov8s.onnx",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest().upper()


def run_staged(image: Path) -> dict:
    print(f"\n=== {image.name} ===")
    with open(image, "rb") as f:
        r = requests.post(
            f"{API}/api/runs",
            files={"file": (image.name, f, "image/jpeg")},
            data={"enable_segmentation": "false", "prediction_mode": "heuristic"},
            timeout=120,
        )
    r.raise_for_status()
    created = r.json()
    run_id = created["run_id"]
    print("created", run_id)

    stages = []
    detect = requests.post(f"{API}/api/runs/{run_id}/detect", timeout=300).json()
    if "error" in detect:
        raise RuntimeError(detect["error"])
    stages.append("detect")
    print("detect cows=", detect["num_cows_detected"], "needs_select=", detect["needs_cow_selection"])

    if detect["needs_cow_selection"]:
        cow_id = detect["selected_cow_id"]
        sel = requests.post(
            f"{API}/api/runs/{run_id}/select-cow",
            json={"cow_id": cow_id},
            timeout=60,
        ).json()
        if "error" in sel:
            raise RuntimeError(sel["error"])
        stages.append("select-cow")
        print("select-cow", cow_id)

    for name in ("pose", "measure", "segment", "scale", "predict"):
        url = f"{API}/api/runs/{run_id}/{name}"
        payload = {"skip": True} if name == "scale" else None
        resp = requests.post(url, json=payload, timeout=300) if payload is not None else requests.post(url, timeout=300)
        data = resp.json()
        if "error" in data:
            raise RuntimeError(f"{name}: {data['error']}")
        stages.append(name)
        print(name, data.get("status"), data.get("final", {}).get("weight_kg") if name == "predict" else "")

    # Copy stage images for report screenshots
    run_dir = ROOT / "frontend_dashboard" / "results" / run_id
    dest = SHOTS / image.stem
    dest.mkdir(parents=True, exist_ok=True)
    for fname in (
        "original_image.jpg",
        "detection_image.jpg",
        "pose_image.jpg",
        "measurements_image.jpg",
    ):
        src = run_dir / fname
        if src.is_file():
            shutil.copy2(src, dest / fname)

    report = requests.get(f"{API}/api/run/{run_id}", timeout=60).json()
    return {
        "image": image.name,
        "run_id": run_id,
        "stages": stages,
        "weight_kg": report.get("final", {}).get("weight_kg"),
        "num_cows": report.get("num_cows_detected"),
        "needs_selection": detect["needs_cow_selection"],
    }


def main() -> int:
    health = requests.get(f"{API}/api/health", timeout=10)
    health.raise_for_status()
    print("health", health.json())

    hashes = {}
    for name, path in MODELS.items():
        if path.is_file():
            hashes[name] = sha256(path)
            print(f"HASH {name}: {hashes[name]}")
        else:
            print(f"MISSING {name}: {path}")

    results = []
    for name in ("img_9.jpg", "cow2.jpg", "img_402.jpg"):
        results.append(run_staged(UPLOADS / name))

    out = {"hashes": hashes, "runs": results}
    out_path = SHOTS / "staged_test_summary.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("\nWrote", out_path)
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
