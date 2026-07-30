"""Flask API for the local cattle weight estimation dashboard."""

from __future__ import annotations

import sys
from pathlib import Path

from flask import Flask, jsonify, request, send_file
from flask_cors import CORS

BACKEND_DIR = Path(__file__).resolve().parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from config import ALLOWED_RESULT_FILES, RESULTS_DIR  # noqa: E402
from pipeline import (  # noqa: E402
    apply_scale_to_run,
    load_run,
    measurement_model_available,
    process_image,
    reselect_cow,
)
from stages import (  # noqa: E402
    create_run,
    stage_detect,
    stage_measure,
    stage_pose,
    stage_predict,
    stage_redraw_four_point_debug,
    stage_scale,
    stage_segment,
    stage_select_cow,
    stage_suggest_four_points,
)
from diagonal_formula import compute_diagonal_formula  # noqa: E402

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 15 * 1024 * 1024
CORS(app, resources={r"/api/*": {"origins": [
    "http://127.0.0.1:5173",
    "http://localhost:5173",
    "http://127.0.0.1:4173",
    "http://localhost:4173",
]}})


@app.get("/api/health")
def health():
    return jsonify({
        "status": "ok",
        "offline": True,
        "measurement_model_available": measurement_model_available(),
    })


@app.get("/api/capabilities")
def capabilities():
    return jsonify({
        "primary_model": {
            "id": "cow_morpho_heuristic",
            "label": "Cow Morpho Heuristic",
            "status": "Experimental",
        },
        "segmentation": True,
        "webcam": True,
    })


# ---------------------------------------------------------------------------
# Staged live pipeline
# ---------------------------------------------------------------------------

@app.post("/api/runs")
def api_create_run():
    try:
        if "file" not in request.files:
            return jsonify({"error": "Missing file field"}), 400
        f = request.files["file"]
        if not f.filename:
            return jsonify({"error": "Empty filename"}), 400
        enable_seg = request.form.get("enable_segmentation", "false").lower() in ("1", "true", "yes")
        mode = request.form.get("prediction_mode", "smartphone_diagonal")
        if mode not in ("heuristic", "measurement", "smartphone_diagonal"):
            return jsonify({
                "error": "prediction_mode must be heuristic, measurement, or smartphone_diagonal",
            }), 400
        return jsonify(create_run(f, enable_segmentation=enable_seg, prediction_mode=mode))
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": str(exc)}), 400


@app.post("/api/runs/<run_id>/detect")
def api_stage_detect(run_id: str):
    try:
        return jsonify(stage_detect(run_id))
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": str(exc)}), 400


@app.post("/api/runs/<run_id>/select-cow")
def api_stage_select_cow(run_id: str):
    try:
        data = request.get_json(force=True) or {}
        cow_id = int(data["cow_id"])
        return jsonify(stage_select_cow(run_id, cow_id))
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": str(exc)}), 400


@app.post("/api/runs/<run_id>/pose")
def api_stage_pose(run_id: str):
    try:
        return jsonify(stage_pose(run_id))
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": str(exc)}), 400


@app.post("/api/runs/<run_id>/measure")
def api_stage_measure(run_id: str):
    try:
        return jsonify(stage_measure(run_id))
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": str(exc)}), 400


@app.post("/api/runs/<run_id>/segment")
def api_stage_segment(run_id: str):
    try:
        return jsonify(stage_segment(run_id))
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": str(exc)}), 400


@app.post("/api/runs/<run_id>/scale")
def api_stage_scale(run_id: str):
    try:
        data = request.get_json(force=True) or {}
        skip = bool(data.get("skip", False))
        reference_px = data.get("reference_px")
        reference_cm = data.get("reference_cm")
        point_a = data.get("point_a")
        point_b = data.get("point_b")
        four_points = data.get("four_points")
        return jsonify(stage_scale(
            run_id,
            reference_px=float(reference_px) if reference_px not in (None, "") else None,
            reference_cm=float(reference_cm) if reference_cm not in (None, "") else None,
            skip=skip,
            point_a=point_a if isinstance(point_a, dict) else None,
            point_b=point_b if isinstance(point_b, dict) else None,
            four_points=four_points if isinstance(four_points, dict) else None,
        ))
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": str(exc)}), 400


def _api_cow_morpho_suggest(run_id: str):
    data = request.get_json(silent=True) or {}
    head = data.get("head_direction")
    if head is not None:
        head = str(head).strip().lower()
        if head not in ("left", "right"):
            return jsonify({"error": "head_direction must be 'left' or 'right'"}), 400
    return jsonify(stage_suggest_four_points(run_id, head_direction=head))


@app.post("/api/runs/<run_id>/four-point-suggest")
def api_four_point_suggest(run_id: str):
    try:
        return _api_cow_morpho_suggest(run_id)
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": str(exc)}), 400


@app.post("/api/runs/<run_id>/four-point-debug")
def api_four_point_debug(run_id: str):
    """Redraw Morpho debug JPEG from current editor keypoints (no re-suggest)."""
    try:
        data = request.get_json(force=True) or {}
        return jsonify(stage_redraw_four_point_debug(
            run_id,
            data.get("keypoints") or {},
            a_end_line=data.get("a_end_line") if isinstance(data.get("a_end_line"), dict) else None,
            lower_chest_guide=(
                data.get("lower_chest_guide")
                if isinstance(data.get("lower_chest_guide"), dict)
                else None
            ),
            head_anchor=data.get("head_anchor") if isinstance(data.get("head_anchor"), dict) else None,
            head_direction=data.get("head_direction"),
        ))
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": str(exc)}), 400


@app.post("/api/runs/<run_id>/cow-morpho-suggest")
def api_cow_morpho_suggest(run_id: str):
    try:
        return _api_cow_morpho_suggest(run_id)
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": str(exc)}), 400


@app.post("/api/diagonal-preview")
def api_diagonal_preview():
    """Client-side live preview helper — no persistence."""
    try:
        data = request.get_json(force=True) or {}
        return jsonify(compute_diagonal_formula(
            data.get("four_points") or {},
            data.get("cm_per_px"),
        ))
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": str(exc)}), 400


@app.post("/api/runs/<run_id>/predict")
def api_stage_predict(run_id: str):
    try:
        return jsonify(stage_predict(run_id))
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": str(exc)}), 400


# ---------------------------------------------------------------------------
# Legacy monolithic endpoints (kept for compatibility)
# ---------------------------------------------------------------------------

@app.post("/api/process-image")
def api_process_image():
    try:
        if "file" not in request.files:
            return jsonify({"error": "Missing file field"}), 400
        f = request.files["file"]
        if not f.filename:
            return jsonify({"error": "Empty filename"}), 400

        enable_seg = request.form.get("enable_segmentation", "false").lower() in ("1", "true", "yes")
        mode = request.form.get("prediction_mode", "smartphone_diagonal")
        if mode not in ("heuristic", "measurement"):
            return jsonify({"error": "prediction_mode must be heuristic or measurement"}), 400

        ref_px = request.form.get("reference_px")
        ref_cm = request.form.get("reference_cm")
        reference_px = float(ref_px) if ref_px not in (None, "") else None
        reference_cm = float(ref_cm) if ref_cm not in (None, "") else None

        report = process_image(
            f,
            enable_segmentation=enable_seg,
            prediction_mode=mode,
            reference_px=reference_px,
            reference_cm=reference_cm,
        )
        return jsonify(report)
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": str(exc)}), 400


@app.post("/api/select-cow")
def api_select_cow():
    try:
        data = request.get_json(force=True)
        run_id = data.get("run_id")
        cow_id = int(data.get("cow_id"))
        enable_seg = bool(data.get("enable_segmentation", False))
        mode = data.get("prediction_mode", "heuristic")
        reference_px = data.get("reference_px")
        reference_cm = data.get("reference_cm")
        report = reselect_cow(
            run_id, cow_id,
            enable_segmentation=enable_seg,
            prediction_mode=mode,
            reference_px=float(reference_px) if reference_px else None,
            reference_cm=float(reference_cm) if reference_cm else None,
        )
        return jsonify(report)
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": str(exc)}), 400


@app.post("/api/convert-scale")
def api_convert_scale():
    try:
        data = request.get_json(force=True)
        run_id = data["run_id"]
        reference_px = float(data["reference_px"])
        reference_cm = float(data["reference_cm"])
        report = apply_scale_to_run(run_id, reference_px, reference_cm)
        return jsonify(report)
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": str(exc)}), 400


@app.get("/api/run/<run_id>")
def api_get_run(run_id: str):
    try:
        return jsonify(load_run(run_id))
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 404


@app.get("/api/run/<run_id>/report")
def api_get_report(run_id: str):
    return api_get_run(run_id)


@app.get("/api/run/<run_id>/file/<path:filename>")
def api_get_file(run_id: str, filename: str):
    name = Path(filename).name
    if name not in ALLOWED_RESULT_FILES:
        return jsonify({"error": "File not allowed"}), 403
    path = RESULTS_DIR / run_id / name
    if not path.is_file():
        return jsonify({"error": "File not found"}), 404
    try:
        path.resolve().relative_to(RESULTS_DIR.resolve())
    except ValueError:
        return jsonify({"error": "Invalid path"}), 403
    resp = send_file(path)
    if name == "four_point_morpho_debug.jpg":
        resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        resp.headers["Pragma"] = "no-cache"
    return resp


@app.get("/api/run/<run_id>/download/json")
def api_download_json(run_id: str):
    path = RESULTS_DIR / run_id / "complete_report.json"
    if not path.is_file():
        return jsonify({"error": "Not found"}), 404
    return send_file(path, as_attachment=True, download_name=f"{run_id}_report.json")


@app.get("/api/run/<run_id>/download/csv")
def api_download_csv(run_id: str):
    path = RESULTS_DIR / run_id / "measurements.csv"
    if not path.is_file():
        return jsonify({"error": "Not found"}), 404
    return send_file(path, as_attachment=True, download_name=f"{run_id}_measurements.csv")


@app.get("/api/run/<run_id>/download/image")
def api_download_image(run_id: str):
    path = RESULTS_DIR / run_id / "measurements_image.jpg"
    if not path.is_file():
        path = RESULTS_DIR / run_id / "pose_image.jpg"
    if not path.is_file():
        return jsonify({"error": "Not found"}), 404
    return send_file(path, as_attachment=True, download_name=f"{run_id}_annotated.jpg")


if __name__ == "__main__":
    print("Dashboard API: http://127.0.0.1:5001")
    print("CORS allowed for Vite: http://127.0.0.1:5173")
    app.run(host="127.0.0.1", port=5001, debug=False, threaded=True)
