"""Flask web UI for offline cattle weight estimation.

Modernized for Python 3.10+ / TensorFlow 2.15+. Uses local models only
(h5model.h5 + model_regression). No cloud inference APIs.
"""

from __future__ import annotations

import os
from pathlib import Path

from flask import Flask, render_template, request
from werkzeug.utils import secure_filename

from cattle_model import CattleWeightEstimator

app = Flask(__name__)
UPLOAD_DIR = Path(__file__).resolve().parent / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

# Lazy-load so `--help`-style imports and cold starts stay light.
_estimator: CattleWeightEstimator | None = None


def get_estimator() -> CattleWeightEstimator:
    global _estimator
    if _estimator is None:
        _estimator = CattleWeightEstimator(backend="auto")
    return _estimator


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/predict", methods=["GET", "POST"])
def upload():
    if request.method != "POST":
        return ""

    f = request.files.get("file")
    if f is None or not f.filename:
        return "No file uploaded", 400

    file_path = UPLOAD_DIR / secure_filename(f.filename)
    f.save(file_path)

    result = get_estimator().predict(file_path)
    return result.message


if __name__ == "__main__":
    # Bind locally only; debug off by default for safer local runs.
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="127.0.0.1", port=port, debug=False)
