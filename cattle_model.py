"""Offline cattle weight estimation models.

Loads the original ``h5model.h5`` (MobileNetV2 feature extractor + Dense-7)
and ``model_regression`` (sklearn LinearRegression) without network access.

TensorFlow is preferred when available. On machines where TF/PyTorch native
DLLs are blocked, a NumPy MobileNetV2 path loads the same H5 weights locally.
"""

from __future__ import annotations

import pickle
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
H5_PATH = ROOT / "h5model.h5"
REGRESSION_PATH = ROOT / "model_regression"

# Class index -> (age_multiplier, kg_offset) matching original app.py heuristics
CLASS_PARAMS: dict[int, tuple[float, float]] = {
    0: (2.0, -150.0),
    1: (3.0, -125.0),
    2: (4.0, -100.0),
    3: (5.0, -80.0),
    4: (6.0, -100.0),
    5: (7.0, 10.0),
    6: (8.0, 20.0),
}

# MobileNetV2 inverted-residual stride schedule (TF Hub feature_vector/4)
BLOCK_STRIDES = {
    "expanded_conv": 1,
    **{f"expanded_conv_{i}": s for i, s in {
        1: 2, 2: 1, 3: 2, 4: 1, 5: 1, 6: 2, 7: 1, 8: 1, 9: 1,
        10: 1, 11: 1, 12: 1, 13: 2, 14: 1, 15: 1, 16: 1,
    }.items()},
}

BN_EPS = 1e-3


@dataclass
class PredictionResult:
    weight_kg: float
    class_index: int
    class_probs: np.ndarray
    confidence: float
    age_proxy: float
    regression_lb: float
    backend: str
    uncertainty_kg: float = 15.0

    @property
    def message(self) -> str:
        return f"Estimated Weight: {self.weight_kg:.2f}±{self.uncertainty_kg:.0f} KG"


def _clamp_confidence(confidence: float) -> float:
    """Mirror original app.py check() heuristic."""
    if confidence < 0.3 or confidence > 1.7:
        return 0.9
    return float(confidence)


def _load_regression(path: Path = REGRESSION_PATH) -> Any:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        with open(path, "rb") as fh:
            return pickle.load(fh)


def verify_artifacts(h5_path: Path = H5_PATH, regression_path: Path = REGRESSION_PATH) -> dict[str, Any]:
    """Check that model files exist, parse, and contain expected tensors."""
    import h5py

    report: dict[str, Any] = {
        "h5_path": str(h5_path),
        "h5_exists": h5_path.is_file(),
        "h5_size_bytes": h5_path.stat().st_size if h5_path.is_file() else 0,
        "regression_path": str(regression_path),
        "regression_exists": regression_path.is_file(),
        "regression_size_bytes": regression_path.stat().st_size if regression_path.is_file() else 0,
        "usable": False,
        "issues": [],
    }

    if not report["h5_exists"]:
        report["issues"].append("h5model.h5 missing")
        return report
    if not report["regression_exists"]:
        report["issues"].append("model_regression missing")
        return report

    with h5py.File(h5_path, "r") as f:
        cfg = f.attrs.get("model_config")
        if isinstance(cfg, bytes):
            cfg = cfg.decode("utf-8")
        report["keras_version"] = (
            f.attrs.get("keras_version").decode()
            if isinstance(f.attrs.get("keras_version"), bytes)
            else str(f.attrs.get("keras_version"))
        )
        report["model_config_snippet"] = str(cfg)[:240] if cfg else None
        layer_names = [
            n.decode() if isinstance(n, bytes) else n
            for n in f["model_weights"].attrs.get("layer_names", [])
        ]
        report["layer_names"] = layer_names

        dense_k = f["model_weights/dense/dense/kernel:0"][()]
        dense_b = f["model_weights/dense/dense/bias:0"][()]
        report["dense_kernel_shape"] = list(dense_k.shape)
        report["dense_bias_shape"] = list(dense_b.shape)

        has_mobilenet = "model_weights/keras_layer/MobilenetV2/Conv/weights:0" in f
        report["has_mobilenet_v2_weights"] = has_mobilenet
        n_tensors = 0

        def _count(_name: str, obj: Any) -> None:
            nonlocal n_tensors
            if isinstance(obj, h5py.Dataset):
                n_tensors += 1

        f.visititems(_count)
        report["n_weight_tensors"] = n_tensors

        if dense_k.shape != (1280, 7) or dense_b.shape != (7,):
            report["issues"].append(f"Unexpected dense shapes {dense_k.shape}, {dense_b.shape}")
        if not has_mobilenet:
            report["issues"].append("MobileNetV2 weights missing inside H5")
        if "keras_layer" not in layer_names or "dense" not in layer_names:
            report["issues"].append(f"Unexpected layers: {layer_names}")

    try:
        reg = _load_regression(regression_path)
        report["regression_type"] = type(reg).__name__
        report["regression_coef"] = float(np.asarray(reg.coef_).reshape(-1)[0])
        report["regression_intercept"] = float(np.asarray(reg.intercept_).reshape(-1)[0])
    except Exception as exc:  # noqa: BLE001
        report["issues"].append(f"model_regression unreadable: {exc}")

    report["usable"] = len(report["issues"]) == 0
    return report


def preprocess_image(image_path: str | Path, size: tuple[int, int] = (224, 224)) -> np.ndarray:
    """Load image -> NHWC float32 in [0, 1], matching original app.py."""
    img = Image.open(image_path).convert("RGB").resize(size, Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


def estimate_weight_from_probs(probs: np.ndarray, regression_model: Any) -> PredictionResult:
    """Apply original class→age-proxy→regression→kg heuristic."""
    probs = np.asarray(probs, dtype=np.float64).reshape(-1)
    class_index = int(np.argmax(probs))
    confidence = float(probs[class_index])
    multiplier, offset = CLASS_PARAMS[class_index]
    age_proxy = _clamp_confidence(confidence) * multiplier
    regression_lb = float(regression_model.predict(np.array([[age_proxy]], dtype=np.float64))[0])
    weight_kg = (regression_lb / 2.205) + offset
    return PredictionResult(
        weight_kg=weight_kg,
        class_index=class_index,
        class_probs=probs.astype(np.float64),
        confidence=confidence,
        age_proxy=age_proxy,
        regression_lb=regression_lb,
        backend="heuristic",
    )


# ---------------------------------------------------------------------------
# NumPy MobileNetV2 (loads trained H5 weights; no network)
# ---------------------------------------------------------------------------


def _same_pad(x: np.ndarray, k: int, stride: int) -> np.ndarray:
    """TensorFlow 'SAME' padding for NHWC tensors."""
    in_h, in_w = x.shape[1], x.shape[2]
    out_h = int(np.ceil(in_h / stride))
    out_w = int(np.ceil(in_w / stride))
    pad_h = max((out_h - 1) * stride + k - in_h, 0)
    pad_w = max((out_w - 1) * stride + k - in_w, 0)
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    return np.pad(x, ((0, 0), (pad_top, pad_bottom), (pad_left, pad_right), (0, 0)))


def _im2col(x_nhwc: np.ndarray, kh: int, kw: int, stride: int) -> tuple[np.ndarray, int, int]:
    """Extract convolution patches as (N*out_h*out_w, kh*kw*C)."""
    n, h, w, c = x_nhwc.shape
    out_h = (h - kh) // stride + 1
    out_w = (w - kw) // stride + 1
    shape = (n, out_h, out_w, kh, kw, c)
    strides = (
        x_nhwc.strides[0],
        stride * x_nhwc.strides[1],
        stride * x_nhwc.strides[2],
        x_nhwc.strides[1],
        x_nhwc.strides[2],
        x_nhwc.strides[3],
    )
    cols = np.lib.stride_tricks.as_strided(x_nhwc, shape=shape, strides=strides)
    return np.ascontiguousarray(cols.reshape(n * out_h * out_w, kh * kw * c)), out_h, out_w


def _conv2d(x: np.ndarray, w_hwio: np.ndarray, stride: int = 1) -> np.ndarray:
    """Standard convolution. w: H W in out."""
    kh, kw, _cin, cout = w_hwio.shape
    x_p = _same_pad(x, kh, stride)
    n = x.shape[0]
    cols, out_h, out_w = _im2col(x_p, kh, kw, stride)
    # HWIO -> (kh*kw*cin, cout)
    flat_w = w_hwio.reshape(-1, cout)
    y = cols @ flat_w
    return y.reshape(n, out_h, out_w, cout).astype(np.float32, copy=False)


def _depthwise_conv2d(x: np.ndarray, w_hwcm: np.ndarray, stride: int = 1) -> np.ndarray:
    """Depthwise conv. w: H W channels multiplier(1)."""
    kh, kw, c, m = w_hwcm.shape
    if m != 1:
        raise ValueError("Only depth multiplier=1 is supported")
    x_p = _same_pad(x, kh, stride)
    n, h, w, _ = x.shape
    out_h = int(np.ceil(h / stride))
    out_w = int(np.ceil(w / stride))
    # (kh, kw, c) 
    kernel = w_hwcm[:, :, :, 0]
    y = np.zeros((n, out_h, out_w, c), dtype=np.float32)
    for i in range(out_h):
        hs = i * stride
        for j in range(out_w):
            ws = j * stride
            patch = x_p[:, hs : hs + kh, ws : ws + kw, :]  # n, kh, kw, c
            y[:, i, j, :] = np.einsum("nhwc,hwc->nc", patch, kernel, optimize=True)
    return y


def _batch_norm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, mean: np.ndarray, var: np.ndarray) -> np.ndarray:
    scale = gamma / np.sqrt(var + BN_EPS)
    return x * scale.reshape(1, 1, 1, -1) + (beta - mean * scale).reshape(1, 1, 1, -1)


def _relu6(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0.0, 6.0)


class NumpyCattleClassifier:
    """MobileNetV2 + Dense(7, softmax) using weights from h5model.h5."""

    def __init__(self, h5_path: Path = H5_PATH) -> None:
        import h5py

        self.weights: dict[str, np.ndarray] = {}
        with h5py.File(h5_path, "r") as f:
            prefix = "model_weights/keras_layer/"

            def _load(name: str, obj: Any) -> None:
                if isinstance(obj, h5py.Dataset) and name.startswith(prefix):
                    key = name[len(prefix) :]
                    self.weights[key] = np.asarray(obj[()], dtype=np.float32)
                if isinstance(obj, h5py.Dataset) and name.startswith("model_weights/dense/"):
                    key = name[len("model_weights/") :]
                    self.weights[key] = np.asarray(obj[()], dtype=np.float32)

            f.visititems(_load)

        required = [
            "MobilenetV2/Conv/weights:0",
            "MobilenetV2/Conv_1/weights:0",
            "dense/dense/kernel:0",
            "dense/dense/bias:0",
        ]
        missing = [k for k in required if k not in self.weights]
        if missing:
            raise RuntimeError(f"Incomplete h5model.h5, missing: {missing}")

    def _bn(self, x: np.ndarray, scope: str) -> np.ndarray:
        g = self.weights[f"{scope}/BatchNorm/gamma:0"]
        b = self.weights[f"{scope}/BatchNorm/beta:0"]
        m = self.weights[f"{scope}/BatchNorm/moving_mean:0"]
        v = self.weights[f"{scope}/BatchNorm/moving_variance:0"]
        return _batch_norm(x, g, b, m, v)

    def _conv_bn_relu6(self, x: np.ndarray, scope: str, stride: int) -> np.ndarray:
        w = self.weights[f"{scope}/weights:0"]
        x = _conv2d(x, w, stride=stride)
        x = self._bn(x, scope)
        return _relu6(x)

    def _inverted_residual(self, x: np.ndarray, block: str, stride: int) -> np.ndarray:
        residual = x
        expand_key = f"MobilenetV2/{block}/expand/weights:0"
        if expand_key in self.weights:
            x = self._conv_bn_relu6(x, f"MobilenetV2/{block}/expand", stride=1)

        w_dw = self.weights[f"MobilenetV2/{block}/depthwise/depthwise_weights:0"]
        x = _depthwise_conv2d(x, w_dw, stride=stride)
        x = self._bn(x, f"MobilenetV2/{block}/depthwise")
        x = _relu6(x)

        w_pw = self.weights[f"MobilenetV2/{block}/project/weights:0"]
        x = _conv2d(x, w_pw, stride=1)
        x = self._bn(x, f"MobilenetV2/{block}/project")

        if stride == 1 and residual.shape[-1] == x.shape[-1]:
            x = x + residual
        return x

    def predict_proba(self, batch_nhwc: np.ndarray) -> np.ndarray:
        x = batch_nhwc.astype(np.float32)
        x = self._conv_bn_relu6(x, "MobilenetV2/Conv", stride=2)

        for block, stride in BLOCK_STRIDES.items():
            x = self._inverted_residual(x, block, stride)

        x = self._conv_bn_relu6(x, "MobilenetV2/Conv_1", stride=1)
        # Global average pool -> (N, 1280)
        features = x.mean(axis=(1, 2))
        logits = features @ self.weights["dense/dense/kernel:0"] + self.weights["dense/dense/bias:0"]
        logits = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(logits)
        return exp / exp.sum(axis=1, keepdims=True)


# ---------------------------------------------------------------------------
# TensorFlow backend (preferred when native DLLs load)
# ---------------------------------------------------------------------------


class TFCattleClassifier:
    """Load original Keras H5 with TF-Hub KerasLayer custom object."""

    def __init__(self, h5_path: Path = H5_PATH) -> None:
        import tensorflow as tf
        import tensorflow_hub as hub

        self.model = tf.keras.models.load_model(
            str(h5_path),
            custom_objects={"KerasLayer": hub.KerasLayer},
            compile=False,
        )

    def predict_proba(self, batch_nhwc: np.ndarray) -> np.ndarray:
        probs = self.model.predict(batch_nhwc, verbose=0)
        return np.asarray(probs, dtype=np.float64)


class CattleWeightEstimator:
    """End-to-end offline estimator: image -> class probs -> kg weight."""

    def __init__(
        self,
        h5_path: Path = H5_PATH,
        regression_path: Path = REGRESSION_PATH,
        backend: str = "auto",
    ) -> None:
        self.regression = _load_regression(regression_path)
        self.backend_name = "numpy"
        self.classifier: Any

        if backend in ("auto", "tf"):
            try:
                self.classifier = TFCattleClassifier(h5_path)
                self.backend_name = "tensorflow"
            except Exception:
                if backend == "tf":
                    raise
                self.classifier = NumpyCattleClassifier(h5_path)
                self.backend_name = "numpy"
        elif backend == "numpy":
            self.classifier = NumpyCattleClassifier(h5_path)
            self.backend_name = "numpy"
        else:
            raise ValueError(f"Unknown backend: {backend}")

    def predict(self, image_path: str | Path) -> PredictionResult:
        batch = preprocess_image(image_path)
        probs = self.classifier.predict_proba(batch)[0]
        result = estimate_weight_from_probs(probs, self.regression)
        result.backend = self.backend_name
        return result
