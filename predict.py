#!/usr/bin/env python3
"""Predict cattle weight from a single image (fully offline).

Usage:
    python predict.py path/to/cow.jpg
    python predict.py path/to/cow.jpg --backend numpy
    python predict.py --verify
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from cattle_model import CattleWeightEstimator, verify_artifacts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Estimate cattle live weight from an image using local trained models."
    )
    parser.add_argument(
        "image",
        nargs="?",
        type=Path,
        help="Path to a cow image (jpg/png).",
    )
    parser.add_argument(
        "--backend",
        choices=("auto", "tf", "numpy"),
        default="auto",
        help="Inference backend. 'auto' uses TensorFlow if available, else NumPy.",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Only verify that h5model.h5 and model_regression are complete/usable.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON instead of a human summary.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.verify or args.image is None:
        report = verify_artifacts()
        if args.json or args.verify:
            print(json.dumps(report, indent=2))
        else:
            print("Usage: python predict.py <image> [--backend auto|tf|numpy]")
            print("       python predict.py --verify")
        return 0 if report.get("usable") else 1

    if not args.image.is_file():
        print(f"Image not found: {args.image}", file=sys.stderr)
        return 1

    estimator = CattleWeightEstimator(backend=args.backend)
    result = estimator.predict(args.image)

    payload = {
        "image": str(args.image),
        "weight_kg": round(result.weight_kg, 2),
        "uncertainty_kg": result.uncertainty_kg,
        "class_index": result.class_index,
        "confidence": round(result.confidence, 6),
        "age_proxy": round(result.age_proxy, 6),
        "regression_lb": round(result.regression_lb, 4),
        "class_probs": [round(float(p), 6) for p in result.class_probs],
        "backend": result.backend,
        "message": result.message,
    }

    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(result.message)
        print(f"class={result.class_index}  confidence={result.confidence:.4f}  backend={result.backend}")
        print(f"age_proxy={result.age_proxy:.4f}  regression_lb={result.regression_lb:.2f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
