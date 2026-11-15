#!/usr/bin/env python3
"""Convert an ONNX graph to explicit FP16 using NVIDIA ModelOpt."""

from __future__ import annotations

import argparse
from pathlib import Path

import onnx
import modelopt.onnx.autocast as autocast


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"ONNX input not found: {args.input}")

    converted = autocast.convert_to_mixed_precision(
        onnx_path=str(args.input),
        low_precision_type="fp16",
        keep_io_types=True,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(converted, str(args.output))
    print(f"[write] {args.output}")


if __name__ == "__main__":
    main()
