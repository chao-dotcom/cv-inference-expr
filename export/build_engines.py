#!/usr/bin/env python3
"""Build TensorRT engines from explicitly typed ONNX graphs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import tensorrt as trt

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", type=Path, default=Path("models/yolo11n.onnx"))
    parser.add_argument("--fp16-onnx", type=Path, default=None)
    parser.add_argument("--int8-onnx", type=Path, default=None)
    parser.add_argument("--engines-dir", type=Path, default=Path("engines"))
    parser.add_argument("--workspace-gb", type=float, default=4.0)
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Engine filename stem. Defaults to the ONNX filename stem.",
    )
    return parser.parse_args()


def build_serialized_engine(
    onnx_path: Path,
    logger: trt.ILogger,
    workspace_bytes: int,
) -> bytes:
    builder = trt.Builder(logger)
    explicit_batch = getattr(trt.NetworkDefinitionCreationFlag, "EXPLICIT_BATCH", None)
    network_flags = 0 if explicit_batch is None else 1 << int(explicit_batch)
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, logger)

    if not parser.parse(onnx_path.read_bytes()):
        details = "\n".join(
            f"  [{i}] {parser.get_error(i)}" for i in range(parser.num_errors)
        )
        raise RuntimeError(f"ONNX parse failed:\n{details}")

    config = builder.create_builder_config()
    if hasattr(config, "set_memory_pool_limit"):
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_bytes)
    else:
        config.max_workspace_size = workspace_bytes

    if hasattr(builder, "build_serialized_network"):
        plan = builder.build_serialized_network(network, config)
        if plan is None:
            raise RuntimeError(f"TensorRT returned None while building {onnx_path}")
        return bytes(plan)

    engine = builder.build_engine(network, config)
    if engine is None:
        raise RuntimeError(f"TensorRT returned None while building {onnx_path}")
    return bytes(engine.serialize())


def main() -> None:
    args = parse_args()
    logger = trt.Logger(trt.Logger.INFO)

    if not args.onnx.exists():
        raise FileNotFoundError(f"ONNX file not found: {args.onnx}")

    args.engines_dir.mkdir(parents=True, exist_ok=True)
    workspace_bytes = int(args.workspace_gb * (1 << 30))
    model_name = args.model_name or args.onnx.stem

    targets = [("fp32", args.onnx)]
    if args.fp16_onnx is not None:
        targets.append(("fp16", args.fp16_onnx))
    if args.int8_onnx is not None:
        targets.append(("int8", args.int8_onnx))

    for precision, onnx_path in targets:
        if not onnx_path.exists():
            raise FileNotFoundError(f"{precision} ONNX file not found: {onnx_path}")
        print(f"[build] precision={precision} onnx={onnx_path}")
        plan = build_serialized_engine(
            onnx_path=onnx_path,
            logger=logger,
            workspace_bytes=workspace_bytes,
        )

        engine_path = args.engines_dir / f"{model_name}_{precision}.engine"
        engine_path.write_bytes(plan)
        print(f"[write] {engine_path} ({len(plan) / (1024 * 1024):.2f} MB)")

    print("[done] build step finished")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[error] {exc}")
        sys.exit(1)
