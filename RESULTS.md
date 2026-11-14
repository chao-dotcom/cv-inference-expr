# Results

**Status: not run.** This file contains no estimated benchmark numbers. Run
the complete workflow in [`colab/RUNBOOK.md`](colab/RUNBOOK.md) on a real
NVIDIA GPU, then replace each `not run` entry with the captured output.

## Summary

Not run. Do not write the headline sentence until the three TensorRT
precisions and the PyTorch reference have been evaluated on the same slice.

## Environment

Record the complete contents of `results/env.txt` here after the Colab run.

- GPU: not run
- Driver: not run
- CUDA: not run
- TensorRT: not run
- cuDNN: not run
- OpenCV: not run
- CMake: not run
- GPU clock/power settings: not run
- OS/runtime image: not run

## Benchmark settings

| Setting | Value |
|---|---|
| Model | YOLO11n |
| Input | 640x640 letterbox, batch 1 |
| Confidence threshold | 0.001 |
| NMS IoU threshold | 0.7 |
| Maximum detections | 300 |
| Evaluation set | deterministic first 1000 COCO val2017 image IDs |
| Calibration set | deterministic first 500 COCO train2017 image IDs |
| Warmup | 50 iterations discarded |
| Timed iterations | 1000 |

## Main results

| Precision | AP@[.5:.95] | AP@.5 | GPU mean (ms) | GPU p50 (ms) | GPU p99 (ms) | E2E mean (ms) | FPS | Engine MB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| PyTorch reference | not run | not run | n/a | n/a | n/a | not run | not run | n/a |
| TRT FP32 | not run | not run | not run | not run | not run | not run | not run | not run |
| TRT FP16 | not run | not run | not run | not run | not run | not run | not run | not run |
| TRT INT8 | not run | not run | not run | not run | not run | not run | not run | not run |

Speedup versus TRT FP32:

| Precision | GPU mean speedup | AP delta @[.5:.95] | AP delta @.5 |
|---|---:|---:|---:|
| FP16 | not run | not run | not run |
| INT8 | not run | not run | not run |

## Verbatim accuracy output

Paste the complete `score_pytorch.log`, `score_fp32.log`,
`score_fp16.log`, and `score_int8.log` output here after execution.

```text
not run
```

## Verbatim benchmark output

Paste the complete `bench_fp32.log`, `bench_fp16.log`, and `bench_int8.log`
output here after execution. Keep the corresponding raw
`latency_*.csv` files with the results.

```text
not run
```

## Calibration study

Build and evaluate INT8 engines using calibration sets of 100, 500, and 1000
images. Record the exact cache and engine names for each run.

| Calibration images | AP@[.5:.95] | AP@.5 | GPU mean (ms) | Engine MB |
|---:|---:|---:|---:|---:|
| 100 | not run | not run | not run | not run |
| 500 | not run | not run | not run | not run |
| 1000 | not run | not run | not run | not run |

Observation: not run.

## Latency breakdown

The current harness records GPU inference and end-to-end samples. Add
preprocess, H2D, inference, D2H, decode, and NMS measurements before filling
this section.

| Precision | Preprocess (ms) | H2D (ms) | Inference (ms) | D2H (ms) | Decode (ms) | NMS (ms) |
|---|---:|---:|---:|---:|---:|---:|
| TRT FP32 | not run | not run | not run | not run | not run | not run |
| TRT FP16 | not run | not run | not run | not run | not run | not run |
| TRT INT8 | not run | not run | not run | not run | not run | not run |

## Exact commands and limitations

Use the commands in [`colab/RUNBOOK.md`](colab/RUNBOOK.md). The benchmark is
currently limited to one Colab GPU, batch 1, the deterministic 1000-image
evaluation slice, and the selected TensorRT/CUDA runtime image.
