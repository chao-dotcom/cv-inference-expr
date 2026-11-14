# trt\_yolo\_bench: Finishing Plan

Goal: turn RESULTS.md from "pending" into a verified FP32 / FP16 / INT8 benchmark that a reviewer can rerun.

## Definition of done

- [ ] FP32 accuracy reproduces the reference model's known AP (sanity gate)
- [ ] FP16 and INT8 engines built, evaluated and timed with the same pipeline
- [ ] RESULTS.md filled in: environment block, exact commands, accuracy / latency / size table
- [ ] Calibration study (at least 3 calibration-set sizes) and latency breakdown added
- [ ] README top section states the headline result in two sentences
- [ ] Repo reruns from a clean clone using only the documented commands

Target effort: about 4 to 6 focused sessions. Phases 0 to 4 are the must-haves; Phases 5 and 6 are the differentiators.

## Fixed benchmark settings (decide once, never change between precisions)

| Setting | Choice |
| --- | --- |
| Model | YOLOv8n first (YOLOv8s only if time remains) |
| Input size | 640x640, letterbox, batch 1 |
| Confidence threshold | 0.001 for AP evaluation (standard for COCOeval) |
| NMS IoU threshold | 0.7 |
| Max detections | 300 |
| Eval set | COCO val2017, fixed image list saved to a file (full 5000 if feasible; otherwise a fixed slice of 1000) |
| Calibration set | COCO train2017 images, disjoint from the eval set |
| Warmup | 50 iterations discarded |
| Timed iterations | 1000 or more per measurement |

Write these into RESULTS.md up front. Any mismatch between precisions makes the comparison invalid.

## Phase 0: Environment and reproducibility (30 minutes)

- [ ] Run on a real NVIDIA GPU (the README says results require one).
- [ ] Record: GPU model, driver, CUDA, TensorRT, cuDNN, OpenCV, CMake versions (`nvidia-smi`, `nvcc --version`, TensorRT version string).
- [ ] Lock the GPU clocks if you can (`nvidia-smi -lgc`), or at least note power mode, so latency numbers are stable.
- [ ] Save the environment block to `results/env.txt` and paste it into RESULTS.md.
- [ ] Pin the Python package versions used for export and eval in `requirements.txt`.

## Phase 1: Export and FP32 baseline (accuracy gate)

- [ ] Export ONNX: `yolo export model=yolov8n.pt format=onnx imgsz=640 opset=17 simplify=True`.
- [ ] Build the FP32 engine with your builder code and save the build log.
- [ ] Run your C++ harness over the fixed eval list and write detections to COCO-format JSON.
- [ ] Evaluate with pycocotools and record AP@\[0.5:0.95\] and AP@0.5.
- [ ] **Baseline check:** run the original PyTorch/Ultralytics model on the same image list with the same thresholds and record its AP.

**Gate:** your FP32 TensorRT AP must be within about 0.5 AP of the PyTorch result on the same list. (For reference, YOLOv8n reports roughly 37 AP on full COCO val; a slice will differ, which is why you compare on the identical list.)

If the gate fails, fix these in order before moving on:

1. Letterbox: scale, padding, and BGR vs RGB.
2. Normalization: divide by 255, channel order, NCHW layout.
3. Output decode: YOLOv8 output is \[1, 84, 8400\] (no objectness score), so check the transpose and class-score handling.
4. Un-letterbox: subtract padding, divide by scale, clip to image bounds.
5. Category-id mapping: model indices 0 to 79 must map to COCO category ids 1 to 90.
6. NMS: per-class, with the right thresholds.

## Phase 2: FP16

- [ ] Build the FP16 engine (same ONNX, FP16 flag set).
- [ ] Evaluate with the identical pipeline.

**Gate:** AP drop vs FP32 of about 0.3 or less. A larger drop suggests a pipeline bug rather than a precision effect.

## Phase 3: INT8

- [ ] Make a calibration list of 500 train2017 images, saved to a file, with no overlap with the eval list (add an assert that checks this).
- [ ] Make sure the calibrator uses the exact same preprocessing as inference.
- [ ] Implement `getBatch`, `readCalibrationCache` and `writeCalibrationCache`. Store the cache file and rerun from the cache to confirm it works.
- [ ] Build the INT8 engine and evaluate.

**Gate:** AP drop vs FP32 of roughly 0.5 to 2. If the drop is above about 3, check: calibration preprocessing mismatch, too few or unrepresentative images, or the calibrator algorithm (entropy vs minmax).

Verify the engine really uses INT8: inspect the build log or per-layer info and check the engine size (INT8 should be clearly smaller than FP16).

## Phase 4: Timing harness and the results table

Implement timing so the numbers are trustworthy:

- Use CUDA events around `enqueueV3` for **GPU-only** time, and a CPU wall clock with a stream sync for **end-to-end** time.
- Discard 50 warmup runs, then time at least 1000 iterations and store every sample.
- Report mean, p50, p99 (compute from the stored samples).
- Report throughput as 1000 / mean latency at batch 1.
- Fix the input once and reuse it for timing runs, so image decoding does not pollute the numbers.

Cross-check your harness against `trtexec` (same engine, report GPU compute time). If the two disagree by more than a few percent, find out why before publishing.

Results table to produce:

| Precision | AP@\[.5:.95\] | AP@.5 | GPU mean (ms) | GPU p99 (ms) | E2E mean (ms) | FPS | Engine MB |
| --- | --- | --- | --- | --- | --- | --- | --- |
| PyTorch (reference) |  |  | n/a | n/a |  |  | n/a |
| TRT FP32 |  |  |  |  |  |  |  |
| TRT FP16 |  |  |  |  |  |  |  |
| TRT INT8 |  |  |  |  |  |  |  |

Also record the speedup and AP delta of FP16 and INT8 relative to FP32.

## Phase 5: Calibration study (high value, cheap)

- [ ] Build INT8 engines with 100, 500 and 1000 calibration images.
- [ ] Optional: one run calibrated on random noise or a single image, as a deliberate bad example.
- [ ] Plot or tabulate AP against calibration size.
- [ ] Write 2 to 3 sentences on what you observed and why.

## Phase 6: Latency breakdown (high value, cheap)

- [ ] Time preprocess (letterbox + normalize), host-to-device copy, inference, device-to-host copy, decode, NMS separately.
- [ ] Show them as a stacked bar per precision.
- [ ] State the bottleneck. Once INT8 is fast, CPU-side preprocessing and NMS often dominate, and that is a finding worth reporting.

Optional extras if you still have time: keep the detect head in FP16 and rest INT8 to show mixed-precision recovery; run YOLOv8s for a second data point; run the same table on a second device and link it to your capstone benchmarking project.

## RESULTS.md structure

1. Summary: two or three sentences with the headline numbers
2. Environment block (GPU, driver, CUDA, TensorRT, versions, clock settings)
3. Benchmark settings (the fixed settings table above)
4. Main results table
5. Calibration study
6. Latency breakdown
7. Commands used (copy-paste reproducible)
8. Known limitations (eval slice size, single GPU, batch 1 only)

Replace every "pending" with a number or with an honest "not run".

## Pitfalls checklist

- [ ] Warmup excluded and CUDA sync used for every timing
- [ ] Same resolution and thresholds across all precisions
- [ ] Calibration images disjoint from eval images
- [ ] Detections written with COCO category ids and xywh boxes in original image coordinates
- [ ] Engine files built on the same GPU and TensorRT version they are benchmarked on
- [ ] Raw per-iteration latency samples saved to disk, not only the summary

## Final step: resume line (fill in only after real numbers exist)

> Built a YOLOv8 TensorRT benchmarking pipeline in C++ (CUDA buffers, custom INT8 entropy calibrator); FP16 gave \[X\]x and INT8 \[Y\]x speedup over FP32 on \[GPU\] with an AP drop of \[A\] / \[B\] on COCO val, p99 latency \[Z\] ms.

Do not write this line until RESULTS.md contains the numbers behind every bracket.
