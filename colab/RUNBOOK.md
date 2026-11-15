# Colab runbook

This is the reproducible Google Colab path for the benchmark. Select
**Runtime -> Change runtime type -> T4 GPU** before starting. Do not copy
numbers into `RESULTS.md` unless the corresponding cell completed on a CUDA
runtime.

## 1. Confirm the runtime

```bash
!nvidia-smi
!nvcc --version
```

Record the GPU, driver, and CUDA output in `results/env.txt`.

## 2. Clone the repository

```bash
!git clone https://github.com/chao-dotcom/trt_yolo_bench.git /content/trt_yolo_bench
%cd /content/trt_yolo_bench
```

## 3. Install TensorRT and its C++ SDK

The Python wheel alone is not enough: the C++ harness needs `NvInfer.h` and
`libnvinfer.so`. First try the development packages exposed by the Colab
image:

```bash
!apt-get update -qq
!apt-get install -y -qq libopencv-dev cmake
!dpkg -l | grep -i nvinfer
!find /usr /usr/local -name "NvInfer.h" 2>/dev/null
```

Colab commonly does not expose `libnvinfer-dev` through apt. The Python wheel
also does not include C++ headers, so fetch the matching header tree for the
harness and point CMake at the pip-installed TensorRT libraries:

```bash
!git clone --depth 1 -q https://github.com/NVIDIA/TensorRT.git /content/TensorRT
```

Do not continue with a Python-only installation unless you are skipping the
C++ harness. A root-enabled GPU VM with the full TensorRT SDK is the fallback
if the Colab TensorRT version and header tree are incompatible.

## 4. Install Python dependencies

```bash
!python -m pip install -q --upgrade pip
!pip install -q ultralytics onnx pycocotools requests opencv-python tensorrt cuda-python
!python -c "import tensorrt as trt; print('TensorRT', trt.__version__)"
!python -c "import tensorrt_libs; print(tensorrt_libs.__file__)"
!git clone --depth 1 -q https://github.com/NVIDIA/TensorRT.git /content/TensorRT
```

## 5. Prepare deterministic data

```bash
!python export/download_coco_slice.py --val-count 1000 --calib-count 500 --insecure-downloads
```

This downloads only the selected COCO images. The script asserts that the
evaluation and calibration image IDs are disjoint.

## 6. Export and build all engines

```bash
!python export/export_onnx.py --weights yolo11n.pt --out-dir models
!python export/convert_fp16.py --input models/yolo11n.onnx --output models/yolo11n_fp16.onnx
!python export/build_engines.py \
    --onnx models/yolo11n.onnx \
    --fp16-onnx models/yolo11n_fp16.onnx \
    --engines-dir engines \
    2>&1 | tee build.log
```

The build produces `yolo11n_fp32.engine` and `yolo11n_fp16.engine`. TensorRT
11 does not accept the legacy Python INT8 calibrator; INT8 must first be
represented by an explicitly quantized ONNX graph with Q/DQ nodes. Once such a
graph exists, add `--int8-onnx models/yolo11n_int8.onnx` to the build command.

## 7. Build the C++ harness

```bash
!bash -lc 'set -e; TRT_HEADER=$(find / -name NvInfer.h 2>/dev/null | head -n 1); TRT_LIB=$(find / -name "libnvinfer.so*" -type f 2>/dev/null | head -n 1); test -n "$TRT_HEADER"; test -n "$TRT_LIB"; cmake -S harness -B harness/build -DTENSORRT_INCLUDE_DIR=$(dirname "$TRT_HEADER") -DTENSORRT_LIB="$TRT_LIB"'
!cmake --build harness/build --parallel
```

## 8. Run FP32 and FP16

The defaults are the fixed plan settings: confidence `0.001`, NMS IoU `0.7`,
maximum 300 detections, 50 warmup iterations, and 1000 timed iterations.
Each run emits CUDA-event GPU latency, end-to-end latency, and a raw CSV.

```bash
!./harness/build/trt_yolo_bench engines/yolo11n_fp32.engine data/val_slice.json --out results_fp32.json --latency-out latency_fp32.csv 2>&1 | tee bench_fp32.log
!./harness/build/trt_yolo_bench engines/yolo11n_fp16.engine data/val_slice.json --out results_fp16.json --latency-out latency_fp16.csv 2>&1 | tee bench_fp16.log
```

INT8 is intentionally deferred here. TensorRT 11 requires an explicitly
quantized ONNX graph containing Q/DQ nodes; the removed legacy calibrator cannot
produce an INT8 engine. Build and benchmark FP32/FP16 first.

## 9. Score accuracy

```bash
!python eval/score_coco.py --dets results_fp32.json | tee score_fp32.log
!python eval/score_coco.py --dets results_fp16.json | tee score_fp16.log
```

Run the PyTorch reference on the exact same image list and thresholds:

```bash
!python eval/run_ultralytics_reference.py --weights yolo11n.pt --out results_pytorch.json
!python eval/score_coco.py --dets results_pytorch.json | tee score_pytorch.log
```

Copy the exact COCOeval summaries and benchmark logs into `RESULTS.md`; do
not estimate or round from memory.

## 10. Record the environment and results

```bash
!mkdir -p results
!{ echo '=== nvidia-smi ==='; nvidia-smi; echo '=== nvcc ==='; nvcc --version; \
   echo '=== TensorRT ==='; python -c "import tensorrt as trt; print(trt.__version__)"; \
   echo '=== Python packages ==='; pip freeze; } | tee results/env.txt
```

Fill `RESULTS.md` only after all three precisions have been evaluated. Include
the AP values, GPU mean/p50/p99, end-to-end mean, FPS, engine sizes, raw CSV
paths, calibration-size study, and any Colab limitations.
