# trt_yolo_bench

**YOLO11 → TensorRT inference harness in hand-written C++.**

End-to-end precision benchmarking (FP32, FP16, INT8) with direct `nvinfer1`
C++ runtime and COCO val2017 accuracy evaluation. TensorRT 11 uses explicitly
typed ONNX graphs for FP16 and INT8; no black-box postprocessing is used.
The point of this repo is not "I called `model.export(format='engine')`" — it's
a real answer to "show me a model you took to silicon and the tradeoff you
owned": what changes at each precision, why, and by how much.

> Status: **implementation ready for measured Colab execution; results pending.**
> Run the end-to-end workflow in [`colab/RUNBOOK.md`](colab/RUNBOOK.md) on a
> Google Colab T4 GPU. [`RESULTS.md`](RESULTS.md) intentionally remains a
> placeholder until real engines have been built and scored.

---

## Core Components

**Export Pipeline** (`export/`)
- PyTorch → ONNX → TensorRT engine builder
- Custom INT8 entropy calibrator with calibration dataset
- Outputs FP32 / FP16 / INT8 engines ready for inference

**Inference Harness** (`harness/`)
- Direct `nvinfer1` C++ API
- CUDA buffer management, letterbox preprocessing
- Hand-decoded YOLO detection head: `[1, 84, 8400]` → bounding boxes + confidence
- Custom NMS (no TensorRT plugin dependency)
- Latency measurement on critical path

**Evaluation** (`eval/`)
- Benchmark output scored against COCO val2017 ground truth
- Official `pycocotools` reference evaluator
- Outputs mAP50, mAP75, per-class metrics

---

## Usage

```bash
# Export YOLO11 to TensorRT engines
python export/download_coco_slice.py
python export/export_onnx.py --weights yolo11n.pt --out-dir models
python export/convert_fp16.py --input models/yolo11n.onnx --output models/yolo11n_fp16.onnx
python export/build_engines.py --onnx models/yolo11n.onnx --fp16-onnx models/yolo11n_fp16.onnx --engines-dir engines

# Run inference and benchmark
cd harness
cmake -S . -B build && cmake --build build --config Release
./build/trt_yolo_bench ../engines/yolo11n_fp16.engine ../data/val_slice.json --out results.json

# Evaluate accuracy
python eval/score_coco.py --dets results.json
```

The harness defaults to confidence `0.001`, NMS IoU `0.7`, 300 maximum
detections, 50 warmups, and 1000 timed iterations. It reports CUDA-event GPU
latency plus end-to-end latency and writes raw per-iteration samples with
`--latency-out`. It expects a YOLO-style output tensor compatible with
`[1, 84, 8400]` at 640x640 input resolution and writes detections in COCO
results JSON format.

TensorRT 11 uses strongly typed ONNX graphs: FP16 is created by
`export/convert_fp16.py`, while INT8 requires an explicitly quantized ONNX
graph containing Q/DQ nodes. Pass that graph as
`--int8-onnx models/yolo11n_int8.onnx` when it is available.
