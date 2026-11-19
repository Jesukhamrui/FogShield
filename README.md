# Fog-Safe Vehicle and Person Detection

**SIH 2026 · Problem Statement SIH26007 · Ministry of Steel**  

This project evaluates whether dehazing helps detect vehicles and people in low-visibility images. Its two-stage pipeline applies classical Dark Channel Prior (DCP) dehazing, then runs a YOLO detector. A detection-only baseline is included for comparison. A Streamlit app provides an interactive image demo.

## Current evaluation

The latest report evaluates 417 hazed frames. DCP improves image similarity to the clear references, but the current evaluation shows a small decrease in detection mAP compared with the hazed-image baseline:

| Setting | mAP@0.5 |
|---|---:|
| Clear frames (reference) | 37.02% |
| Hazed, detection without dehazing | 38.45% |
| Hazed, DCP dehazing then detection | 36.06% |

| Image quality | Hazed input | Dehazed output |
|---|---:|---:|
| PSNR (dB) | 8.24 | 15.51 |
| SSIM | 0.49 | 0.72 |

Mean latency was 83.2 ms/frame for detection only and 191.5 ms/frame for the two-stage pipeline (182.0 ms median, 300.5 ms p95). These measurements are hardware- and dataset-dependent. See [results/report.md](results/report.md) for the generated report and per-class metrics in [results/report.json](results/report.json).

[results/report_pretrained_100.md](results/report_pretrained_100.md) is a separate 100-frame pretrained-model run. Its scores should not be combined with the 417-frame report above.

## Pipeline and demo

1. **Dehaze:** DCP is the default and requires no additional model. An optional ONNX backend is supported.
2. **Detect:** YOLO detects `person` and `vehicle` (car, motorcycle, bus, truck, bicycle).
3. **Compare:** evaluation reports detection-only and dehaze-plus-detection mAP, image quality, and latency.

The Streamlit app lets you upload an image, choose model weights, adjust confidence and DCP strength, compare with detection without dehazing, and download the dehazed image and annotated detections.

## Setup

Python 3.10 or newer is recommended. Create and activate a virtual environment, then install dependencies:

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

## Run the web demo

```bash
streamlit run app.py
```

Upload a JPG, JPEG, PNG, or BMP image. The app uses `models/yolo_fog.pt` by default when available; `yolov8n.pt` is also an option. The custom weights are produced by the training workflow below and may need to be generated locally.

## Run a single image from the command line

```bash
python scripts/run_pipeline.py --image path/to/foggy.jpg --weights models/yolo_fog.pt
```

The command writes `dehazed.png` and `detections.png` under `results/demo/` by default. To use the ONNX dehazer, provide `--dehazer onnx --onnx-path models/dehazer.onnx`.

## Data and training

The dataset workflow uses clear COCO street-scene images, synthesizes fog variants, and creates labels and YOLO train/validation splits. The current reports describe experimental runs on generated data, not a benchmark on naturally fogged imagery.

1. Fetch clear COCO images and human COCO labels (the script downloads the COCO 2017 validation annotations on first use):

   ```bash
   python scripts/fetch_coco_subset.py --num 700
   ```

2. Generate three fog variants per clear image:

   ```bash
   python scripts/make_hazed_dataset.py --clear data/clear --out data/hazed --per-image 3
   ```

3. Create `data/annotations.json`, YOLO image/label splits, and `data/fog.yaml` from the human labels:

   ```bash
   python scripts/make_labels.py --labels data/clear_labels.json
   ```

   The split is made by source image so variants of one clear scene do not appear in both training and validation. Frames without annotated people or vehicles are excluded. The script also supports detector-generated pseudo-labels, but those measure agreement with the labeling detector rather than ground-truth accuracy.

4. Fine-tune YOLO and copy the best checkpoint to `models/yolo_fog.pt`:

   ```bash
   python scripts/train_detector.py --data data/fog.yaml --base yolov8n.pt --epochs 50 --out models/yolo_fog.pt
   ```

The generation scripts accept configurable image counts and fog parameters. The project has scripts for making a larger dataset, but the available evaluation report covers 417 frames; it does not establish that the 2,000-frame target has been met.

## Evaluate

Set paths and model settings in [configs/default.yaml](configs/default.yaml), then run:

```bash
python -m src.evaluate --config configs/default.yaml
```

The evaluation writes `results/report.json` and `results/report.md`. To evaluate only the held-out YOLO validation split, use:

```bash
python -m src.evaluate --config configs/default.yaml --split val
```

For a learned dehazer, export a model with NCHW float32 input and output in `[0, 1]`, place it at `models/dehazer.onnx`, and set `dehazer.name: onnx` in the config. The ONNX backend runs on CPU.

## Project structure

```text
app.py                       Streamlit image demo
configs/default.yaml         Dehazer, detector, synthesis, and data settings
src/hazing.py                Synthetic atmospheric fog model
src/dehaze.py                DCP and ONNX dehazing backends
src/detector.py              YOLO detection and COCO class mapping
src/pipeline.py              Two-stage and detection-only pipelines
src/metrics.py               Detection and image-quality metrics
src/evaluate.py              Evaluation and Markdown/JSON report generation
scripts/                     Data preparation, training, and CLI demo
tests/                       Unit tests
models/                      Fine-tuned model and optional ONNX model
results/                     Evaluation reports and demo outputs
```

## Limitations

- DCP relies on dark pixels in local image patches and can perform poorly in bright, low-texture scenes.
- Fog is synthesized with a ground-plane depth proxy; it does not reproduce all real-world fog conditions.
- The current 417-frame result shows better PSNR/SSIM after dehazing but lower mAP than the detection-only baseline. Dehazing therefore does not guarantee improved detection.
- Results depend on the evaluation data, detector checkpoint, confidence threshold, and hardware. Report these alongside any new measurements.
