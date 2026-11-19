"""Run the two-stage pipeline on a single image and save an annotated result.

Usage:
  python scripts/run_pipeline.py --image path/to/foggy.jpg --weights models/yolo_fog.pt
"""
import argparse
import os
import sys

import cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.dehaze import build_dehazer  # noqa: E402
from src.detector import Detector  # noqa: E402
from src.pipeline import TwoStagePipeline  # noqa: E402


def draw(img, dets):
    out = img.copy()
    color = {"person": (0, 255, 255), "vehicle": (0, 255, 0)}
    for box, score, label in zip(dets["boxes"], dets["scores"], dets["labels"]):
        x1, y1, x2, y2 = map(int, box)
        c = color.get(label, (255, 0, 0))
        cv2.rectangle(out, (x1, y1), (x2, y2), c, 2)
        cv2.putText(out, f"{label} {score:.2f}", (x1, max(y1 - 5, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--weights", default="models/yolo_fog.pt")
    ap.add_argument("--dehazer", default="dcp", choices=["dcp", "onnx"])
    ap.add_argument("--onnx-path", default="models/dehazer.onnx")
    ap.add_argument("--out", default="results/demo")
    args = ap.parse_args()

    img = cv2.imread(args.image)
    if img is None:
        raise SystemExit(f"Could not read {args.image}")

    dehazer_kwargs = {"model_path": args.onnx_path} if args.dehazer == "onnx" else {}
    pipe = TwoStagePipeline(build_dehazer(args.dehazer, **dehazer_kwargs),
                            Detector(args.weights))
    result = pipe.run(img)

    os.makedirs(args.out, exist_ok=True)
    cv2.imwrite(os.path.join(args.out, "dehazed.png"), result["enhanced"])
    cv2.imwrite(os.path.join(args.out, "detections.png"), draw(result["enhanced"], result["detections"]))
    print(f"Detections: {len(result['detections']['labels'])}")
    print("Timings (ms):", {k: round(v, 1) for k, v in result["timings_ms"].items()})
    print(f"Saved to {args.out}/")


if __name__ == "__main__":
    main()
