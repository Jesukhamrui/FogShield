"""Fine-tune YOLO on hazed frames (and optionally dehazed frames) for vehicle/person classes.

Expects a YOLO-format dataset described by a data YAML, e.g. data/fog.yaml:
  path: data
  train: images/train
  val: images/val
  names: {0: person, 1: vehicle}

Usage:
  python scripts/train_detector.py --data data/fog.yaml --base yolov8n.pt --epochs 50
"""
import argparse
import shutil

from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--base", default="yolov8n.pt")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default="models/yolo_fog.pt")
    args = ap.parse_args()

    model = YOLO(args.base)
    results = model.train(data=args.data, epochs=args.epochs, imgsz=args.imgsz,
                          batch=args.batch, device=args.device, project="runs",
                          name="fog_detector", exist_ok=True)
    best = results.save_dir / "weights" / "best.pt"
    shutil.copy(best, args.out)
    print(f"Best weights copied to {args.out}")


if __name__ == "__main__":
    main()
