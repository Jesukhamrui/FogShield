"""Stage 2: vehicle and person detection (Ultralytics YOLO).

COCO classes are mapped onto the two classes the problem statement needs:
'person' and 'vehicle' (car, motorcycle, bus, truck, bicycle). If you fine-tune on
fog data with those two labels directly, set `class_map=None` to use raw labels.
"""
import numpy as np

DEFAULT_COCO_MAP = {0: "person", 1: "vehicle", 2: "vehicle", 3: "vehicle",
                    5: "vehicle", 7: "vehicle"}


class Detector:
    def __init__(self, weights: str, conf: float = 0.25, imgsz: int = 640,
                 class_map: dict | None = DEFAULT_COCO_MAP, device: str | None = None):
        from ultralytics import YOLO
        self.model = YOLO(weights)
        self.conf = conf
        self.imgsz = imgsz
        self.class_map = class_map
        self.device = device

    def __call__(self, img_bgr: np.ndarray) -> dict:
        res = self.model.predict(img_bgr, conf=self.conf, imgsz=self.imgsz,
                                 device=self.device, verbose=False)[0]
        boxes, scores, labels = [], [], []
        names = res.names
        for xyxy, score, cls in zip(res.boxes.xyxy.cpu().numpy(),
                                    res.boxes.conf.cpu().numpy(),
                                    res.boxes.cls.cpu().numpy().astype(int)):
            if self.class_map is None:
                label = names[cls]
            elif cls in self.class_map:
                label = self.class_map[cls]
            else:
                continue
            boxes.append(xyxy)
            scores.append(float(score))
            labels.append(label)
        return {
            "boxes": np.array(boxes, dtype=np.float32).reshape(-1, 4),
            "scores": np.array(scores, dtype=np.float32),
            "labels": labels,
        }
