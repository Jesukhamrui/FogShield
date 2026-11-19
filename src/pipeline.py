"""End-to-end pipelines: dehaze -> detect (two-stage) and detect-only (mandatory baseline)."""
import time


class TwoStagePipeline:
    def __init__(self, dehazer, detector):
        self.dehazer = dehazer
        self.detector = detector

    def run(self, img_bgr):
        t0 = time.perf_counter()
        enhanced = self.dehazer.dehaze(img_bgr)
        t1 = time.perf_counter()
        dets = self.detector(enhanced)
        t2 = time.perf_counter()
        return {
            "enhanced": enhanced,
            "detections": dets,
            "timings_ms": {
                "dehaze": (t1 - t0) * 1000,
                "detect": (t2 - t1) * 1000,
                "total": (t2 - t0) * 1000,
            },
        }


class BaselinePipeline:
    """Detection without dehazing: the comparison required by the problem statement."""

    def __init__(self, detector):
        self.detector = detector

    def run(self, img_bgr):
        t0 = time.perf_counter()
        dets = self.detector(img_bgr)
        t1 = time.perf_counter()
        return {
            "enhanced": img_bgr,
            "detections": dets,
            "timings_ms": {"dehaze": 0.0, "detect": (t1 - t0) * 1000, "total": (t1 - t0) * 1000},
        }
