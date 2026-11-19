"""Stage 1: dehazing.

Two interchangeable backends behind the same interface `dehaze(img_bgr) -> img_bgr`:
  * DarkChannelDehazer  - classical He et al. (2009) dark channel prior. No weights needed.
  * OnnxDehazer         - any learned dehazer (e.g. FFA-Net, AOD-Net, DehazeFormer)
                          exported to ONNX with input/output NCHW float32 in [0, 1].
"""
import numpy as np
import cv2


class DarkChannelDehazer:
    def __init__(self, patch: int = 15, omega: float = 0.95, t0: float = 0.1,
                 guided_radius: int = 40, guided_eps: float = 1e-3):
        self.patch = patch
        self.omega = omega
        self.t0 = t0
        self.guided_radius = guided_radius
        self.guided_eps = guided_eps

    def _dark_channel(self, img: np.ndarray) -> np.ndarray:
        min_rgb = img.min(axis=2)
        kernel = np.ones((self.patch, self.patch), np.uint8)
        return cv2.erode(min_rgb, kernel)

    def _airlight(self, img: np.ndarray, dark: np.ndarray) -> np.ndarray:
        n = max(int(dark.size * 0.001), 1)
        idx = np.argsort(dark.ravel())[-n:]
        candidates = img.reshape(-1, 3)[idx]
        return candidates[np.argmax(candidates.sum(axis=1))]

    def _refine(self, t: np.ndarray, gray: np.ndarray) -> np.ndarray:
        gray32 = gray.astype(np.float32)
        t32 = t.astype(np.float32)
        ximgproc = getattr(cv2, "ximgproc", None)
        if ximgproc is not None and hasattr(ximgproc, "guidedFilter"):
            return ximgproc.guidedFilter(gray32, t32, self.guided_radius, self.guided_eps)
        return cv2.bilateralFilter(t32, d=self.guided_radius // 2, sigmaColor=0.1, sigmaSpace=15)

    def dehaze(self, img_bgr: np.ndarray) -> np.ndarray:
        img = img_bgr.astype(np.float32) / 255.0
        dark = self._dark_channel(img)
        A = self._airlight(img, dark)
        t_coarse = 1.0 - self.omega * self._dark_channel(img / A)
        t = self._refine(t_coarse, cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        t = np.maximum(t, self.t0)[..., None]
        J = (img - A) / t + A
        return np.clip(J * 255.0, 0, 255).astype(np.uint8)


class OnnxDehazer:
    def __init__(self, model_path: str, input_size: int = 512):
        import onnxruntime as ort
        self.session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.input_size = input_size

    def dehaze(self, img_bgr: np.ndarray) -> np.ndarray:
        h, w = img_bgr.shape[:2]
        s = self.input_size
        resized = cv2.resize(img_bgr, (s, s), interpolation=cv2.INTER_LINEAR)
        x = resized[..., ::-1].astype(np.float32) / 255.0  # BGR -> RGB
        x = np.transpose(x, (2, 0, 1))[None]
        y = self.session.run(None, {self.input_name: x})[0][0]
        y = np.clip(np.transpose(y, (1, 2, 0)), 0.0, 1.0)
        out = (y[..., ::-1] * 255.0).astype(np.uint8)  # RGB -> BGR
        return cv2.resize(out, (w, h), interpolation=cv2.INTER_LINEAR)


def build_dehazer(name: str, **kwargs):
    if name == "dcp":
        return DarkChannelDehazer(**kwargs)
    if name == "onnx":
        return OnnxDehazer(**kwargs)
    raise ValueError(f"Unknown dehazer: {name}")
