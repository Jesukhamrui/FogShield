"""Web frontend: upload a foggy image -> dehaze -> detect people and vehicles.

Run:  streamlit run app.py
"""
import glob
import os

import cv2
import numpy as np
import streamlit as st

from src.dehaze import DarkChannelDehazer, OnnxDehazer
from src.detector import Detector
from src.pipeline import BaselinePipeline, TwoStagePipeline
from src.viz import count_labels, draw_detections, encode_png, limit_size

st.set_page_config(page_title="Fog-Safe Mine Vehicle Detection", layout="wide")
st.title("Fog-Safe Vehicle and Person Detection")
st.caption("SIH 2026 · SIH26007 · Upload a foggy frame. It is dehazed first, then vehicles and people are detected.")


@st.cache_resource(show_spinner="Loading detector...")
def load_detector(weights: str):
    return Detector(weights)


@st.cache_resource(show_spinner="Loading dehazer...")
def load_dehazer(kind: str, omega: float):
    if kind == "Learned (ONNX)":
        return OnnxDehazer("models/dehazer.onnx")
    return DarkChannelDehazer(omega=omega)


# ---------------- Sidebar settings ----------------
st.sidebar.header("Settings")
available = [w for w in ["models/yolo_fog.pt", "yolov8n.pt"] if os.path.exists(w) or w == "yolov8n.pt"]
available += [w for w in glob.glob("models/*.pt") if w not in available]
weights = st.sidebar.selectbox("Detector weights", available)
conf = st.sidebar.slider("Detection confidence", 0.05, 0.95, 0.25, 0.05)

dehaze_options = ["Dark Channel Prior"]
if os.path.exists("models/dehazer.onnx"):
    dehaze_options.append("Learned (ONNX)")
dehaze_kind = st.sidebar.selectbox("Dehazing method", dehaze_options)
omega = st.sidebar.slider("Dehazing strength", 0.5, 0.99, 0.95, 0.01,
                          disabled=dehaze_kind != "Dark Channel Prior")
show_baseline = st.sidebar.checkbox("Also show detection WITHOUT dehazing", value=True)

# ---------------- Upload ----------------
uploaded = st.file_uploader("Upload a foggy image", type=["jpg", "jpeg", "png", "bmp"])

if uploaded is None:
    st.info("Choose an image to begin. Try a file from data/hazed/ if you have no foggy photo.")
    st.stop()

data = np.frombuffer(uploaded.read(), np.uint8)
img = cv2.imdecode(data, cv2.IMREAD_COLOR)
if img is None:
    st.error("Could not read that file as an image.")
    st.stop()
img = limit_size(img)

try:
    detector = load_detector(weights)
    detector.conf = conf
    dehazer = load_dehazer(dehaze_kind, omega)
except Exception as e:  # missing weights / onnx model etc.
    st.error(f"Could not load model: {e}")
    st.stop()

with st.spinner("Dehazing and detecting..."):
    result = TwoStagePipeline(dehazer, detector).run(img)
    base = BaselinePipeline(detector).run(img) if show_baseline else None

enhanced = result["enhanced"]
dets = result["detections"]
annotated = draw_detections(enhanced, dets)

# ---------------- Results ----------------
cols = st.columns(3 if not show_baseline else 4)
cols[0].subheader("1. Original (foggy)")
cols[0].image(img, channels="BGR", use_container_width=True)
cols[1].subheader("2. Dehazed")
cols[1].image(enhanced, channels="BGR", use_container_width=True)
cols[2].subheader("3. Detections (after dehazing)")
cols[2].image(annotated, channels="BGR", use_container_width=True)
if base is not None:
    cols[3].subheader("Detections (no dehazing)")
    cols[3].image(draw_detections(img, base["detections"]), channels="BGR", use_container_width=True)

st.divider()
c = count_labels(dets)
m1, m2, m3, m4 = st.columns(4)
m1.metric("Vehicles found", c["vehicle"], delta=(c["vehicle"] - count_labels(base["detections"])["vehicle"]) if base else None)
m2.metric("People found", c["person"], delta=(c["person"] - count_labels(base["detections"])["person"]) if base else None)
m3.metric("Dehaze time", f"{result['timings_ms']['dehaze']:.0f} ms")
m4.metric("Total time", f"{result['timings_ms']['total']:.0f} ms")
if base is not None:
    st.caption("Green/red arrows show how many more (or fewer) objects were found compared with no dehazing.")

d1, d2 = st.columns(2)
d1.download_button("Download dehazed image", encode_png(enhanced), "dehazed.png", "image/png")
d2.download_button("Download detection result", encode_png(annotated), "detections.png", "image/png")
