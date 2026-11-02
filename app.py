from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image
from ultralytics import YOLO

MODEL_PATH = Path(__file__).parent / "best.pt"

# Value of each class in Bahraini dinar (1 BD = 1000 fils)
CLASS_VALUES = {
    "5 fils": 0.005,
    "10 fils": 0.010,
    "25 fils": 0.025,
    "50 fils": 0.050,
    "100 fils": 0.100,
    "0.5 BD": 0.5,
    "1 BD": 1.0,
    "5 BD": 5.0,
    "10 BD": 10.0,
    "20 BD": 20.0,
}

st.set_page_config(page_title="Bahraini Currency Detector", page_icon="💵", layout="wide")


@st.cache_resource
def load_model():
    return YOLO(str(MODEL_PATH))


def detect(model, bgr_image, conf, iou):
    """Run the model on a BGR image. Returns annotated RGB image and list of (name, confidence)."""
    result = model.predict(bgr_image, conf=conf, iou=iou, verbose=False)[0]
    annotated = cv2.cvtColor(result.plot(), cv2.COLOR_BGR2RGB)
    detections = [
        (result.names[int(c)], float(p))
        for c, p in zip(result.boxes.cls.tolist(), result.boxes.conf.tolist())
    ]
    return annotated, detections


def show_results(annotated, detections):
    left, right = st.columns([3, 2])
    left.image(annotated, caption="Detections", width="stretch")

    with right:
        if not detections:
            st.info("No currency detected. Try a clearer photo or lower the confidence threshold.")
            return

        total = sum(CLASS_VALUES.get(name, 0.0) for name, _ in detections)
        st.metric("Estimated total", f"{total:.3f} BD")

        counts = {}
        for name, p in detections:
            counts.setdefault(name, []).append(p)
        rows = [
            {
                "Denomination": name,
                "Count": len(ps),
                "Avg confidence": f"{np.mean(ps):.0%}",
                "Subtotal (BD)": f"{CLASS_VALUES.get(name, 0.0) * len(ps):.3f}",
            }
            for name, ps in counts.items()
        ]
        st.dataframe(rows, hide_index=True, width="stretch")


def run_on_pil(model, pil_image, conf, iou):
    rgb = np.array(pil_image.convert("RGB"))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    with st.spinner("Detecting..."):
        annotated, detections = detect(model, bgr, conf, iou)
    show_results(annotated, detections)


# ---------------------------------------------------------------- UI
st.title("💵 Bahraini Currency Detector")
st.caption("Detects Bahraini banknotes and coins with a YOLO26 model and adds up the total value.")

if not MODEL_PATH.exists():
    st.error(f"Model file not found: {MODEL_PATH.name}. Put best.pt next to app.py.")
    st.stop()

model = load_model()

with st.sidebar:
    st.header("Settings")
    conf = st.slider("Confidence threshold", 0.05, 0.95, 0.25, 0.05)
    iou = st.slider("NMS IoU threshold", 0.1, 0.9, 0.7, 0.05)
    st.divider()
    st.markdown("**Classes**")
    st.write(", ".join(model.names.values()))

tab_upload, tab_camera, tab_live = st.tabs(["Upload image", "Take a photo", "Live camera"])

with tab_upload:
    files = st.file_uploader(
        "Choose one or more images", type=["jpg", "jpeg", "png", "webp"], accept_multiple_files=True
    )
    for f in files or []:
        st.subheader(f.name)
        run_on_pil(model, Image.open(f), conf, iou)

with tab_camera:
    shot = st.camera_input("Point the camera at the money and take a photo")
    if shot is not None:
        run_on_pil(model, Image.open(shot), conf, iou)

with tab_live:
    try:
        import av
        from streamlit_webrtc import VideoProcessorBase, webrtc_streamer
    except ImportError:
        st.warning("Live mode needs `streamlit-webrtc`. Run: pip install streamlit-webrtc")
    else:

        class CurrencyProcessor(VideoProcessorBase):
            def __init__(self):
                self.model = model
                self.conf = 0.25
                self.iou = 0.7

            def recv(self, frame):
                img = frame.to_ndarray(format="bgr24")
                result = self.model.predict(img, conf=self.conf, iou=self.iou, verbose=False)[0]
                return av.VideoFrame.from_ndarray(result.plot(), format="bgr24")

        ctx = webrtc_streamer(
            key="currency-live",
            video_processor_factory=CurrencyProcessor,
            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )
        if ctx.video_processor:
            ctx.video_processor.conf = conf
            ctx.video_processor.iou = iou
        st.caption("Click START and allow camera access. Confidence and IoU follow the sidebar sliders.")