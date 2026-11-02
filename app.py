from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image
from ultralytics import YOLO

MODEL_PATH = Path(__file__).parent / "best.pt"
CONFIDENCE = 0.25
IOU = 0.7

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

st.set_page_config(page_title="Bahraini currency detector", page_icon=":material/payments:")


@st.cache_resource
def load_model():
    return YOLO(str(MODEL_PATH))


def detect(model, bgr_image):
    """Run the model on a BGR image. Returns annotated RGB image and list of denomination names."""
    result = model.predict(bgr_image, conf=CONFIDENCE, iou=IOU, verbose=False)[0]
    annotated = cv2.cvtColor(result.plot(), cv2.COLOR_BGR2RGB)
    detections = [result.names[int(c)] for c in result.boxes.cls.tolist()]
    return annotated, detections


def show_results(annotated, detections):
    if not detections:
        st.image(annotated, width="stretch")
        st.info("No currency detected. Try a clearer photo.")
        return

    total = sum(CLASS_VALUES.get(name, 0.0) for name in detections)
    st.metric("Estimated total", f"{total:.3f} BD")
    st.image(annotated, width="stretch")

    rows = [
        {
            "Denomination": name,
            "Count": count,
            "Subtotal (BD)": f"{CLASS_VALUES.get(name, 0.0) * count:.3f}",
        }
        for name, count in Counter(detections).items()
    ]
    st.dataframe(rows, hide_index=True, width="stretch")


def run_on_pil(model, pil_image):
    rgb = np.array(pil_image.convert("RGB"))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    with st.spinner("Detecting..."):
        annotated, detections = detect(model, bgr)
    show_results(annotated, detections)


# ---------------------------------------------------------------- UI
st.title("Bahraini currency detector")
st.caption("Finds Bahraini banknotes and coins and adds up their value.")

if not MODEL_PATH.exists():
    st.error(f"Model file not found: {MODEL_PATH.name}. Put best.pt next to app.py.")
    st.stop()

model = load_model()

tab_upload, tab_camera, tab_live = st.tabs(["Upload", "Photo", "Live"])

with tab_upload:
    files = st.file_uploader(
        "Choose one or more images", type=["jpg", "jpeg", "png", "webp"], accept_multiple_files=True
    )
    for f in files or []:
        st.subheader(f.name)
        run_on_pil(model, Image.open(f))

with tab_camera:
    shot = st.camera_input("Point the camera at the money and take a photo")
    if shot is not None:
        run_on_pil(model, Image.open(shot))

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

            def recv(self, frame):
                img = frame.to_ndarray(format="bgr24")
                result = self.model.predict(img, conf=CONFIDENCE, iou=IOU, verbose=False)[0]
                return av.VideoFrame.from_ndarray(result.plot(), format="bgr24")

        webrtc_streamer(
            key="currency-live",
            video_processor_factory=CurrencyProcessor,
            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )
        st.caption("Click START and allow camera access.")