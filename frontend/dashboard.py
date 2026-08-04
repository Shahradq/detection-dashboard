import streamlit as st
import requests
import os
from streamlit_autorefresh import st_autorefresh


# BACKEND_URL: used by this script to call the backend.
# BACKEND_PUBLIC_URL: used in the page HTML.
BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000")
BACKEND_PUBLIC_URL = os.environ.get("BACKEND_PUBLIC_URL", "http://localhost:8000")

st.set_page_config(page_title="Object Detection Dashboard", layout="wide")
st.title("Real-Time Object Detection Dashboard")

st.sidebar.subheader("Video Source")
uploaded_file = st.sidebar.file_uploader("Upload a video", type=["mp4", "avi", "mov"])

if uploaded_file is not None and st.sidebar.button("Use this video"):
    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
    response = requests.post(f"{BACKEND_URL}/source", files=files)
    if response.ok:
        st.sidebar.success(f"Switched to {uploaded_file.name}")
    else:
        st.sidebar.error("Failed to switch source")

# Re-runs the script every 3s to fetch fresh detections from the backend.
st_autorefresh(interval=3000, key="refresh")

col_video, col_data = st.columns([2, 1])

with col_video:
    st.subheader("Live Stream")
    # The browser connects straight to the video stream and updates itself.
    st.markdown(
        f'<img src="{BACKEND_PUBLIC_URL}/stream" width="100%">',
        unsafe_allow_html=True,
    )

with col_data:
    st.subheader("Detections")

    try:
        response = requests.get(f"{BACKEND_URL}/detections", timeout=10)
        response.raise_for_status()
        data = response.json()

        st.metric("FPS", data["fps"])
        st.metric("Inference Time (ms)", data["inference_time_ms"])

        if data["detections"]:
            st.table([
                {"Class": d["class"], "Confidence": f"{d['confidence']:.2f}"}
                for d in data["detections"]
            ])
        else:
            st.info("No objects detected in current frame.")

    except requests.exceptions.RequestException:
        st.error("Backend unavailable. Is the FastAPI server running?")