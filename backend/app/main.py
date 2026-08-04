import os
import shutil
import threading
import cv2
import time

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse

from app.state import state
from app.worker import run_worker

app = FastAPI()

UPLOAD_DIR = "data/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.on_event("startup")
def start_worker():
    # Detection runs in the background so requests just read the latest result.
    thread = threading.Thread(target=run_worker, daemon=True)
    thread.start()

@app.post("/source")
async def upload_source(file: UploadFile = File(...)):
    # Strip any path info from the filename so uploads can't escape UPLOAD_DIR.
    safe_filename = os.path.basename(file.filename)
    if not safe_filename:
        return {"status": "error", "detail": "Invalid filename"}
    file_path = os.path.join(UPLOAD_DIR, safe_filename)
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    state.set_source(file_path)
    return {"status": "ok", "source": file_path}

def mjpeg_generator():
    # Sends whatever frame the worker last produced, not a live video read.
    while True:
        frame, _, _, _ = state.read()
        if frame is None:
            time.sleep(0.1)
            continue

        ok, buffer = cv2.imencode(".jpg", frame)
        if not ok:
            continue

        jpeg_bytes = buffer.tobytes()
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + jpeg_bytes + b"\r\n"
        )
        time.sleep(0.03)  # ~30fps cap
@app.get("/stream")
def stream():
    return StreamingResponse(
        mjpeg_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/detections")
def get_detections():
    _, detections, fps, inference_time_ms = state.read()
    return {
        "detections": detections,
        "fps": round(fps, 2),
        "inference_time_ms": round(inference_time_ms, 2),
    }