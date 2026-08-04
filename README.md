# Real-Time Object Detection Dashboard

A simple app that runs YOLOv8 object detection on an uploaded video and shows the live results in a web dashboard.

## What it does

- Upload a video file (or use the included sample video by default)
- Backend detects objects frame-by-frame using YOLOv8 (PyTorch + OpenCV)
- Annotated video streams live to the dashboard
- Detected objects, confidence scores, FPS, and inference time are shown alongside it

## Tech Stack

- **Detection:** YOLOv8n (Ultralytics) + PyTorch
- **Video processing:** OpenCV
- **Backend:** FastAPI
- **Frontend:** Streamlit
- **Containerization:** Docker + Docker Compose

## Getting Started

Requires Docker installed.

```bash
git clone https://github.com/Shahradq/detection-dashboard.git
cd detection-dashboard
docker compose up --build
```

Open `http://localhost:8501`. A sample video is included by default, so detection starts automatically — no upload needed to try it out. Upload your own video via the sidebar and click **Use this video** to switch sources.

First run takes a few minutes (downloading dependencies and model weights). Later runs are much faster.

## API Endpoints

Backend routes served by FastAPI on **port 8000** (not the dashboard's 8501) — called automatically by the dashboard, but also testable directly, e.g. `curl http://localhost:8000/health`.

| Endpoint | Method | Purpose |
|---|---|---|
| `/health` | GET | Check the server is running |
| `/stream` | GET | Live annotated video feed |
| `/detections` | GET | Current detections, FPS, inference time |
| `/source` | POST | Upload a new video |

## Known Limitations

- Uses the pretrained YOLOv8n model (not fine-tuned on custom data)
- No webcam or RTSP support — file upload only
- No detection history is saved between sessions
- No authentication — meant for local/demo use
