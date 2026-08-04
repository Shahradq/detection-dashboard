import time
import cv2
from app.video_source import read_video
from app.detector import Detector
from app.state import state

def run_worker():
    # Runs forever: reads frames from the current source, detects objects,
    # and saves the result to shared state for the API to serve.
    detector = Detector()
    prev_time = time.time()

    while True:
        current_source = state.get_source()
        state.source_changed.clear()

        try:
            for frame in read_video(current_source):
                # A new video was uploaded — stop this one and switch now.
                if state.source_changed.is_set():
                    break

                frame = cv2.resize(frame, (640, 480))
                start = time.time()
                result = detector.infer(frame)
                inference_time_ms = (time.time() - start) * 1000

                now = time.time()
                fps = 1.0 / (now - prev_time) if now > prev_time else 0.0
                prev_time = now

                detections = [
                    {
                        "class": result.names[int(box.cls[0])],
                        "confidence": float(box.conf[0]),
                        "bbox": box.xyxy[0].tolist(),
                    }
                    for box in result.boxes
                ]

                annotated = result.plot()  # frame with boxes/labels drawn on it

                state.update(annotated, detections, fps, inference_time_ms)
        except Exception as exc:
            print(f"[worker] error reading source '{current_source}': {exc}")
            time.sleep(1.0)