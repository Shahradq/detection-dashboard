from ultralytics import YOLO

# Small wrapper around the YOLO model.
class Detector:
    def __init__(self, weights: str = "app/models/yolov8n.pt"):
        # Auto-downloads the weights if the file isn't there yet.
        self.model = YOLO(weights)

    def infer(self, frame):
        # Only one frame at a time will be passed.
        results = self.model(frame, verbose=False)
        return results[0]