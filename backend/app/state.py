import threading

# Holds the latest frame/detections so the API and the worker thread can
# share them safely. source_changed tells the worker to switch videos.
class SharedState:
    def __init__(self):
        self.lock = threading.Lock()
        self.latest_frame = None
        self.latest_detections = []
        self.fps = 0.0
        self.inference_time_ms = 0.0
        self.source_path = "data/file_example002.mp4"  # default video on first run
        self.source_changed = threading.Event()

    def update(self, frame, detections, fps, inference_time_ms):
        with self.lock:
            self.latest_frame = frame
            self.latest_detections = detections
            self.fps = fps
            self.inference_time_ms = inference_time_ms

    def read(self):
        with self.lock:
            return self.latest_frame, self.latest_detections, self.fps, self.inference_time_ms

    def set_source(self, path: str):
        with self.lock:
            self.source_path = path
        self.source_changed.set()

    def get_source(self):
        with self.lock:
            return self.source_path

state = SharedState()