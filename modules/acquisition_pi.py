"""Module 1 (Pi): Acquisition using Picamera2 for Raspberry Pi Camera v2.1."""

import threading
import queue
import numpy as np
from picamera2 import Picamera2
from config import FRAME_WIDTH, FRAME_HEIGHT, FPS, BUFFER_SIZE


class CameraCapture:
    """Threaded Pi camera capture with circular buffer."""

    def __init__(self):
        self.buffer = queue.Queue(maxsize=BUFFER_SIZE)
        self._running = False
        self._thread = None
        self._cam = None

    def start(self):
        self._cam = Picamera2()
        config = self._cam.create_preview_configuration(
            main={"size": (FRAME_WIDTH, FRAME_HEIGHT), "format": "RGB888"},
            controls={"FrameRate": FPS},
        )
        self._cam.configure(config)
        self._cam.start()
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def _capture_loop(self):
        while self._running:
            frame = self._cam.capture_array()
            if self.buffer.full():
                try:
                    self.buffer.get_nowait()
                except queue.Empty:
                    pass
            self.buffer.put((frame, True))

    def read(self, timeout=1.0):
        try:
            return self.buffer.get(timeout=timeout)
        except queue.Empty:
            return None, False

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)
        if self._cam:
            self._cam.stop()
