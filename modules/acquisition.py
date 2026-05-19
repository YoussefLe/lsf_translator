"""Module 1: Acquisition - Threaded webcam capture with circular buffer and MOG2 motion detection."""

import threading
import queue
import cv2
import numpy as np
from config import CAMERA_ID, FRAME_WIDTH, FRAME_HEIGHT, FPS, BUFFER_SIZE


class MotionDetector:
    """MOG2-based motion detector to trigger pipeline only on hand movement."""

    def __init__(self, history=100, threshold=25, min_area=500):
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=history, varThreshold=threshold, detectShadows=False
        )
        self.min_area = min_area

    def detect(self, frame):
        mask = self.bg_subtractor.apply(frame)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return any(cv2.contourArea(c) > self.min_area for c in contours)


class CameraCapture:
    """Threaded webcam capture with circular buffer (producer/consumer via queue.Queue)."""

    def __init__(self):
        self.buffer = queue.Queue(maxsize=BUFFER_SIZE)
        self.motion_detector = MotionDetector()
        self._running = False
        self._thread = None
        self._cap = None

    def start(self):
        self._cap = cv2.VideoCapture(CAMERA_ID)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        self._cap.set(cv2.CAP_PROP_FPS, FPS)
        if not self._cap.isOpened():
            raise RuntimeError("Cannot open webcam")
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def _capture_loop(self):
        while self._running:
            ret, frame = self._cap.read()
            if not ret:
                continue
            motion = self.motion_detector.detect(frame)
            if self.buffer.full():
                try:
                    self.buffer.get_nowait()
                except queue.Empty:
                    pass
            self.buffer.put((frame, motion))

    def read(self, timeout=1.0):
        """Get next frame and motion flag from buffer."""
        try:
            return self.buffer.get(timeout=timeout)
        except queue.Empty:
            return None, False

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)
        if self._cap:
            self._cap.release()
