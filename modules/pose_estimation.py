"""Module 2: Pose Estimation - MediaPipe Hand Landmarker (task-based API)."""

import os
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "hand_landmarker.task")


class PoseEstimator:
    """Extract 21 3D hand landmarks using MediaPipe HandLandmarker task API."""

    def __init__(self, max_hands=1, min_detection_conf=0.5, min_tracking_conf=0.5):
        options = vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=vision.RunningMode.IMAGE,
            num_hands=max_hands,
            min_hand_detection_confidence=min_detection_conf,
            min_tracking_confidence=min_tracking_conf,
        )
        self.landmarker = vision.HandLandmarker.create_from_options(options)

    def extract(self, frame_rgb):
        """Return list of (21,3) arrays for each detected hand, or empty list."""
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        result = self.landmarker.detect(mp_image)
        if not result.hand_landmarks:
            return []
        hands = []
        for hand in result.hand_landmarks:
            landmarks = np.array(
                [[lm.x, lm.y, lm.z] for lm in hand], dtype=np.float32
            )
            hands.append(landmarks)
        return hands

    def release(self):
        self.landmarker.close()
