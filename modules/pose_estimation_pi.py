"""Module 2 (Pi): Pose Estimation using mediapipe-rpi4 (legacy solutions API)."""

import numpy as np
import mediapipe as mp


class PoseEstimator:
    """Extract 21 3D hand landmarks using MediaPipe Hands (legacy API for Pi)."""

    def __init__(self, max_hands=1, min_detection_conf=0.6, min_tracking_conf=0.5):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_hands,
            min_detection_confidence=min_detection_conf,
            min_tracking_confidence=min_tracking_conf,
        )

    def extract(self, frame_rgb):
        """Return list of (21,3) arrays for each detected hand, or empty list."""
        results = self.hands.process(frame_rgb)
        if not results.multi_hand_landmarks:
            return []
        hands = []
        for hand_landmarks in results.multi_hand_landmarks:
            landmarks = np.array(
                [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark],
                dtype=np.float32,
            )
            hands.append(landmarks)
        return hands

    def release(self):
        self.hands.close()
