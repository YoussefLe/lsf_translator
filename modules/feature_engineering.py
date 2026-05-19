"""Module 3: Feature Engineering - Joint angles, normalized coordinates, trajectory vectors."""

import numpy as np
from collections import deque
from config import NUM_LANDMARKS, STATIC_FEATURE_DIM, SEQUENCE_LENGTH, MIN_WINDOW, MAX_WINDOW

# Finger joint triplets for angle computation (15 angles)
ANGLE_TRIPLETS = [
    (0, 1, 2), (1, 2, 3), (2, 3, 4),       # Thumb
    (0, 5, 6), (5, 6, 7), (6, 7, 8),       # Index
    (0, 9, 10), (9, 10, 11), (10, 11, 12), # Middle
    (0, 13, 14), (13, 14, 15), (14, 15, 16),  # Ring
    (0, 17, 18), (17, 18, 19), (18, 19, 20),  # Pinky
]


def compute_joint_angles(landmarks):
    """Compute 15 joint angles using dot product of bone vectors (rotation invariant)."""
    angles = np.zeros(15, dtype=np.float32)
    for i, (a, b, c) in enumerate(ANGLE_TRIPLETS):
        v1 = landmarks[a] - landmarks[b]
        v2 = landmarks[c] - landmarks[b]
        cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
        angles[i] = np.arccos(np.clip(cos, -1.0, 1.0))
    return angles


def normalize_landmarks(landmarks):
    """Normalize 21 landmarks to wrist, scaled by wrist-to-middle-finger distance (63 values)."""
    wrist = landmarks[0]
    centered = landmarks - wrist
    scale = np.linalg.norm(centered[9]) + 1e-8  # wrist to middle finger MCP
    normalized = centered / scale
    return normalized.flatten()  # 63 values


def compute_static_features(landmarks):
    """78-dim feature vector: 15 angles + 63 normalized coords."""
    angles = compute_joint_angles(landmarks)
    coords = normalize_landmarks(landmarks)
    return np.concatenate([angles, coords])


class FeatureExtractor:
    """Maintains sliding window for dynamic gesture features with adaptive window size."""

    def __init__(self):
        self.window = deque(maxlen=MAX_WINDOW)
        self.current_window_size = SEQUENCE_LENGTH

    def update_window_size(self, optical_flow_magnitude):
        """Adapt window size based on gesture speed (optical flow magnitude)."""
        if optical_flow_magnitude > 5.0:
            self.current_window_size = MIN_WINDOW  # Fast gesture
        elif optical_flow_magnitude < 1.0:
            self.current_window_size = MAX_WINDOW  # Slow gesture
        else:
            t = (optical_flow_magnitude - 1.0) / 4.0
            self.current_window_size = int(MAX_WINDOW - t * (MAX_WINDOW - MIN_WINDOW))

    def add_frame(self, landmarks, optical_flow_mag=None):
        """Add a frame's features to the sliding window."""
        features = compute_static_features(landmarks)
        self.window.append(features)
        if optical_flow_mag is not None:
            self.update_window_size(optical_flow_mag)

    def get_static_features(self, landmarks):
        """Return 78-dim static feature vector."""
        return compute_static_features(landmarks)

    def get_dynamic_features(self):
        """Return (window_size, 78) sequence for dynamic classification."""
        n = self.current_window_size
        if len(self.window) < n:
            # Pad with zeros if not enough frames
            pad = [np.zeros(STATIC_FEATURE_DIM, dtype=np.float32)] * (n - len(self.window))
            seq = list(pad) + list(self.window)
        else:
            seq = list(self.window)[-n:]
        return np.array(seq, dtype=np.float32)

    def is_ready(self):
        """Check if enough frames for dynamic classification."""
        return len(self.window) >= self.current_window_size

    def reset(self):
        self.window.clear()
