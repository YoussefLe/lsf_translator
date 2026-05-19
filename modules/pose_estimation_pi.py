"""Module 2 (Pi): Pose Estimation without mediapipe package.

Uses OpenCV DNN for palm detection, then estimates 21 landmarks
from the hand region using a TFLite model extracted from MediaPipe.

Alternative: Install Python 3.11 and use mediapipe there.
"""

import os
import numpy as np
import cv2
import urllib.request

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")

# MediaPipe hand landmark lite model (standalone TFLite, not .task bundle)
LANDMARK_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
LANDMARK_MODEL_PATH = os.path.join(MODEL_DIR, "hand_landmark_lite.tflite")
PALM_MODEL_PATH = os.path.join(MODEL_DIR, "palm_detection_lite.tflite")

# Direct TFLite model URLs (standalone, not bundled in .task)
PALM_URL = "https://storage.googleapis.com/mediapipe-assets/palm_detection_lite.tflite"
LANDMARK_URL = "https://storage.googleapis.com/mediapipe-assets/hand_landmark_lite.tflite"


def ensure_models():
    """Download palm detection and hand landmark TFLite models."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    for url, path in [(PALM_URL, PALM_MODEL_PATH), (LANDMARK_URL, LANDMARK_MODEL_PATH)]:
        if not os.path.exists(path):
            print(f"[Pose] Downloading {os.path.basename(path)}...")
            urllib.request.urlretrieve(url, path)


def _load_interpreter(model_path):
    """Load TFLite interpreter with best available runtime."""
    try:
        import tflite_runtime.interpreter as tflite
        return tflite.Interpreter(model_path=model_path, num_threads=4)
    except ImportError:
        pass
    try:
        import tensorflow as tf
        return tf.lite.Interpreter(model_path=model_path, num_threads=4)
    except ImportError:
        pass
    raise RuntimeError("No TFLite runtime available. Install tflite-runtime.")


class PoseEstimator:
    """Hand landmark detection using raw TFLite models (no mediapipe package)."""

    def __init__(self, max_hands=1, min_detection_conf=0.5, min_tracking_conf=0.5):
        ensure_models()
        self.min_conf = min_detection_conf

        # Load palm detection model
        self.palm_interp = _load_interpreter(PALM_MODEL_PATH)
        self.palm_interp.allocate_tensors()
        self.palm_input = self.palm_interp.get_input_details()[0]
        self.palm_outputs = self.palm_interp.get_output_details()

        # Load hand landmark model
        self.lm_interp = _load_interpreter(LANDMARK_MODEL_PATH)
        self.lm_interp.allocate_tensors()
        self.lm_input = self.lm_interp.get_input_details()[0]
        self.lm_outputs = self.lm_interp.get_output_details()

    def _detect_palm(self, frame_rgb):
        """Run palm detection, return bounding box [x, y, w, h] normalized or None."""
        h, w = frame_rgb.shape[:2]
        input_size = self.palm_input['shape'][1]  # typically 192
        img = cv2.resize(frame_rgb, (input_size, input_size))
        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=0)

        self.palm_interp.set_tensor(self.palm_input['index'], img)
        self.palm_interp.invoke()

        # Palm detection outputs: boxes and scores
        # Output format varies by model version
        scores = None
        boxes = None
        for out in self.palm_outputs:
            tensor = self.palm_interp.get_tensor(out['index'])
            if len(tensor.shape) == 3 and tensor.shape[-1] == 1:
                scores = tensor[0, :, 0]
            elif len(tensor.shape) == 3 and tensor.shape[-1] >= 4:
                boxes = tensor[0]

        if scores is None or boxes is None:
            # Try alternative output interpretation
            out0 = self.palm_interp.get_tensor(self.palm_outputs[0]['index'])[0]
            out1 = self.palm_interp.get_tensor(self.palm_outputs[1]['index'])[0]
            if out0.shape[-1] > out1.shape[-1]:
                boxes, scores = out0, out1.flatten()
            else:
                boxes, scores = out1, out0.flatten()

        # Find best detection
        best_idx = np.argmax(scores)
        if scores[best_idx] < self.min_conf:
            return None

        # Extract box (center_x, center_y, w, h format typically)
        box = boxes[best_idx]
        cx, cy = box[0], box[1]
        bw, bh = box[2], box[3]

        # Convert to corner format, add padding
        pad = 0.3
        x1 = max(0, cx - bw * (0.5 + pad))
        y1 = max(0, cy - bh * (0.5 + pad))
        x2 = min(1, cx + bw * (0.5 + pad))
        y2 = min(1, cy + bh * (0.5 + pad))

        return [x1, y1, x2 - x1, y2 - y1]

    def _get_landmarks(self, frame_rgb, bbox):
        """Run hand landmark model on cropped hand region."""
        h, w = frame_rgb.shape[:2]
        x, y, bw, bh = bbox
        x1, y1 = int(x * w), int(y * h)
        x2, y2 = int((x + bw) * w), int((y + bh) * h)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        if x2 <= x1 or y2 <= y1:
            return None

        crop = frame_rgb[y1:y2, x1:x2]
        input_size = self.lm_input['shape'][1]  # typically 224
        img = cv2.resize(crop, (input_size, input_size))
        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=0)

        self.lm_interp.set_tensor(self.lm_input['index'], img)
        self.lm_interp.invoke()

        # Landmark output: (1, 63) = 21 landmarks * 3 coords
        lm_tensor = self.lm_interp.get_tensor(self.lm_outputs[0]['index'])[0]

        # Check for presence/confidence score
        if len(self.lm_outputs) > 1:
            presence = self.lm_interp.get_tensor(self.lm_outputs[1]['index'])[0]
            if hasattr(presence, '__len__'):
                conf = float(presence[0]) if len(presence) > 0 else float(presence)
            else:
                conf = float(presence)
            if conf < self.min_conf:
                return None

        # Reshape to (21, 3) and map back to full frame coordinates
        landmarks = lm_tensor.reshape(21, -1)[:, :3].copy()
        # Landmarks are in crop-relative coords [0,1]
        landmarks[:, 0] = landmarks[:, 0] * bw + x
        landmarks[:, 1] = landmarks[:, 1] * bh + y
        # z stays relative

        return landmarks.astype(np.float32)

    def extract(self, frame_rgb):
        """Return list of (21,3) arrays for each detected hand, or empty list."""
        bbox = self._detect_palm(frame_rgb)
        if bbox is None:
            return []

        landmarks = self._get_landmarks(frame_rgb, bbox)
        if landmarks is None:
            return []

        return [landmarks]

    def release(self):
        pass
