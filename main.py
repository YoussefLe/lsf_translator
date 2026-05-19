"""Main Pipeline - Real-time LSF Translator with multithreading and GPU acceleration."""

import os
import sys
import time
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(__file__))

from config import SEQUENCE_LENGTH
from modules.acquisition import CameraCapture
from modules.pose_estimation import PoseEstimator
from modules.feature_engineering import FeatureExtractor
from modules.model_training import TFLiteClassifier, setup_gpu, MODEL_DIR
from modules.accessibility import AccessibilityInterface

# Label mappings - load from trained model if available
LABELS_PATH = os.path.join(MODEL_DIR, "labels.npy")
if os.path.exists(LABELS_PATH):
    STATIC_LABELS = list(np.load(LABELS_PATH, allow_pickle=True))
else:
    STATIC_LABELS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + ["del", "nothing", "space"]
DYNAMIC_LABELS = ["bonjour", "merci", "oui", "non", "aide", "eau", "manger", "dormir"]


class LSFTranslator:
    """Full real-time LSF translation pipeline with thread-balanced inference."""

    def __init__(self, static_model_path=None, dynamic_model_path=None):
        self.has_gpu = setup_gpu()
        print(f"[Pipeline] GPU available: {self.has_gpu}")

        # Initialize modules
        self.camera = CameraCapture()
        self.pose = PoseEstimator()
        self.features = FeatureExtractor()
        self.interface = AccessibilityInterface()

        # Load TFLite models if available
        self.static_clf = None
        self.dynamic_clf = None
        static_path = static_model_path or os.path.join(MODEL_DIR, "static_mobilenetv3_fp32.tflite")
        dynamic_path = dynamic_model_path or os.path.join(MODEL_DIR, "dynamic_lstm_int8.tflite")
        if os.path.exists(static_path):
            self.static_clf = TFLiteClassifier(static_path)
            print(f"[Pipeline] Static model loaded: {static_path}")
        if os.path.exists(dynamic_path):
            self.dynamic_clf = TFLiteClassifier(dynamic_path)
            print(f"[Pipeline] Dynamic model loaded: {dynamic_path}")

        # Thread pool for parallel inference
        self.executor = ThreadPoolExecutor(max_workers=os.cpu_count() or 4)
        self._prev_gray = None
        self._running = False

        # Sentence accumulation
        self._sentence = ""
        self._last_char = ""
        self._char_count = 0
        self._stable_threshold = 15  # frames before accepting a character
        self._current_pred = ""
        self._current_count = 0

    def _compute_optical_flow(self, frame):
        """Compute optical flow magnitude for adaptive window sizing."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self._prev_gray is None:
            self._prev_gray = gray
            return 0.0
        flow = cv2.calcOpticalFlowFarneback(self._prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag = np.mean(np.sqrt(flow[..., 0]**2 + flow[..., 1]**2))
        self._prev_gray = gray
        return mag

    def _classify_static(self, landmarks):
        """Classify static sign from single frame."""
        if not self.static_clf:
            return None, 0.0
        features = self.features.get_static_features(landmarks)
        idx, conf = self.static_clf.predict(features)
        if idx < len(STATIC_LABELS):
            return STATIC_LABELS[idx], conf
        return None, 0.0

    def _classify_dynamic(self):
        """Classify dynamic sign from frame sequence."""
        if not self.dynamic_clf or not self.features.is_ready():
            return None, 0.0
        seq = self.features.get_dynamic_features()
        # Pad/trim to SEQUENCE_LENGTH for model input
        if seq.shape[0] != SEQUENCE_LENGTH:
            padded = np.zeros((SEQUENCE_LENGTH, seq.shape[1]), dtype=np.float32)
            n = min(seq.shape[0], SEQUENCE_LENGTH)
            padded[:n] = seq[:n]
            seq = padded
        idx, conf = self.dynamic_clf.predict(seq)
        if idx < len(DYNAMIC_LABELS):
            return DYNAMIC_LABELS[idx], conf
        return None, 0.0

    def run(self):
        """Main loop: capture → pose → features → classify → output."""
        self._running = True
        self.camera.start()
        self.interface.start()
        print("[Pipeline] Running... Press Ctrl+C to stop.")

        try:
            while self._running:
                frame, motion = self.camera.read()
                if frame is None:
                    continue

                # Pose estimation (always run, don't gate on motion)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                hands = self.pose.extract(frame_rgb)

                if not hands:
                    self.interface.output(frame, "", 0.0)
                    self._show_local(frame, "", 0.0, None)
                    continue

                landmarks = hands[0]  # Primary hand

                # Optical flow for adaptive window
                flow_mag = self._compute_optical_flow(frame)
                self.features.add_frame(landmarks, flow_mag)

                # Parallel classification: static + dynamic
                static_future = self.executor.submit(self._classify_static, landmarks)
                dynamic_future = self.executor.submit(self._classify_dynamic)

                static_pred, static_conf = static_future.result()
                dynamic_pred, dynamic_conf = dynamic_future.result()

                # Choose best prediction
                if dynamic_pred and dynamic_conf > static_conf:
                    prediction, confidence = dynamic_pred, dynamic_conf
                elif static_pred:
                    prediction, confidence = static_pred, static_conf
                else:
                    prediction, confidence = "", 0.0

                self.interface.output(frame, prediction, confidence, hands)

                # Local OpenCV window (fast, no encoding overhead)
                self._show_local(frame, prediction, confidence, hands)

        except KeyboardInterrupt:
            print("\n[Pipeline] Stopping...")
        finally:
            self.stop()

    def _show_local(self, frame, prediction, confidence, hands):
        """Display frame with landmark overlay and sentence panel."""
        display = frame.copy()
        h, w = display.shape[:2]

        # Draw landmarks
        if hands:
            for landmarks in hands:
                pts = (landmarks[:, :2] * [w, h]).astype(int)
                for pt in pts:
                    cv2.circle(display, tuple(pt), 3, (0, 255, 0), -1)

        # Accumulate stable predictions into sentence
        if prediction and confidence > 0.6:
            if prediction == self._current_pred:
                self._current_count += 1
            else:
                self._current_pred = prediction
                self._current_count = 1

            if self._current_count == self._stable_threshold:
                if prediction == "space":
                    self._sentence += " "
                elif prediction == "del":
                    self._sentence = self._sentence[:-1]
                elif prediction == "nothing":
                    pass
                else:
                    self._sentence += prediction
        else:
            self._current_count = 0
            self._current_pred = ""

        # Show current detected sign
        if prediction and confidence > 0.6:
            cv2.putText(display, f"{prediction} ({confidence:.0%})", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

        # Draw sentence panel at bottom
        panel_h = 60
        panel = np.zeros((panel_h, w, 3), dtype=np.uint8)
        panel[:] = (50, 50, 50)
        # Show accumulated sentence
        display_text = self._sentence if self._sentence else "Show signs to translate..."
        cv2.putText(panel, display_text, (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        # Stack frame + panel
        combined = np.vstack([display, panel])
        cv2.imshow("LSF Translator", combined)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            self._running = False
        elif key == ord("c"):  # clear sentence
            self._sentence = ""

    def stop(self):
        self._running = False
        self.camera.stop()
        self.pose.release()
        self.interface.stop()
        self.executor.shutdown(wait=False)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    translator = LSFTranslator()
    translator.run()
