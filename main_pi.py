"""Main Pipeline for Raspberry Pi 4B - LSF Translator."""

import os
import sys
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(__file__))

from config import SEQUENCE_LENGTH
from modules.acquisition_pi import CameraCapture
from modules.pose_estimation_pi import PoseEstimator
from modules.feature_engineering import FeatureExtractor
from modules.accessibility import AccessibilityInterface

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
LABELS_PATH = os.path.join(MODEL_DIR, "labels.npy")

if os.path.exists(LABELS_PATH):
    STATIC_LABELS = list(np.load(LABELS_PATH, allow_pickle=True))
else:
    STATIC_LABELS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + ["del", "nothing", "space"]


class TFLiteClassifierPi:
    """TFLite inference using best available runtime."""

    def __init__(self, model_path):
        self.interpreter = self._make_interpreter(model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def _make_interpreter(self, model_path):
        try:
            from tflite_runtime.interpreter import Interpreter
            return Interpreter(model_path=model_path, num_threads=4)
        except ImportError:
            pass
        try:
            from ai_edge_litert import interpreter as litert
            return litert.Interpreter(model_path=model_path)
        except (ImportError, AttributeError):
            pass
        import tensorflow as tf
        return tf.lite.Interpreter(model_path=model_path, num_threads=4)

    def predict(self, features):
        input_data = features[np.newaxis].astype(self.input_details[0]["dtype"])
        self.interpreter.set_tensor(self.input_details[0]["index"], input_data)
        self.interpreter.invoke()
        output = self.interpreter.get_tensor(self.output_details[0]["index"])[0]
        output = output.astype(np.float32)
        class_idx = int(np.argmax(output))
        confidence = float(output[class_idx])
        return class_idx, confidence


class LSFTranslatorPi:
    """Optimized pipeline for Raspberry Pi 4B."""

    def __init__(self):
        self.camera = CameraCapture()
        self.pose = PoseEstimator(max_hands=1, min_detection_conf=0.6)
        self.features = FeatureExtractor()
        self.interface = AccessibilityInterface()

        static_path = os.path.join(MODEL_DIR, "static_mobilenetv3_fp16.tflite")
        if not os.path.exists(static_path):
            static_path = os.path.join(MODEL_DIR, "static_mobilenetv3_fp32.tflite")
        self.static_clf = TFLiteClassifierPi(static_path)
        print(f"[Pi] Model loaded: {static_path}")

        self.executor = ThreadPoolExecutor(max_workers=4)
        self._running = False
        self._sentence = ""
        self._current_pred = ""
        self._current_count = 0
        self._stable_threshold = 15

    def _classify_static(self, landmarks):
        features = self.features.get_static_features(landmarks)
        idx, conf = self.static_clf.predict(features)
        if idx < len(STATIC_LABELS):
            return STATIC_LABELS[idx], conf
        return None, 0.0

    def run(self):
        self._running = True
        self.camera.start()
        self.interface.start()
        print("[Pi] Running... Press Ctrl+C to stop.")
        print(f"[Pi] Stream at http://<PI_IP>:5000/stream")

        try:
            while self._running:
                frame, _ = self.camera.read()
                if frame is None:
                    continue

                # picamera2 gives RGB directly
                hands = self.pose.extract(frame)

                # Convert to BGR for display
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                if not hands:
                    self._show(frame_bgr, "", 0.0, None)
                    continue

                landmarks = hands[0]
                prediction, confidence = self._classify_static(landmarks)

                if not prediction:
                    prediction, confidence = "", 0.0

                # Accumulate
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
                        elif prediction != "nothing":
                            self._sentence += prediction
                else:
                    self._current_count = 0
                    self._current_pred = ""

                self._show(frame_bgr, prediction, confidence, hands)
                self.interface.output(frame_bgr, prediction, confidence, hands)

        except KeyboardInterrupt:
            print("\n[Pi] Stopping...")
        finally:
            self.stop()

    def _show(self, frame, prediction, confidence, hands):
        h, w = frame.shape[:2]
        display = frame.copy()
        if hands:
            for lm in hands:
                pts = (lm[:, :2] * [w, h]).astype(int)
                for pt in pts:
                    cv2.circle(display, tuple(pt), 3, (0, 255, 0), -1)
        if prediction and confidence > 0.6:
            cv2.putText(display, f"{prediction} ({confidence:.0%})", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

        panel = np.zeros((60, w, 3), dtype=np.uint8)
        panel[:] = (50, 50, 50)
        text = self._sentence if self._sentence else "Show signs..."
        cv2.putText(panel, text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        combined = np.vstack([display, panel])
        cv2.imshow("LSF Translator", combined)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            self._running = False
        elif key == ord("c"):
            self._sentence = ""

    def stop(self):
        self._running = False
        self.camera.stop()
        self.pose.release()
        self.interface.stop()
        self.executor.shutdown(wait=False)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    translator = LSFTranslatorPi()
    translator.run()
