"""Main Pipeline for Raspberry Pi 5 - LSF Translator.
Supporte : PiCamera2, caméra USB, ou flux IP (iPhone via DroidCam/Camo/EpocCam).
"""

import os
import sys
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(__file__))

from config import SEQUENCE_LENGTH
from modules.feature_engineering import FeatureExtractor
from modules.accessibility import AccessibilityInterface

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")

STATIC_LABELS_PATH = os.path.join(MODEL_DIR, "labels.npy")
STATIC_LABELS = list(np.load(STATIC_LABELS_PATH, allow_pickle=True)) if os.path.exists(STATIC_LABELS_PATH) \
    else list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + ["del", "nothing", "space"]

DYNAMIC_LABELS_PATH = os.path.join(MODEL_DIR, "dynamic_labels.npy")
DYNAMIC_LABELS = [str(l) for l in np.load(DYNAMIC_LABELS_PATH, allow_pickle=True)] if os.path.exists(DYNAMIC_LABELS_PATH) else []


# ── TFLite Classifier ─────────────────────────────────────────────────────────

class TFLiteClassifierPi:
    def __init__(self, model_path):
        self.interpreter = self._load(model_path)
        self.interpreter.allocate_tensors()
        self.inp = self.interpreter.get_input_details()
        self.out = self.interpreter.get_output_details()

    def _load(self, path):
        try:
            from tflite_runtime.interpreter import Interpreter
            return Interpreter(model_path=path, num_threads=4)
        except ImportError:
            pass
        try:
            from ai_edge_litert import interpreter as litert
            return litert.Interpreter(model_path=path)
        except (ImportError, AttributeError):
            pass
        import tensorflow as tf
        return tf.lite.Interpreter(model_path=path, num_threads=4)

    def predict(self, features):
        data = features[np.newaxis].astype(self.inp[0]["dtype"])
        self.interpreter.set_tensor(self.inp[0]["index"], data)
        self.interpreter.invoke()
        output = self.interpreter.get_tensor(self.out[0]["index"])[0].astype(np.float32)
        idx = int(np.argmax(output))
        return idx, float(output[idx])


# ── Camera: PiCamera2 ou flux IP (iPhone) ────────────────────────────────────

def make_camera(source="picamera"):
    """
    source = "picamera"        → PiCamera2 (caméra officielle Pi)
    source = "http://IP:PORT/" → flux MJPEG iPhone (Camo, EpocCam, DroidCam)
    source = 0, 1, ...         → webcam USB
    """
    if source == "picamera":
        from modules.acquisition_pi import CameraCapture
        cam = CameraCapture()
        cam.start()
        return cam
    else:
        # Flux IP ou USB — wrapper OpenCV standard
        return _CVCapture(source)


class _CVCapture:
    def __init__(self, source):
        self._cap = cv2.VideoCapture(source)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open camera: {source}")

    def read(self, timeout=1.0):
        ret, frame = self._cap.read()
        if not ret:
            return None, False
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), True

    def stop(self):
        self._cap.release()


# ── Main Translator ───────────────────────────────────────────────────────────

class LSFTranslatorPi:

    def __init__(self, camera_source="picamera"):
        # Pose estimation — utilise mediapipe si dispo, sinon TFLite custom
        try:
            from modules.pose_estimation import PoseEstimator
        except Exception:
            from modules.pose_estimation_pi import PoseEstimator
        self.pose = PoseEstimator(max_hands=2, min_detection_conf=0.5)

        self.features = FeatureExtractor()
        self.interface = AccessibilityInterface()
        self.camera = make_camera(camera_source)

        # Static model (lettres)
        static_path = os.path.join(MODEL_DIR, "static_mobilenetv3_fp16.tflite")
        if not os.path.exists(static_path):
            static_path = os.path.join(MODEL_DIR, "static_mobilenetv3_fp32.tflite")
        self.static_clf = TFLiteClassifierPi(static_path)
        print(f"[Pi] Static model: {static_path}")

        # Dynamic model (mots)
        self.dynamic_clf = None
        dynamic_path = os.path.join(MODEL_DIR, "dynamic_lstm_fp32.tflite")
        if os.path.exists(dynamic_path):
            try:
                self.dynamic_clf = TFLiteClassifierPi(dynamic_path)
                print(f"[Pi] Dynamic model: {dynamic_path}")
            except Exception as e:
                print(f"[Pi] Dynamic model failed: {e}")

        self.executor = ThreadPoolExecutor(max_workers=4)
        self._running = False
        self._sentence = ""
        self._current_pred = ""
        self._current_count = 0
        self._stable_threshold = 15
        self._debug_static = ("", 0.0)
        self._debug_dynamic = ("", 0.0)

    def _classify_static(self, landmarks):
        features = self.features.get_static_features(landmarks)
        idx, conf = self.static_clf.predict(features)
        if idx < len(STATIC_LABELS):
            return STATIC_LABELS[idx], conf
        return None, 0.0

    def _classify_dynamic(self):
        if not self.dynamic_clf or not self.features.is_ready():
            return None, 0.0
        seq = self.features.get_dynamic_features()
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
        self._running = True
        self.interface.start()
        print(f"[Pi] Stream: http://<PI_IP>:5000/stream")
        print("[Pi] Running... Ctrl+C to stop.")

        try:
            while self._running:
                frame_rgb, _ = self.camera.read()
                if frame_rgb is None:
                    continue

                hands = self.pose.extract(frame_rgb)
                frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

                if not hands:
                    self._show(frame_bgr, "", 0.0, None)
                    continue

                landmarks = hands[0]
                self.features.add_frame(landmarks)

                static_future = self.executor.submit(self._classify_static, landmarks)
                dynamic_pred, dynamic_conf = self._classify_dynamic()
                static_pred, static_conf = static_future.result()

                self._debug_static = (static_pred or "", static_conf)
                self._debug_dynamic = (dynamic_pred or "", dynamic_conf)

                if dynamic_pred and dynamic_conf >= 0.4:
                    prediction, confidence = dynamic_pred, dynamic_conf
                elif static_pred:
                    prediction, confidence = static_pred, static_conf
                else:
                    prediction, confidence = "", 0.0

                self.interface.output(frame_bgr, prediction, confidence, hands)
                self._show(frame_bgr, prediction, confidence, hands)

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

        # Debug scores
        cv2.putText(display, f"S:{self._debug_static[0]}({self._debug_static[1]:.0%}) D:{self._debug_dynamic[0]}({self._debug_dynamic[1]:.0%})",
                    (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 0), 1)

        # Phrase panel
        panel = np.zeros((80, w, 3), dtype=np.uint8)
        panel[:] = (50, 50, 50)
        text = self._sentence if self._sentence else "Show signs to translate..."
        cv2.putText(panel, text, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        cv2.putText(panel, "[Enter]=Read  [C]=Clear  [Q]=Quit", (10, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

        cv2.imshow("LSF Translator Pi", np.vstack([display, panel]))
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            self._running = False
        elif key == ord("c"):
            self._sentence = ""
        elif key == 13 and self._sentence.strip():
            self.interface.tts.speak(self._sentence.strip())
        elif prediction and confidence > 0.6:
            # Accumulate stable predictions
            if prediction == self._current_pred:
                self._current_count += 1
            else:
                self._current_pred = prediction
                self._current_count = 1
            if self._current_count == self._stable_threshold:
                if prediction == "space":
                    self._sentence += " "
                    self.interface.tts.speak("space")
                elif prediction == "del":
                    self._sentence = self._sentence[:-1]
                elif prediction != "nothing":
                    self._sentence += prediction
                    self.interface.tts.speak(prediction)
        else:
            self._current_count = 0
            self._current_pred = ""

    def stop(self):
        self._running = False
        self.camera.stop()
        self.pose.release()
        self.interface.stop()
        self.executor.shutdown(wait=False)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", default="picamera",
                        help="picamera | http://IP:PORT/ | 0 (USB)")
    args = parser.parse_args()
    LSFTranslatorPi(camera_source=args.camera).run()
