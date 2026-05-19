"""Local-only LSF Translator - OpenCV window, no Flask/MQTT."""

import os
import sys
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(__file__))

from config import SEQUENCE_LENGTH
from modules.pose_estimation import PoseEstimator
from modules.feature_engineering import FeatureExtractor
from modules.model_training import TFLiteClassifier, setup_gpu, MODEL_DIR

LABELS_PATH = os.path.join(MODEL_DIR, "labels.npy")
STATIC_LABELS = list(np.load(LABELS_PATH, allow_pickle=True)) if os.path.exists(LABELS_PATH) else list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + ["del", "nothing", "space"]


class LSFLocal:
    def __init__(self):
        setup_gpu()
        self.pose = PoseEstimator()
        self.features = FeatureExtractor()
        static_path = os.path.join(MODEL_DIR, "static_mobilenetv3_fp32.tflite")
        self.clf = TFLiteClassifier(static_path) if os.path.exists(static_path) else None
        self.executor = ThreadPoolExecutor(max_workers=4)
        self._sentence = ""
        self._current_pred = ""
        self._current_count = 0
        self._stable_threshold = 15

    def run(self):
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        print("[Local] Running... 'q' to quit, 'c' to clear text")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            hands = self.pose.extract(rgb)

            prediction, confidence = "", 0.0
            if hands and self.clf:
                features = self.features.get_static_features(hands[0])
                idx, conf = self.clf.predict(features)
                if idx < len(STATIC_LABELS):
                    prediction, confidence = STATIC_LABELS[idx], conf

            self._draw(frame, prediction, confidence, hands)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("c"):
                self._sentence = ""

        cap.release()
        cv2.destroyAllWindows()
        self.pose.release()

    def _draw(self, frame, prediction, confidence, hands):
        h, w = frame.shape[:2]

        if hands:
            for lm in hands:
                pts = (lm[:, :2] * [w, h]).astype(int)
                for pt in pts:
                    cv2.circle(frame, tuple(pt), 3, (0, 255, 0), -1)

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
            cv2.putText(frame, f"{prediction} ({confidence:.0%})", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        else:
            self._current_count = 0
            self._current_pred = ""

        # Sentence panel
        panel = np.zeros((60, w, 3), dtype=np.uint8)
        panel[:] = (50, 50, 50)
        text = self._sentence if self._sentence else "Show signs to translate..."
        cv2.putText(panel, text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        cv2.imshow("LSF Translator", np.vstack([frame, panel]))


if __name__ == "__main__":
    LSFLocal().run()
