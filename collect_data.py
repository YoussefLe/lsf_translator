"""Data Collection - Record hand gesture samples from webcam for training."""

import os
import sys
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from config import FRAME_WIDTH, FRAME_HEIGHT, SEQUENCE_LENGTH
from modules.pose_estimation import PoseEstimator
from modules.feature_engineering import compute_static_features

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def collect_static(label, num_samples=100):
    """Collect static gesture samples. Press 's' to save a sample, 'q' to quit."""
    save_dir = os.path.join(DATA_DIR, "static", label)
    os.makedirs(save_dir, exist_ok=True)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    pose = PoseEstimator()
    samples = []

    print(f"Collecting static samples for '{label}'. Press 's' to capture, 'q' to quit.")

    while len(samples) < num_samples:
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        hands = pose.extract(rgb)

        status = f"Samples: {len(samples)}/{num_samples}"
        cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        if hands:
            cv2.putText(frame, "Hand detected", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Collect Static", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("s") and hands:
            features = compute_static_features(hands[0])
            samples.append(features)
            print(f"  Captured {len(samples)}/{num_samples}")
        elif key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    pose.release()

    if samples:
        path = os.path.join(save_dir, f"samples_{len(os.listdir(save_dir))}.npy")
        np.save(path, np.array(samples))
        print(f"Saved {len(samples)} samples to {path}")


def collect_dynamic(label, num_sequences=50):
    """Collect dynamic gesture sequences. Press 'r' to start recording, 'q' to quit."""
    save_dir = os.path.join(DATA_DIR, "dynamic", label)
    os.makedirs(save_dir, exist_ok=True)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    pose = PoseEstimator()
    sequences = []
    recording = False
    current_seq = []

    print(f"Collecting dynamic sequences for '{label}'. Press 'r' to record, 'q' to quit.")

    while len(sequences) < num_sequences:
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        hands = pose.extract(rgb)

        status = f"Sequences: {len(sequences)}/{num_sequences}"
        if recording:
            status += f" | Recording: {len(current_seq)}/{SEQUENCE_LENGTH}"
        cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255) if recording else (0, 255, 0), 2)
        cv2.imshow("Collect Dynamic", frame)

        if recording and hands:
            features = compute_static_features(hands[0])
            current_seq.append(features)
            if len(current_seq) >= SEQUENCE_LENGTH:
                sequences.append(np.array(current_seq))
                current_seq = []
                recording = False
                print(f"  Captured {len(sequences)}/{num_sequences}")

        key = cv2.waitKey(1) & 0xFF
        if key == ord("r") and not recording:
            recording = True
            current_seq = []
        elif key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    pose.release()

    if sequences:
        path = os.path.join(save_dir, f"sequences_{len(os.listdir(save_dir))}.npy")
        np.save(path, np.array(sequences))
        print(f"Saved {len(sequences)} sequences to {path}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python collect_data.py <static|dynamic> <label>")
        print("  Example: python collect_data.py static A")
        print("  Example: python collect_data.py dynamic bonjour")
        sys.exit(1)

    mode = sys.argv[1]
    label = sys.argv[2]

    if mode == "static":
        collect_static(label)
    elif mode == "dynamic":
        collect_dynamic(label)
    else:
        print("Mode must be 'static' or 'dynamic'")
