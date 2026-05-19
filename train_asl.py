"""Train static model using ASL Alphabet Kaggle dataset.

Dataset: https://www.kaggle.com/datasets/grassknoted/asl-alphabet
Processes images through MediaPipe Hands to extract 78-dim feature vectors,
then trains MobileNetV3-Small with INT8/FP16/FP32 quantization comparison.
"""

import os
import sys
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(__file__))

from config import STATIC_FEATURE_DIM
from modules.pose_estimation import PoseEstimator
from modules.feature_engineering import compute_static_features
from modules.model_training import (
    setup_gpu, build_static_model, train_model,
    quantize_model, save_tflite, compare_quantization, compare_feature_types,
)

# Paths
TRAIN_DIR = r"C:\Users\Youssef Lamine\sign-language-translator-raspi5\data\raw\asl_alphabet_train\asl_alphabet_train"
TEST_DIR = r"C:\Users\Youssef Lamine\sign-language-translator-raspi5\data\raw\asl_alphabet_test\asl_alphabet_test"
FEATURES_CACHE = os.path.join(os.path.dirname(__file__), "data", "asl_features.npz")
os.makedirs(os.path.dirname(FEATURES_CACHE), exist_ok=True)


def extract_features_from_image(img_path, pose_estimator):
    """Extract 78-dim features from a single image. Returns None if no hand detected."""
    img = cv2.imread(img_path)
    if img is None:
        return None
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    hands = pose_estimator.extract(rgb)
    if not hands:
        return None
    return compute_static_features(hands[0])


def process_dataset(data_dir, max_per_class=None):
    """Process all images in dataset, extract features. Returns X, y, labels."""
    labels = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    print(f"Found {len(labels)} classes: {labels}")

    pose = PoseEstimator(max_hands=1, min_detection_conf=0.5, min_tracking_conf=0.5)
    X, y = [], []

    for idx, label in enumerate(labels):
        label_dir = os.path.join(data_dir, label)
        files = [f for f in os.listdir(label_dir) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
        if max_per_class:
            files = files[:max_per_class]

        count = 0
        for i, fname in enumerate(files):
            features = extract_features_from_image(os.path.join(label_dir, fname), pose)
            if features is not None:
                X.append(features)
                y.append(idx)
                count += 1

            if (i + 1) % 500 == 0:
                print(f"  [{label}] {i+1}/{len(files)} processed, {count} valid")

        print(f"  [{label}] Done: {count}/{len(files)} valid samples")

    pose.release()
    return np.array(X, dtype=np.float32), np.array(y), labels


def main():
    setup_gpu()

    # Check for cached features
    if os.path.exists(FEATURES_CACHE):
        print(f"Loading cached features from {FEATURES_CACHE}")
        data = np.load(FEATURES_CACHE, allow_pickle=True)
        X, y, labels = data["X"], data["y"], data["labels"]
    else:
        print("Extracting features from dataset (this may take a while)...")
        X, y, labels = process_dataset(TRAIN_DIR)
        np.savez(FEATURES_CACHE, X=X, y=y, labels=labels)
        print(f"Cached features to {FEATURES_CACHE}")

    print(f"\nDataset: {X.shape[0]} samples, {len(labels)} classes, {STATIC_FEATURE_DIM}-dim features")
    num_classes = len(labels)

    # Filter out classes with too few samples
    min_samples = 50
    unique, counts = np.unique(y, return_counts=True)
    valid_classes = unique[counts >= min_samples]
    mask = np.isin(y, valid_classes)
    X, y = X[mask], y[mask]
    # Remap labels
    labels = [labels[i] for i in valid_classes]
    label_map = {old: new for new, old in enumerate(valid_classes)}
    y = np.array([label_map[v] for v in y])
    num_classes = len(labels)
    print(f"After filtering (min {min_samples} samples): {X.shape[0]} samples, {num_classes} classes")

    # Split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"Train: {X_train.shape[0]}, Val: {X_val.shape[0]}")

    # Train MobileNetV3-Small
    print("\n" + "="*60)
    print("TRAINING MobileNetV3-Small STATIC MODEL")
    print("="*60)
    model = build_static_model(num_classes)
    model.summary()
    train_model(model, X_train, y_train, X_val, y_val)

    loss, acc = model.evaluate(X_val, y_val, verbose=0)
    print(f"\nFinal Validation Accuracy: {acc:.4f}, Loss: {loss:.4f}")

    # Quantization comparison: INT8 vs FP16 vs FP32
    print("\n" + "="*60)
    print("QUANTIZATION COMPARISON")
    print("="*60)
    compare_quantization(model, X_train, "static_mobilenetv3")

    # Feature type comparison: angles only vs coords only
    print("\n" + "="*60)
    print("FEATURE TYPE COMPARISON (angles vs coords)")
    print("="*60)
    X_angles = X_train[:, :15]
    X_coords = X_train[:, 15:]
    compare_feature_types(X_angles, X_coords, y_train, num_classes)

    # Save label mapping
    label_path = os.path.join(os.path.dirname(__file__), "models", "labels.npy")
    np.save(label_path, labels)
    print(f"\nLabels saved to {label_path}")
    print("\n✓ Training complete! Models saved in ./models/")


if __name__ == "__main__":
    main()
