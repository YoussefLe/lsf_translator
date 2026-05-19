"""Training script - Collect data and train all models with comparative evaluation."""

import os
import sys
import numpy as np
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(__file__))

from config import STATIC_FEATURE_DIM, SEQUENCE_LENGTH
from modules.model_training import (
    setup_gpu, build_static_model, train_model,
    quantize_model, save_tflite, apply_structural_pruning, strip_pruning,
    compare_dynamic_models, compare_quantization, compare_feature_types,
)

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DATA_DIR, exist_ok=True)

STATIC_LABELS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + [str(i) for i in range(10)]
DYNAMIC_LABELS = ["bonjour", "merci", "oui", "non", "aide", "eau", "manger", "dormir"]


def load_data(data_path, mode="static"):
    """Load .npy data files. Expected structure: data/{mode}/{label}/samples.npy"""
    X, y = [], []
    base = os.path.join(data_path, mode)
    if not os.path.exists(base):
        return None, None
    labels = sorted(os.listdir(base))
    for idx, label in enumerate(labels):
        label_dir = os.path.join(base, label)
        for f in os.listdir(label_dir):
            if f.endswith(".npy"):
                samples = np.load(os.path.join(label_dir, f))
                if samples.ndim == 1:
                    X.append(samples)
                    y.append(idx)
                else:
                    for s in samples:
                        X.append(s)
                        y.append(idx)
    return np.array(X, dtype=np.float32), np.array(y)


def train_static():
    """Train static sign classifier (MobileNetV3-Small) with quantization comparison."""
    print("\n" + "="*60)
    print("TRAINING STATIC MODEL (MobileNetV3-Small)")
    print("="*60)

    X, y = load_data(DATA_DIR, "static")
    if X is None:
        print("[!] No static data found. Place .npy files in data/static/{label}/")
        print("    Generating synthetic data for demonstration...")
        num_classes = len(STATIC_LABELS)
        X = np.random.randn(1000, STATIC_FEATURE_DIM).astype(np.float32)
        y = np.random.randint(0, num_classes, 1000)
    else:
        num_classes = len(set(y))

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    model = build_static_model(num_classes)
    model.summary()
    train_model(model, X_train, y_train, X_val, y_val)

    # Quantization comparison
    print("\n--- Quantization Comparison ---")
    compare_quantization(model, X_train, "static_mobilenetv3")

    # Feature type comparison (angles vs coords)
    print("\n--- Feature Type Comparison ---")
    X_angles = X_train[:, :15]
    X_coords = X_train[:, 15:]
    compare_feature_types(X_angles, X_coords, y_train, num_classes)

    return model


def train_dynamic():
    """Train dynamic sign classifiers (LSTM, GRU, Transformer) with comparative evaluation."""
    print("\n" + "="*60)
    print("TRAINING DYNAMIC MODELS (LSTM / GRU / Transformer)")
    print("="*60)

    X, y = load_data(DATA_DIR, "dynamic")
    if X is None:
        print("[!] No dynamic data found. Place .npy files in data/dynamic/{label}/")
        print("    Generating synthetic data for demonstration...")
        num_classes = len(DYNAMIC_LABELS)
        X = np.random.randn(500, SEQUENCE_LENGTH, STATIC_FEATURE_DIM).astype(np.float32)
        y = np.random.randint(0, num_classes, 500)
    else:
        num_classes = len(set(y))

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Comparative evaluation: LSTM vs GRU vs Transformer
    results = compare_dynamic_models(X_train, y_train, X_val, y_val, num_classes)

    # Structural pruning on best LSTM
    print("\n--- Structural Pruning (LSTM) ---")
    try:
        lstm_model = results["LSTM"]["model"]
        pruned = apply_structural_pruning(lstm_model, pruning_ratio=0.3)
        pruned.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)
        pruned = strip_pruning(pruned)
        tflite = quantize_model(pruned, "INT8", X_train)
        path = save_tflite(tflite, "dynamic_lstm_pruned_int8")
        print(f"  Pruned model saved: {path}")
    except ImportError:
        print("  [!] tensorflow-model-optimization not installed. Skipping pruning.")

    return results


if __name__ == "__main__":
    setup_gpu()
    train_static()
    train_dynamic()
    print("\n\n✓ Training complete. Models saved in ./models/")
