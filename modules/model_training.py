"""Module 4: Model Training - MobileNetV3-Small (static), LSTM/GRU/Transformer (dynamic).

Includes INT8/FP16/FP32 quantization, structural pruning, and comparative evaluation.
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from config import STATIC_FEATURE_DIM, SEQUENCE_LENGTH, LSTM_UNITS, LSTM_LAYERS, BATCH_SIZE, EPOCHS

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
os.makedirs(MODEL_DIR, exist_ok=True)


# ─── GPU Setup ───────────────────────────────────────────────────────────────

def setup_gpu():
    """Enable GPU memory growth if CUDA available."""
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    return len(gpus) > 0


# ─── Static Model: MobileNetV3-Small ─────────────────────────────────────────

def build_static_model(num_classes):
    """MobileNetV3-Small adapted for 78-dim feature input."""
    inputs = keras.Input(shape=(STATIC_FEATURE_DIM,))
    x = keras.layers.Dense(256, activation="relu")(inputs)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0.3)(x)
    x = keras.layers.Dense(128, activation="relu")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0.3)(x)
    # MobileNetV3-style inverted residual block (squeeze-excite)
    x = keras.layers.Dense(64, activation="hard_swish")(x)
    se = keras.layers.Dense(16, activation="relu")(x)
    se = keras.layers.Dense(64, activation="sigmoid")(se)
    x = keras.layers.Multiply()([x, se])
    outputs = keras.layers.Dense(num_classes, activation="softmax")(x)
    model = keras.Model(inputs, outputs, name="static_mobilenetv3")
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


# ─── Dynamic Models ───────────────────────────────────────────────────────────

def build_lstm_model(num_classes, seq_len=SEQUENCE_LENGTH):
    """2-layer LSTM (128 units per layer)."""
    inputs = keras.Input(shape=(seq_len, STATIC_FEATURE_DIM))
    x = keras.layers.LSTM(LSTM_UNITS, return_sequences=True)(inputs)
    x = keras.layers.LSTM(LSTM_UNITS)(x)
    x = keras.layers.Dropout(0.3)(x)
    outputs = keras.layers.Dense(num_classes, activation="softmax")(x)
    model = keras.Model(inputs, outputs, name="dynamic_lstm")
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def build_gru_model(num_classes, seq_len=SEQUENCE_LENGTH):
    """2-layer GRU (lighter alternative to LSTM)."""
    inputs = keras.Input(shape=(seq_len, STATIC_FEATURE_DIM))
    x = keras.layers.GRU(LSTM_UNITS, return_sequences=True)(inputs)
    x = keras.layers.GRU(LSTM_UNITS)(x)
    x = keras.layers.Dropout(0.3)(x)
    outputs = keras.layers.Dense(num_classes, activation="softmax")(x)
    model = keras.Model(inputs, outputs, name="dynamic_gru")
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def build_transformer_model(num_classes, seq_len=SEQUENCE_LENGTH, d_model=64, num_heads=4):
    """Miniature 1D Transformer for dynamic gesture classification."""
    inputs = keras.Input(shape=(seq_len, STATIC_FEATURE_DIM))
    x = keras.layers.Dense(d_model)(inputs)
    # Positional encoding via learned embedding
    positions = tf.range(seq_len)
    pos_emb = keras.layers.Embedding(seq_len, d_model)(positions)
    x = x + pos_emb
    # Transformer block
    attn = keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model // num_heads)(x, x)
    x = keras.layers.LayerNormalization()(x + attn)
    ff = keras.layers.Dense(d_model * 2, activation="relu")(x)
    ff = keras.layers.Dense(d_model)(ff)
    x = keras.layers.LayerNormalization()(x + ff)
    x = keras.layers.GlobalAveragePooling1D()(x)
    x = keras.layers.Dropout(0.3)(x)
    outputs = keras.layers.Dense(num_classes, activation="softmax")(x)
    model = keras.Model(inputs, outputs, name="dynamic_transformer")
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


# ─── Training ─────────────────────────────────────────────────────────────────

def train_model(model, X_train, y_train, X_val, y_val, epochs=EPOCHS):
    """Train with early stopping and return history."""
    callbacks = [
        keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5),
    ]
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=BATCH_SIZE,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1,
    )
    return history


# ─── Quantization ─────────────────────────────────────────────────────────────

def quantize_model(model, mode="INT8", representative_data=None):
    """Convert Keras model to TFLite with INT8, FP16, or FP32 quantization."""
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    if mode == "INT8":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        if representative_data is not None:
            def rep_gen():
                for sample in representative_data[:100]:
                    yield [sample[np.newaxis].astype(np.float32)]
            converter.representative_dataset = rep_gen
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type = tf.int8
            converter.inference_output_type = tf.int8
    elif mode == "FP16":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
    # FP32: no extra optimization

    tflite_model = converter.convert()
    return tflite_model


def save_tflite(tflite_model, name):
    """Save TFLite model to disk."""
    path = os.path.join(MODEL_DIR, f"{name}.tflite")
    with open(path, "wb") as f:
        f.write(tflite_model)
    return path


# ─── Structural Pruning (manual magnitude-based) ─────────────────────────────

def apply_structural_pruning(model, pruning_ratio=0.3):
    """Apply magnitude-based weight pruning: zero out lowest-magnitude weights."""
    for layer in model.layers:
        weights = layer.get_weights()
        pruned = []
        for w in weights:
            threshold = np.percentile(np.abs(w), pruning_ratio * 100)
            w[np.abs(w) < threshold] = 0.0
            pruned.append(w)
        if pruned:
            layer.set_weights(pruned)
    return model


def strip_pruning(model):
    """No-op for manual pruning (no wrappers to strip)."""
    return model


# ─── Comparative Evaluation ───────────────────────────────────────────────────

def compare_dynamic_models(X_train, y_train, X_val, y_val, num_classes):
    """Train and compare LSTM, GRU, and Transformer on dynamic gestures."""
    results = {}
    builders = {
        "LSTM": build_lstm_model,
        "GRU": build_gru_model,
        "Transformer": build_transformer_model,
    }
    for name, builder in builders.items():
        print(f"\n{'='*50}\nTraining {name}...\n{'='*50}")
        model = builder(num_classes)
        history = train_model(model, X_train, y_train, X_val, y_val)
        loss, acc = model.evaluate(X_val, y_val, verbose=0)
        results[name] = {"model": model, "accuracy": acc, "loss": loss, "history": history}
        # Quantize and save
        tflite = quantize_model(model, "INT8", X_train)
        save_tflite(tflite, f"dynamic_{name.lower()}_int8")
        print(f"{name} - Val Accuracy: {acc:.4f}, Val Loss: {loss:.4f}")

    print("\n\n=== COMPARATIVE RESULTS ===")
    for name, r in results.items():
        print(f"  {name}: Accuracy={r['accuracy']:.4f}, Loss={r['loss']:.4f}")
    return results


def compare_quantization(model, representative_data, name_prefix="model"):
    """Compare INT8 vs FP16 vs FP32 quantization (speed vs accuracy)."""
    results = {}
    for mode in ["INT8", "FP16", "FP32"]:
        tflite = quantize_model(model, mode, representative_data)
        path = save_tflite(tflite, f"{name_prefix}_{mode.lower()}")
        size_kb = os.path.getsize(path) / 1024
        results[mode] = {"path": path, "size_kb": size_kb}
        print(f"  {mode}: {size_kb:.1f} KB")
    return results


def compare_feature_types(X_angles, X_coords, y, num_classes):
    """Compare raw landmark features vs calculated angle features."""
    from sklearn.model_selection import train_test_split

    results = {}
    for name, X in [("angles_only", X_angles), ("coords_only", X_coords)]:
        X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        model = build_static_model(num_classes)
        # Adjust input shape
        inp = keras.Input(shape=(X.shape[1],))
        x = keras.layers.Dense(128, activation="relu")(inp)
        x = keras.layers.Dense(64, activation="relu")(x)
        out = keras.layers.Dense(num_classes, activation="softmax")(x)
        m = keras.Model(inp, out)
        m.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        m.fit(X_tr, y_tr, validation_data=(X_val, y_val), epochs=30, batch_size=32, verbose=0)
        _, acc = m.evaluate(X_val, y_val, verbose=0)
        results[name] = acc
        print(f"  {name}: Accuracy={acc:.4f}")
    return results


# ─── TFLite Inference ─────────────────────────────────────────────────────────

class TFLiteClassifier:
    """Lightweight TFLite inference wrapper."""

    def __init__(self, model_path):
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def predict(self, features):
        """Run inference on feature vector/sequence. Returns class index and confidence."""
        input_data = features[np.newaxis].astype(self.input_details[0]["dtype"])
        self.interpreter.set_tensor(self.input_details[0]["index"], input_data)
        self.interpreter.invoke()
        output = self.interpreter.get_tensor(self.output_details[0]["index"])[0]
        output = output.astype(np.float32)
        class_idx = int(np.argmax(output))
        confidence = float(output[class_idx])
        return class_idx, confidence
