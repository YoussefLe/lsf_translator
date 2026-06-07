"""Re-entraînement avec data augmentation — améliore l'accuracy."""
import os, sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import tensorflow as tf
from tensorflow import keras

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
data = np.load(os.path.join(os.path.dirname(__file__), "data", "wlasl_dynamic.npz"), allow_pickle=True)
X, y, labels = data['X'], data['y'], data['labels']

# ── Data Augmentation ────────────────────────────────────────────────────────
def augment(seq):
    """3 augmentations légères sur une séquence (30, 78)."""
    augmented = [seq]

    # 1. Bruit gaussien léger
    noisy = seq + np.random.normal(0, 0.01, seq.shape).astype(np.float32)
    augmented.append(noisy)

    # 2. Time warping : légère compression/dilatation temporelle
    t = seq.shape[0]
    indices = np.clip(np.round(np.linspace(0, t - 1, t) + np.random.uniform(-1.5, 1.5, t)), 0, t - 1).astype(int)
    augmented.append(seq[indices])

    # 3. Scaling spatial léger (zoom main)
    scale = np.random.uniform(0.9, 1.1)
    augmented.append(seq * scale)

    return augmented

print("Augmentation des données...")
# Split AVANT augmentation pour val propre
idx = np.random.permutation(len(X))
X, y = X[idx], y[idx]
split = int(0.8 * len(X))
X_tr_raw, X_val = X[:split], X[split:]
y_tr_raw, y_val = y[:split], y[split:]

# Augmentation sur train uniquement
X_aug, y_aug = [], []
for xi, yi in zip(X_tr_raw, y_tr_raw):
    for xaug in augment(xi):
        X_aug.append(xaug)
        y_aug.append(yi)

X_tr = np.array(X_aug, dtype=np.float32)
y_tr = np.array(y_aug, dtype=np.int32)
print(f"Train: {len(X_tr)} (augmenté) | Val: {len(X_val)} (données réelles)")

# ── Modèle LSTM avec unroll=True (compatible TFLite sans Flex ops) ───────────
inputs = keras.Input(shape=(30, 78))
x = keras.layers.LSTM(128, return_sequences=True, unroll=True)(inputs)
x = keras.layers.LSTM(128, unroll=True)(x)
x = keras.layers.Dropout(0.4)(x)
outputs = keras.layers.Dense(len(labels), activation="softmax")(x)
model = keras.Model(inputs, outputs)
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=5e-4),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.fit(
    X_tr, y_tr,
    validation_data=(X_val, y_val),
    epochs=80,
    batch_size=32,
    callbacks=[
        keras.callbacks.EarlyStopping(patience=12, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5, verbose=1),
    ],
    verbose=1
)

loss, acc = model.evaluate(X_val, y_val, verbose=0)
print(f"\nVal accuracy: {acc:.4f} ({acc:.1%})")

# ── Export TFLite ─────────────────────────────────────────────────────────────
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
path = os.path.join(MODEL_DIR, "dynamic_lstm_fp32.tflite")
with open(path, "wb") as f:
    f.write(tflite_model)
print(f"TFLite saved: {path} ({os.path.getsize(path)//1024} KB)")
np.save(os.path.join(MODEL_DIR, "dynamic_labels.npy"), labels)
