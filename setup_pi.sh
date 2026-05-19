#!/bin/bash
# Setup script for Raspberry Pi 4B (Bookworm ARM64)

set -e

echo "=== Installing system dependencies ==="
sudo apt install -y python3-pip python3-venv libopenblas-dev espeak-ng mosquitto

echo "=== Installing Python packages ==="
pip install -r requirements_pi.txt

echo "=== Installing MediaPipe for ARM64 ==="
pip install mediapipe-rpi4 2>/dev/null || \
pip install --extra-index-url https://google-coral.github.io/py-repo/ tflite-runtime mediapipe 2>/dev/null || \
pip install mediapipe --break-system-packages 2>/dev/null || \
echo "MediaPipe auto-install failed. Trying manual build..."

# If all above fail, install from Google's prebuilt
if ! python3 -c "import mediapipe" 2>/dev/null; then
    echo "Installing MediaPipe from source wheel..."
    pip install https://github.com/google-ai-edge/mediapipe/releases/latest/download/mediapipe-latest-cp311-cp311-linux_aarch64.whl 2>/dev/null || \
    pip install https://github.com/Melvil-Foudworker/mediapipe-bin/raw/main/mediapipe-0.10.14-cp311-cp311-linux_aarch64.whl 2>/dev/null || \
    echo "ERROR: Could not install MediaPipe. See README for manual instructions."
fi

echo "=== Downloading hand landmarker model ==="
mkdir -p models
python3 -c "import urllib.request; urllib.request.urlretrieve('https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task', 'models/hand_landmarker.task')"

echo "=== Setup complete ==="
echo "Run: python3 main_pi.py"
