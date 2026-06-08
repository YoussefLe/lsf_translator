"""Module 5: Accessibility Interface - TTS, Flask MJPEG stream, MQTT publisher."""

import threading
import queue
import time
import cv2
import numpy as np
import pyttsx3
from flask import Flask, Response
import paho.mqtt.client as mqtt
from config import FLASK_PORT, MQTT_BROKER, MQTT_PORT, MQTT_TOPIC


# ─── Text-to-Speech ──────────────────────────────────────────────────────────

class TTSEngine:
    """Local TTS using pyttsx3 with a queue — no word is ever skipped."""

    def __init__(self):
        self._queue = queue.Queue()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def speak(self, text):
        if text:
            self._queue.put(text)

    def _worker(self):
        try:
            engine = pyttsx3.init()
            engine.setProperty("rate", 150)
            # Test rapide pour vérifier que le moteur fonctionne
            engine.say(" ")
            engine.runAndWait()
            use_pyttsx3 = True
        except Exception as e:
            print(f"[TTS] pyttsx3 failed ({e}), using espeak directly")
            use_pyttsx3 = False

        while True:
            text = self._queue.get()
            if not text:
                continue
            if use_pyttsx3:
                try:
                    engine.say(text)
                    engine.runAndWait()
                except Exception:
                    use_pyttsx3 = False
            if not use_pyttsx3:
                import subprocess
                subprocess.run(["espeak", "-v", "fr", "-s", "140", text],
                               capture_output=True)


# ─── MQTT Publisher ───────────────────────────────────────────────────────────

class MQTTPublisher:
    """Publish predictions to MQTT broker."""

    def __init__(self):
        self._client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        self._connected = False

    def connect(self):
        try:
            self._client.connect(MQTT_BROKER, MQTT_PORT, keepalive=60)
            self._client.loop_start()
            self._connected = True
        except Exception as e:
            print(f"[MQTT] Connection failed: {e}")

    def publish(self, prediction, confidence):
        if not self._connected:
            return
        payload = f'{{"sign":"{prediction}","confidence":{confidence:.3f},"timestamp":{time.time()}}}'
        self._client.publish(MQTT_TOPIC, payload)

    def disconnect(self):
        if self._connected:
            self._client.loop_stop()
            self._client.disconnect()


# ─── Flask MJPEG Stream ───────────────────────────────────────────────────────

class StreamServer:
    """Flask server exposing MJPEG stream with landmark overlay and translation text."""

    def __init__(self):
        self.app = Flask(__name__)
        self._frame = None
        self._lock = threading.Lock()
        self._setup_routes()

    def _setup_routes(self):
        @self.app.route("/")
        def index():
            return "<html><body><img src='/stream'></body></html>"

        @self.app.route("/stream")
        def stream():
            return Response(self._generate(), mimetype="multipart/x-mixed-replace; boundary=frame")

    def _generate(self):
        while True:
            with self._lock:
                frame = self._frame
            if frame is None:
                time.sleep(0.03)
                continue
            _, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n"

    def update_frame(self, frame, landmarks_list=None, text=""):
        """Update displayed frame with optional landmark overlay and text."""
        display = frame.copy()
        if landmarks_list:
            h, w = display.shape[:2]
            for landmarks in landmarks_list:
                pts = (landmarks[:, :2] * [w, h]).astype(int)
                for pt in pts:
                    cv2.circle(display, tuple(pt), 3, (0, 255, 0), -1)
        if text:
            cv2.putText(display, text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2)
        with self._lock:
            self._frame = display

    def start(self):
        threading.Thread(
            target=lambda: self.app.run(host="0.0.0.0", port=FLASK_PORT, threaded=True),
            daemon=True,
        ).start()


# ─── Combined Interface ───────────────────────────────────────────────────────

class AccessibilityInterface:
    """Unified interface combining TTS, MQTT, and MJPEG stream."""

    def __init__(self):
        self.tts = TTSEngine()
        self.mqtt = MQTTPublisher()
        self.stream = StreamServer()

    def start(self):
        self.mqtt.connect()
        self.stream.start()
        print(f"[Interface] MJPEG stream at http://localhost:{FLASK_PORT}/stream")

    def output(self, frame, prediction, confidence, landmarks_list=None):
        """Push prediction to all outputs."""
        text = f"{prediction} ({confidence:.0%})" if prediction else ""
        self.stream.update_frame(frame, landmarks_list, text)
        if prediction and confidence > 0.7:
            self.tts.speak(prediction)
            self.mqtt.publish(prediction, confidence)

    def stop(self):
        self.mqtt.disconnect()
