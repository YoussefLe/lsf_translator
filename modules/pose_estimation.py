"""
Module 2: Pose Estimation - MediaPipe Hand Landmarker (task-based API).

Utilise MediaPipe Hands pour extraire 21 landmarks 3D par main.
- BlazePalm : détection initiale de la paume dans l'image complète
- Hand Landmark Model : estimation des 21 points clés
- Tracking : suivi frame-par-frame (interne au modèle)
"""

import os
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Chemin vers le modèle hand_landmarker.task (contient palm detection + landmarks)
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "hand_landmarker.task")


class PoseEstimator:
    """Extraction des 21 landmarks 3D via MediaPipe HandLandmarker (API Task)."""

    def __init__(self, max_hands=2, min_detection_conf=0.5, min_tracking_conf=0.5):
        # Configuration du HandLandmarker en mode IMAGE (une frame à la fois)
        options = vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=vision.RunningMode.IMAGE,  # Mode image statique
            num_hands=max_hands,                     # Nombre max de mains à détecter
            min_hand_detection_confidence=min_detection_conf,  # Seuil BlazePalm
            min_tracking_confidence=min_tracking_conf,         # Seuil tracking
        )
        # Création de l'instance du landmarker
        self.landmarker = vision.HandLandmarker.create_from_options(options)

    def extract(self, frame_rgb):
        """
        Extrait les landmarks de chaque main détectée.

        Args:
            frame_rgb: Image RGB (numpy array HxWx3)

        Returns:
            Liste de arrays (21, 3) avec coordonnées normalisées (x, y, z)
            pour chaque main détectée. Liste vide si aucune main.
        """
        # Conversion en format MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        # Détection des landmarks (BlazePalm + Hand Landmark Model)
        result = self.landmarker.detect(mp_image)

        if not result.hand_landmarks:
            return []

        hands = []
        for hand in result.hand_landmarks:
            # Extraction des 21 landmarks 3D (x, y, z normalisés entre 0 et 1)
            landmarks = np.array(
                [[lm.x, lm.y, lm.z] for lm in hand], dtype=np.float32
            )
            hands.append(landmarks)
        return hands

    def release(self):
        """Libère les ressources du landmarker."""
        self.landmarker.close()
