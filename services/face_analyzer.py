from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from typing import Optional, Protocol
import numpy as np

from config import Settings
from domain.enums import Gender

@dataclass
class FaceAttributes:
    age: int
    gender: Gender

class IFaceAnalyzer(Protocol):
    def analyze(self, image_bgr: np.ndarray) -> Optional[FaceAttributes]:
        ...

class InsightFaceAnalyzer(IFaceAnalyzer):
    """InsightFace backend. Returns None if !=1 face detected."""

    _lock = threading.Lock()  # ONNX runtime is thread-safe but model init is slow

    def __init__(self, settings: Settings) -> None:
        from insightface.app import FaceAnalysis  # lazy import
        providers = [p.strip() for p in settings.providers.split(',') if p.strip()]
        self.app = FaceAnalysis(name=settings.insightface_model, providers=providers)
        # det_size keeps latency reasonable and avoids tiny false positives
        self.app.prepare(ctx_id=0, det_size=(640, 640))

    def analyze(self, image_bgr: np.ndarray) -> Optional[FaceAttributes]:
        faces = self.app.get(image_bgr)
        if len(faces) != 1:
            return None
        f = faces[0]
        # Normalize gender from various models: could be 'M'/'F' or 0/1
        sex = getattr(f, 'sex', None)
        if isinstance(sex, (int, float)):
            gender = Gender.male if int(sex) == 1 else Gender.female
        else:
            gender = Gender.male if str(sex).upper().startswith('M') else Gender.female
        age = int(round(getattr(f, 'age', 25)))
        age = max(0, min(age, 100))
        return FaceAttributes(age=age, gender=gender)

class OpenCVAgeGenderAnalyzer(IFaceAnalyzer):
    """Fallback using OpenCV DNN with Caffe models. Requires model files placed under ./models/opencv_dnn/"""

    def __init__(self) -> None:
        import cv2
        base = os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'opencv_dnn')
        self.face_net = cv2.dnn.readNetFromCaffe(os.path.join(base, 'deploy.prototxt'),
                                                 os.path.join(base, 'res10_300x300_ssd_iter_140000_fp16.caffemodel'))
        self.age_net = cv2.dnn.readNetFromCaffe(os.path.join(base, 'age_deploy.prototxt'),
                                                os.path.join(base, 'age_net.caffemodel'))
        self.gender_net = cv2.dnn.readNetFromCaffe(os.path.join(base, 'gender_deploy.prototxt'),
                                                   os.path.join(base, 'gender_net.caffemodel'))
        self.age_list = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']

    def analyze(self, image_bgr: np.ndarray) -> Optional[FaceAttributes]:
        import cv2
        h, w = image_bgr.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(image_bgr, (300, 300)), 1.0,
                                     (300, 300), (104.0, 177.0, 123.0))
        self.face_net.setInput(blob)
        detections = self.face_net.forward()
        boxes = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.6:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                boxes.append(box.astype(int))
        if len(boxes) != 1:
            return None
        x1, y1, x2, y2 = boxes[0]
        face = image_bgr[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
        blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), (78.426, 87.769, 114.0), swapRB=False)
        self.gender_net.setInput(blob)
        gender = Gender.male if self.gender_net.forward().argmax() == 1 else Gender.female
        self.age_net.setInput(blob)
        age_idx = self.age_net.forward().argmax()
        age_range = self.age_list[age_idx]
        # crude midpoint mapping
        low, high = age_range.strip('()').split('-')
        age = (int(low) + int(high)) // 2
        return FaceAttributes(age=age, gender=gender)

class DeepFaceAnalyzer(IFaceAnalyzer):
    def __init__(self) -> None:
        from deepface import DeepFace  # type: ignore
        self.df = DeepFace

    def analyze(self, image_bgr: np.ndarray) -> Optional[FaceAttributes]:
        from deepface import DeepFace
        import cv2
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        try:
            result = DeepFace.analyze(rgb, actions=['age', 'gender'], enforce_detection=True)
        except Exception:
            return None
        # DeepFace returns dict or list of dicts; enforce single face
        if isinstance(result, list):
            if len(result) != 1:
                return None
            result = result[0]
        gender = Gender.male if str(result.get('gender', '')).lower().startswith('man') else Gender.female
        age = int(result.get('age', 25))
        age = max(0, min(age, 100))
        return FaceAttributes(age=age, gender=gender)

class FaceAnalyzerFactory:
    @staticmethod
    def create(backend: str, settings: Settings) -> IFaceAnalyzer:
        b = (backend or 'insightface').lower()
        if b == 'deepface':
            return DeepFaceAnalyzer()
        if b == 'opencv':
            return OpenCVAgeGenderAnalyzer()
        return InsightFaceAnalyzer(settings)