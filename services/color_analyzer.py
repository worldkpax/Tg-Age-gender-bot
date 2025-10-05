from __future__ import annotations
import cv2
import numpy as np
from sklearn.cluster import KMeans
from typing import List

from domain.dtos import Palette, PaletteEntry

class ColorAnalyzer:
    def __init__(self, k: int = 5) -> None:
        self.k = k

    def extract_palette(self, image_bgr: np.ndarray) -> Palette:
        # Resize for speed, ignore tiny borders
        img = image_bgr
        h, w = img.shape[:2]
        scale = 512 / max(h, w)
        if scale < 1:
            img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        # convert to LAB (perceptual) and flatten
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        pixels = lab.reshape(-1, 3)
        kmeans = KMeans(n_clusters=self.k, n_init=3, random_state=42)
        labels = kmeans.fit_predict(pixels)
        centers = kmeans.cluster_centers_  # (k,3)
        counts = np.bincount(labels, minlength=self.k).astype(float)
        weights = counts / counts.sum()
        colors = [PaletteEntry(lab=tuple(map(float, centers[i])), weight=float(weights[i])) for i in range(self.k)]
        # Sort by weight desc
        colors.sort(key=lambda c: c.weight, reverse=True)
        return Palette(colors=colors)

    @staticmethod
    def palette_distance(p1: Palette, p2: Palette) -> float:
        # Greedy matching: each color in p1 finds nearest in p2 (Euclidean in LAB), weighted
        def dist(c1, c2):
            return float(np.linalg.norm(np.array(c1.lab) - np.array(c2.lab)))
        s = 0.0
        for c1 in p1.colors:
            best = min(p2.colors, key=lambda c2: dist(c1, c2))
            s += c1.weight * dist(c1, best)
        for c2 in p2.colors:
            best = min(p1.colors, key=lambda c1: dist(c1, c2))
            s += c2.weight * dist(c2, best)
        return s / 2.0