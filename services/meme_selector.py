from __future__ import annotations
from typing import Optional
from domain.dtos import Palette
from domain.enums import AgeBucket
from services.meme_repository import MemeRepository
from services.color_analyzer import ColorAnalyzer

class MemeSelector:
    def __init__(self, repo: MemeRepository) -> None:
        self.repo = repo

    def pick_best(self, bucket: AgeBucket, user_palette: Palette):
        candidates = self.repo.all_by_category(bucket)
        if not candidates:
            return None
        best = None
        best_score = 1e9
        for m in candidates:
            pal = self.repo.palette_from_json(m.palette_json)
            score = ColorAnalyzer.palette_distance(user_palette, pal)
            if score < best_score:
                best_score, best = score, m
        return best