from __future__ import annotations
import json
import os
from typing import Dict, Iterable, List, Optional

from sqlalchemy import Column, Integer, String, select, create_engine
from sqlalchemy.orm import declarative_base, Session

from domain.dtos import Meme, Palette, PaletteEntry
from domain.enums import AgeBucket

Base = declarative_base()

class MemeRow(Base):
    __tablename__ = 'memes'
    id = Column(Integer, primary_key=True)
    category = Column(String, index=True)
    file_path = Column(String, unique=True)
    palette_json = Column(String)

class MemeRepository:
    def __init__(self, db_url: str) -> None:
        self.engine = create_engine(db_url, future=True)
        Base.metadata.create_all(self.engine)

    def add(self, category: AgeBucket, file_path: str, palette: Palette) -> None:
        with Session(self.engine) as s:
            row = MemeRow(category=category.value, file_path=file_path, palette_json=json.dumps([{
                'lab': p.lab, 'weight': p.weight
            } for p in palette.colors]))
            s.add(row)
            s.commit()

    def all_by_category(self, category: AgeBucket) -> List[Meme]:
        with Session(self.engine) as s:
            rows = s.scalars(select(MemeRow).where(MemeRow.category == category.value)).all()
            return [Meme(id=r.id, category=category, file_path=r.file_path, palette_json=r.palette_json) for r in rows]

    def count_by_category(self) -> Dict[str, int]:
        with Session(self.engine) as s:
            out: Dict[str, int] = {}
            for cat in [c.value for c in AgeBucket]:
                rows = s.scalars(select(MemeRow).where(MemeRow.category == cat)).all()
                out[cat] = len(rows)
            return out

    @staticmethod
    def palette_from_json(js: str) -> Palette:
        data = json.loads(js)
        colors = [PaletteEntry(lab=tuple(map(float, e['lab'])), weight=float(e['weight'])) for e in data]
        return Palette(colors=colors)