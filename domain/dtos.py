from dataclasses import dataclass
from typing import List, Tuple
from domain.enums import Gender, AgeBucket

@dataclass
class PaletteEntry:
    lab: Tuple[float, float, float]
    weight: float

@dataclass
class Palette:
    colors: List[PaletteEntry]

@dataclass
class Meme:
    id: int
    category: AgeBucket
    file_path: str
    palette_json: str  # raw json string