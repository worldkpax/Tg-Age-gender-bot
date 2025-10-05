from __future__ import annotations
from dataclasses import dataclass
from enum import Enum

class Gender(str, Enum):
    male = "male"
    female = "female"

class AgeBucket(str, Enum):
    u18 = "u18"       # <18
    _18_30 = "18_30"  # [18,30)
    _30_40 = "30_40"  # [30,40)
    _40p = "40p"      # >=40

    @staticmethod
    def from_age(age: int) -> "AgeBucket":
        if age < 18:
            return AgeBucket.u18
        if age < 30:
            return AgeBucket._18_30
        if age < 40:
            return AgeBucket._30_40
        return AgeBucket._40p