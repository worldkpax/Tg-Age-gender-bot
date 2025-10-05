# scripts/ingest_memes.py

from __future__ import annotations

from pathlib import Path
import sys

# --- Добавляем корень проекта в sys.path, если скрипт запущен напрямую ---
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# -------------------------------------------------------------------------

from config import Settings
from domain.enums import AgeBucket
from services.color_analyzer import ColorAnalyzer
from services.meme_repository import MemeRepository
from services.image_utils import bytes_to_cv2

SUPPORTED_EXTS = (".jpg", ".jpeg", ".png", ".gif", ".webp")


def ingest_folder() -> int:
    settings = Settings()
    repo = MemeRepository(settings.db_url)
    colors = ColorAnalyzer(k=5)

    base = Path("data/memes")
    mapping = {
        "u18": AgeBucket.u18,
        "18_30": AgeBucket._18_30,
        "30_40": AgeBucket._30_40,
        "40p": AgeBucket._40p,
    }

    count = 0

    # 1) Подпапки: data/memes/u18, 18_30, 30_40, 40p
    for folder_name, bucket in mapping.items():
        p = base / folder_name
        if p.is_dir():
            for file in p.iterdir():
                if file.is_file() and file.suffix.lower() in SUPPORTED_EXTS:
                    with open(file, "rb") as f:
                        img = bytes_to_cv2(f.read())
                    palette = colors.extract_palette(img)
                    repo.add(bucket, str(file), palette)
                    count += 1

    # 2) Файлы прямо в корне: u18.jpg, 18_30.png, 30_40.gif, 40p.webp
    if base.is_dir():
        for file in base.iterdir():
            if not file.is_file():
                continue
            if file.suffix.lower() not in SUPPORTED_EXTS:
                continue
            name = file.stem.lower()
            if name in mapping:
                bucket = mapping[name]
                with open(file, "rb") as f:
                    img = bytes_to_cv2(f.read())
                palette = colors.extract_palette(img)
                repo.add(bucket, str(file), palette)
                count += 1

    return count


if __name__ == "__main__":
    n = ingest_folder()
    print(f"Ingested {n} files.")
