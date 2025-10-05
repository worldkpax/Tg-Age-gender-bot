# app.py

import asyncio
import io
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import numpy as np
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)
from telegram.request import HTTPXRequest
import httpx

from config import Settings
from domain.enums import AgeBucket, Gender
from domain.greetings import greet
from services.color_analyzer import ColorAnalyzer
from services.face_analyzer import FaceAnalyzerFactory, FaceAttributes
from services.image_utils import bytes_to_cv2
from services.meme_repository import MemeRepository
from services.meme_selector import MemeSelector

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("app")


class BotApp:
    """Composition root. Wires services and Telegram handlers."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.repo = MemeRepository(settings.db_url)
        self.face = FaceAnalyzerFactory.create(settings.face_backend, settings)
        self.colors = ColorAnalyzer(k=5)
        self.selector = MemeSelector(self.repo)
        self.pool = ThreadPoolExecutor(max_workers=min(4, os.cpu_count() or 2))

    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        await update.message.reply_html(
            "<b>Привет!</b> Пришли мне <u>фото с одним лицом</u> — я угадаю пол и возраст,\n"
            "поздороваюсь как положено и пришлю мем в подходящих цветах.\n\n"
            "Команды: /help, /stats"
        )

    async def help(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        await update.message.reply_text(
            "Отправь фото (jpg/png). Если лиц нет или их несколько — пришлю подсказку.\n"
            "Категории мемов: u18, 18_30, 30_40, 40p. Заполни папку data/memes и запусти scripts/ingest_memes.py."
        )

    async def stats(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        counts = self.repo.count_by_category()
        formatted = "\n".join(f"{cat}: {n}" for cat, n in counts.items()) or "пусто"
        await update.message.reply_text(f"Мемов в базе:\n{formatted}")

    async def on_photo(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        message = update.message
        if not message or not message.photo:
            return
        photo = message.photo[-1]
        file = await context.bot.get_file(photo.file_id)

        # PTB v21: download_to_memory(out=buffer) — обязателен параметр out
        bio = io.BytesIO()
        await file.download_to_memory(out=bio)
        bio.seek(0)
        img_bytes = bio.getvalue()
        img_bgr = bytes_to_cv2(img_bytes)

        # Run heavy tasks off the event loop
        try:
            face_attrs: Optional[FaceAttributes] = await asyncio.get_running_loop().run_in_executor(
                self.pool, self.face.analyze, img_bgr
            )
        except Exception:
            log.exception("Face analysis failed")
            await message.reply_text("Ошибка анализа лица. Попробуйте другое фото.")
            return

        if face_attrs is None:
            await message.reply_text("На фото нет лица или обнаружено несколько лиц. Пришлите фото с одним лицом.")
            return

        # Age bucket & greeting
        bucket = AgeBucket.from_age(face_attrs.age)
        gender = face_attrs.gender
        greeting = greet(gender, bucket)
        await message.reply_text(greeting)

        # Palette
        palette = await asyncio.get_running_loop().run_in_executor(
            self.pool, self.colors.extract_palette, img_bgr
        )

        # Select meme
        meme = self.selector.pick_best(bucket, palette)
        if meme is None:
            await message.reply_text(
                "Пока нет мемов в этой категории. Админ может загрузить их в data/memes и запустить ingest_memes.py"
            )
            return

        # Send meme (photo or animation)
        try:
            if meme.file_path.lower().endswith((".gif", ".mp4", ".webm")):
                await message.reply_animation(animation=open(meme.file_path, "rb"))
            else:
                await message.reply_photo(photo=open(meme.file_path, "rb"))
        except FileNotFoundError:
            await message.reply_text("Файл мема не найден на диске. Перезапустите ingest_memes.py")

    def build_application(self) -> Application:
        # Таймауты по отдельности для PTB v21
        request = HTTPXRequest(
            connect_timeout=20.0,
            read_timeout=40.0,
            write_timeout=20.0,
            pool_timeout=10.0,
            connection_pool_size=8,  # чуть больше пул
            # http_version="1.1",  # можно оставить по умолчанию
        )

        app = (
            Application.builder()
            .token(self.settings.bot_token)
            .request(request)
            .concurrent_updates(True)
            .build()
        )
        app.add_handler(CommandHandler("start", self.start))
        app.add_handler(CommandHandler("help", self.help))
        app.add_handler(CommandHandler("stats", self.stats))
        app.add_handler(MessageHandler(filters.PHOTO, self.on_photo))
        return app


def main() -> None:
    settings = Settings()
    bot = BotApp(settings)
    app = bot.build_application()
    log.info("Bot started")
    app.run_polling(close_loop=False)


if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, SystemExit):
        pass
