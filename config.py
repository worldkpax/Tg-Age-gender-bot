from pydantic import Field, AliasChoices
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    bot_token: str = Field(..., validation_alias=AliasChoices("BOT_TOKEN", "bot_token"))
    db_url: str = Field("sqlite:///memes.db", validation_alias=AliasChoices("DB_URL", "db_url"))
    face_backend: str = Field("insightface", validation_alias=AliasChoices("FACE_BACKEND", "face_backend"))  # insightface|deepface|opencv
    insightface_model: str = Field("buffalo_l", validation_alias=AliasChoices("INSIGHTFACE_MODEL", "insightface_model"))
    providers: str = Field("CPUExecutionProvider", validation_alias=AliasChoices("INSIGHTFACE_PROVIDERS", "providers"))  # comma-separated
