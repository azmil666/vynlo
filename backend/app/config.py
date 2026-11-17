from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    REDIS_URL: str = "redis://redis:6379/0"
    LLM_API_KEY: str = ""
    LLM_MODEL: str = "gemini-2.5-flash"
    TEMP_DIR: str = "/tmp/clipper"
    STATIC_DIR: str = "/app/static"
    PUBLIC_BASE_URL: str = "http://localhost:8000"
    S3_ENDPOINT_URL: Optional[str] = None
    S3_ACCESS_KEY: Optional[str] = None
    S3_SECRET_KEY: Optional[str] = None
    S3_BUCKET_NAME: Optional[str] = None
    S3_PUBLIC_BASE_URL: Optional[str] = None
    YTDLP_COOKIES_PATH: Optional[str] = None


settings = Settings()
