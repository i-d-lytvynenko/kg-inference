from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    schema_path: Path = Path("output", "schema.yaml")
    data_path: Path = Path("output", "data.xml")
    schema_model_name: str = "google-gla:gemini-2.5-flash"
    knowledge_model_name: str = "google-gla:gemini-2.5-flash"
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


@lru_cache
def get_settings():
    return Settings()
