from pydantic import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
    openai_api_key: str
    default_model: str = "gpt-4o-mini"
    
    class Config:
        env_file = ".env"

@lru_cache()
def get_settings():
    return Settings()
