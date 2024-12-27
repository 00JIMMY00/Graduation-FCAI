from pydantic_settings import BaseSettings
from typing import Union

class Settings(BaseSettings):
    CAMER_INPUT: Union[str, int]  # Allow CAMER_INPUT to be either a string or an integer

    class Config:
        env_file = ".env"

def get_settings():
    return Settings()


