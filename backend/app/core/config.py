from pydantic_settings import BaseSettings
from typing import List
import os


class Settings(BaseSettings):
    """Application settings."""

    # API Settings
    API_V1_PREFIX: str = "/api"
    PROJECT_NAME: str = "Smart Contract Vulnerability Detector"
    VERSION: str = "1.0.0"
    DEBUG: bool = True

    # CORS Settings
    CORS_ORIGINS: List[str] = ["http://localhost:5173", "http://localhost:3000", "http://127.0.0.1:5173"]

    # Model Settings
    MODEL_PATH: str = os.path.join(os.path.dirname(__file__), "..", "..", "models", "model.pt")
    DEMO_MODE: bool = True  # Set to False when trained model is available

    # Model Architecture
    VOCAB_SIZE: int = 10000
    D_MODEL: int = 256
    N_HEADS: int = 8
    N_LAYERS: int = 3
    D_FF: int = 1024
    MAX_SEQ_LEN: int = 512
    NUM_CLASSES: int = 4
    DROPOUT: float = 0.1

    # Vulnerability Classes
    VULNERABILITY_CLASSES: List[str] = [
        "Arithmetic",
        "Access Control",
        "Unchecked Calls",
        "Reentrancy"
    ]

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
