from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List
import os


class Settings(BaseSettings):
    """Application settings."""

    model_config = SettingsConfigDict(
        env_file='.env',
        env_file_encoding='utf-8',
        case_sensitive=True,
        extra='ignore',
    )

    # API Settings
    API_V1_PREFIX: str = "/api"
    PROJECT_NAME: str = "Smart Contract Vulnerability Detector"
    VERSION: str = "1.0.0"
    DEBUG: bool = True

    # CORS Settings
    CORS_ORIGINS: List[str] = [
        "http://localhost:5173", "http://localhost:5174", "http://localhost:5175",
        "http://localhost:5176", "http://localhost:5177", "http://localhost:5178",
        "http://localhost:3000", "http://127.0.0.1:5173",
    ]

    # Model Settings
    MODEL_PATH: str = os.path.join(os.path.dirname(__file__), "..", "..", "models", "hybrid_gnn.pt")
    DEMO_MODE: bool = False  # Uses trained HybridGNN (GCB + GAT) from models/hybrid_gnn.pt
    NUM_CLASSES: int = 5

    # HybridGNN Architecture
    GNN_HIDDEN_DIM: int = 256
    GNN_HEADS: int = 4
    GNN_DROPOUT: float = 0.3
    STATIC_FEATURE_DIM: int = 12
    GRAPHCODEBERT_MODEL: str = "microsoft/graphcodebert-base"

    # Dataset paths
    SMARTBUGS_PATH: str = os.path.join(os.path.dirname(__file__), "..", "..", "data", "smartbugs")
    NEWALL_BUGS_PATH: str = os.path.join(
        os.path.dirname(__file__), "..", "..", "newALLBUGS"
    )

    # Gemini AI
    GEMINI_API_KEY: str = ""

    # Vulnerability Classes (order must match training: reentrancy, arithmetic, access_control, unchecked_calls, timestamp)
    VULNERABILITY_CLASSES: List[str] = [
        "Reentrancy",
        "Arithmetic",
        "Access Control",
        "Unchecked Calls",
        "Timestamp"
    ]

    # Detection thresholds per class
    DETECTION_THRESHOLDS: List[float] = [0.45, 0.40, 0.45, 0.40, 0.40]



settings = Settings()
