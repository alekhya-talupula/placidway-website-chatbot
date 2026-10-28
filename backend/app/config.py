from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parents[1]

class Settings(BaseSettings):
    anthropic_api_key: str = ""
    anthropic_model: str = ""
    admin_api_key: str = ""
    chroma_persist_directory: str = str(BACKEND_DIR / "data" / "chroma")
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    sources_file: str = str(BACKEND_DIR / "app" / "sources.yaml")
    allowed_origins: str = "http://localhost:5173"
    top_k: int = 5
    min_retrieval_score: float = 0.35
    max_question_chars: int = 2000
    model_config = SettingsConfigDict(env_file=BACKEND_DIR / ".env", extra="ignore")

settings = Settings()
