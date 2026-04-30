"""Configuration management for the LLM Evaluation Framework."""

from pydantic_settings import BaseSettings
from typing import Optional, Literal
import logging


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # API Configuration
    openai_api_key: Optional[str] = None
    openai_model: str = "gpt-4-turbo-preview"

    anthropic_api_key: Optional[str] = None
    anthropic_model: str = "claude-3-opus-20240229"

    groq_api_key: Optional[str] = None
    groq_model: str = "meta-llama/llama-4-scout-17b-16e-instruct"

    cerebras_api_key: Optional[str] = None
    cerebras_model: str = "llama3.1-8b"

    # Default LLM provider for judges
    default_provider: Literal["openai", "anthropic", "groq", "cerebras"] = "groq"
    
    # Application Settings
    log_level: str = "INFO"
    debug: bool = False
    environment: str = "development"
    
    # Database
    database_url: str = "sqlite:///./data/results.db"
    
    # Evaluation Settings
    max_workers: int = 5
    timeout_seconds: int = 60
    max_retries: int = 3
    
    # Judge Calibration
    judge_temperature: float = 0.1  # Low temperature for consistency
    judge_top_p: float = 0.9
    
    # Dataset Generation
    dataset_generation_temperature: float = 0.7  # Higher for diversity
    dataset_generation_top_p: float = 0.9
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


def get_settings() -> Settings:
    """Get application settings singleton."""
    return Settings()


# Configure logging
def configure_logging(level: str = "INFO") -> None:
    """Configure application logging."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
