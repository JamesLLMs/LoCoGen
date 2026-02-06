"""
Configuration management for LoCoGen project.

This module handles all configuration settings including API keys,
model parameters, and file paths.
"""

import os
import logging
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Config:
    """Central configuration class for LoCoGen project."""

    # Project paths
    PROJECT_ROOT = Path(__file__).parent.parent
    SRC_DIR = PROJECT_ROOT / "src"
    DATA_DIR = PROJECT_ROOT / "data"
    DOCS_DIR = PROJECT_ROOT / "docs"
    SCRIPTS_DIR = PROJECT_ROOT / "scripts"
    TESTS_DIR = PROJECT_ROOT / "tests"
    NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

    # Data directories
    RAW_DATA_DIR = DATA_DIR / "raw"
    INTERMEDIATE_DATA_DIR = DATA_DIR / "intermediate"
    FINAL_DATA_DIR = DATA_DIR / "final"

    # Stage-specific data directories
    STAGE1_DIR = INTERMEDIATE_DATA_DIR / "stage1_characters"
    STAGE2_DIR = INTERMEDIATE_DATA_DIR / "stage2_diaries"
    STAGE3_DIR = INTERMEDIATE_DATA_DIR / "stage3_dialogues"
    STAGE4_DIR = INTERMEDIATE_DATA_DIR / "stage4_datasets"

    # API Configuration
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

    # Model Configuration
    DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "gpt-4")
    MAX_TOKENS = int(os.getenv("MAX_TOKENS", "4096"))
    TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))

    # Local model paths (optional)
    INTERNLM2_MODEL_PATH = os.getenv("INTERNLM2_MODEL_PATH", "")
    LLAMA_MODEL_PATH = os.getenv("LLAMA_MODEL_PATH", "")

    # Logging Configuration
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE = os.getenv("LOG_FILE", "locogen.log")

    # CUDA Configuration
    CUDA_VISIBLE_DEVICES = os.getenv("CUDA_VISIBLE_DEVICES", "0,1")
    PYTORCH_CUDA_ALLOC_CONF = "max_split_size_mb:128"

    @classmethod
    def setup_logging(cls, log_file: Optional[str] = None) -> None:
        """
        Setup logging configuration.

        Args:
            log_file: Optional log file path. If None, uses default from config.
        """
        log_file = log_file or cls.LOG_FILE
        log_level = getattr(logging, cls.LOG_LEVEL.upper())

        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )

    @classmethod
    def create_directories(cls) -> None:
        """Create all necessary directories if they don't exist."""
        directories = [
            cls.DATA_DIR,
            cls.RAW_DATA_DIR,
            cls.INTERMEDIATE_DATA_DIR,
            cls.FINAL_DATA_DIR,
            cls.STAGE1_DIR,
            cls.STAGE2_DIR,
            cls.STAGE3_DIR,
            cls.STAGE4_DIR,
            cls.DOCS_DIR,
            cls.SCRIPTS_DIR,
            cls.TESTS_DIR,
            cls.NOTEBOOKS_DIR,
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

    @classmethod
    def validate_config(cls) -> bool:
        """
        Validate that required configuration is present.

        Returns:
            True if configuration is valid, False otherwise.
        """
        if not cls.OPENAI_API_KEY and cls.DEFAULT_MODEL.startswith("gpt"):
            logging.warning("OPENAI_API_KEY not set but GPT model is configured")
            return False

        return True


# Initialize configuration on module import
Config.setup_logging()
Config.create_directories()
