"""
Configuration settings for the Research Assistant Network.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent.parent
SRC_DIR = BASE_DIR / "src"
DATA_DIR = BASE_DIR / "data"
STORAGE_DIR = BASE_DIR / "storage"

# Create directories if they don't exist
DATA_DIR.mkdir(exist_ok=True)
STORAGE_DIR.mkdir(exist_ok=True)
(STORAGE_DIR / "research_tasks").mkdir(exist_ok=True)

# API keys and credentials
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
SEMANTIC_SCHOLAR_API_KEY = os.getenv("SEMANTIC_SCHOLAR_API_KEY", "")
BING_SEARCH_API_KEY = os.getenv("BING_SEARCH_API_KEY", "")

# LLM settings
DEFAULT_LLM_MODEL = "gpt-4"
DEFAULT_EMBEDDING_MODEL = "text-embedding-ada-002"

# API settings
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "5000"))
API_DEBUG = os.getenv("API_DEBUG", "False").lower() == "true"

# Vector database settings
VECTOR_DB_DIR = STORAGE_DIR / "vector_db"
VECTOR_DB_DIR.mkdir(exist_ok=True)

# Logging settings
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

# Task settings
MAX_CONCURRENT_TASKS = int(os.getenv("MAX_CONCURRENT_TASKS", "5"))
TASK_TIMEOUT_SECONDS = int(os.getenv("TASK_TIMEOUT_SECONDS", "3600"))  # 1 hour default

# Web interface settings
WEB_INTERFACE_ENABLED = os.getenv("WEB_INTERFACE_ENABLED", "True").lower() == "true"
