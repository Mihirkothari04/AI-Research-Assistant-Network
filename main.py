"""
Main entry point for the Research Assistant Network.
"""
import os
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.api.api import run_api_server
from src.config.config import STORAGE_DIR

def main():
    """Main entry point for the application."""
    # Create storage directories if they don't exist
    os.makedirs(STORAGE_DIR, exist_ok=True)
    os.makedirs(os.path.join(STORAGE_DIR, "research_tasks"), exist_ok=True)
    
    # Run the API server
    run_api_server()

if __name__ == "__main__":
    main()
