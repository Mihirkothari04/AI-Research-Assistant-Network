"""
Modified Flask server startup script to ensure correct imports
"""
import sys
import os
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).resolve().parent
sys.path.append(str(project_root))

# Import the Flask app
from src.api.api import app

# Run the app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
