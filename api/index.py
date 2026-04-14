import sys
import os
from pathlib import Path

# Add the project root to sys.path
root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir))

# Import the FastAPI app from backend/server.py
from backend.server import app

# Vercel needs the 'app' variable to be exposed
# No further code needed as FastAPI app is imported
