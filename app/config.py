"""
Configuration loader.
Reads settings from .env file and makes them available to the rest of the app.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# XAI API settings
XAI_API_KEY = os.getenv("XAI_API_KEY")
XAI_BASE_URL = "https://api.x.ai/v1"

# Dashboard credentials
DASHBOARD_USERNAME = os.getenv("DASHBOARD_USERNAME", "admin")
DASHBOARD_PASSWORD = os.getenv("DASHBOARD_PASSWORD", "changeme123")

# Database path — each app gets its own SQLite file inside this folder
DATABASE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "processed")
