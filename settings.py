"""Local configuration shared by the desktop app, bot, and server."""
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env", override=False)
SERVER_URL = os.getenv("SERVER_URL", "http://127.0.0.1:8000").rstrip("/")


def get_bot_token():
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    if not token:
        raise RuntimeError("Set TELEGRAM_BOT_TOKEN in the local .env file before starting the server or bot.")
    return token
