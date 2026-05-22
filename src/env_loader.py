from pathlib import Path
from dotenv import load_dotenv

# Resolve project root (one level above src/) and load .env once.
# Safe to import multiple times — load_dotenv is idempotent.
load_dotenv(Path(__file__).resolve().parent.parent / ".env")
