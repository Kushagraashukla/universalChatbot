from pathlib import Path

from dotenv import load_dotenv

# Read the project's .env (if any) once, so CLI and API see the same settings. Real env vars win.
load_dotenv(Path(__file__).resolve().parents[1] / ".env")
