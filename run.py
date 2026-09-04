#!/usr/bin/env python3
"""Start Researcher. Usage: python run.py [--port 8600]"""
import os
import sys
from pathlib import Path

# Auto-detect local virtualenv if current interpreter lacks required packages
BASE_DIR = Path(__file__).resolve().parent
VENV_DIR = BASE_DIR / ".venv"
VENV_PYTHON = VENV_DIR / ("Scripts" if os.name == "nt" else "bin") / ("python.exe" if os.name == "nt" else "python")

if VENV_PYTHON.exists() and sys.executable != str(VENV_PYTHON.resolve()):
    try:
        import uvicorn
        import fastapi
    except ImportError:
        # Re-execute seamlessly within the project's .venv
        os.execv(str(VENV_PYTHON), [str(VENV_PYTHON)] + sys.argv)

import argparse
import threading
import webbrowser

import uvicorn

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start Researcher Desktop")
    parser.add_argument("--port", type=int, default=8600)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    if not args.no_browser:
        threading.Timer(1.5, lambda: webbrowser.open(f"http://{args.host}:{args.port}")).start()

    uvicorn.run("app.main:app", host=args.host, port=args.port, log_level="info")
