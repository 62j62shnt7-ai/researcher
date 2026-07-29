#!/bin/bash
cd "$(dirname "$0")"

# --- pick the best Python: OCR libraries need 3.10-3.13, so prefer those over 3.14+ ---
PY=""
for cand in python3.13 python3.12 python3.11 python3.10; do
    if command -v "$cand" >/dev/null 2>&1; then PY="$cand"; break; fi
done
if [ -z "$PY" ]; then
    if command -v python3 >/dev/null 2>&1; then
        PY=python3
    else
        echo "ERROR: python3 is not installed."
        echo "Install it from https://www.python.org/downloads/ (3.10-3.13 recommended), then run this script again."
        exit 1
    fi
fi
if ! "$PY" -c 'import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)'; then
    echo "ERROR: $($PY --version) found, but 3.10 or newer is required."
    exit 1
fi
echo "Using $($PY --version)"

# --- detect a venv copied from another OS (Windows uses Scripts\, not bin/) and rebuild ---
if [ -d .venv ] && [ ! -f .venv/bin/activate ]; then
    echo "Found a .venv from another OS - rebuilding for Mac..."
    rm -rf .venv
fi

# --- create venv + install deps on first run ---
if [ ! -d .venv ]; then
    echo "First run: creating virtual environment (this takes a few minutes)..."
    "$PY" -m venv .venv || exit 1
    source .venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt || { echo "Dependency install failed - see errors above."; exit 1; }
    echo "Installing optional OCR extras (for scanned PDFs)..."
    if ! pip install -r requirements-ocr.txt; then
        pip install pypdfium2 pillow rapidocr || \
        echo "NOTE: OCR extras could not be installed (your Python version may be too new). The app works fine without them - scanned PDF pages will be skipped. For OCR, install Python 3.12 or 3.13 and delete the .venv folder."
    fi
else
    source .venv/bin/activate
fi

python run.py
