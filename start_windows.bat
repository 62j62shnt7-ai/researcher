@echo off
cd /d "%~dp0"
if exist .venv if not exist .venv\Scripts\activate.bat (
    echo Found a .venv from another OS - rebuilding for Windows...
    rmdir /s /q .venv
)
if not exist .venv (
    echo First run: creating virtual environment...
    python -m venv .venv
    call .venv\Scripts\activate.bat
    pip install --upgrade pip
    pip install -r requirements.txt
    echo Installing optional OCR extras...
    pip install -r requirements-ocr.txt
    if errorlevel 1 (
        pip install pypdfium2 pillow rapidocr
        if errorlevel 1 echo NOTE: OCR extras could not be installed. The app works without them - scanned PDF pages will be skipped. Use Python 3.12/3.13 for OCR support.
    )
) else (
    call .venv\Scripts\activate.bat
)
python run.py
pause
