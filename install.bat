@echo off
echo Creating venv...
python -m venv venv
call venv\Scripts\activate.bat
echo Installing dependencies...
pip install -r requirements.txt
echo.
echo Done! Copy .env.example to .env and add your OpenAI key.
echo Then fill in knowledge_base\personal\ files and run: python ingest.py
pause
