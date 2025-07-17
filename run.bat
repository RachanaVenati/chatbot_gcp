@echo off
REM Create virtual environment
python -m venv .venv

REM Activate virtual environment
call .venv\Scripts\activate

REM Upgrade pip
python -m pip install --upgrade pip

REM Install dependencies
pip install streamlit openai tiktoken weaviate-client python-dotenv

REM Run the Streamlit app
streamlit run UIbot.py
