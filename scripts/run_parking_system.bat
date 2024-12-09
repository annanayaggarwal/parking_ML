@echo off
TITLE Parking System Setup

:: Check Python installation
python --version > nul 2>&1
if errorlevel 1 (
    echo Python not found! Please install Python first.
    pause
    exit
)

:: Setup virtual environment
echo Creating virtual environment...
python -m venv venv
call venv\Scripts\activate

:: Install requirements
echo Installing dependencies...
pip install tensorflow opencv-python numpy pandas scikit-learn streamlit plotly matplotlib seaborn tqdm

:: Create directories
echo Creating directory structure...
mkdir data output output\models logs

:: Download dataset
echo Downloading dataset...
cd data
curl -LO https://github.com/fabiocarrara/deep-parking/releases/download/archive/CNR-EXT_FULL_IMAGE_1000x750.tar
tar -xf CNR-EXT_FULL_IMAGE_1000x750.tar
del CNR-EXT_FULL_IMAGE_1000x750.tar
cd ..

:: Start application
echo Starting Streamlit dashboard...
streamlit run app.py

pause