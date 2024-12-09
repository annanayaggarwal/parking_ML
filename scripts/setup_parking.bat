@echo off
TITLE Parking System Setup
python -m venv venv
call venv\Scripts\activate
pip install tensorflow opencv-python numpy pandas scikit-learn streamlit plotly matplotlib seaborn tqdm
mkdir data output output\models logs
echo Download CNR-EXT dataset and extract to data\FULL_IMAGE_1000x750