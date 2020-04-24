import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))  # This is the Project Root
DATA_DIR = os.path.join(ROOT_DIR)  # This is the data directory
RAW_DATA_DIR = os.path.join(DATA_DIR, 'PlantDoc-Dataset')  # This is the raw data directory
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'cropped')  # This is the processed data directory
