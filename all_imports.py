import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import seaborn as sns
from dotenv import load_dotenv
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split

load_dotenv()

BASE_DIR = Path(__file__).parent
CSV_PATH = BASE_DIR / "datasets" / "heart_disease_uci.csv"
