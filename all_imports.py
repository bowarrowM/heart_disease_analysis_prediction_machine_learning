import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import psycopg2
import scipy
from dotenv import load_dotenv
import os
import joblib
import streamlit as st
# dev note: I code & retrieve files from .env or such files, but you can also write the path on any file you want
# dev note: I also prefer to code different functionalities (plotting, analysis, libraries etc.) in separate files for better organization for larger projects.
# dev not: But! All can be done in a single file as well. Totally up to you. Have fun, do it your own style ^^.


# KAGGLE dataset | UCI Heart Disease
load_dotenv()
data_path = os.getenv("DATA_PATH")
df = pd.read_csv(data_path)


