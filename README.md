# Heart Disease Analysis & Prediction Project with Random Forest

-Note: This PROJECT and README are all to be updated. Feedback and suggestions are welcomed and appreciated! Thank you.

This is a Python-based data analysis and machine learning project using the UCI Heart Disease Dataset (from Kaggle).
The project covers data preprocessing, exploratory data analysis (EDA), visualization, SQL database integration (PostgreSQL), and predictive modeling.
Comments and reasoning are included throughout the code for clarity.

<img src="images/appscreenshot.png" width="600">


## Project Overview

The purpose of this project is to analyze heart disease data, explore correlations and distributions, and build a machine learning model to predict the presence of heart disease.
Key aspects of the project include:

Exploratory Data Analysis (EDA): summary statistics, missing values, duplicates

Data Visualization: categorical and numerical feature distributions

Data Cleaning & Preprocessing: handling nulls, encoding categorical variables

Database Integration: optional PostgreSQL connection for storing/querying the dataset. Currently commented out in my version but modify as wanted.

Model Training & Evaluation: predictive modeling with performance metrics and confusion matrix

## Model Results and Project Conclusion

### Note:
   ""
   Missing values in multiple feature categories: 
   - ca, 611, (~66%) | slope, 309, (~33%) | thal, 486, (~53%) | trestbps, chol, thalch, oldpeak, exang : under 10%
   - Solutions: ca was dropped, mode and median techniques were used to handle other vals with high missing values.
  
   ""

### model_results = 
- Train shape: (736, 20)
- Test shape: (184, 20)
- Accuracy: 0.86 binary class
- F1 Score: 0.858 binary

### Classification Report: For all  levels of heart disease stated in the dataset (0 : no disease, 1: disease)
   -0: precision 0.86, recall 0.82, f1 0.84
   -1: precision 0.86, recall 0.89, f1 0.88

  
### Top 10 Features by Importance:
  age, thalch, oldpeak, chol, trestbpsy, exang_True, restecg_normal, thal_normal, cp_atypical angina

## Project Structure
├── .env                  # Environment variables / database credentials (gitignored)

├── all_imports.py        # Library imports and dataset loading

├── app.py                # Streamlit frontend interface for users

├── data_analysis.py      # EDA: info, shape, nulls, summary stats

├── plots_visualization.py# Visualizations for EDA

├── sqlDB_connection.py   # PostgreSQL database connection (optional)

├── model.py              # Preprocessing, training, evaluation, confusion matrix

├── heart_disease_model.pkl

├── model_columns.pkl

└── requirements.txt      # Required Python packages


## Setup Instructions

### Clone the repository
from terminal/powershell (bash):

git clone repo-url

cd heart-disease-project


### Install dependencies

* pip install -r requirements.txt 

* or install with the package manager of your choice.
* (All the libraries used: pandas numpy matplotlib seaborn scikit-learn psycopg2-binary python-dotenv sqlalchemy)


Open the project in your preferred IDE (VS Code, PyCharm, etc.)

Create your .env file (not included in the repo for security) and add your credentials (listed below):

### PostgreSQL database URL
DB_URL="postgresql://username:password@localhost:5432/db_name"

### Local dataset path (Kaggle CSV)
DATA_PATH="path/to/heart_disease_data.csv"


Alternatively, you can add the paths/credentials directly into your Python scripts if preferred.

After completing the setup, open your terminal and launch the app to access the user interface:

""" 
bash
streamlit run app.py
"""

If any warning messages appear or Streamlit asks for an email, simply press Enter to skip.

## Notes

All scripts contain comments explaining code logic and reasoning.

SQL integration is optional; you can run the project fully with CSV data.

Visualizations and model evaluation results are generated automatically via the scripts.
