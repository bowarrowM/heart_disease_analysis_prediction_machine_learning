# Heart Disease Risk Prediction

A machine learning web app that predicts the presence of heart disease from patient vitals. Built with Python, scikit-learn (Random Forest), and Streamlit. Uses the UCI Heart Disease dataset (Cleveland, Hungarian, Switzerland, VA Long Beach).

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://heartdiseaseanalysispredictionmachinelearning-ctwydqk4zj5ahx5y.streamlit.app/)

<img src="images/appscreenshot.png" width="700">

---

## Model Performance

Binary classification: 0 = no disease, 1 = disease present (severity levels 1-4 collapsed).

| Metric | Score |
|--------|-------|
| Accuracy | 0.86 |
| F1 Score (weighted) | 0.858 |
| Train / Test split | 736 / 184 (80/20, stratified) |

**Classification report**

| Class | Precision | Recall | F1 |
|-------|-----------|--------|----|
| 0 - No disease | 0.86 | 0.82 | 0.84 |
| 1 - Disease | 0.86 | 0.89 | 0.88 |

**Top 10 features by importance:** age, thalch, oldpeak, chol, trestbps, exang_True, restecg_normal, thal_normal, cp_atypical angina

**Missing value handling:** `ca` (~66% missing) was dropped entirely. Remaining missing values were imputed with column median (numerical) or mode (categorical).

---

## Setup

### Docker (recommended)

```bash
git clone <repo-url>
cd heart_disease_analysis_prediction_machine_learning
docker compose up --build
```

Open http://localhost:8501. The pre-built model files are committed so the app starts immediately without retraining.

### Local

```bash
git clone <repo-url>
cd heart_disease_analysis_prediction_machine_learning
pip install -r requirements.txt
streamlit run app.py
```

### PostgreSQL (optional)

The database integration is optional. The app runs fully without it using only the CSV and `.pkl` files.

Copy `.env.example` to `.env`:

```bash
cp .env.example .env
```

With Docker the database is already included in `docker-compose.yml`. For local development, start the container and connect via localhost:

```
DB_URL=postgresql://heart:heart@localhost:5432/heart_disease
```

Then use `sqlDB_connection.py` to upload the dataset or run queries:

```python
from sqlDB_connection import upload_dataframe, query
import pandas as pd

df = pd.read_csv("datasets/heart_disease_uci.csv")
upload_dataframe(df)
```

---

## Rebuilding the Model

The `.pkl` files are committed and ready to use. To retrain from scratch:

```bash
python data_analysis.py        # EDA summary to stdout
python plots_visualization.py  # saves plots to images/eda/
python model.py                # trains and saves heart_disease_model.pkl + model_columns.pkl
```

---

## Project Structure

```
.
+-- app.py                    # Streamlit UI
+-- model.py                  # Preprocessing, training, evaluation
+-- data_analysis.py          # EDA script
+-- plots_visualization.py    # EDA visualizations (saves to images/eda/)
+-- all_imports.py            # Shared imports and path constants
+-- sqlDB_connection.py       # Optional PostgreSQL integration
+-- Dockerfile
+-- docker-compose.yml
+-- requirements.txt
+-- .env.example
+-- datasets/
|   +-- heart_disease_uci.csv
+-- images/
|   +-- appscreenshot.png
|   +-- confmatx.png
+-- heart_disease_model.pkl
+-- model_columns.pkl
```
