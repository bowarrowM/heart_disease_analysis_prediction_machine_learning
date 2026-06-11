# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Docker (recommended)
docker compose up --build        # Starts app (port 8501) + PostgreSQL (port 5432)
docker compose down              # Stop and remove containers
docker compose down -v           # Also remove the postgres_data volume

# Local development
pip install -r requirements.txt
streamlit run app.py             # Launch app (pre-built .pkl files are committed)

# ML pipeline — only needed to rebuild the model
python data_analysis.py          # Prints EDA summary to stdout
python plots_visualization.py    # Saves EDA plots to images/eda/
python model.py                  # Trains model, saves heart_disease_model.pkl + model_columns.pkl
```

## Architecture

**Entry points**

| File | Role |
|------|------|
| `all_imports.py` | Shared imports + `BASE_DIR` / `CSV_PATH` path constants. No data loading. |
| `app.py` | Streamlit UI: loads `.pkl` artifacts via `@st.cache_resource`, runs inference. |
| `model.py` | Full ML pipeline (functions + `__main__` guard): preprocess → train → evaluate → save `.pkl`. |
| `data_analysis.py` | EDA script (`__main__` guard): prints summary stats, null counts, duplicates. |
| `plots_visualization.py` | EDA visualization script (`__main__` guard): saves countplots to `images/eda/`. |
| `sqlDB_connection.py` | Optional PostgreSQL module: `upload_dataframe()` and `query()`. Reads `DB_URL` from `.env`. |

**Data flow**

`datasets/heart_disease_uci.csv` → `model.py` (preprocess + train) → `.pkl` files → `app.py` (inference).

`app.py` only loads the `.pkl` files — it never reads the CSV.

**Preprocessing decisions (in `model.py::preprocess`)**

- Drops `ca` (~66% missing) and `id` (no predictive value).
- Numerical nulls filled with column median; categorical with column mode.
- One-hot encodes all categorical columns via `pd.get_dummies(drop_first=True)`.
- Target `num` binarized: `0` = no disease, `1` = any disease (severity levels 1–4 collapsed).
- `model_columns.pkl` stores the exact `Index` from training; `app.py` builds a zero-initialized row and sets only the relevant columns to avoid shape mismatches.

**App input → model column mapping**

The form collects 8 inputs. Numerical fields set directly as `float`. Categorical fields map to one-hot columns:

| Input | Column |
|-------|--------|
| Sex = Male | `sex_Male` |
| Exercise Angina = Yes | `exang_True` |
| Chest Pain = Atypical Angina | `cp_atypical angina` |
| Chest Pain = Non-anginal Pain | `cp_non-anginal pain` |
| Chest Pain = Asymptomatic | `cp_asymptomatic` |

Typical Angina sets no column (it is the `drop_first` baseline).

## Environment

Copy `.env.example` → `.env`. Credentials match `docker-compose.yml` defaults:

```
DB_URL=postgresql://heart:heart@localhost:5432/heart_disease
```

When running inside Docker Compose the hostname is `db` (the service name), not `localhost` — the compose file sets `DB_URL` automatically for the `app` service, so the `.env` file is only needed for local development runs against the Dockerized database.

## Model performance

- Accuracy: 0.86 | F1: 0.858 (binary, 80/20 split, stratified)
- Top features: `age`, `thalch`, `oldpeak`, `chol`, `trestbps`, `exang_True`, `restecg_normal`, `thal_normal`, `cp_atypical angina`
