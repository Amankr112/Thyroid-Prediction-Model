# 🦋 HypoDetect — Hypothyroid Prediction System

Production-ready, end-to-end hypothyroid classification system
using 5 ML models with explainability, a Streamlit UI, and a FastAPI endpoint.

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train all models (≈ 3–5 min)
python train.py

# 3. Launch Streamlit app
streamlit run app.py

# 4. (Optional) Launch FastAPI endpoint
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
# Swagger UI: http://localhost:8000/docs
```

---

## Docker

```bash
# Build
docker build -t hypodetect .

# Run Streamlit (train first on host, mount artifacts)
docker run -p 8501:8501 -v $(pwd)/artifacts:/app/artifacts hypodetect

# Run FastAPI
docker run -p 8000:8000 -v $(pwd)/artifacts:/app/artifacts hypodetect \
  uvicorn api:app --host 0.0.0.0 --port 8000
```

---

## Project Structure

```
hypothyroid_system/
├── preprocessing.py        # Data cleaning, encoding, imputation
├── feature_engineering.py  # Medical feature creation
├── train.py                # Train 5 models + evaluation + SHAP
├── app.py                  # Streamlit web app
├── api.py                  # FastAPI REST endpoint
├── requirements.txt
├── Dockerfile
├── artifacts/              # (created by train.py)
│   ├── best_model.pkl / best_model.keras
│   ├── preprocess_artifacts.pkl
│   ├── scaler.pkl
│   ├── model_comparison.csv
│   ├── confusion_matrix.png
│   ├── rf_feature_importance.png
│   ├── xgb_feature_importance.png
│   └── shap_summary.png
└── hypothyroid_classification.csv
```

---

## Models Trained

| # | Model               | Notes                             |
|---|---------------------|-----------------------------------|
| 1 | Logistic Regression | Baseline, scaled features         |
| 2 | Random Forest       | 300 trees, balanced class weight  |
| 3 | XGBoost             | scale_pos_weight, 300 estimators  |
| 4 | SVM (RBF kernel)    | Balanced class weight             |
| 5 | Deep Learning (DL)  | 3 hidden layers, BN, Dropout, ES  |

**Best model selection**: 0.5 × ROC-AUC + 0.5 × Recall  
(Recall is critical — missing a hypothyroid case is worse than a false alarm.)

---

## API Endpoints

| Method | Path             | Description                     |
|--------|------------------|---------------------------------|
| GET    | /health          | Health check                    |
| POST   | /predict         | Single patient prediction       |
| POST   | /predict/batch   | Batch predictions (JSON array)  |
| GET    | /model/info      | Model metadata + comparison     |

---

## Key Features

- **Preprocessing**: KNN imputation, OneHotEncoding, boolean t/f conversion
- **Feature Engineering**: Hormone ratios, clinical interaction flags, age groups
- **Class Imbalance**: SMOTE oversampling on training set
- **Explainability**: SHAP values per prediction + global summary plot
- **Streamlit App**: Single patient form + batch CSV upload + model insights
- **FastAPI**: Swagger-documented REST API
- **Dockerized**: Single image for both UI and API
