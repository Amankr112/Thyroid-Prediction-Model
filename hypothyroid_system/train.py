"""
train.py
--------
End-to-end training pipeline:
  1. Load & preprocess data
  2. Feature engineering
  3. Train 5 models (LR, RF, XGB, SVM, DL)
  4. Evaluate & compare
  5. Save best model + preprocessing artifacts
"""

import logging
import os
import sys
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay,
)
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from preprocessing import load_raw, preprocess
from feature_engineering import engineer_features

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

# Resolve CSV path: CLI arg → same folder → current working directory
_CSV_NAME = "hypothyroid_classification.csv"
if len(sys.argv) > 1:
    DATA_PATH = sys.argv[1]
elif os.path.exists(os.path.join(BASE_DIR, _CSV_NAME)):
    DATA_PATH = os.path.join(BASE_DIR, _CSV_NAME)
elif os.path.exists(os.path.join(os.getcwd(), _CSV_NAME)):
    DATA_PATH = os.path.join(os.getcwd(), _CSV_NAME)
else:
    print(
        f"\n❌  ERROR: Cannot find '{_CSV_NAME}'.\n\n"
        f"   Fix options (pick ONE):\n"
        f"   1. Copy the CSV into the same folder as train.py:\n"
        f"      {BASE_DIR}\\\n\n"
        f"   2. Pass the full path as an argument:\n"
        f"      python train.py C:\\path\\to\\{_CSV_NAME}\n"
    )
    sys.exit(1)


# ── Helpers ───────────────────────────────────────────────────────────────────
def evaluate(model, X_test, y_test, name: str, is_keras=False) -> dict:
    if is_keras:
        y_prob = model.predict(X_test).ravel()
        y_pred = (y_prob > 0.5).astype(int)
    elif hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = model.predict(X_test)
    else:  # SVM with decision_function
        y_prob = model.decision_function(X_test)
        y_prob = (y_prob - y_prob.min()) / (y_prob.max() - y_prob.min() + 1e-9)
        y_pred = model.predict(X_test)

    return {
        "Model": name,
        "Accuracy": round(accuracy_score(y_test, y_pred), 4),
        "Precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "Recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
        "F1": round(f1_score(y_test, y_pred, zero_division=0), 4),
        "ROC-AUC": round(roc_auc_score(y_test, y_prob), 4),
        "_y_pred": y_pred,
        "_y_prob": y_prob,
    }


def build_dl_model(input_dim: int) -> keras.Model:
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(128, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(64, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        layers.Dense(32, activation="relu"),
        layers.Dropout(0.1),
        layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy", keras.metrics.AUC(name="auc")],
    )
    return model


# ── Main pipeline ─────────────────────────────────────────────────────────────
def main():
    # 1. Load + preprocess ────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("HYPOTHYROID PREDICTION — TRAINING PIPELINE")
    logger.info("=" * 60)

    raw = load_raw(DATA_PATH)
    X, y, preprocess_artifacts = preprocess(raw, fit=True)
    X = engineer_features(X)

    logger.info("Final feature matrix: %s", X.shape)
    logger.info("Class distribution — Hypothyroid=1: %d  Normal=0: %d",
                y.sum(), (y == 0).sum())

    # 2. Train / test split ───────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )

    # 3. SMOTE on training set ────────────────────────────────────────────────
    sm = SMOTE(random_state=42)
    X_train_sm, y_train_sm = sm.fit_resample(X_train, y_train)
    logger.info("After SMOTE — train size: %d  (Hypo=%d, Normal=%d)",
                len(y_train_sm), y_train_sm.sum(), (y_train_sm == 0).sum())

    # 4. Scale (needed for LR and SVM) ────────────────────────────────────────
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train_sm)
    X_test_sc = scaler.transform(X_test)

    # 5. Train models ─────────────────────────────────────────────────────────
    logger.info("Training models …")

    # Logistic Regression
    lr = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    lr.fit(X_train_sc, y_train_sm)

    # Random Forest
    rf = RandomForestClassifier(
        n_estimators=300, class_weight="balanced",
        max_depth=None, random_state=42, n_jobs=-1
    )
    rf.fit(X_train_sm, y_train_sm)

    # XGBoost
    scale_pos = (y_train_sm == 0).sum() / (y_train_sm == 1).sum()
    xgb = XGBClassifier(
        n_estimators=300, learning_rate=0.05, max_depth=6,
        scale_pos_weight=scale_pos, use_label_encoder=False,
        eval_metric="logloss", random_state=42, n_jobs=-1,
    )
    xgb.fit(X_train_sm, y_train_sm,
            eval_set=[(X_test, y_test)], verbose=False)

    # SVM
    svm = SVC(kernel="rbf", class_weight="balanced", probability=False, random_state=42)
    svm.fit(X_train_sc, y_train_sm)

    # Deep Learning
    dl = build_dl_model(X_train_sm.shape[1])
    cb = [
        keras.callbacks.EarlyStopping(monitor="val_auc", patience=10,
                                       restore_best_weights=True, mode="max"),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5),
    ]
    dl.fit(
        X_train_sc, y_train_sm,
        validation_split=0.15,
        epochs=100, batch_size=32,
        callbacks=cb, verbose=0,
    )

    logger.info("All models trained.")

    # 6. Evaluate ─────────────────────────────────────────────────────────────
    results = [
        evaluate(lr,  X_test_sc, y_test, "Logistic Regression"),
        evaluate(rf,  X_test,    y_test, "Random Forest"),
        evaluate(xgb, X_test,    y_test, "XGBoost"),
        evaluate(svm, X_test_sc, y_test, "SVM"),
        evaluate(dl,  X_test_sc, y_test, "Deep Learning", is_keras=True),
    ]

    # Build comparison table
    display_keys = ["Model", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
    comparison = pd.DataFrame([{k: r[k] for k in display_keys} for r in results])
    comparison.set_index("Model", inplace=True)

    logger.info("\n\n%s\n", "=" * 60)
    logger.info("MODEL COMPARISON TABLE")
    logger.info("\n%s\n", comparison.to_string())

    # 7. Select best model (ROC-AUC + Recall equally weighted) ───────────────
    comparison["score"] = 0.5 * comparison["ROC-AUC"] + 0.5 * comparison["Recall"]
    best_name = comparison["score"].idxmax()
    best_idx  = comparison.index.tolist().index(best_name)
    best_result = results[best_idx]
    logger.info("Best model: %s (score=%.4f)", best_name, comparison.loc[best_name, "score"])

    # 8. Confusion matrix for best model ──────────────────────────────────────
    cm = confusion_matrix(y_test, best_result["_y_pred"])
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(cm, display_labels=["Normal", "Hypothyroid"]).plot(ax=ax, cmap="Blues")
    ax.set_title(f"Confusion Matrix — {best_name}")
    cm_path = os.path.join(ARTIFACTS_DIR, "confusion_matrix.png")
    fig.savefig(cm_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    logger.info("Confusion matrix saved to %s", cm_path)

    # 9. Feature importance (RF + XGB) ────────────────────────────────────────
    feature_names = X.columns.tolist()

    # Random Forest importance
    fi_rf = pd.Series(rf.feature_importances_, index=feature_names).nlargest(20)
    fig, ax = plt.subplots(figsize=(8, 6))
    fi_rf.sort_values().plot(kind="barh", ax=ax, color="steelblue")
    ax.set_title("Random Forest — Top 20 Feature Importances")
    ax.set_xlabel("Importance")
    fig.tight_layout()
    fig.savefig(os.path.join(ARTIFACTS_DIR, "rf_feature_importance.png"), dpi=150)
    plt.close(fig)

    # XGBoost importance
    fi_xgb = pd.Series(xgb.feature_importances_, index=feature_names).nlargest(20)
    fig, ax = plt.subplots(figsize=(8, 6))
    fi_xgb.sort_values().plot(kind="barh", ax=ax, color="darkorange")
    ax.set_title("XGBoost — Top 20 Feature Importances")
    ax.set_xlabel("Importance")
    fig.tight_layout()
    fig.savefig(os.path.join(ARTIFACTS_DIR, "xgb_feature_importance.png"), dpi=150)
    plt.close(fig)

    # 10. SHAP values (XGBoost) ───────────────────────────────────────────────
    logger.info("Computing SHAP values …")
    explainer = shap.TreeExplainer(xgb)
    shap_vals = explainer.shap_values(X_test.iloc[:200])
    fig = plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_vals, X_test.iloc[:200], feature_names=feature_names,
                      show=False, max_display=20)
    plt.title("SHAP Summary Plot (XGBoost, first 200 test samples)")
    plt.tight_layout()
    fig.savefig(os.path.join(ARTIFACTS_DIR, "shap_summary.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    # Save explainer for app
    joblib.dump(explainer, os.path.join(ARTIFACTS_DIR, "shap_explainer.pkl"))
    logger.info("SHAP plots saved.")

    # 11. Save artifacts ──────────────────────────────────────────────────────
    joblib.dump(preprocess_artifacts, os.path.join(ARTIFACTS_DIR, "preprocess_artifacts.pkl"))
    joblib.dump(scaler, os.path.join(ARTIFACTS_DIR, "scaler.pkl"))
    joblib.dump(rf,  os.path.join(ARTIFACTS_DIR, "rf_model.pkl"))
    joblib.dump(xgb, os.path.join(ARTIFACTS_DIR, "xgb_model.pkl"))
    joblib.dump(lr,  os.path.join(ARTIFACTS_DIR, "lr_model.pkl"))
    joblib.dump(svm, os.path.join(ARTIFACTS_DIR, "svm_model.pkl"))
    dl.save(os.path.join(ARTIFACTS_DIR, "dl_model.keras"))

    # Save best model reference
    model_map = {
        "Logistic Regression": lr,
        "Random Forest": rf,
        "XGBoost": xgb,
        "SVM": svm,
    }
    if best_name in model_map:
        joblib.dump(model_map[best_name], os.path.join(ARTIFACTS_DIR, "best_model.pkl"))
        joblib.dump({"name": best_name, "type": "sklearn"},
                    os.path.join(ARTIFACTS_DIR, "best_model_meta.pkl"))
    else:
        dl.save(os.path.join(ARTIFACTS_DIR, "best_model.keras"))
        joblib.dump({"name": best_name, "type": "keras"},
                    os.path.join(ARTIFACTS_DIR, "best_model_meta.pkl"))

    # Save comparison results
    comparison.to_csv(os.path.join(ARTIFACTS_DIR, "model_comparison.csv"))
    logger.info("All artifacts saved to %s/", ARTIFACTS_DIR)

    logger.info("\n%s", "=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info("Best model : %s", best_name)
    logger.info("ROC-AUC   : %.4f", comparison.loc[best_name, "ROC-AUC"])
    logger.info("Recall    : %.4f", comparison.loc[best_name, "Recall"])
    logger.info("%s\n", "=" * 60)

    return comparison, best_name


if __name__ == "__main__":
    main()
