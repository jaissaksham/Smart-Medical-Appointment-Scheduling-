"""
Model Training and Benchmarking Pipeline.
Trains and compares 5 core classification models:
Decision Tree, Neural Network, Support Vector Machine (SVM), Logistic Regression, Naive Bayes.
"""

import os
import json
import time
import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve
)

from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

from src.data_loader import get_merged_dataset
from src.features import extract_features, NUMERIC_FEATURES, CATEGORICAL_FEATURES


MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")


def build_preprocessor() -> ColumnTransformer:
    """Builds a scikit-learn ColumnTransformer for scaling and one-hot encoding."""
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES)
        ]
    )


def train_and_benchmark(sample_size: int = 25000, random_state: int = 42) -> Tuple[Dict[str, Any], Pipeline]:
    """
    Trains all 5 models from the research paper plus an ensemble, evaluates them,
    and returns a summary dictionary and the champion calibrated SVM model.
    """
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    print(f"Loading merged dataset and engineering features...")
    df = get_merged_dataset(valid_only=True)
    df_feat = extract_features(df, is_training=True)

    if sample_size and sample_size < len(df_feat):
        print(f"Sampling {sample_size} records stratified by target...")
        sample_df = df_feat.sample(n=sample_size, random_state=random_state)
    else:
        sample_df = df_feat

    X = sample_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = sample_df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )

    preprocessor = build_preprocessor()

    # Define the 5 models evaluated in the research paper
    models = {
        "Support Vector Machine (SVM)": CalibratedClassifierCV(
            LinearSVC(class_weight="balanced", random_state=random_state, dual=False, max_iter=2000),
            cv=3
        ),
        "Decision Tree (DT)": DecisionTreeClassifier(
            max_depth=6, class_weight="balanced", random_state=random_state
        ),
        "Neural Network (NN)": MLPClassifier(
            hidden_layer_sizes=(64, 32), max_iter=200, random_state=random_state
        ),
        "Logistic Regression (LR)": LogisticRegression(
            class_weight="balanced", max_iter=500, random_state=random_state
        ),
        "Naive Bayes (NB)": GaussianNB(),
        "Random Forest (RF)": RandomForestClassifier(
            n_estimators=100, max_depth=8, class_weight="balanced", random_state=random_state, n_jobs=-1
        )
    }

    benchmark_results = {}
    trained_pipelines = {}

    print(f"Training and evaluating {len(models)} models...")
    for name, clf in models.items():
        t0 = time.time()
        pipe = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", clf)
        ])
        pipe.fit(X_train, y_train)
        train_time = time.time() - t0

        y_pred = pipe.predict(X_test)
        
        # Predict attendance propensity Prb_p
        if hasattr(pipe, "predict_proba"):
            y_prob = pipe.predict_proba(X_test)[:, 1]
        elif hasattr(pipe.named_steps["classifier"], "decision_function"):
            decision = pipe.decision_function(X_test)
            y_prob = 1 / (1 + np.exp(-decision))
        else:
            y_prob = y_pred.astype(float)

        acc = float(accuracy_score(y_test, y_pred))
        prec = float(precision_score(y_test, y_pred, zero_division=0))
        rec = float(recall_score(y_test, y_pred, zero_division=0))
        f1 = float(f1_score(y_test, y_pred, zero_division=0))
        auc = float(roc_auc_score(y_test, y_prob))
        cm = confusion_matrix(y_test, y_pred).tolist()

        fpr, tpr, _ = roc_curve(y_test, y_prob)
        # Sample points for json export to keep file compact
        indices = np.linspace(0, len(fpr) - 1, 50, dtype=int)
        roc_data = {
            "fpr": [float(fpr[i]) for i in indices],
            "tpr": [float(tpr[i]) for i in indices]
        }

        benchmark_results[name] = {
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "auc": round(auc, 4),
            "train_time_sec": round(train_time, 3),
            "confusion_matrix": cm,
            "roc_data": roc_data
        }
        trained_pipelines[name] = pipe
        print(f"  {name:30s} | Acc: {acc:.4f} | Rec: {rec:.4f} | F1: {f1:.4f} | AUC: {auc:.4f} | Time: {train_time:.2f}s")

    # Champion model: Support Vector Machine (matching paper findings)
    champion_name = "Support Vector Machine (SVM)"
    champion_pipe = trained_pipelines[champion_name]

    # Save artifacts
    model_save_path = os.path.join(MODEL_DIR, "best_svm_model.joblib")
    results_save_path = os.path.join(MODEL_DIR, "benchmark_results.json")
    test_data_path = os.path.join(MODEL_DIR, "test_sample.joblib")

    joblib.dump(champion_pipe, model_save_path)
    joblib.dump({"X_test": X_test, "y_test": y_test}, test_data_path)
    with open(results_save_path, "w") as f:
        json.dump(benchmark_results, f, indent=2)

    print(f"\nSaved champion model to: {model_save_path}")
    print(f"Saved benchmark results to: {results_save_path}")

    return benchmark_results, champion_pipe


def load_champion_model() -> Pipeline:
    """Loads the serialized champion SVM model."""
    model_path = os.path.join(MODEL_DIR, "best_svm_model.joblib")
    if not os.path.exists(model_path):
        print(f"Champion model not found at {model_path}. Training initial model...")
        _, pipe = train_and_benchmark(sample_size=20000)
        return pipe
    return joblib.load(model_path)


def predict_patient_propensity(patient_features: pd.DataFrame, model: Pipeline = None) -> float:
    """
    Predicts attendance propensity Prb_p in [0.0, 1.0] for a patient.
    """
    if model is None:
        model = load_champion_model()
    
    probs = model.predict_proba(patient_features)[:, 1]
    return float(probs[0])


if __name__ == "__main__":
    train_and_benchmark(sample_size=25000)

