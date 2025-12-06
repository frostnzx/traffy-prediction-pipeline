import logging
from pathlib import Path
from typing import Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

BASE_DIR = Path("/opt/airflow")
DATA_DIR = BASE_DIR / "data"
FEATURE_DIR = DATA_DIR / "features"
MODEL_DIR = BASE_DIR / "models"
PREDICTION_DIR = DATA_DIR / "predictions"

FEATURE_FILE = FEATURE_DIR / "traffy_features.csv"
MODEL_PATH = MODEL_DIR / "traffy_rf_model.joblib"
PREDICTIONS_PATH = PREDICTION_DIR / "traffy_with_predictions.csv"


def _require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")


def _load_features(path: Path) -> pd.DataFrame:
    _require_file(path)
    df = pd.read_csv(path)
    logger.info("Loaded features: %s rows, %s columns", df.shape[0], df.shape[1])
    return df


def _split_X_y(df: pd.DataFrame, target: str = "is_late") -> Tuple[pd.DataFrame, pd.Series]:
    if target not in df.columns:
        raise KeyError(f"Target column '{target}' not found in features file")
    X = df.drop(columns=[target])
    y = df[target]
    logger.info("Split features X shape %s, y shape %s", X.shape, y.shape)
    return X, y


def _build_pipeline(numeric_cols, categorical_cols) -> Pipeline:
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            # Keep sparse to avoid huge dense matrices; handle_unknown avoids errors
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        n_jobs=-1,
        random_state=42,
    )

    clf = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )
    # Ensure default (sparse) output to avoid pandas conversion and huge dense arrays
    clf.set_output(transform="default")
    return clf


def run(feature_file: Optional[Path] = None) -> Path:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTION_DIR.mkdir(parents=True, exist_ok=True)

    feature_path = feature_file or FEATURE_FILE
    df = _load_features(feature_path)

    X, y = _split_X_y(df)

    numeric_cols = X.select_dtypes(include=["int64", "float64", "bool"]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()

    clf = _build_pipeline(numeric_cols, categorical_cols)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf.fit(X_train, y_train)
    logger.info("Training complete")

    y_pred = clf.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=False)
    cm = confusion_matrix(y_test, y_pred)

    logger.info("Classification report:\n%s", report)
    logger.info("Confusion matrix:\n%s", cm)

    joblib.dump(clf, MODEL_PATH)
    logger.info("Model saved to %s", MODEL_PATH)

    df_out = df.copy()
    df_out["pred_is_late"] = clf.predict(X)
    df_out["pred_proba_late"] = clf.predict_proba(X)[:, 1]

    df_out.to_csv(PREDICTIONS_PATH, index=False, encoding="utf-8-sig")
    logger.info("Batch predictions saved to %s", PREDICTIONS_PATH)

    return MODEL_PATH


if __name__ == "__main__":  # pragma: no cover
    run()
