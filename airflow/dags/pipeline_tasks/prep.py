import logging
from pathlib import Path
from typing import Optional

import pandas as pd
from sklearn import set_config

set_config(transform_output="pandas")

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

BASE_DIR = Path("/opt/airflow")
DATA_DIR = BASE_DIR / "data"
RAW_PATH = DATA_DIR / "bangkok_traffy.csv"
CLEAN_PATH = DATA_DIR / "bangkok_traffy_clean.csv"

TARGET_ROWS = 300_000
LATE_THRESHOLD_HOURS = 168  # 7 days


def _require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")


def _stratified_sample(df: pd.DataFrame, target_rows: int) -> pd.DataFrame:
    if len(df) == 0:
        raise ValueError("Input dataframe is empty; cannot sample.")
    sample_frac = min(target_rows / len(df), 1.0)
    logger.info("Applying stratified sampling with fraction %.4f", sample_frac)
    sampled_df = (
        df.groupby("month")
        .apply(lambda x: x.sample(frac=sample_frac, random_state=42))
        .reset_index(drop=True)
    )
    logger.info("Sampled %d rows from %d", len(sampled_df), len(df))
    return sampled_df


def _prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["timestamp"] = pd.to_datetime(
        df["timestamp"].astype(str).str.strip(), errors="coerce", utc=True
    )
    df = df.dropna(subset=["timestamp"])
    df["month"] = df["timestamp"].dt.to_period("M")

    df = _stratified_sample(df, TARGET_ROWS)

    df["district"] = df["district"].astype(str).str.strip()

    cols_to_drop = [
        "ticket_id",
        "photo",
        "photo_after",
        "address",
        "subdistrict",
        "province",
        "state",
    ]
    df = df.drop(columns=cols_to_drop, errors="ignore")

    coords_split = df["coords"].astype(str).str.split(",", expand=True)
    df["lat"] = pd.to_numeric(coords_split[0], errors="coerce")
    df["lon"] = pd.to_numeric(coords_split[1], errors="coerce")
    df = df.drop(columns=["coords"], errors="ignore")

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    df["hour"] = df["timestamp"].dt.hour
    df["dayofweek"] = df["timestamp"].dt.dayofweek
    df["month"] = df["timestamp"].dt.month
    df["year"] = df["timestamp"].dt.year

    df["last_activity"] = pd.to_datetime(
        df["last_activity"], errors="coerce", utc=True
    )
    df["resolution_time_hours"] = (
        df["last_activity"] - df["timestamp"]
    ).dt.total_seconds() / 3600
    df = df.drop(columns=["last_activity"], errors="ignore")

    df["is_late"] = (df["resolution_time_hours"] > LATE_THRESHOLD_HOURS).astype(int)

    df["star"] = df["star"].fillna(0)
    df["count_reopen"] = df["count_reopen"].fillna(0)
    df["organization"] = df["organization"].fillna("Unknown")

    return df


def run(raw_path: Optional[Path] = None, output_path: Optional[Path] = None) -> Path:
    """Clean raw Traffy data and write the cleaned CSV."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    src_path = raw_path or RAW_PATH
    dst_path = output_path or CLEAN_PATH

    _require_file(src_path)
    logger.info("Loading raw Traffy data from %s", src_path)
    df = pd.read_csv(src_path)

    df_clean = _prepare_dataframe(df)

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_csv(dst_path, index=False, encoding="utf-8-sig")
    logger.info("Saved cleaned data to %s", dst_path)
    return dst_path


if __name__ == "__main__":  # pragma: no cover
    run()
