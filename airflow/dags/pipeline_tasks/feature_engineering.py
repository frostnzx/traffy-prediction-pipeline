import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

BASE_DIR = Path(__file__).resolve().parents[3]
DATA_DIR = BASE_DIR / "data"
EXTERNAL_DIR = DATA_DIR / "external"
FEATURE_DIR = DATA_DIR / "features"

CLEAN_PATH = DATA_DIR / "bangkok_traffy_clean.csv"
HOSPITAL_PATH = EXTERNAL_DIR / "bangkok_hospitals_network.csv"
WEATHER_PATH = EXTERNAL_DIR / "bangkok_hourly_weather_2023.csv"
FEATURES_OUT = FEATURE_DIR / "traffy_features.csv"


def _require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")


def _normalize_district(value):
    if pd.isna(value):
        return None
    return str(value).replace("เขต", "").strip().lower()


def _load_inputs(
    clean_path: Path,
    hospital_path: Path,
    weather_path: Path,
):
    _require_file(clean_path)
    _require_file(hospital_path)
    _require_file(weather_path)

    df = pd.read_csv(clean_path)
    df_hosp = pd.read_csv(hospital_path)
    df_weather = pd.read_csv(weather_path)

    logger.info("Loaded clean traffy %s rows, hospital %s rows, weather %s rows", len(df), len(df_hosp), len(df_weather))
    return df, df_hosp, df_weather


def _merge_hospitals(df: pd.DataFrame, df_hosp: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df.columns = df.columns.str.strip()
    df["timestamp"] = pd.to_datetime(df["timestamp"].astype(str).str.strip(), errors="coerce")
    df = df.dropna(subset=["timestamp"]).copy()

    df["district"] = df["district"].map(_normalize_district)

    if "district_th" in df_hosp.columns:
        df_hosp["district_th"] = df_hosp["district_th"].map(_normalize_district)
    elif "district" in df_hosp.columns:
        df_hosp["district_th"] = df_hosp["district"].map(_normalize_district)
    else:
        raise RuntimeError("Cannot find district column in hospital dataset.")

    hosp_count = (
        df_hosp.groupby("district_th")
        .size()
        .reset_index(name="num_hospitals_in_district")
    )

    df = df.merge(
        hosp_count,
        how="left",
        left_on="district",
        right_on="district_th",
    )

    df["num_hospitals_in_district"] = df["num_hospitals_in_district"].fillna(0).astype(int)
    df = df.drop(columns=["district_th"], errors="ignore")
    return df


def _merge_weather(df: pd.DataFrame, df_weather: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df_weather.columns = df_weather.columns.str.strip()
    df_weather["datetime"] = pd.to_datetime(
        df_weather["datetime"].astype(str).str.strip(), errors="coerce"
    )

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])

    df_weather["datetime"] = df_weather["datetime"].dt.tz_localize(None)
    df["report_hour"] = df["timestamp"].dt.tz_localize(None).dt.floor("h")

    df = df.merge(
        df_weather,
        how="left",
        left_on="report_hour",
        right_on="datetime",
    )

    rename_map = {
        "precipitation": "rain_mm",
        "temperature_2m": "temperature",
        "wind_speed_10m": "wind_speed",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    df["rain_mm"] = df.get("rain_mm", 0)
    df["rain_mm"] = df["rain_mm"].fillna(0)
    df = df.sort_values("report_hour")

    df["is_rainy_hour"] = (df["rain_mm"] > 0.5).astype(int)

    rain_rolling = (
        df.groupby(df["report_hour"].dt.date)["rain_mm"]
        .rolling(window=3, min_periods=1)
        .sum()
        .reset_index(level=0, drop=True)
    )
    df["rain_last_3h"] = rain_rolling

    if "temperature" in df.columns:
        df["high_temperature"] = (df["temperature"] > 33).astype(int)
    else:
        df["temperature"] = np.nan
        df["high_temperature"] = 0

    if "wind_speed" not in df.columns:
        df["wind_speed"] = np.nan

    return df


def _select_features(df: pd.DataFrame) -> pd.DataFrame:
    feature_cols = [
        "type",
        "organization",
        "district",
        "lat",
        "lon",
        "star",
        "count_reopen",
        "hour",
        "dayofweek",
        "month",
        "year",
        "num_hospitals_in_district",
        "rain_mm",
        "is_rainy_hour",
        "rain_last_3h",
        "temperature",
        "high_temperature",
        "wind_speed",
        "is_late",
    ]
    existing = [c for c in feature_cols if c in df.columns]
    df_features = df[existing].copy()
    logger.info("Final feature table shape: %s rows, %s columns", df_features.shape[0], df_features.shape[1])
    return df_features


def run(
    clean_path: Optional[Path] = None,
    hospital_path: Optional[Path] = None,
    weather_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
) -> Path:
    FEATURE_DIR.mkdir(parents=True, exist_ok=True)

    c_path = clean_path or CLEAN_PATH
    h_path = hospital_path or HOSPITAL_PATH
    w_path = weather_path or WEATHER_PATH
    out_path = output_path or FEATURES_OUT

    df, df_hosp, df_weather = _load_inputs(c_path, h_path, w_path)
    df = _merge_hospitals(df, df_hosp)
    df = _merge_weather(df, df_weather)

    df_features = _select_features(df)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_features.to_csv(out_path, index=False, encoding="utf-8-sig")
    logger.info("Saved feature table to %s", out_path)
    return out_path


if __name__ == "__main__":  # pragma: no cover
    run()
