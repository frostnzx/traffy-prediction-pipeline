import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

BASE_DIR = Path("/opt/airflow")
DATA_DIR = BASE_DIR / "data"
EXTERNAL_DIR = DATA_DIR / "external"

HOSPITAL_URL = "https://www.thaihealth.co.th/en/list-of-network-hospital-bangkok-area/"
WEATHER_URL = "https://archive-api.open-meteo.com/v1/era5"

BANGKOK_LAT = 13.75
BANGKOK_LON = 100.50
START_DATE = "2023-01-01"
END_DATE = "2023-12-31"


def _ensure_dirs() -> None:
    EXTERNAL_DIR.mkdir(parents=True, exist_ok=True)


def _scrape_bangkok_hospitals(url: str = HOSPITAL_URL) -> pd.DataFrame:
    logger.info("Requesting hospital page: %s", url)
    resp = requests.get(url, timeout=15)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")
    table = soup.find("table")
    if table is None:
        raise RuntimeError("Cannot find <table> on hospital page.")

    rows = table.find_all("tr")
    records: List[Dict[str, str]] = []
    for tr in rows[1:]:
        cells = [td.get_text(strip=True) for td in tr.find_all("td")]
        if not cells:
            continue

        def safe(i: int) -> str:
            return cells[i] if i < len(cells) else ""

        record = {
            "hospital_th": safe(1),
            "district_th": safe(2),
            "tel_th": safe(3),
            "hospital_en": safe(4),
            "district_en": safe(5),
            "tel_en": safe(6),
        }
        records.append(record)

    df = pd.DataFrame(records)
    logger.info("Parsed %d hospital rows", len(df))
    return df


def _download_weather(
    start_date: str = START_DATE,
    end_date: str = END_DATE,
    lat: float = BANGKOK_LAT,
    lon: float = BANGKOK_LON,
) -> pd.DataFrame:
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ["temperature_2m", "precipitation", "wind_speed_10m"],
        "timezone": "Asia/Bangkok",
    }
    logger.info("Requesting hourly weather from Open-Meteo (%s to %s)", start_date, end_date)
    resp = requests.get(WEATHER_URL, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    hourly = data.get("hourly")
    if hourly is None:
        raise RuntimeError("Weather response missing 'hourly' field")
    df_weather = pd.DataFrame(hourly)
    logger.info("Weather rows: %d", len(df_weather))
    return df_weather


def run(hospital_url: Optional[str] = None) -> None:
    """Scrape hospital list and download weather data, saving to data/external."""
    _ensure_dirs()

    # Hospitals
    df_hosp = _scrape_bangkok_hospitals(hospital_url or HOSPITAL_URL)
    df_hosp["district_th"] = df_hosp["district_th"].str.strip()
    df_hosp["district_en"] = df_hosp["district_en"].str.strip()

    hospital_path = EXTERNAL_DIR / "bangkok_hospitals_network.csv"
    df_hosp.to_csv(hospital_path, index=False, encoding="utf-8-sig")
    logger.info("Saved hospital network data to %s", hospital_path)

    # Weather
    df_weather = _download_weather()
    df_weather.rename(columns={"time": "datetime"}, inplace=True)

    weather_path = EXTERNAL_DIR / "bangkok_hourly_weather_2023.csv"
    df_weather.to_csv(weather_path, index=False, encoding="utf-8-sig")
    logger.info("Saved hourly weather data to %s", weather_path)


if __name__ == "__main__":  # pragma: no cover
    run()
