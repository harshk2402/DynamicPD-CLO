import os

import pandas as pd
from dotenv import load_dotenv

from src.utils.connections import get_fred_client

load_dotenv()

START_DATE = "2000-01-01"
END_DATE = "2024-12-31"


def fetch_treasury_rates() -> pd.DataFrame:
    fred = get_fred_client()
    t3m = fred.get_series(
        "DGS3MO", observation_start=START_DATE, observation_end=END_DATE
    )
    t10y = fred.get_series(
        "DGS10", observation_start=START_DATE, observation_end=END_DATE
    )
    df = pd.DataFrame({"treasury_3m": t3m, "treasury_10y": t10y})
    df["term_spread"] = df["treasury_10y"] - df["treasury_3m"]
    df.index.name = "date"
    return df.sort_index()


def fetch_gdp_growth() -> pd.DataFrame:
    fred = get_fred_client()
    s = fred.get_series(
        "A191RL1Q225SBEA", observation_start=START_DATE, observation_end=END_DATE
    )
    df = s.rename("gdp_growth_qoq").to_frame()
    df.index.name = "date"
    return df.sort_index()


def fetch_fed_funds() -> pd.DataFrame:
    fred = get_fred_client()
    s = fred.get_series(
        "FEDFUNDS", observation_start=START_DATE, observation_end=END_DATE
    )
    df = s.rename("fed_funds_rate").to_frame()
    df.index.name = "date"
    return df.sort_index()


def fetch_unemployment() -> pd.DataFrame:
    fred = get_fred_client()
    s = fred.get_series(
        "UNRATE", observation_start=START_DATE, observation_end=END_DATE
    )
    df = s.rename("unemployment_rate").to_frame()
    df["unemployment_change"] = df["unemployment_rate"].diff()
    df.index.name = "date"
    return df.sort_index()


def fetch_cpi() -> pd.DataFrame:
    fred = get_fred_client()
    s = fred.get_series(
        "CPIAUCSL", observation_start=START_DATE, observation_end=END_DATE
    )
    df = s.rename("cpi").to_frame()
    df["cpi_yoy"] = df["cpi"].pct_change(12) * 100
    df.index.name = "date"
    return df.sort_index()
