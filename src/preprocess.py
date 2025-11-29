import json
import ast
from pathlib import Path
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_MOVIES_PATH = DATA_DIR / "movies_metadata.csv"
CLEAN_CACHE_DIR = DATA_DIR


def _safe_json_list(val):
    """Parse JSON-like strings safely into list of dicts or names."""
    if isinstance(val, list):
        return val
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return []
    if not isinstance(val, str):
        return []
    val = val.strip()
    if val == "" or val.lower() == "nan":
        return []
    for parser in (json.loads, ast.literal_eval):
        try:
            parsed = parser(val)
            if isinstance(parsed, list):
                return parsed
        except Exception:
            continue
    # fallback: pipe/comma-delimited strings
    if "|" in val:
        return [{"name": x.strip()} for x in val.split("|") if x.strip()]
    if "," in val:
        return [{"name": x.strip()} for x in val.split(",") if x.strip()]
    return []


def _safe_collection_name(val):
    """Extract collection/franchise name from JSON or dict."""
    if isinstance(val, dict):
        return val.get("name")
    if isinstance(val, float) and np.isnan(val):
        return None
    if isinstance(val, str):
        val = val.strip()
        if not val or val.lower() == "nan":
            return None
        for parser in (json.loads, ast.literal_eval):
            try:
                parsed = parser(val)
                if isinstance(parsed, dict):
                    return parsed.get("name")
            except Exception:
                continue
    return None



def _extract_names(lst):
    if not isinstance(lst, list):
        return []
    return [d.get("name") for d in lst if isinstance(d, dict) and d.get("name")]


def _scale_0_1(series: pd.Series) -> pd.Series:
    s = series.copy()
    s = s.replace([np.inf, -np.inf], np.nan)
    s = s.clip(lower=s.quantile(0.01), upper=s.quantile(0.99))
    s = (s - s.min()) / (s.max() - s.min() + 1e-9)
    return s.fillna(0)


def preprocess_movies_full(
    movies_raw: pd.DataFrame,
    credits: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Clean and enrich movies data (full dataset)."""
    df = movies_raw.copy()

    # IDs and deduplication
    if "id" in df.columns:
        df["id"] = pd.to_numeric(df["id"], errors="coerce")
    df = df.drop_duplicates(subset=["id"]).reset_index(drop=True)

    numeric_cols = ["budget", "revenue", "popularity", "runtime", "vote_average", "vote_count"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Dates
    df["release_date"] = pd.to_datetime(df.get("release_date"), errors="coerce")
    df["release_year"] = df["release_date"].dt.year
    df["release_month"] = df["release_date"].dt.month
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    df["release_month_name"] = df["release_month"].apply(
        lambda m: month_names[int(m) - 1] if pd.notna(m) and 1 <= int(m) <= 12 else "Unknown"
    )

    # Genres / countries
    df["genres_json"] = df.get("genres", "").apply(_safe_json_list)
    df["genres_list"] = df["genres_json"].apply(_extract_names)
    df["main_genre"] = df["genres_list"].apply(lambda lst: lst[0] if lst else "Unknown")

    df["countries_json"] = df.get("production_countries", "").apply(_safe_json_list)
    df["production_countries_list"] = df["countries_json"].apply(_extract_names)
    df["main_country"] = df["production_countries_list"].apply(lambda lst: lst[0] if lst else "Unknown")

    # Companies
    df["companies_json"] = df.get("production_companies", "").apply(_safe_json_list)
    df["production_companies_list"] = df["companies_json"].apply(_extract_names)
    df["main_company"] = df["production_companies_list"].apply(lambda lst: lst[0] if lst else "Unknown")

    # Franchise / collection
    df["collection_name"] = df.get("belongs_to_collection").apply(_safe_collection_name).fillna("Unknown")
    df["is_franchise"] = (df["collection_name"] != "Unknown").astype(int)

    # Handle zeros/missing budgets & revenues
    df["budget"] = df["budget"].where(df["budget"] > 0)
    df["revenue"] = df["revenue"].where(df["revenue"] > 0)
    df["runtime"] = df["runtime"].where(df["runtime"] > 0)

    # Genre-level median fill, then global median
    for col in ["budget", "revenue", "runtime"]:
        genre_median = df.groupby("main_genre")[col].transform("median")
        df[col] = df[col].fillna(genre_median)
        df[col] = df[col].fillna(df[col].median())

    # ROI and profit
    df["roi"] = (df["revenue"] - df["budget"]) / df["budget"].replace(0, np.nan)
    df["roi"] = df["roi"].replace([np.inf, -np.inf], np.nan).fillna(0)
    df["profit"] = df["revenue"] - df["budget"]

    # Drop rows with clearly non-sensical values after type casting
    df = df[(df["runtime"].isna()) | (df["runtime"].between(20, 400))]
    df = df[(df["budget"].isna()) | (df["budget"] >= 50_000)]
    df = df[(df["revenue"].isna()) | (df["revenue"] >= 0)]

    # Blockbuster scoring (hybrid profitability + demand)
    revenue_scaled = _scale_0_1(df["revenue"])
    roi_scaled = _scale_0_1(df["roi"])
    popularity_scaled = _scale_0_1(df["popularity"])
    votes_scaled = _scale_0_1(np.log1p(df["vote_count"]))

    df["blockbuster_score"] = (
        0.35 * revenue_scaled
        + 0.3 * roi_scaled
        + 0.2 * popularity_scaled
        + 0.15 * votes_scaled
    )

    rev_cut = df["revenue"].quantile(0.85)
    roi_cut = df["roi"].quantile(0.75)
    pop_cut = df["popularity"].quantile(0.75)
    score_cut = df["blockbuster_score"].quantile(0.9)

    df["is_blockbuster"] = (
        (
            (df["revenue"] >= rev_cut)
            & (df["roi"] >= roi_cut)
            & (df["popularity"] >= pop_cut)
        )
        | (df["blockbuster_score"] >= score_cut)
    ).astype(int)

    season_map = {12: "Winter", 1: "Winter", 2: "Winter", 3: "Spring", 4: "Spring", 5: "Spring", 6: "Summer", 7: "Summer", 8: "Summer", 9: "Fall", 10: "Fall", 11: "Fall"}
    df["season"] = df["release_month"].map(season_map).fillna("Unknown")
    df["budget_band"] = pd.cut(
        df["budget"],
        bins=[-1, 10_000_000, 40_000_000, 100_000_000, 250_000_000, np.inf],
        labels=["<10M", "10–40M", "40–100M", "100–250M", "250M+"],
    )

    # Drop rows with more than one missing among key fields
    key_fields = ["budget", "revenue", "popularity", "runtime", "vote_average", "vote_count", "release_year", "main_genre", "main_country"]
    missing_counts = df[key_fields].isna().sum(axis=1)
    df = df[missing_counts <= 1]

    # Drop unusable rows (no ID or year)
    df = df.dropna(subset=["id", "release_year"])

    # Attach people signals if provided
    if credits is not None and not credits.empty:
        df = df.merge(credits, on="id", how="left")

    return df


def preprocess_movies(
    movies_raw: pd.DataFrame | None = None,
    data_path: str | Path = RAW_MOVIES_PATH,
    credits: pd.DataFrame | None = None,
    use_disk_cache: bool = True,
) -> pd.DataFrame:
    """Wrapper that caches cleaned data to disk to avoid repeated heavy parsing."""
    raw_path = Path(data_path)
    cache_path = CLEAN_CACHE_DIR / "movies_clean_full.pkl"

    needed_cols = {"collection_name", "is_franchise"}

    if use_disk_cache and cache_path.exists() and cache_path.stat().st_mtime >= raw_path.stat().st_mtime:
        try:
            cached = pd.read_pickle(cache_path)
            if needed_cols.issubset(set(cached.columns)):
                return cached
        except Exception:
            pass

    if movies_raw is None:
        from src.data_loader import load_movies
        movies_raw = load_movies(raw_path, fast=False)

    df = preprocess_movies_full(movies_raw, credits=credits)

    if use_disk_cache:
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_pickle(cache_path)
        except Exception:
            pass

    return df
