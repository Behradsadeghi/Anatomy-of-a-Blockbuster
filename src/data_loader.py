import json
import ast
import pandas as pd
from pathlib import Path

# Resolve project root so reads work no matter where `streamlit run` is invoked.
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

# Only load the columns we actually use across the app to speed up IO.
MOVIE_COLS = [
    "id",
    "original_title",
    "budget",
    "revenue",
    "popularity",
    "runtime",
    "vote_average",
    "vote_count",
    "release_date",
    "genres",
    "production_countries",
    "production_companies",
    "spoken_languages",
    "belongs_to_collection",
]

_movies_cache: dict[str, pd.DataFrame] = {}
_credits_cache: dict[str, pd.DataFrame] = {}
_ratings_cache: dict[str, pd.DataFrame] = {}


def load_movies(path: str | Path = DATA_DIR / "movies_metadata.csv", fast: bool = False, sample_size: int = 5000) -> pd.DataFrame:
    """Load movies metadata."""
    key = str(Path(path).resolve())
    if key in _movies_cache:
        return _movies_cache[key]

    csv_path = Path(path)
    cache_path = csv_path.with_suffix(".pkl")

    if cache_path.exists() and cache_path.stat().st_mtime >= csv_path.stat().st_mtime:
        df = pd.read_pickle(cache_path)
        _movies_cache[key] = df
        return df

    try:
        df = pd.read_csv(csv_path, usecols=MOVIE_COLS, low_memory=False)
    except ValueError:
        df = pd.read_csv(csv_path, low_memory=False)

    df.to_pickle(cache_path)
    _movies_cache[key] = df
    return df


def _first_cast_name(x):
    if not isinstance(x, str):
        return None
    for parser in (json.loads, ast.literal_eval):
        try:
            data = parser(x)
            if isinstance(data, list) and data and isinstance(data[0], dict):
                return data[0].get("name")
        except Exception:
            continue
    return None


def _first_director_name(x):
    if not isinstance(x, str):
        return None
    for parser in (json.loads, ast.literal_eval):
        try:
            data = parser(x)
            if isinstance(data, list):
                for person in data:
                    if isinstance(person, dict) and person.get("job") == "Director":
                        return person.get("name")
        except Exception:
            continue
    return None


def load_credits(path: str | Path = DATA_DIR / "credits.csv", lean: bool = True) -> pd.DataFrame:
    """Load credits. In lean mode, keep only id + lead_actor + director."""
    key = str(Path(path).resolve()) + f"::lean={lean}"
    if key in _credits_cache:
        return _credits_cache[key]

    csv_path = Path(path)
    cache_suffix = "_lead.pkl" if lean else ".pkl"
    cache_path = csv_path.with_name(csv_path.stem + cache_suffix)

    if cache_path.exists() and cache_path.stat().st_mtime >= csv_path.stat().st_mtime:
        df = pd.read_pickle(cache_path)
        _credits_cache[key] = df
        return df

    usecols = ["id", "cast", "crew"] if lean else None
    df = pd.read_csv(csv_path, usecols=usecols)

    if lean:
        df["lead_actor"] = df["cast"].apply(_first_cast_name)
        df["director"] = df["crew"].apply(_first_director_name)
        df["id"] = pd.to_numeric(df["id"], errors="coerce")
        df = df[["id", "lead_actor", "director"]]

    df.to_pickle(cache_path)
    _credits_cache[key] = df
    return df


def load_ratings(path: str | Path = DATA_DIR / "ratings_small.csv") -> pd.DataFrame:
    key = str(Path(path).resolve())
    if key in _ratings_cache:
        return _ratings_cache[key]
    df = pd.read_csv(path)
    _ratings_cache[key] = df
    return df


def load_all(
    data_dir: str | Path = DATA_DIR,
    include_credits: bool = False,
    include_ratings: bool = False,
    lean_credits: bool = True,
) -> dict:
    base = Path(data_dir)
    movies = load_movies(base / "movies_metadata.csv")
    credits = load_credits(base / "credits.csv", lean=lean_credits) if include_credits else None
    ratings = load_ratings(base / "ratings_small.csv") if include_ratings else None

    return {"movies": movies, "credits": credits, "ratings": ratings}
