import pandas as pd
import numpy as np
import ast

# -------------------------------------------------
# Helper: safe parser for JSON-like strings
# -------------------------------------------------
def safe_parse(x):
    try:
        if pd.isna(x):
            return []
        if isinstance(x, list):
            return x
        return ast.literal_eval(x)
    except:
        return []

# -------------------------------------------------
# Compute ROI
# -------------------------------------------------
def compute_roi(df):
    df = df.copy()
    df["roi"] = 0.0

    valid = df["budget"] > 0
    df.loc[valid, "roi"] = (
        (df.loc[valid, "revenue"] - df.loc[valid, "budget"]) 
        / df.loc[valid, "budget"]
    )

    df["roi"] = df["roi"].replace([np.inf, -np.inf], 0).fillna(0)
    return df

# -------------------------------------------------
# Blockbuster flag (default threshold ROI ≥ 1.5)
# -------------------------------------------------
def add_blockbuster_flag(df, threshold=1.5):
    df = df.copy()
    df["is_blockbuster"] = (df["roi"] >= threshold).astype(int)
    return df

# -------------------------------------------------
# Assign season from release_month
# -------------------------------------------------
def assign_season(df):
    df = df.copy()
    season_map = {
        12:"Winter",1:"Winter",2:"Winter",
        3:"Spring",4:"Spring",5:"Spring",
        6:"Summer",7:"Summer",8:"Summer",
        9:"Fall",10:"Fall",11:"Fall"
    }
    df["season"] = df["release_month"].map(season_map)
    return df

# -------------------------------------------------
# Extract Director from crew JSON
# -------------------------------------------------
def extract_director(df):
    df = df.copy()

    def get_director(crew_str):
        crew = safe_parse(crew_str)
        for person in crew:
            if isinstance(person, dict) and person.get("job") == "Director":
                return person.get("name")
        return None

    df["director"] = df["crew"].apply(get_director)
    return df

# -------------------------------------------------
# Extract main cast (first 5 actors)
# -------------------------------------------------
def extract_cast(df):
    df = df.copy()

    df["main_cast"] = (
        df["cast"]
        .apply(safe_parse)
        .apply(lambda lst: [p.get("name") for p in lst[:5] if isinstance(p, dict)])
    )
    return df

# -------------------------------------------------
# Compute director score (how many blockbusters made)
# -------------------------------------------------
def compute_director_score(df):
    df = df.copy()
    director_stats = (
        df.groupby("director")["is_blockbuster"]
        .mean()
        .rename("director_score")
        .reset_index()
    )
    df = df.merge(director_stats, on="director", how="left")
    df["director_score"] = df["director_score"].fillna(0)
    return df

# -------------------------------------------------
# Compute cast score (average blockbuster rate for top actors)
# -------------------------------------------------
def compute_cast_score(df):
    df = df.copy()

    # explode cast list
    exploded = df[["id", "main_cast", "is_blockbuster"]].explode("main_cast")

    cast_stats = (
        exploded.groupby("main_cast")["is_blockbuster"]
        .mean()
        .rename("cast_score")
        .reset_index()
    )

    df = df.merge(
        cast_stats, 
        left_on="main_cast",
        right_on="main_cast",
        how="left"
    )

    df["cast_score"] = df["cast_score"].fillna(0)
    return df
