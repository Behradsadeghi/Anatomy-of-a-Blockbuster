import pandas as pd
import numpy as np
import streamlit as st


@st.cache_data
def overview_metrics(df: pd.DataFrame) -> dict:
    return {
        "movies": int(len(df)),
        "blockbusters": int(df["is_blockbuster"].sum()),
        "blockbuster_share": float(df["is_blockbuster"].mean()) * 100,
        "median_budget": float(df["budget"].median()),
        "median_revenue": float(df["revenue"].median()),
        "median_roi": float(df["roi"].median()),
        "avg_popularity": float(df["popularity"].mean()),
    }


@st.cache_data
def yearly_blockbuster_trend(df: pd.DataFrame) -> pd.DataFrame:
    grp = (
        df.dropna(subset=["release_year"])
        .groupby("release_year")
        .agg(
            movies=("id", "count"),
            blockbuster_share=("is_blockbuster", "mean"),
            mean_revenue=("revenue", "mean"),
        )
        .reset_index()
    )
    grp["blockbuster_share_pct"] = grp["blockbuster_share"] * 100
    return grp


@st.cache_data
def season_blockbuster_stats(df: pd.DataFrame) -> pd.DataFrame:
    grp = (
        df.groupby(["season", "is_blockbuster"])
        .size()
        .unstack(fill_value=0)
        .rename(columns={0: "Non-Blockbuster", 1: "Blockbuster"})
        .reset_index()
    )
    grp["Total"] = grp["Non-Blockbuster"] + grp["Blockbuster"]
    grp["Blockbuster_pct"] = grp["Blockbuster"] / grp["Total"] * 100
    return grp


@st.cache_data
def month_blockbuster_stats(df: pd.DataFrame) -> pd.DataFrame:
    grp = (
        df.groupby(["release_month", "release_month_name", "is_blockbuster"])
        .size()
        .unstack(fill_value=0)
        .rename(columns={0: "Non-Blockbuster", 1: "Blockbuster"})
        .reset_index()
    )
    grp["Total"] = grp["Non-Blockbuster"] + grp["Blockbuster"]
    grp["Blockbuster_pct"] = grp["Blockbuster"] / grp["Total"] * 100
    return grp.sort_values("release_month")


@st.cache_data
def genre_performance(df: pd.DataFrame, drop_unknown: bool = True) -> pd.DataFrame:
    data = df.copy()
    if drop_unknown:
        data = data[data["main_genre"] != "Unknown"]
    grp = (
        data.groupby("main_genre")
        .agg(
            movies=("id", "count"),
            blockbuster_rate=("is_blockbuster", "mean"),
            avg_revenue=("revenue", "mean"),
            median_roi=("roi", "median"),
            popularity=("popularity", "mean"),
        )
        .reset_index()
    )
    grp["blockbuster_rate_pct"] = grp["blockbuster_rate"] * 100
    return grp.sort_values("movies", ascending=False)


@st.cache_data
def genre_year_heatmap(df: pd.DataFrame, drop_unknown: bool = True) -> pd.DataFrame:
    data = df.copy()
    if drop_unknown:
        data = data[data["main_genre"] != "Unknown"]
    grp = (
        data.dropna(subset=["release_year"])
        .groupby(["main_genre", "release_year"])
        .agg(blockbuster_rate=("is_blockbuster", "mean"), movies=("id", "count"))
        .reset_index()
    )
    grp["blockbuster_rate_pct"] = grp["blockbuster_rate"] * 100
    return grp


@st.cache_data
def budget_revenue_points(df: pd.DataFrame, year_range: tuple[int, int] | None = None) -> pd.DataFrame:
    data = df.copy()
    if year_range:
        data = data[(data["release_year"] >= year_range[0]) & (data["release_year"] <= year_range[1])]
    return data[
        ["budget", "revenue", "roi", "original_title", "release_year", "main_genre", "is_blockbuster"]
    ]


@st.cache_data
def budget_band_stats(df: pd.DataFrame) -> pd.DataFrame:
    grp = (
        df.groupby("budget_band")
        .agg(
            movies=("id", "count"),
            blockbuster_rate=("is_blockbuster", "mean"),
            median_roi=("roi", "median"),
        )
        .reset_index()
    )
    grp["blockbuster_rate_pct"] = grp["blockbuster_rate"] * 100
    return grp


@st.cache_data
def country_revenue(df: pd.DataFrame, drop_unknown: bool = True) -> pd.DataFrame:
    data = df.copy()
    if drop_unknown:
        data = data[data["main_country"] != "Unknown"]
    grp = (
        data.groupby("main_country")
        .agg(
            total_revenue=("revenue", "sum"),
            movies=("id", "count"),
            blockbusters=("is_blockbuster", "sum"),
            blockbuster_rate=("is_blockbuster", "mean"),
        )
        .reset_index()
    )
    grp["blockbuster_rate_pct"] = grp["blockbuster_rate"] * 100
    return grp.sort_values("total_revenue", ascending=False)


@st.cache_data
def blockbuster_correlations(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Correlation of is_blockbuster with core numeric drivers (no buckets/dummies)."""
    if df.empty or "is_blockbuster" not in df.columns:
        return pd.DataFrame(columns=["feature", "correlation"])

    data = df.copy()
    data = data.dropna(subset=["is_blockbuster"])
    feature_frames = []

    numeric_cols = [
        "budget",
        "revenue",
        "roi",
        "popularity",
        "runtime",
        "vote_average",
        "vote_count",
        "blockbuster_score",
        "is_franchise",
    ]
    numeric_cols = [c for c in numeric_cols if c in data.columns]
    if numeric_cols:
        feature_frames.append(data[numeric_cols])

    # Season (compact categorical)
    if "season" in data.columns:
        seasons = pd.get_dummies(data["season"], dummy_na=False).add_prefix("season: ")
        feature_frames.append(seasons)

    if not feature_frames:
        return pd.DataFrame(columns=["feature", "correlation"])

    features = pd.concat(feature_frames, axis=1).fillna(0)
    corr = features.corrwith(data["is_blockbuster"]).dropna()
    corr = corr.reindex(corr.abs().sort_values(ascending=False).index)
    return corr.reset_index().rename(columns={"index": "feature", 0: "correlation"})


@st.cache_data
def franchise_summary(df: pd.DataFrame) -> pd.DataFrame:
    if "is_franchise" not in df.columns:
        return pd.DataFrame()
    grp = (
        df.groupby("is_franchise")
        .agg(
            movies=("id", "count"),
            blockbusters=("is_blockbuster", "sum"),
            blockbuster_rate=("is_blockbuster", "mean"),
            avg_revenue=("revenue", "mean"),
            avg_roi=("roi", "mean"),
        )
        .reset_index()
    )
    grp["label"] = grp["is_franchise"].map({0: "Standalone", 1: "Franchise"})
    grp["blockbuster_rate_pct"] = grp["blockbuster_rate"] * 100
    return grp.sort_values("is_franchise", ascending=False)


@st.cache_data
def top_franchises(df: pd.DataFrame, min_movies: int = 3) -> pd.DataFrame:
    if "collection_name" not in df.columns:
        return pd.DataFrame()
    data = df[df["collection_name"] != "Unknown"]
    grp = (
        data.groupby("collection_name")
        .agg(
            movies=("id", "count"),
            blockbusters=("is_blockbuster", "sum"),
            blockbuster_rate=("is_blockbuster", "mean"),
            avg_revenue=("revenue", "mean"),
        )
        .reset_index()
    )
    grp = grp[grp["movies"] >= min_movies]
    grp["blockbuster_rate_pct"] = grp["blockbuster_rate"] * 100
    return grp.sort_values("blockbusters", ascending=False)


@st.cache_data
def top_people(df: pd.DataFrame, role_col: str = "lead_actor", min_movies: int = 4) -> pd.DataFrame:
    if role_col not in df.columns:
        return pd.DataFrame()
    grp = (
        df.dropna(subset=[role_col])
        .groupby(role_col)
        .agg(
            movies=("id", "count"),
            avg_revenue=("revenue", "mean"),
            avg_roi=("roi", "mean"),
            blockbuster_rate=("is_blockbuster", "mean"),
        )
        .reset_index()
    )
    grp = grp[grp["movies"] >= min_movies]
    grp["blockbuster_rate_pct"] = grp["blockbuster_rate"] * 100
    return grp.sort_values("avg_revenue", ascending=False)


@st.cache_data
def top_actor_director_pairs(df: pd.DataFrame, min_movies: int = 3) -> pd.DataFrame:
    if not {"lead_actor", "director"}.issubset(df.columns):
        return pd.DataFrame()
    grp = (
        df.dropna(subset=["lead_actor", "director"])
        .groupby(["director", "lead_actor"])
        .agg(
            movies=("id", "count"),
            blockbuster_rate=("is_blockbuster", "mean"),
            avg_revenue=("revenue", "mean"),
        )
        .reset_index()
    )
    grp = grp[grp["movies"] >= min_movies]
    grp["blockbuster_rate_pct"] = grp["blockbuster_rate"] * 100
    return grp.sort_values("avg_revenue", ascending=False)


@st.cache_data
def corr_matrix(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "budget",
        "revenue",
        "roi",
        "popularity",
        "runtime",
        "vote_average",
        "vote_count",
        "blockbuster_score",
        "is_franchise",
    ]
    cols = [c for c in cols if c in df.columns]
    if not cols:
        return pd.DataFrame()
    numeric = df[cols].copy()
    corr = numeric.corr()
    # Keep only lower triangle to avoid duplicate info
    mask = np.tril(np.ones_like(corr, dtype=bool))
    corr = corr.where(mask)
    return corr


@st.cache_data
def company_stats(df: pd.DataFrame, min_movies: int = 8) -> pd.DataFrame:
    if "production_companies_list" not in df.columns:
        return pd.DataFrame()
    exploded = df.explode("production_companies_list")
    exploded = exploded[
        exploded["production_companies_list"].notna()
        & (exploded["production_companies_list"] != "Unknown")
    ]
    grp = (
        exploded.groupby("production_companies_list")
        .agg(
            movies=("id", "count"),
            blockbusters=("is_blockbuster", "sum"),
            blockbuster_rate=("is_blockbuster", "mean"),
            avg_revenue=("revenue", "mean"),
            avg_roi=("roi", "mean"),
        )
        .reset_index()
    )
    grp = grp[grp["movies"] >= min_movies]
    grp["blockbuster_rate_pct"] = grp["blockbuster_rate"] * 100
    return grp.sort_values("blockbusters", ascending=False)
