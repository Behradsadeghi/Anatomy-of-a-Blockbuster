import streamlit as st
import pandas as pd
import pydeck as pdk
import numpy as np
import json
from pathlib import Path

# Country centroids (approx) for visualization
COUNTRY_COORDS = {
    "United States of America": (37.0902, -95.7129),
    "United Kingdom": (55.3781, -3.4360),
    "France": (46.2276, 2.2137),
    "Canada": (56.1304, -106.3468),
    "Germany": (51.1657, 10.4515),
    "India": (20.5937, 78.9629),
    "Japan": (36.2048, 138.2529),
    "Australia": (-25.2744, 133.7751),
    "China": (35.8617, 104.1954),
    "Italy": (41.8719, 12.5674),
    "Spain": (40.4637, -3.7492),
    "Brazil": (-14.2350, -51.9253),
    "Mexico": (23.6345, -102.5528),
    "South Korea": (35.9078, 127.7669),
    "Russian Federation": (61.5240, 105.3188),
    "Argentina": (-38.4161, -63.6167),
    "Netherlands": (52.1326, 5.2913),
    "Sweden": (60.1282, 18.6435),
    "New Zealand": (-40.9006, 174.8860),
    "Belgium": (50.5039, 4.4699),
    "Ireland": (53.1424, -7.6921),
    "Switzerland": (46.8182, 8.2275),
    "Austria": (47.5162, 14.5501),
    "Denmark": (56.2639, 9.5018),
}


@st.cache_data
def aggregate_country_revenue(movies: pd.DataFrame) -> pd.DataFrame:
    """Aggregate revenue by main production country and attach lat/long + blockbuster stats."""
    df = (
        movies.groupby("main_country")
        .agg(
            total_revenue=("revenue", "sum"),
            movie_count=("id", "count"),
            blockbusters=("is_blockbuster", "sum"),
        )
        .reset_index()
    )
    df = df[df["main_country"].isin(COUNTRY_COORDS.keys())]

    df["lat"] = df["main_country"].apply(lambda c: COUNTRY_COORDS[c][0])
    df["lon"] = df["main_country"].apply(lambda c: COUNTRY_COORDS[c][1])
    df["blockbuster_share_pct"] = np.where(
        df["movie_count"] > 0, (df["blockbusters"] / df["movie_count"]) * 100, 0
    ).round(1)
    df["non_blockbusters"] = df["movie_count"] - df["blockbusters"]

    max_revenue = df["total_revenue"].max() or 1
    max_blockbusters = df["blockbusters"].max() or 1
    # Normalize to soften skew and feed the map aesthetics
    df["revenue_norm"] = np.sqrt(df["total_revenue"].clip(lower=0)) / np.sqrt(max_revenue)
    df["block_norm"] = (df["blockbusters"] / max_blockbusters) ** 0.6
    df["elevation_blocks"] = df["blockbusters"].clip(lower=0)
    df["blockbusters_rank"] = df["blockbusters"].rank(method="dense", ascending=False).astype(int)

    # Discrete 5-step ramp for clearer contrast (light blush to deep crimson)
    palette = [
        np.array([180, 215, 255]),  # pale blue for low
        np.array([255, 210, 210]),
        np.array([248, 130, 130]),
        np.array([224, 48, 48]),
        np.array([158, 0, 0]),      # deep red for high
    ]

    def _step_color(val: float) -> list[int]:
        val = float(np.clip(val, 0, 1))
        idx = int(np.floor(val * (len(palette) - 1)))
        color = palette[idx]
        return [int(color[0]), int(color[1]), int(color[2]), 210]

    df["fill_color"] = df["block_norm"].apply(_step_color)
    df["radius"] = 150000 + (700000 * df["block_norm"])
    df["total_revenue_label"] = df["total_revenue"].apply(lambda x: f"${x:,.0f}")
    df["blockbusters_label"] = df["blockbusters"].apply(lambda x: f"{int(x):,}")
    df["movies_label"] = df["movie_count"].apply(lambda x: f"{int(x):,}")
    df["blockbuster_pct_label"] = df["blockbuster_share_pct"].apply(lambda x: f"{x:.1f}%")
    df["blockbusters_rank_label"] = df["blockbusters_rank"].apply(lambda x: f"#{x}")
    return df


@st.cache_data
def load_country_geojson(path: str | Path = Path(__file__).resolve().parent.parent / "data" / "world_countries.geojson") -> dict:
    with open(path, "r") as f:
        return json.load(f)


@st.cache_resource
def revenue_map_deck(country_agg: pd.DataFrame) -> pdk.Deck:
    """3D column map of revenue by country."""
    if country_agg.empty:
        return pdk.Deck()

    geojson = load_country_geojson()
    stats_by_country = country_agg.set_index("main_country").to_dict(orient="index")

    for feature in geojson.get("features", []):
        name = feature.get("properties", {}).get("name")
        stat = stats_by_country.get(name, {})
        fill = stat.get("fill_color", [220, 220, 220, 80])
        props = feature.setdefault("properties", {})
        props["fill_color"] = fill
        props["total_revenue_label"] = stat.get("total_revenue_label", "$0")
        props["movies_label"] = stat.get("movies_label", "0")
        props["blockbusters_label"] = stat.get("blockbusters_label", "0")
        props["blockbuster_pct_label"] = stat.get("blockbuster_pct_label", "0.0%")
        props["blockbuster_share_pct"] = stat.get("blockbuster_share_pct", 0)
        props["blockbusters_rank_label"] = stat.get("blockbusters_rank_label", "#-")

    geo_layer = pdk.Layer(
        "GeoJsonLayer",
        data=geojson,
        stroked=True,
        filled=True,
        get_fill_color="properties.fill_color",
        get_line_color=[180, 180, 180, 120],
        pickable=True,
        auto_highlight=True,
        opacity=0.8,
    )

    view_state = pdk.ViewState(
        latitude=20, longitude=0, zoom=1.3, pitch=40
    )

    tooltip = {
        "html": """
            <div style="font-size:12px; font-family: 'Inter', system-ui, sans-serif; min-width: 180px;">
                <b>{name}</b><br/>
                Revenue: {total_revenue_label}<br/>
                Movies: {movies_label}<br/>
                Blockbusters: {blockbusters_label} ({blockbuster_pct_label})<br/>
                Rank by blockbusters: {blockbusters_rank_label}
                <div style="margin-top:8px; width:120px; height:120px; border-radius:8px; background:#f5f5f5; display:flex; align-items:center; justify-content:center;">
                    <div style="
                        width:95px; height:95px; border-radius:50%;
                        background: conic-gradient(#b30000 {blockbuster_share_pct}%, #e0e0e0 {blockbuster_share_pct}% 100%);
                        position: relative;
                    ">
                        <span style="
                            position:absolute;
                            left:50%;
                            top:72%;
                            transform: translate(-50%, -50%);
                            color:#b30000;
                            font-weight:600;
                            font-size:12px;
                        ">{blockbuster_share_pct}%</span>
                    </div>
                </div>
            </div>
        """,
        "style": {"backgroundColor": "#ffffff", "color": "#111111", "border": "1px solid #ddd"},
    }

    return pdk.Deck(
        layers=[geo_layer],
        initial_view_state=view_state,
        tooltip=tooltip,
        map_provider="carto",
        # Use a public Carto dark basemap so country outlines/labels render without any token.
        map_style="https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json",
    )
