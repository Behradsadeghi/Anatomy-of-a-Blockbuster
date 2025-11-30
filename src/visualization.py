import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

CONTINENT_MAP = {
    "United States of America": "Americas",
    "Canada": "Americas",
    "Mexico": "Americas",
    "Brazil": "Americas",
    "Argentina": "Americas",
    "Chile": "Americas",
    "Colombia": "Americas",
    "Peru": "Americas",
    "United Kingdom": "Europe",
    "France": "Europe",
    "Germany": "Europe",
    "Spain": "Europe",
    "Italy": "Europe",
    "Netherlands": "Europe",
    "Sweden": "Europe",
    "Belgium": "Europe",
    "Ireland": "Europe",
    "Switzerland": "Europe",
    "Austria": "Europe",
    "Denmark": "Europe",
    "Finland": "Europe",
    "Norway": "Europe",
    "Poland": "Europe",
    "Russia": "Europe",
    "Russian Federation": "Europe",
    "Turkey": "Europe",
    "India": "Asia",
    "China": "Asia",
    "Japan": "Asia",
    "South Korea": "Asia",
    "Hong Kong": "Asia",
    "Taiwan": "Asia",
    "Thailand": "Asia",
    "Vietnam": "Asia",
    "Indonesia": "Asia",
    "Pakistan": "Asia",
    "Bangladesh": "Asia",
    "Singapore": "Asia",
    "United Arab Emirates": "Asia",
    "Saudi Arabia": "Asia",
    "Iran": "Asia",
    "Australia": "Oceania",
    "New Zealand": "Oceania",
    "South Africa": "Africa",
    "Nigeria": "Africa",
    "Egypt": "Africa",
    "Kenya": "Africa",
    "Morocco": "Africa",
}

BUDGET_BAND_ORDER = ["<10M", "10–50M", "50–100M", "100–250M", "250M+"]

def fig_yearly_trend(yearly: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=yearly["release_year"],
            y=yearly["movies"],
            name="Movies",
            marker_color="#7FB3FF",
            opacity=0.6,
            yaxis="y1",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=yearly["release_year"],
            y=yearly["blockbuster_share_pct"],
            name="Blockbuster %",
            marker_color="#FF7A7A",
            mode="lines+markers",
            yaxis="y2",
        )
    )
    fig.update_layout(
        title="Yearly Output & Blockbuster Share",
        yaxis=dict(title="Movies"),
        yaxis2=dict(title="Blockbuster %", overlaying="y", side="right"),
        hovermode="x unified",
    )
    return fig


def fig_budget_revenue_scatter(points: pd.DataFrame) -> go.Figure:
    fig = px.scatter(
        points,
        x="budget",
        y="revenue",
        color="is_blockbuster",
        hover_data=["original_title", "release_year", "main_genre", "roi"],
        color_discrete_map={0: "#7FB3FF", 1: "#FF7A7A"},
        title="Budget vs Revenue (log-log)",
        opacity=0.4,
        size_max=8,
    )
    fig.update_xaxes(type="log", title="Budget")
    fig.update_yaxes(type="log", title="Revenue")
    return fig


def fig_genre_roi_box(df: pd.DataFrame) -> go.Figure:
    data = df.copy()
    data["roi"] = pd.to_numeric(data["roi"], errors="coerce")
    data["roi"] = data["roi"].replace([np.inf, -np.inf], np.nan)
    data["roi_clipped"] = data["roi"].clip(upper=300)
    fig = px.box(
        data,
        x="main_genre",
        y="roi_clipped",
        color="is_blockbuster",
        color_discrete_map={0: "#7FB3FF", 1: "#FF7A7A"},
        title="ROI by Genre (ROI clipped at 300)",
    )
    fig.update_xaxes(title="Genre")
    fig.update_yaxes(title="ROI (clipped ≤300)")
    return fig


def fig_genre_blockbuster_bar(genre_stats: pd.DataFrame) -> go.Figure:
    top = genre_stats.sort_values("movies", ascending=False).head(15)
    top["text_pct"] = top["blockbuster_rate_pct"].round(1)
    fig = px.bar(
        top,
        x="main_genre",
        y="blockbuster_rate_pct",
        color="avg_revenue",
        title="Top Genres by Blockbuster %",
        labels={"blockbuster_rate_pct": "Blockbuster %", "avg_revenue": "Avg Revenue"},
        color_continuous_scale="Bluered",
        text="text_pct",
    )
    fig.update_layout(
        xaxis_title="Genre",
        yaxis_title="Blockbuster %",
        yaxis=dict(range=[0, 42]),
    )
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside", cliponaxis=False)
    return fig


def fig_season_blockbuster(season_stats: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=season_stats["season"],
            y=season_stats["Non-Blockbuster"],
            name="Non-Blockbuster",
            marker_color="#7FB3FF",
        )
    )
    fig.add_trace(
        go.Bar(
            x=season_stats["season"],
            y=season_stats["Blockbuster"],
            name="Blockbuster",
            marker_color="#FF7A7A",
        )
    )
    fig.update_layout(
        barmode="stack",
        title="Release Season Mix",
        xaxis_title="Season",
        yaxis_title="Movies",
        hovermode="x",
    )
    return fig


def fig_month_blockbuster(month_stats: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=month_stats["release_month_name"],
            y=month_stats["Total"],
            name="Total",
            marker_color="#7FB3FF",
            opacity=0.5,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=month_stats["release_month_name"],
            y=month_stats["Blockbuster_pct"],
            name="Blockbuster %",
            marker_color="#FF7A7A",
            mode="lines+markers",
            yaxis="y2",
        )
    )
    fig.update_layout(
        title="Monthly Volume & Blockbuster %",
        yaxis=dict(title="Movies"),
        yaxis2=dict(title="Blockbuster %", overlaying="y", side="right"),
        hovermode="x",
    )
    return fig


def fig_roi_hist(df: pd.DataFrame) -> go.Figure:
    roi_pos = df.copy()
    roi_pos["roi"] = pd.to_numeric(roi_pos["roi"], errors="coerce")
    roi_pos["roi"] = roi_pos["roi"].replace([np.inf, -np.inf], np.nan)
    roi_pos = roi_pos[roi_pos["roi"] > 0].dropna(subset=["roi"])
    if roi_pos.empty:
        return go.Figure()

    # Clip extreme outliers so the log scale does not flatten the plot
    low, high = roi_pos["roi"].quantile([0.01, 0.99])
    high = high if high > 0 else roi_pos["roi"].max()
    x_min = max(low, 1e-6)
    if high <= x_min:
        high = x_min * 1.1
    roi_pos["roi_clipped"] = roi_pos["roi"].clip(lower=x_min, upper=high)

    fig = px.histogram(
        roi_pos,
        x="roi_clipped",
        nbins=40,
        color="is_blockbuster",
        color_discrete_map={0: "#7FB3FF", 1: "#FF7A7A"},
        title="ROI Distribution (Blockbuster vs Others, ROI > 0, 1–99th pct clipped)",
        log_x=True,
        opacity=0.75,
        labels={"roi_clipped": "ROI (clipped)"},
        range_x=(x_min, high),
    )
    fig.update_yaxes(title="Movies")
    return fig


def fig_popularity_hist(df: pd.DataFrame) -> go.Figure:
    data = df.copy()
    data["popularity"] = pd.to_numeric(data["popularity"], errors="coerce")
    data = data.dropna(subset=["popularity"])
    if data.empty:
        return go.Figure()
    low, high = data["popularity"].quantile([0.01, 0.99])
    data["popularity_clipped"] = data["popularity"].clip(lower=low, upper=high)
    fig = px.histogram(
        data,
        x="popularity_clipped",
        nbins=40,
        color="is_blockbuster",
        color_discrete_map={0: "#7FB3FF", 1: "#FF7A7A"},
        opacity=0.75,
        title="Popularity Distribution (clipped 1st–99th pct)",
        labels={"popularity_clipped": "Popularity (clipped)"},
    )
    fig.update_yaxes(title="Movies")
    return fig


def fig_popularity_box(df: pd.DataFrame) -> go.Figure:
    data = df.copy()
    data["popularity"] = pd.to_numeric(data["popularity"], errors="coerce")
    data = data.dropna(subset=["popularity"])
    if data.empty:
        return go.Figure()
    fig = px.box(
        data,
        x="is_blockbuster",
        y="popularity",
        color="is_blockbuster",
        color_discrete_map={0: "#7FB3FF", 1: "#FF7A7A"},
        title="Popularity vs Blockbuster Label",
        labels={"is_blockbuster": "Is Blockbuster", "popularity": "Popularity"},
    )
    fig.update_xaxes(tickvals=[0, 1], ticktext=["No", "Yes"])
    return fig


def fig_popularity_rate_curve(df: pd.DataFrame, bins: int = 12) -> go.Figure:
    data = df.copy()
    data["popularity"] = pd.to_numeric(data["popularity"], errors="coerce")
    data = data.dropna(subset=["popularity", "is_blockbuster"])
    if data.empty:
        return go.Figure()
    quantiles = data["popularity"].quantile(np.linspace(0, 1, bins + 1)).values
    quantiles[0] = -np.inf
    quantiles[-1] = np.inf
    data["pop_bin"] = pd.cut(data["popularity"], bins=quantiles, include_lowest=True, duplicates="drop")
    agg = (
        data.groupby("pop_bin")
        .agg(
            movies=("id", "count"),
            blockbuster_rate=("is_blockbuster", "mean"),
            median_pop=("popularity", "median"),
        )
        .reset_index()
    )
    agg = agg[agg["movies"] > 0]
    agg["blockbuster_rate_pct"] = agg["blockbuster_rate"] * 100
    fig = px.line(
        agg,
        x="median_pop",
        y="blockbuster_rate_pct",
        markers=True,
        title="Blockbuster likelihood by popularity (quantile bins)",
        labels={"median_pop": "Median popularity in bin", "blockbuster_rate_pct": "Blockbuster %"},
    )
    fig.update_traces(marker_color="#FDC267", line_color="#FDC267")
    fig.update_yaxes(range=[0, max(105, agg["blockbuster_rate_pct"].max() * 1.1)])
    return fig


def fig_popularity_runtime_animation(df: pd.DataFrame) -> go.Figure:
    data = df.copy()
    data["popularity"] = pd.to_numeric(data["popularity"], errors="coerce")
    data["runtime"] = pd.to_numeric(data["runtime"], errors="coerce")
    data = data.dropna(subset=["popularity", "runtime"])
    if data.empty:
        return go.Figure()

    bins = [0, 60, 90, 120, 150, 180, np.inf]
    labels = ["<60", "60–90", "90–120", "120–150", "150–180", "180+"]
    data["runtime_bin"] = pd.cut(data["runtime"], bins=bins, labels=labels, include_lowest=True)

    fig = px.histogram(
        data,
        x="popularity",
        color="is_blockbuster",
        color_discrete_map={0: "#7FB3FF", 1: "#FF7A7A"},
        nbins=35,
        animation_frame="runtime_bin",
        opacity=0.75,
        title="Popularity distribution across runtime bins",
        labels={"popularity": "Popularity", "runtime_bin": "Runtime (mins)"},
    )
    fig.update_layout(bargap=0.05, legend_title="is_blockbuster")
    return fig


def fig_popularity_density(df: pd.DataFrame) -> go.Figure:
    data = df.copy()
    data["popularity"] = pd.to_numeric(data["popularity"], errors="coerce")
    data = data.dropna(subset=["popularity", "is_blockbuster"])
    if data.empty:
        return go.Figure()
    # Clip to reduce long-tail compression
    low, high = data["popularity"].quantile([0.01, 0.99])
    data["popularity_clipped"] = data["popularity"].clip(lower=low, upper=high)

    fig = go.Figure()
    for label, color, name in [
        (1, "#FF7A7A", "Blockbuster"),
        (0, "#7FB3FF", "Non-Blockbuster"),
    ]:
        subset = data[data["is_blockbuster"] == label]
        fig.add_trace(
            go.Histogram(
                x=subset["popularity_clipped"],
                name=name,
                histnorm="probability density",
                opacity=0.5,
                marker_color=color,
                nbinsx=50,
            )
        )
    fig.update_layout(
        barmode="overlay",
        title="Popularity density by blockbuster label",
        xaxis_title="Popularity (clipped 1st–99th pct)",
        yaxis_title="Density",
        legend_title="",
    )
    fig.update_yaxes(showgrid=True)
    return fig


def fig_popularity_cdf(df: pd.DataFrame) -> go.Figure:
    data = df.copy()
    data["popularity"] = pd.to_numeric(data["popularity"], errors="coerce")
    data = data.dropna(subset=["popularity", "is_blockbuster"])
    if data.empty:
        return go.Figure()

    fig = go.Figure()
    for label, color in [(0, "#7FB3FF"), (1, "#FF7A7A")]:
        subset = data[data["is_blockbuster"] == label]["popularity"].sort_values()
        if subset.empty:
            continue
        y = np.linspace(0, 100, len(subset))
        fig.add_trace(
            go.Scatter(
                x=subset.values,
                y=y,
                mode="lines",
                name="Blockbuster" if label == 1 else "Non-Blockbuster",
                line=dict(color=color),
            )
        )
    fig.update_layout(
        title="Cumulative distribution of popularity",
        xaxis_title="Popularity",
        yaxis_title="Cumulative %",
        legend_title="",
    )
    return fig


def fig_popularity_logistic(df: pd.DataFrame, bins: int = 15) -> go.Figure:
    data = df.copy()
    data["popularity"] = pd.to_numeric(data["popularity"], errors="coerce")
    data = data.dropna(subset=["popularity", "is_blockbuster"])
    if data.empty:
        return go.Figure()

    # Bin by quantiles to stabilize rates
    quantiles = data["popularity"].quantile(np.linspace(0, 1, bins + 1)).values
    quantiles[0] = -np.inf
    quantiles[-1] = np.inf
    data["pop_bin"] = pd.cut(data["popularity"], bins=quantiles, include_lowest=True, duplicates="drop")
    agg = (
        data.groupby("pop_bin")
        .agg(
            rate=("is_blockbuster", "mean"),
            median_pop=("popularity", "median"),
            movies=("id", "count"),
        )
        .reset_index()
    )
    agg = agg[(agg["movies"] > 0) & agg["rate"].between(0, 1)]
    if agg.empty:
        return go.Figure()

    eps = 1e-4
    agg["rate_clipped"] = agg["rate"].clip(eps, 1 - eps)
    agg["logit"] = np.log(agg["rate_clipped"] / (1 - agg["rate_clipped"]))
    coef = np.polyfit(agg["median_pop"], agg["logit"], 1)
    x_min, x_max = data["popularity"].min(), data["popularity"].max()
    x_grid = np.linspace(x_min, x_max, 200)
    logits = coef[0] * x_grid + coef[1]
    preds = 1 / (1 + np.exp(-logits))

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=agg["median_pop"],
            y=agg["rate"] * 100,
            mode="markers",
            marker=dict(color="#FDC267", size=8),
            name="Bin rates",
            hovertemplate="Popularity (median): %{x:.2f}<br>Blockbuster %: %{y:.1f}%<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x_grid,
            y=preds * 100,
            mode="lines",
            line=dict(color="#FF7A7A"),
            name="Logistic fit",
            hovertemplate="Popularity: %{x:.2f}<br>Pred. Blockbuster %: %{y:.1f}%<extra></extra>",
        )
    )
    fig.update_layout(
        title="Probability of blockbuster vs popularity (logistic fit)",
        xaxis_title="Popularity",
        yaxis_title="Blockbuster %",
        legend_title="",
        yaxis=dict(range=[0, 105]),
    )
    return fig


def fig_budget_band_roi(df: pd.DataFrame) -> go.Figure:
    fig = px.box(
        df,
        x="budget_band",
        y="roi",
        color="is_blockbuster",
        color_discrete_map={0: "#7FB3FF", 1: "#FF7A7A"},
        title="ROI by Budget Band",
        labels={"budget_band": "Budget Band", "roi": "ROI"},
        category_orders={"budget_band": BUDGET_BAND_ORDER},
    )
    fig.update_layout(
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.18,
            xanchor="center",
            x=0.5,
            title="is_blockbuster",
        ),
        margin=dict(l=60, r=40, t=70, b=90),
    )
    fig.update_xaxes(categoryorder="array", categoryarray=BUDGET_BAND_ORDER)
    return fig


def fig_budget_band_success_rate(df: pd.DataFrame) -> go.Figure:
    data = df.dropna(subset=["budget_band"]).copy()
    if data.empty:
        return go.Figure()

    data["budget_band"] = pd.Categorical(data["budget_band"], categories=BUDGET_BAND_ORDER, ordered=True)
    data["is_blockbuster_label"] = data["is_blockbuster"].map({0: "0", 1: "1"}).fillna("0")

    counts = (
        data.groupby(["budget_band", "is_blockbuster_label"])
        .size()
        .reset_index(name="movies")
        .pivot(index="budget_band", columns="is_blockbuster_label", values="movies")
        .fillna(0)
        .reindex(BUDGET_BAND_ORDER)
    )
    counts = counts.rename(columns={"0": "Non-Blockbuster", "1": "Blockbuster"})
    totals = counts.sum(axis=1)
    rates = (counts.get("Blockbuster", 0) / totals.replace(0, np.nan) * 100).fillna(0)
    min_sample = 30
    rate_plot = rates.where(totals >= min_sample, other=np.nan)
    low_sample_points = rates.where(totals < min_sample, other=np.nan)

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Bar(
            x=counts.index,
            y=counts.get("Non-Blockbuster", 0),
            name="Non-Blockbuster",
            marker_color="#7FB3FF",
            hovertemplate="Budget band: %{x}<br>Non-Blockbuster: %{y}<extra></extra>",
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Bar(
            x=counts.index,
            y=counts.get("Blockbuster", 0),
            name="Blockbuster",
            marker_color="#FF7A7A",
            hovertemplate="Budget band: %{x}<br>Blockbuster: %{y}<extra></extra>",
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=rates.index,
            y=rate_plot,
            mode="lines+markers",
            name="Blockbuster %",
            marker_color="#FDC267",
            hovertemplate="Budget band: %{x}<br>Blockbuster %: %{y:.1f}%<extra></extra>",
            connectgaps=False,
        ),
        secondary_y=True,
    )
    fig.add_trace(
        go.Scatter(
            x=rates.index,
            y=low_sample_points,
            mode="markers",
            name="Blockbuster % (low n)",
            marker_color="#FDC267",
            marker=dict(symbol="circle-open", size=10, line=dict(color="#FDC267", width=1)),
            hovertemplate="Budget band: %{x}<br>Blockbuster %: %{y:.1f}%<br>n < 30 (low sample)<extra></extra>",
        ),
        secondary_y=True,
    )

    fig.update_layout(
        barmode="stack",
        title="Movies and Blockbuster Rate by Budget Band",
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.2,
            xanchor="center",
            x=0.5,
        ),
        legend_title_text="",
        bargap=0.25,
        margin=dict(l=70, r=30, t=70, b=120),
    )
    fig.update_xaxes(title="Budget Band", categoryorder="array", categoryarray=BUDGET_BAND_ORDER)
    fig.update_yaxes(title_text="Movies", secondary_y=False)
    fig.update_yaxes(title_text="Blockbuster %", secondary_y=True, range=[0, 105])
    fig.add_annotation(
        text="Blockbuster % shown only when band has ≥30 movies; hollow markers = low sample",
        xref="paper",
        yref="paper",
        x=0,
        y=1.08,
        showarrow=False,
        font=dict(size=11, color="#aaa"),
    )
    return fig


def fig_corr_heatmap(corr: pd.DataFrame) -> go.Figure:
    if corr.empty:
        return go.Figure()
    fig = px.imshow(
        corr,
        text_auto=".2f",
        color_continuous_scale="RdBu",
        zmin=-1,
        zmax=1,
        title="Correlation Heatmap",
    )
    fig.update_layout(height=800, width=1100)
    return fig


def fig_genre_year_heatmap(df: pd.DataFrame) -> go.Figure:
    fig = px.density_heatmap(
        df,
        x="release_year",
        y="main_genre",
        z="blockbuster_rate_pct",
        color_continuous_scale="Viridis",
        histfunc="avg",
        title="Blockbuster Activity by Genre and Year",
        labels={"blockbuster_rate_pct": "Blockbuster %"},
    )
    fig.update_xaxes(title="Year")
    fig.update_yaxes(title="Genre")
    return fig


def fig_people_bar(df: pd.DataFrame, name_col: str, value_col: str, title: str) -> go.Figure:
    top = df.head(15)
    fig = px.bar(
        top,
        x=value_col,
        y=name_col,
        orientation="h",
        title=title,
        color=value_col,
        color_continuous_scale="Bluered",
    )
    return fig


def fig_pair_scatter(df: pd.DataFrame) -> go.Figure:
    top = df.head(30)
    fig = px.scatter(
        top,
        x="movies",
        y="avg_revenue",
        color="blockbuster_rate_pct",
        hover_data=["director", "lead_actor"],
        title="Director–Actor Pairs",
        color_continuous_scale="Bluered",
    )
    fig.update_xaxes(title="Movies Together")
    fig.update_yaxes(title="Avg Revenue")
    return fig


def fig_blockbuster_share_pie(df: pd.DataFrame) -> go.Figure:
    counts = df["is_blockbuster"].value_counts().rename({0: "Other", 1: "Blockbuster"})
    fig = px.pie(
        names=counts.index,
        values=counts.values,
        title="Blockbuster Share",
        hole=0.4,
        color=counts.index,
        color_discrete_map={"Blockbuster": "#FF7A7A", "Other": "#7FB3FF"},
    )
    return fig


def fig_genre_mix_pie(df: pd.DataFrame) -> go.Figure:
    top_genres = (
        df[df["main_genre"] != "Unknown"]
        .groupby("main_genre")
        .size()
        .sort_values(ascending=False)
        .head(10)
        .reset_index(name="count")
    )
    fig = px.pie(
        top_genres,
        names="main_genre",
        values="count",
        title="Genre Mix (Top 10)",
        hole=0.35,
    )
    return fig


def fig_country_mix_pie(df: pd.DataFrame) -> go.Figure:
    top_countries = (
        df[df["main_country"] != "Unknown"]
        .groupby("main_country")
        .size()
        .sort_values(ascending=False)
        .head(10)
        .reset_index(name="count")
    )
    fig = px.pie(
        top_countries,
        names="main_country",
        values="count",
        title="Country Mix (Top 10)",
        hole=0.35,
    )
    return fig


SUNBURST_BAND_COLORS = {
    "17%+": "#b30000",
    "15–17%": "#d63b3b",
    "12–15%": "#f88686",
    "8–12%": "#8bb8ff",
    "<8%": "#3169c3",
}


def fig_country_sunburst(df: pd.DataFrame) -> go.Figure:
    if df.empty or "main_country" not in df.columns:
        return go.Figure()
    data = df[df["main_country"] != "Unknown"].copy()
    if data.empty:
        return go.Figure()
    data["continent"] = data["main_country"].apply(lambda c: CONTINENT_MAP.get(c))
    data = data[data["continent"].notna()]
    agg = (
        data.groupby(["continent", "main_country"])
        .agg(movies=("id", "count"), blockbusters=("is_blockbuster", "sum"))
        .reset_index()
    )
    agg = agg[agg["movies"] > 0]
    if agg.empty:
        return go.Figure()

    # Use blockbuster counts for sizing; fallback to movies when zero to keep visibility
    agg["size_value"] = agg["blockbusters"].where(agg["blockbusters"] > 0, agg["movies"])

    # Shares within continent based on size_value
    cont_totals = agg.groupby("continent")["size_value"].transform("sum")
    cont_totals = cont_totals.replace(0, np.nan)
    agg["share_in_continent_pct"] = agg["size_value"] / cont_totals * 100

    agg["blockbuster_rate_pct"] = agg["blockbusters"] / agg["movies"].replace(0, np.nan) * 100
    bins = [
        (agg["blockbuster_rate_pct"] >= 17, "17%+"),
        ((agg["blockbuster_rate_pct"] >= 15) & (agg["blockbuster_rate_pct"] < 17), "15–17%"),
        ((agg["blockbuster_rate_pct"] >= 12) & (agg["blockbuster_rate_pct"] < 15), "12–15%"),
        ((agg["blockbuster_rate_pct"] >= 8) & (agg["blockbuster_rate_pct"] < 12), "8–12%"),
        (agg["blockbuster_rate_pct"] < 8, "<8%"),
    ]
    agg["rate_band"] = "<8%"
    for mask, label in bins:
        agg.loc[mask, "rate_band"] = label

    # Show continents as root + their top 5 countries (by blockbuster count) using raw values
    agg = agg.sort_values(["continent", "size_value"], ascending=[True, False])
    top5 = agg.groupby("continent").head(5)
    if top5.empty:
        return go.Figure()

    labels = []
    parents = []
    values = []
    colors = []
    custom = []

    africa_boost = 8.0
    americas_damp = 0.7
    tiny_boost = 2.0

    for continent, grp in top5.groupby("continent"):
        cont_children = []
        cont_sum = 0.0
        cont_multiplier = africa_boost if continent == "Africa" else americas_damp if continent == "Americas" else 1.0
        for _, row in grp.iterrows():
            val = row["size_value"]
            if val <= 0:
                continue
            val_scaled = val * cont_multiplier
            if row["share_in_continent_pct"] < 3:
                val_scaled *= tiny_boost
            cont_children.append((row, val_scaled))
            cont_sum += val_scaled
        if cont_sum <= 0:
            continue
        labels.append(continent)
        parents.append("")
        values.append(cont_sum)
        colors.append("#888888")
        custom.append((continent, 100.0, ""))
        for row, val_scaled in cont_children:
            labels.append(row["main_country"])
            parents.append(continent)
            values.append(val_scaled)
            colors.append(SUNBURST_BAND_COLORS.get(row["rate_band"], "#3169c3"))
            custom.append((continent, row["share_in_continent_pct"], row["rate_band"]))

    fig = go.Figure(
        go.Sunburst(
            labels=labels,
            parents=parents,
            values=values,
            branchvalues="total",
            marker=dict(colors=colors, line=dict(color="#111", width=0.5)),
            hovertemplate="<b>%{label}</b><br>Continent: %{customdata[0]}<br>Share in continent: %{customdata[1]:.1f}%<br>Band: %{customdata[2]}<extra></extra>",
            customdata=custom,
            textinfo="label+percent entry",
        )
    )
    fig.update_layout(height=820, margin=dict(t=60, b=0, l=0, r=0), showlegend=True, title="Blockbuster mix by continent and country (top 5 per continent)")
    return fig


def fig_continent_pie(df: pd.DataFrame, continent: str | None) -> go.Figure:
    if df.empty or continent is None:
        return go.Figure()
    return go.Figure()

def fig_blockbuster_corr_bar(corr_df: pd.DataFrame) -> go.Figure:
    if corr_df.empty:
        return go.Figure()
    df = corr_df.copy()
    df["correlation"] = df["correlation"].round(2)
    fig = px.bar(
        df,
        x="correlation",
        y="feature",
        orientation="h",
        title="Correlation with Being a Blockbuster",
        labels={"correlation": "Correlation", "feature": "Feature"},
        color="correlation",
        color_continuous_scale="RdBu",
        range_color=[-1, 1],
        color_continuous_midpoint=0,
    )
    fig.update_layout(
        height=700,
        xaxis=dict(range=[-1, 1], zeroline=True, zerolinecolor="#888"),
    )
    return fig


def fig_company_blockbusters(df: pd.DataFrame) -> go.Figure:
    if df.empty:
        return go.Figure()
    top = df.head(20)
    fig = px.bar(
        top,
        x="blockbusters",
        y="production_companies_list",
        orientation="h",
        color="blockbuster_rate_pct",
        labels={
            "blockbusters": "Blockbusters",
            "production_companies_list": "Company",
            "blockbuster_rate_pct": "Blockbuster %",
        },
        title="Top Production Companies by Blockbusters",
        color_continuous_scale="RdBu",
    )
    return fig


def fig_franchise_split(df: pd.DataFrame) -> go.Figure:
    if df.empty:
        return go.Figure()
    df = df.copy()
    df["text_pct"] = df["blockbuster_rate_pct"].round(1)
    fig = px.bar(
        df,
        x="label",
        y="blockbuster_rate_pct",
        color="avg_revenue",
        labels={"blockbuster_rate_pct": "Blockbuster %", "label": ""},
        title="Franchise vs Standalone: Blockbuster Rate",
        color_continuous_scale="RdBu",
        text="text_pct",
    )
    fig.update_layout(yaxis_title="Blockbuster %", yaxis=dict(range=[0, 40]))
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside", cliponaxis=False)
    return fig


def fig_top_franchises(df: pd.DataFrame) -> go.Figure:
    if df.empty:
        return go.Figure()
    top = df.head(15)
    top = top.copy()
    top["text_blockbusters"] = top["blockbusters"]
    fig = px.bar(
        top,
        x="blockbusters",
        y="collection_name",
        orientation="h",
        color="blockbuster_rate_pct",
        labels={"blockbusters": "Blockbusters", "collection_name": "Franchise", "blockbuster_rate_pct": "Blockbuster %"},
        title="Top Franchises by Blockbuster Count",
        color_continuous_scale="RdBu",
        hover_data={"avg_revenue": ":,.0f"},
        text="text_blockbusters",
    )
    fig.update_traces(texttemplate="%{text}", textposition="outside", cliponaxis=False)
    return fig
