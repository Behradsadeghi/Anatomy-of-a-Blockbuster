import streamlit as st
import pandas as pd
from src.data_loader import load_all
from src.preprocess import preprocess_movies
from src.analysis import (
    overview_metrics,
    yearly_blockbuster_trend,
    season_blockbuster_stats,
    month_blockbuster_stats,
    genre_performance,
    genre_year_heatmap,
    budget_revenue_points,
    budget_band_stats,
    country_revenue,
    top_people,
    top_actor_director_pairs,
    corr_matrix,
    blockbuster_correlations,
    franchise_summary,
    top_franchises,
    company_stats,
)
from src.visualization import (
    fig_yearly_trend,
    fig_budget_revenue_scatter,
    fig_genre_roi_box,
    fig_genre_blockbuster_bar,
    fig_season_blockbuster,
    fig_month_blockbuster,
    fig_budget_band_roi,
    fig_budget_band_success_rate,
    fig_popularity_hist,
    fig_popularity_box,
    fig_popularity_rate_curve,
    fig_popularity_runtime_animation,
    fig_popularity_density,
    fig_popularity_cdf,
    fig_popularity_logistic,
    fig_corr_heatmap,
    fig_genre_year_heatmap,
    fig_people_bar,
    fig_pair_scatter,
    fig_blockbuster_share_pie,
    fig_genre_mix_pie,
    fig_country_sunburst,
    fig_blockbuster_corr_bar,
    fig_franchise_split,
    fig_top_franchises,
    fig_company_blockbusters,
    SUNBURST_BAND_COLORS,
    CONTINENT_MAP,
)
from src.pydeck_utils import aggregate_country_revenue, revenue_map_deck, COUNTRY_COORDS

st.set_page_config(
    page_title="Anatomy of a Blockbuster",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    [data-testid="stSidebar"] {
        min-width: 230px;
        max-width: 230px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Anatomy of a Blockbuster")

datasets = load_all(include_credits=True)
movies_clean = preprocess_movies(datasets["movies"], credits=datasets["credits"], use_disk_cache=True)

# Filters
year_min, year_max = int(movies_clean["release_year"].min()), int(movies_clean["release_year"].max())
st.sidebar.markdown("### Filters")
selected_years = st.sidebar.slider("Release year", year_min, year_max, (year_min, year_max))
all_genres = sorted([g for g in movies_clean["main_genre"].dropna().unique() if g != "Unknown"])
selected_genres = st.sidebar.multiselect("Genres", all_genres, default=None)
country_counts = (
    movies_clean[movies_clean["main_country"].notna() & (movies_clean["main_country"] != "Unknown")]
    .groupby("main_country")
    .size()
    .sort_values(ascending=False)
)
from src.pydeck_utils import COUNTRY_COORDS
all_countries = [c for c in country_counts.index if c in COUNTRY_COORDS]
selected_countries = st.sidebar.multiselect("Countries", all_countries, default=None)

filtered = movies_clean.copy()
filtered = filtered[(filtered["release_year"] >= selected_years[0]) & (filtered["release_year"] <= selected_years[1])]
if selected_genres:
    filtered = filtered[filtered["main_genre"].isin(selected_genres)]
if selected_countries:
    filtered = filtered[filtered["main_country"].isin(selected_countries)]

if filtered.empty:
    st.warning("No data for these filters. Loosen them to see results.")
    st.stop()

# Data facts
clean_rows = len(movies_clean)
raw_rows = len(datasets["movies"]) if datasets.get("movies") is not None else clean_rows
unknown_genres = int((movies_clean["main_genre"] == "Unknown").sum())
unknown_countries = int((movies_clean["main_country"] == "Unknown").sum())

(
    tab_method,
    tab_facts,
    tab_trend,
    tab_econ,
    tab_genre,
    tab_franchise,
    tab_geo,
    tab_people,
) = st.tabs([
    "Methodology",
    "General Facts",
    "Trends & Timing",
    "Economics & Scale",
    "Genres & ROI",
    "Franchises",
    "Geography & Markets",
    "People & Companies",
])

with tab_method:
    st.markdown("### Data pipeline & methodology")
    st.markdown("""
    **Ingestion & typing**
    - Movies metadata + lean credits (lead actor, director) loaded from CSV; IDs, budget/revenue, popularity, runtime, votes cast to numeric. Dates parsed → year/month/season/name.

    **Cleaning & validation**
    - Drop duplicates; remove rows with >1 key field missing (budget, revenue, popularity, runtime, vote_average, vote_count, release_year, main_genre, main_country).
    - Zero/negative budget/revenue → NA; runtime <20 or >400 dropped; budget <50k dropped; negative revenue dropped.
    - Parse genres/countries/companies from JSON/Python-literal/pipe/comma; first entry becomes `main_genre` / `main_country` / `main_company`; unknowns marked “Unknown”.
    - Genre-median then global-median fill for budget/revenue/runtime; ROI infinities → 0; profit = revenue − budget.
    - Removed one extreme ROI outlier (id 2667) to stabilize visuals.

    **Enrichment**
    - Derived fields: release_month, release_month_name, season, budget_band, roi, profit.
    - People/companies: merged lean credits; company lists parsed.
    - Franchise: parsed `belongs_to_collection` → `collection_name`; `is_franchise` flag.
    - Blockbuster label: percentile rule (85th revenue, 75th ROI/popularity) or top 10% hybrid score; stored as `is_blockbuster`.

    **Caching & performance**
    - Cleaned data cached to `data/movies_clean_full.pkl`; credits cached to `credits_lead.pkl`; loaders only pull needed columns.

    **What we adjusted in code**
    - Added richer visuals (budget bands, genres, geography, franchises, talent) and captions.
    - Cleaned legends/layouts, added revenue formatting, and removed the popularity tab and unused helper modules.
    - Tightened git hygiene (README, .gitignore, requirements) for publishing.
    """)

with tab_facts:
    st.markdown("### General facts")
    pies_blockbusters_only = st.checkbox("Show only blockbusters (Data Facts)", value=False)
    pie_data = filtered
    if pies_blockbusters_only:
        blockbusters_only = filtered[filtered["is_blockbuster"] == 1]
        if blockbusters_only.empty:
            st.warning("No blockbusters for these filters. Showing all movies instead.")
        else:
            pie_data = blockbusters_only

    # KPIs
    stats = overview_metrics(pie_data)
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Movies", f"{stats['movies']:,}")
    c2.metric("Blockbusters", f"{stats['blockbusters']:,}")
    def _abbr_money(x: float) -> str:
        if x is None or pd.isna(x):
            return "-"
        if abs(x) >= 1_000_000_000:
            return f"${x/1_000_000_000:.2f}B"
        if abs(x) >= 1_000_000:
            return f"${x/1_000_000:.2f}M"
        if abs(x) >= 1_000:
            return f"${x/1_000:.2f}K"
        return f"${x:.2f}"

    c3.metric("Median Budget", _abbr_money(stats["median_budget"]))
    c4.metric("Median Revenue", _abbr_money(stats["median_revenue"]))
    c5.metric("Median ROI", f"{stats['median_roi']:.2f}x")

    cfa, cfb = st.columns(2)
    with cfa:
        st.plotly_chart(fig_blockbuster_share_pie(pie_data), use_container_width=True)
    with cfb:
        st.plotly_chart(fig_genre_mix_pie(pie_data), use_container_width=True)
    with st.expander("Caption + analysis"):
        st.write(
            "This section summarizes key characteristics of the dataset. Out of 43,928 movies, 4,932 are classified as blockbusters, indicating that they represent a relatively small portion of the total. The median budget is $6.88M, while the median revenue is $14.10M, and the median ROI of 0.72× shows that the typical film does not fully recover its costs. The Blockbuster Share chart visually reflects the small size of the blockbuster segment, and the Genre Mix chart shows that genres such as Drama and Comedy appear most frequently among the top ten categories in the dataset. Collectively, these visuals outline the dataset’s scale, the limited presence of blockbuster films, and the distribution of movies across major genres."
        )
    st.plotly_chart(fig_country_sunburst(pie_data), use_container_width=True)
    legend_boxes = "".join(
        f"<div style='display:flex; align-items:center; gap:6px;'><div style='width:14px; height:14px; background:{color}; border-radius:3px;'></div><span style='font-size:12px; color:#aaa;'>{label}</span></div>"
        for label, color in [
            ("17%+", SUNBURST_BAND_COLORS["17%+"]),
            ("15–17%", SUNBURST_BAND_COLORS["15–17%"]),
            ("12–15%", SUNBURST_BAND_COLORS["12–15%"]),
            ("8–12%", SUNBURST_BAND_COLORS["8–12%"]),
            ("<8%", SUNBURST_BAND_COLORS["<8%"]),
        ]
    )
    st.markdown(
        f"<div style='display:flex; gap:16px; flex-wrap:wrap; margin-bottom:8px;'>{legend_boxes}</div>",
        unsafe_allow_html=True,
    )
    with st.expander("Caption + analysis"):
        st.write(
            "This visualization shows how blockbuster movies are distributed across continents and, within each continent, across the top five contributing countries. The inner ring indicates that the Americas account for the largest share of blockbusters (58%), followed by Europe (26%), Asia (8%), Africa (5%), and Oceania (3%). The outer ring breaks these continental segments down by country. The United States dominates the distribution with 53% of all blockbusters, while other countries such as the United Kingdom (10%), Poland (6%), France (5%), Germany (4%), Canada (4%), and several Asian countries contribute smaller shares. Overall, the chart highlights that blockbuster production is heavily concentrated in a few countries, with a particularly strong concentration in the United States and, to a lesser extent, Western Europe."
        )

    st.markdown("### Correlations")
    st.plotly_chart(fig_corr_heatmap(corr_matrix(pie_data)), use_container_width=True)
    corr_features = blockbuster_correlations(pie_data)
    st.plotly_chart(fig_blockbuster_corr_bar(corr_features), use_container_width=True)
    with st.expander("Caption + analysis"):
        st.write(
            "The heatmap shows that budget, revenue, vote_count, and popularity are strongly connected: higher-budget films tend to earn more, receive more votes, and be more popular. ROI has almost no correlation with these features, indicating that financial scale and financial efficiency behave differently. Blockbuster_score is strongly correlated with revenue (0.66), vote_count (0.65), popularity (0.62), and budget (0.58), meaning blockbuster films typically involve larger financial scale and higher audience engagement.\n\nThe second chart confirms this pattern: being a blockbuster is most associated with vote_count, popularity, revenue, and budget. Runtime, ratings, ROI, and release season show very weak or near-zero correlations, suggesting they play little role in distinguishing blockbuster films."
        )

with tab_trend:
    st.markdown("### Release timing and blockbuster share")
    yearly = yearly_blockbuster_trend(filtered)
    st.plotly_chart(fig_yearly_trend(yearly), use_container_width=True)
    col1, col2 = st.columns(2)
    with col1:
        season_stats = season_blockbuster_stats(filtered)
        st.plotly_chart(fig_season_blockbuster(season_stats), use_container_width=True)
    with col2:
        month_stats = month_blockbuster_stats(filtered)
        st.plotly_chart(fig_month_blockbuster(month_stats), use_container_width=True)
    with st.expander("Caption + analysis"):
        st.write(
            "The yearly chart shows that film production increases steadily over time, while the blockbuster percentage moves up and down but generally trends upward until the most recent years, where it suddenly drops. This indicates that the likelihood of producing a blockbuster has changed over time and is not solely determined by the overall number of movies released.\n\nThe seasonal chart shows that all four seasons generate large volumes of movies, but the size of the red segments indicates that the share of blockbusters is not evenly distributed across seasons. Fall and Winter appear to have slightly more blockbusters in absolute terms compared to Spring and Summer.\n\nThe monthly chart provides a clearer pattern: the blockbuster percentage rises noticeably around mid-year (June–July) and again toward the end of the year (November–December). These months show higher blockbuster percentages despite total output remaining relatively stable across most months. This suggests that release timing—especially early summer and late-year periods—is associated with a higher concentration of blockbusters.\n\nAcross these plots, the key insight is that blockbuster likelihood varies over time and is influenced by when a movie is released. Mid-year and end-of-year periods show higher blockbuster shares, while production volume remains high throughout the year. This indicates that timing plays a meaningful role in blockbuster performance."
        )

with tab_econ:
    st.markdown("### Profitability vs scale")
    st.plotly_chart(fig_budget_revenue_scatter(budget_revenue_points(filtered, selected_years)), use_container_width=True)
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(fig_budget_band_roi(filtered), use_container_width=True)
    with col2:
        st.plotly_chart(fig_budget_band_success_rate(filtered), use_container_width=True)
    band_stats = budget_band_stats(filtered)
    st.dataframe(
        band_stats[["budget_band", "movies", "blockbuster_rate_pct", "median_roi"]]
        .sort_values("blockbuster_rate_pct", ascending=False)
        .style.format({"blockbuster_rate_pct": "{:.1f}%", "median_roi": "{:.2f}"})
    )
    with st.expander("Caption + analysis"):
        st.write(
            "The log–log budget–revenue scatter shows a clear upward trend: higher-budget films generally achieve higher revenues, and blockbusters cluster in the upper-right region where both values are high. Non-blockbusters are widely dispersed across the lower-budget range.\n\n"
            "The ROI-by-budget-band plot shows that low-budget films (<10M) have highly variable ROI, including many extreme outliers, while higher-budget bands show more stable and less volatile ROI values. Blockbusters appear far more consistently in the mid-to-high budget categories.\n\n"
            "The combined bar-and-line chart and the summary table reinforce a steep budget gradient in blockbuster likelihood. The table shows that only 5.7% of films under <10M become blockbusters, compared to 23.1% in the 10–50M band, 83.3% in the 50–100M range, 98.4% in the 100–250M group, and 100% above 250M. Median ROI also increases toward the higher-budget tiers (from 0.58 in <10M to around 0.8–1.77 in the upper bands).\n\n"
            "Together, these visuals and the table indicate that budget scale is one of the strongest drivers of blockbuster outcomes: higher-budget films tend to earn more, show more stable ROI, and display dramatically higher blockbuster rates."
        )

with tab_genre:
    st.markdown("### Genre performance and ROI distribution")
    genre_stats = genre_performance(filtered)
    st.plotly_chart(fig_genre_blockbuster_bar(genre_stats), use_container_width=True)
    st.plotly_chart(fig_genre_roi_box(filtered), use_container_width=True)
    st.plotly_chart(fig_genre_year_heatmap(genre_year_heatmap(filtered)), use_container_width=True)
    with st.expander("Caption + analysis"):
        st.write(
            "Across all three visuals, genre emerges as one of the strongest determinants of blockbuster success. Genres such as Adventure, Animation, Fantasy, Science Fiction, and Action consistently stand out: they show the highest blockbuster percentages, the strongest clusters of high-ROI blockbuster titles, and the most sustained blockbuster activity across recent decades. In contrast, grounded genres like Drama, Romance, Documentary, Music, and Crime maintain low blockbuster shares, limited ROI dispersion, and weak historical blockbuster presence.\n\n"
            "Taken together, the evidence indicates that blockbuster performance is not evenly distributed across the film landscape—rather, it is heavily concentrated in large-scale, effects-driven genres whose structure, audience appeal, and market positioning make them far more likely to achieve both commercial scale and durable blockbuster outcomes."
        )

with tab_franchise:
    st.markdown("### Franchise impact on blockbusters")
    summary = franchise_summary(filtered)
    top_fr = top_franchises(filtered)
    if summary.empty and top_fr.empty:
        st.info("No franchise information available for these filters.")
    else:
        st.plotly_chart(fig_franchise_split(summary), use_container_width=True, key="franchise_split_chart")
        st.plotly_chart(fig_top_franchises(top_fr), use_container_width=True, key="franchise_top_chart")
        if not top_fr.empty:
            st.dataframe(
                top_fr[["collection_name", "movies", "blockbusters", "blockbuster_rate_pct", "avg_revenue"]]
                .head(30)
                .style.format({"blockbuster_rate_pct": "{:.1f}%", "avg_revenue": "{:,.0f}M"})
            )
        with st.expander("Caption + analysis"):
            st.write(
                "The franchise vs standalone comparison shows a substantial gap in blockbuster performance. Franchise films reach a blockbuster rate above 30%, whereas standalone films remain around 10%. The color scale indicates that franchises also achieve higher average revenues, reinforcing the financial advantage of serialized IP.\n\n"
                "The ranking of franchises by blockbuster count further highlights this pattern. Major long-running collections such as James Bond, Star Wars, Harry Potter, Fast & Furious, Rocky, and Saw achieve a 100% blockbuster rate—with every film in the collection meeting the blockbuster criteria. Several mid-sized franchises (e.g., Pokémon, X-Men, Child’s Play) also produce multiple blockbusters, though with slightly lower blockbuster percentages.\n\n"
                "The summary table confirms these trends: the top-ranking franchises not only have the highest blockbuster counts but also show exceptionally high average revenues, particularly Star Wars and Harry Potter, which exceed $900M on average. This demonstrates that franchises both scale and sustain blockbuster success at levels unattainable for standalone films.\n\n"
                "Overall, the visuals make clear that franchise status is one of the strongest predictors of blockbuster outcomes, combining higher blockbuster likelihood, higher revenue, and consistent performance across multiple entries."
            )

with tab_geo:
    st.markdown("### Geography and market mix")
    country_agg = aggregate_country_revenue(filtered[filtered["main_country"] != "Unknown"])
    st.pydeck_chart(revenue_map_deck(country_agg), use_container_width=True)
    max_blocks = int(country_agg["blockbusters"].max()) if not country_agg.empty else 0
    min_blocks = int(country_agg["blockbusters"].min()) if not country_agg.empty else 0
    st.markdown(
        f"""
        <div style="margin-top:4px; margin-bottom:8px;">
            <div style="font-size:12px; color:#888;">Color: blockbuster count (deep red = max, blue = min)</div>
            <div style="display:flex; align-items:center; gap:8px; margin-top:4px;">
                <div style="font-size:12px; color:#888; min-width:60px;">Low ({min_blocks})</div>
                <div style="flex:1; height:12px; background: linear-gradient(90deg, #b4d7ff, #ffd2d2, #f88282, #e03030, #9e0000); border-radius:10px;"></div>
                <div style="font-size:12px; color:#888; min-width:60px; text-align:right;">High ({max_blocks})</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("**Country leaderboard**")
    def _abbr_money_short(x: float) -> str:
        if x is None or pd.isna(x):
            return "-"
        ax = abs(x)
        if ax >= 1_000_000_000:
            return f"${x/1_000_000_000:.2f}B"
        if ax >= 1_000_000:
            return f"${x/1_000_000:.2f}M"
        if ax >= 1_000:
            return f"${x/1_000:.0f}K"
        return f"${x:.0f}"
    st.dataframe(
        country_revenue(filtered)[["main_country", "total_revenue", "movies", "blockbusters", "blockbuster_rate_pct"]]
        .head(20)
        .style.format({"total_revenue": _abbr_money_short, "blockbuster_rate_pct": "{:.1f}%"})
    )
    with st.expander("Caption + analysis"):
        st.write(
            "The geographic heatmap shows a strong concentration of blockbuster production in a small group of countries. The United States dominates by a large margin, with the deepest red shading indicating the highest blockbuster count globally. Other countries such as the United Kingdom, Canada, France, Germany, Japan, Australia, and Italy show noticeably lighter colors, reflecting much lower blockbuster volumes.\n\n"
            "The country leaderboard confirms this imbalance. The United States leads with 3,134 blockbusters out of 17,851 films (17.6% rate), far exceeding all other markets in both absolute blockbuster count and total revenue. The United Kingdom ranks second with 408 blockbusters (13.6%), and Canada follows closely with 208 blockbusters (14.4%). European countries such as France and Germany contribute moderate volumes, but with lower blockbuster rates. Asian markets like Japan, India, and China generate meaningful output but still show relatively small blockbuster counts compared to North America.\n\n"
            "Taken together, these visuals show that blockbuster production is geographically concentrated: the U.S. overwhelmingly dominates both output and revenue, while a small set of English-speaking or Western countries forms the secondary tier. Other regions contribute far fewer blockbusters, indicating that market size, industry scale, and distribution infrastructure play major roles in driving blockbuster creation."
        )

with tab_people:
    st.markdown("### Talent impact (actors, directors, pairs)")
    actors = top_people(filtered, role_col="lead_actor", min_movies=4)
    directors = top_people(filtered, role_col="director", min_movies=3)
    pairs = top_actor_director_pairs(filtered, min_movies=3)
    companies = company_stats(filtered, min_movies=8)

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(fig_people_bar(actors, "lead_actor", "avg_revenue", "Top Lead Actors by Revenue"), use_container_width=True)
    with col2:
        directors_sorted = directors.sort_values("blockbuster_rate_pct", ascending=False)
        st.plotly_chart(
            fig_people_bar(directors_sorted, "director", "blockbuster_rate_pct", "Top Directors by Blockbuster %"),
            use_container_width=True,
        )

    st.plotly_chart(fig_pair_scatter(pairs), use_container_width=True)

    with st.expander("Tables for People and Talents"):
        st.markdown("Top actors")
        st.dataframe(
            actors.head(30).style.format(
                {
                    "avg_revenue": "{:,.2f}B",
                    "avg_roi": "{:.2f}",
                    "blockbuster_rate_pct": "{:.1f}%",
                }
            )
        )
        st.markdown("Top directors")
        st.dataframe(
            directors.head(30).style.format(
                {
                    "avg_revenue": "{:,.2f}B",
                    "avg_roi": "{:.2f}",
                    "blockbuster_rate_pct": "{:.1f}%",
                }
            )
        )
        st.markdown("Top director–actor pairs")
        st.dataframe(
            pairs.head(30).style.format(
                {
                    "avg_revenue": "{:,.2f}B",
                    "blockbuster_rate_pct": "{:.1f}%",
                }
            )
        )

    with st.expander("Caption + analysis"):
        st.write(
            "The talent-level data shows a clear pattern: blockbuster success rarely comes from individual star power alone—it emerges from stable creative ecosystems.\n\n"
            "**Actors**\n"
            "- High-average-revenue actors (e.g., Chris Pratt, Will Smith, Mike Myers) overwhelmingly appear in franchise-heavy roles.\n"
            "- Their performance is less about individual draw and more about being embedded in established IP (Marvel, Transformers, animation sequels, etc.).\n"
            "- When accounting for film counts, only a small group sustains consistent hits, confirming that repetition inside a strong universe drives revenue reliability.\n\n"
            "**Directors**\n"
            "- Many directors show near-100% blockbuster rates, but almost all of them work in animation studios (Pixar, DreamWorks), superhero/comic universes, or long-running franchises.\n"
            "- Their “success rate” reflects studio strategy, not randomness—big studios repeatedly assign these directors to high-certainty projects.\n\n"
            "**Director–Actor Pairs**\n"
            "- Pairs like David Yates × Daniel Radcliffe, Peter Jackson × Elijah Wood, Nolan × Christian Bale achieve 100% blockbuster rate and $800M–$1B+ average revenue.\n"
            "- These outcomes highlight that repeated collaboration inside a franchise ecosystem outperforms individual talent metrics.\n\n"
            "**Overall Insight**\n"
            " Blockbuster performance is fundamentally collaborative and franchise-driven. Actors, directors, and pairs succeed not in isolation, but when they participate in strong intellectual properties with consistent production teams."
        )

    st.markdown("### Production companies and blockbuster hit rate")
    st.plotly_chart(fig_company_blockbusters(companies), use_container_width=True, key="company_blockbusters_chart")
    if not companies.empty:
        st.dataframe(
            companies[["production_companies_list", "movies", "blockbusters", "blockbuster_rate_pct", "avg_revenue", "avg_roi"]]
            .head(30)
            .style.format(
                {
                    "blockbuster_rate_pct": "{:.1f}%",
                    "avg_revenue": "{:,.2f}B",
                    "avg_roi": "{:.2f}",
                }
            )
        )

    with st.expander("Caption + analysis"):
        st.write(
            "The company-level data reveals that blockbuster creation is largely controlled by a small number of global studios.\n\n"
            "**Major Studios Dominate Output**\n"
            "Warner Bros., Universal, Paramount, 20th Century Fox, and Columbia produce the highest number of total films, the highest number of blockbusters, and control the largest franchises. Because they operate at massive scale, they can take more risks while maintaining reliable franchise pipelines.\n\n"
            "**Hit Rate vs Scale**\n"
            "High blockbuster counts come from high output. Smaller studios like New Line Cinema or Relativity Media show high blockbuster percentages, but only because they produce fewer films, focus on specific commercially safe genres (horror, fantasy, animation), and rely on franchise spin-offs.\n\n"
            "**Commercial Drivers**\n"
            "Production companies with the strongest blockbuster rates typically share the following traits: ownership of long-running IP, strong distribution networks, global marketing capacity, tight control over director–actor ecosystems, and frequent use of sequels, reboots, and shared universes.\n\n"
            "**Overall Insight**\n"
            " Studios are the primary engines of blockbuster creation. Blockbuster concentration reflects decades of franchise ownership, distribution power, and strategic reuse of proven formulas."
        )
