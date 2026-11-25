"""
NBA Rising Star Predictor - streamlit_app.py  (OPTION 2: 10-feature model)

Option 2 uses ten features: age, minutes, points, usage %, true shooting %, assist %,
free throw attempts, BPM, win shares per 48 and the change in BPM from last season.

Deployed the MIS 545 way: Streamlit Community Cloud + a GitHub Codespace.
The notebook (NBA_Rising_Star_Predictor.ipynb) trains the models in Colab and saves
everything this app needs into ONE file with joblib:  nba_model.joblib
Put that file next to this script in the repo root.

What is inside nba_model.joblib:
  - models:   the three final models (scikit-learn pipelines) trained on 2000-2019
  - features: the feature columns the models expect
  - tables:   walk-forward scores 2006-2024, per-player SHAP values, salary paths,
              and the 2021-2024 test results (stored as plain Python lists)
  - versions: the scikit-learn / pandas / numpy versions used in Colab

Scores are shown as a 0-100 percentile rank within the season (100 = most likely
future All-Star). The models' raw probabilities are not calibrated, so the app
does not present them as chances.
"""
import altair as alt
import joblib
import numpy as np
import pandas as pd
import sklearn
import streamlit as st

MODEL_FILE = "nba_model.joblib"

st.set_page_config(page_title="NBA Rising Star Predictor", page_icon="🏀", layout="wide")

MODEL_COLS = {
    "Gradient Boosting": "score_gradient_boosting",
    "Random Forest": "score_random_forest",
    "Logistic Regression": "score_logistic_regression",
}

FEATURE_LABELS = {
    "pts_per_game": "Points per game", "per": "PER", "fta_per_game": "Free throws attempted per game",
    "ws": "Win shares", "age": "Age", "vorp": "VORP", "rookie_scale": "On rookie-scale deal",
    "trb_per_game": "Rebounds per game", "ws_48": "Win shares per 48", "ast_per_game": "Assists per game",
    "tov_per_game": "Turnovers per game", "experience": "Years in league", "tov_percent": "Turnover %",
    "salary_pctile": "Salary percentile", "mp_per_game": "Minutes per game", "bpm": "Box plus/minus",
    "obpm": "Offensive BPM", "dbpm": "Defensive BPM", "usg_percent": "Usage %", "ts_percent": "True shooting %",
    "e_fg_percent": "Effective FG %", "ft_percent": "Free throw %", "fga_per_game": "Shots per game",
    "x3pa_per_game": "3PA per game", "stl_per_game": "Steals per game", "blk_per_game": "Blocks per game",
    "g": "Games", "gs": "Games started", "total_mp": "Total minutes", "overall_pick": "Draft pick",
    "undrafted": "Undrafted", "ast_percent": "Assist %", "trb_percent": "Rebound %",
    "stl_percent": "Steal %", "blk_percent": "Block %",
}


def label(feature: str) -> str:
    if feature.startswith("missingindicator_"):
        return f"No data: {label(feature.removeprefix('missingindicator_'))}"
    if feature.endswith("_delta"):
        return f"Change in {label(feature.removesuffix('_delta')).lower()} vs last season"
    if feature.startswith("pos_"):
        return f"Position: {feature.removeprefix('pos_')}"
    return FEATURE_LABELS.get(feature, feature)


@st.cache_resource
def load_bundle():
    """Load the trained models and app data saved by the notebook with joblib."""
    return joblib.load(MODEL_FILE)


bundle = load_bundle()
tables = {name: pd.DataFrame(cols) for name, cols in bundle["tables"].items()}
scores, shap_long = tables["scores"], tables["shap"]
salary_paths, test_results = tables["salary_paths"], tables["test_results"]
models, NUMERIC, CATEGORICAL = bundle["models"], bundle["numeric"], bundle["categorical"]

if sklearn.__version__ != bundle["versions"]["scikit-learn"]:
    st.warning(f"This app runs scikit-learn {sklearn.__version__}, but the model was trained with "
               f"{bundle['versions']['scikit-learn']}. Pin the same version with uv add if predictions fail.")
seasons = sorted(scores["season"].unique())
latest = seasons[-1]


def season_label(s: int) -> str:
    return f"{s - 1}-{str(s)[-2:]}"


def outcome_text(r) -> str:
    if r["allstar_next1"] == 1:
        return f"✅ {season_label(r['season'] + 1)}"
    if r["allstar_next2"] == 1:
        return f"✅ {season_label(r['season'] + 2)}"
    return "—"


HEADSHOT = "https://cdn.nba.com/headshots/nba/latest/1040x760/{}.png"
FALLBACK = "https://cdn.nba.com/headshots/nba/latest/1040x760/fallback.png"


def headshot_url(nba_id):
    """Official NBA.com headshot, loaded by the viewer's browser (nothing is stored in the app).
    nba_id is the NBA.com person id that came with the salary data."""
    if "nba_id" not in scores.columns or pd.isna(nba_id):
        return None
    return HEADSHOT.format(int(nba_id))


def photo(nba_id, width: int = 180) -> None:
    url = headshot_url(nba_id) or FALLBACK
    st.markdown(f'<img src="{url}" width="{width}" style="border-radius:8px;background:#f0f2f6" '
                f'onerror="this.onerror=null;this.src=\'{FALLBACK}\';">', unsafe_allow_html=True)


def tier(score: float) -> str:
    return "High potential" if score >= 97 else "Watch list" if score >= 90 else "—"


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.title("🏀 Rising Star Predictor")
st.sidebar.caption("**Option 2:** 10-feature model")
page = st.sidebar.radio("Page", ["Emerging Players", "Player Predictor", "Model Explainability",
                                 "Salary Analysis"])
model_name = st.sidebar.selectbox("Model", list(MODEL_COLS), index=0,
                                  help="Player-level explanations use Gradient Boosting.")
score_col = MODEL_COLS[model_name]
midtier_only = st.sidebar.checkbox("Mid-tier salaries only (25th-75th percentile)", value=True)
show_outcome = st.sidebar.checkbox("Show what actually happened", value=True,
                                   help="Turn off for a 'blind' view, as a front office would see it at the time.")
st.sidebar.caption(
    "Eligible players: never an All-Star, 20+ games that season. "
    "**Score** = percentile rank among that season's eligible players (100 = top). "
    "Every score comes from a model trained only on earlier seasons."
)


def pool(season: int) -> pd.DataFrame:
    d = scores[scores["season"] == season].copy()
    if midtier_only:
        d = d[d["eligible_midtier"] == 1]
    d = d.sort_values(score_col, ascending=False)
    d["rank"] = range(1, len(d) + 1)
    return d


# ---------------------------------------------------------------------------
# Page 1: Emerging Players dashboard
# ---------------------------------------------------------------------------
if page == "Emerging Players":
    st.title("Emerging Players")
    c1, c2 = st.columns([1, 2])
    season = c1.selectbox("Season", seasons[::-1], format_func=season_label)
    top_n = c2.slider("Players to show", 5, 50, 15)
    d = pool(season)
    st.caption(f"{len(d)} eligible players in {season_label(season)}"
               f"{' with mid-tier salaries' if midtier_only else ''}. Ranked by {model_name}.")

    if show_outcome:
        hits = int(d.head(10)["target"].sum())
        m1, m2, m3 = st.columns(3)
        m1.metric("Top 10 who became first-time All-Stars", f"{hits} of 10")
        m2.metric("Base rate (random pick)", f"{d['target'].mean():.1%}")
        m3.metric("Future first-time All-Stars in this pool", int(d["target"].sum()))

    view = d.head(top_n).assign(
        **{"Salary ($M)": lambda x: (x["salary"] / 1e6).round(2),
           "Salary pct": lambda x: x["salary_pctile"].round(0),
           "Score": lambda x: x[score_col],
           "Tier": lambda x: x[score_col].map(tier),
           "All-Star within 2 seasons": lambda x: x.apply(outcome_text, axis=1),
           "Photo": lambda x: x.get("nba_id", pd.Series(index=x.index, dtype=float)).map(headshot_url)})
    cols = ["rank", "Photo", "player", "team", "pos", "age", "Salary ($M)", "Salary pct", "pts_per_game",
            "per", "bpm", "Score", "Tier"] + (["All-Star within 2 seasons"] if show_outcome else [])
    st.dataframe(
        view[cols].rename(columns={"rank": "Rank", "player": "Player", "team": "Team", "pos": "Pos",
                                   "age": "Age", "pts_per_game": "PTS", "per": "PER", "bpm": "BPM"}),
        hide_index=True, width="stretch",
        column_config={"Score": st.column_config.ProgressColumn("Score", min_value=0, max_value=100,
                                                                format="%.1f"),
                       "Photo": st.column_config.ImageColumn("", width="small")})
    st.caption("Tiers: High potential = score 97+, Watch list = 90-97.")

    st.subheader("Score vs salary")
    chart = alt.Chart(d).mark_circle(size=60, opacity=0.7).encode(
        x=alt.X("salary_pctile:Q", title="Salary percentile"),
        y=alt.Y(f"{score_col}:Q", title="Score (0-100)"),
        color=alt.Color("target:N", title="Became All-Star",
                        scale=alt.Scale(domain=[0, 1], range=["#9aa5b1", "#e4572e"]))
        if show_outcome else alt.value("#4c78a8"),
        tooltip=["player", "team", "age", alt.Tooltip(score_col, title="Score"),
                 alt.Tooltip("salary_pctile", format=".0f", title="Salary pct")],
    ).properties(height=380)
    st.altair_chart(chart, width="stretch")

# ---------------------------------------------------------------------------
# Page 2: Player Predictor
# ---------------------------------------------------------------------------
elif page == "Player Predictor":
    st.title("Player Predictor")
    c1, c2 = st.columns([1, 2])
    season = c1.selectbox("Season", seasons[::-1], format_func=season_label)
    d = pool(season)
    names = d["player"].tolist()
    player = c2.selectbox("Player (sorted by score)", names)
    r = d[d["player"] == player].iloc[0]

    pic, info = st.columns([1, 4])
    with pic:
        photo(r.get("nba_id"))
    with info:
        st.markdown(f"### {r['player']}  \n{r['team']} · {r['pos']} · age {int(r['age'])} · {season_label(season)}")
        m = st.columns(4)
    m[0].metric("Score (of 100)", f"{r[score_col]:.1f}")
    m[1].metric("Rank", f"{int(r['rank'])} of {len(d)}")
    m[2].metric("Salary", f"${r['salary'] / 1e6:.1f}M" if pd.notna(r["salary"]) else "n/a")
    m[3].metric("Salary percentile", f"{r['salary_pctile']:.0f}" if pd.notna(r["salary_pctile"]) else "n/a")
    st.markdown(f"**Tier:** {tier(r[score_col])}")
    if show_outcome:
        out = outcome_text(r)
        st.info(f"What happened: {'first All-Star selection in ' + out[2:] if out != '—' else 'no All-Star selection in the next two seasons'}"
                + (f" (first selected in {season_label(int(r['first_allstar_season']))})"
                   if out == "—" and pd.notna(r["first_allstar_season"]) else ""))

    st.subheader("Season stats")
    stats = {"Age": r["age"], "Years in league": r["experience"], "Games": r["g"],
             "Minutes": r["mp_per_game"], "Points": r["pts_per_game"], "Rebounds": r["trb_per_game"],
             "Assists": r["ast_per_game"], "TS%": r["ts_percent"], "PER": r["per"],
             "Usage %": r["usg_percent"], "BPM": r["bpm"], "VORP": r["vorp"],
             "Points vs last season": r["pts_per_game_delta"], "BPM vs last season": r["bpm_delta"]}
    st.dataframe(pd.DataFrame([stats]).round(2), hide_index=True, width="stretch")

    st.subheader("Why the model scored him this way (Gradient Boosting, SHAP)")
    s = shap_long[(shap_long["player_id"] == r["player_id"]) & (shap_long["season"] == season)].copy()
    if s.empty:
        st.write("No explanation available for this player-season.")
    else:
        s["Feature"] = s["feature"].map(label)
        s = s.reindex(s["shap"].abs().sort_values(ascending=False).index).head(12)
        s["Direction"] = s["shap"].map(lambda v: "Raises score" if v > 0 else "Lowers score")
        bars = alt.Chart(s).mark_bar().encode(
            x=alt.X("shap:Q", title="Contribution (log-odds)"),
            y=alt.Y("Feature:N", sort=alt.EncodingSortField("shap", op="sum", order="descending"),
                    title=None, axis=alt.Axis(labelLimit=320)),
            color=alt.Color("Direction:N", scale=alt.Scale(domain=["Raises score", "Lowers score"],
                                                           range=["#e4572e", "#4c78a8"])),
            tooltip=["Feature", alt.Tooltip("shap:Q", format=".3f")],
        ).properties(height=34 * len(s))
        st.altair_chart(bars, width="stretch")

    hist = scores[scores["player_id"] == r["player_id"]].sort_values("season")
    if len(hist) > 1:
        st.subheader("Score across seasons")
        st.altair_chart(alt.Chart(hist).mark_line(point=True).encode(
            x=alt.X("season:O", title="Season (end year)"),
            y=alt.Y(f"{score_col}:Q", title="Score", scale=alt.Scale(domain=[0, 100]))),
            width="stretch")

    st.subheader("What if? Change his numbers and re-score him")
    st.caption(f"Uses the saved {model_name} model from nba_model.joblib (trained on 2000-2019). "
               "The new score is his rank among this season's eligible players under that model. "
               "For seasons up to 2019 the model has already seen these players, so treat it as a "
               "what-if tool, not a backtest. All ten model inputs can be changed below. "
               "unrealistic combinations can move the tree models in odd directions; Logistic Regression "
               "responds most predictably.")
    model = models[model_name]
    season_pool = scores[scores["season"] == season]
    base_probs = model.predict_proba(season_pool[NUMERIC + CATEGORICAL])[:, 1]
    row = scores[(scores["player_id"] == r["player_id"]) & (scores["season"] == season)][NUMERIC + CATEGORICAL].copy()

    sliders = [("age", "Age", 18.0, 40.0, 1.0),
               ("mp_per_game", "Minutes per game", 0.0, 42.0, 0.5),
               ("pts_per_game", "Points per game", 0.0, 40.0, 0.1),
               ("fta_per_game", "Free throws attempted per game", 0.0, 15.0, 0.1),
               ("usg_percent", "Usage %", 5.0, 40.0, 0.5),
               ("ts_percent", "True shooting %", 0.30, 0.75, 0.005),
               ("ast_percent", "Assist %", 0.0, 50.0, 0.5),
               ("bpm", "Box plus/minus", -10.0, 12.0, 0.1),
               ("ws_48", "Win shares per 48", -0.10, 0.30, 0.005),
               ("bpm_delta", "Change in BPM vs last season", -8.0, 8.0, 0.1)]
    cols = st.columns(4)
    for i, (feat, lab, lo, hi, step) in enumerate(sliders):
        cur = row[feat].iloc[0]
        cur = float(np.clip(cur if pd.notna(cur) else (lo + hi) / 2, lo, hi))
        row[feat] = cols[i % 4].slider(lab, lo, hi, cur, step, key=f"{feat}_{r['player_id']}_{season}")

    new_prob = model.predict_proba(row)[:, 1][0]
    new_score = 100 * (base_probs <= new_prob).mean()
    old_score = 100 * (base_probs <= model.predict_proba(
        season_pool.loc[season_pool["player_id"] == r["player_id"], NUMERIC + CATEGORICAL])[:, 1][0]).mean()
    w1, w2 = st.columns(2)
    w1.metric("Score with his real numbers", f"{old_score:.1f}")
    w2.metric("Score with your numbers", f"{new_score:.1f}", delta=f"{new_score - old_score:+.1f}")

# ---------------------------------------------------------------------------
# Page 3: Model Explainability
# ---------------------------------------------------------------------------
elif page == "Model Explainability":
    st.title("Model Explainability")

    st.subheader("Hit rate of each season's top 10 (walk-forward backtest)")
    mid = scores[scores["eligible_midtier"] == 1] if midtier_only else scores
    rows = []
    for name, col in MODEL_COLS.items():
        for season, g in mid.groupby("season"):
            rows.append({"Model": name, "season": season, "Top-10 hit rate": g.nlargest(10, col)["target"].mean()})
    for season, g in mid.groupby("season"):
        rows.append({"Model": "Random pick (base rate)", "season": season, "Top-10 hit rate": g["target"].mean()})
    hr = pd.DataFrame(rows)
    st.altair_chart(alt.Chart(hr).mark_line(point=True).encode(
        x=alt.X("season:O", title="Season (end year)"),
        y=alt.Y("Top-10 hit rate:Q", axis=alt.Axis(format="%")),
        color=alt.Color("Model:N", scale=alt.Scale(
            domain=["Gradient Boosting", "Random Forest", "Logistic Regression", "Random pick (base rate)"],
            range=["#e4572e", "#2e86ab", "#76b041", "#9aa5b1"])),
        strokeDash=alt.condition(alt.datum.Model == "Random pick (base rate)", alt.value([4, 4]), alt.value([1, 0])),
        tooltip=["Model", "season", alt.Tooltip("Top-10 hit rate:Q", format=".0%")],
    ).properties(height=320), width="stretch")
    avg = hr.groupby("Model")["Top-10 hit rate"].mean().sort_values(ascending=False)
    st.write("Average across seasons: " + ", ".join(f"**{m}** {v:.0%}" for m, v in avg.items()))

    st.subheader("Held-out test results, 2021-2024")
    st.caption("Models tuned on 2017-2019, trained on 2000-2019, scored once on 2021-2024.")
    st.dataframe(test_results, hide_index=True, width="stretch")

    st.subheader("What drives the predictions")
    imp = (shap_long.assign(abs_shap=shap_long["shap"].abs())
           .groupby("feature")["abs_shap"].sum() / scores.shape[0])
    imp = imp.sort_values(ascending=False).head(15).rename("Mean |SHAP|").reset_index()
    imp["Feature"] = imp["feature"].map(label)
    st.altair_chart(alt.Chart(imp).mark_bar(color="#e4572e").encode(
        x="Mean |SHAP|:Q", y=alt.Y("Feature:N", sort="-x", title=None, axis=alt.Axis(labelLimit=320))
    ).properties(height=420),
        width="stretch")
    st.caption("Average size of each feature's contribution, Gradient Boosting, all scored seasons. "
               "Getting to the free throw line, big minutes, young age and high scoring push predictions up.")

# ---------------------------------------------------------------------------
# Page 4: Salary Analysis
# ---------------------------------------------------------------------------
else:
    st.title("Salary Analysis")
    st.caption("Do players the model flags go on to earn more? Mid-tier players only, "
               "seasons 2006-2021, so salary is known three seasons later (salary data ends in 2023-24). "
               "This is descriptive. It does not prove that a player was undervalued.")
    top_k = st.slider("Treat each season's top-K scored players as 'flagged'", 5, 30, 10)
    mid = scores[(scores["eligible_midtier"] == 1) & (scores["season"] <= 2021)].copy()
    mid["flagged"] = mid.groupby("season")[score_col].rank(ascending=False) <= top_k
    mid = mid.merge(salary_paths.drop(columns=["salary", "salary_pctile"]), on=["season", "player_id"])
    mid["in_league_t3"] = mid["salary_t3"].notna().astype(float)
    mid["group"] = mid["flagged"].map({True: f"Flagged (top {top_k})", False: "Other mid-tier"})

    long = []
    for k, col in [(0, "salary_pctile"), (1, "salary_pctile_t1"), (2, "salary_pctile_t2"), (3, "salary_pctile_t3")]:
        g = mid.groupby("group")[col].median().reset_index(name="Median salary percentile")
        g["Seasons later"] = k
        long.append(g)
    long = pd.concat(long)
    st.altair_chart(alt.Chart(long).mark_line(point=True).encode(
        x=alt.X("Seasons later:O", axis=alt.Axis(labelAngle=0)),
        y=alt.Y("Median salary percentile:Q", scale=alt.Scale(domain=[0, 100])),
        color=alt.Color("group:N", title=None, scale=alt.Scale(range=["#e4572e", "#9aa5b1"]))
    ).properties(height=320), width="stretch")

    summ = mid.groupby("group").agg(
        players=("player_id", "size"),
        became_allstar=("target", "mean"),
        median_salary_now=("salary", "median"),
        median_salary_3_later=("salary_t3", "median"),
        still_in_league_3_later=("in_league_t3", "mean"),
    )
    summ["median_salary_now"] = (summ["median_salary_now"] / 1e6).round(2)
    summ["median_salary_3_later"] = (summ["median_salary_3_later"] / 1e6).round(2)
    summ["became_allstar"] = (summ["became_allstar"] * 100).round(1)
    summ["still_in_league_3_later"] = (summ["still_in_league_3_later"] * 100).round(0)
    st.dataframe(summ.rename(columns={
        "players": "Player-seasons", "became_allstar": "Became All-Star (%)",
        "median_salary_now": "Median salary now ($M)", "median_salary_3_later": "Median salary 3 later ($M)",
        "still_in_league_3_later": "On a payroll 3 later (%)"}), width="stretch")

    st.subheader("Flagged players who did not become All-Stars")
    st.caption("False positives are often still valuable players. Their salary three seasons later, "
               "and their first All-Star season if it came after the two-season window:")
    fp = mid[mid["flagged"] & (mid["target"] == 0)].assign(
        salary_now=lambda x: (x["salary"] / 1e6).round(2),
        salary_3_later=lambda x: (x["salary_t3"] / 1e6).round(2),
        later_allstar=lambda x: x["first_allstar_season"].map(
            lambda v: season_label(int(v)) if pd.notna(v) else "—"))
    st.dataframe(fp.sort_values("salary_3_later", ascending=False)[
        ["season", "player", "team", "salary_now", "salary_3_later", "salary_pctile_t3",
         "later_allstar"]].rename(columns={"later_allstar": "First All-Star (later)",
            "season": "Season", "player": "Player", "team": "Team", "salary_now": "Salary then ($M)",
            "salary_3_later": "Salary 3 seasons later ($M)", "salary_pctile_t3": "Salary pct 3 later"}).round(1),
        hide_index=True, width="stretch", height=320)
