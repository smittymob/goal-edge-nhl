"""
GoalEdge NHL — player ratings and goal projection
=================================================
pip install streamlit pandas numpy requests
streamlit run goal_edge_nhl.py

Data: MoneyPuck season summaries (credited in the app) + NHL schedule API.
Early-season rates are blended with the prior season so small samples do not dominate.
"""

from __future__ import annotations

import io
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests
import streamlit as st

MP_SKATERS = "https://moneypuck.com/moneypuck/playerData/seasonSummary/{year}/regular/skaters.csv"
MP_TEAMS = "https://moneypuck.com/moneypuck/playerData/seasonSummary/{year}/regular/teams.csv"
NHL_SCHEDULE = "https://api-web.nhle.com/v1/schedule/{date}"

CURRENT_SEASON = 2026  # 2026-27; MoneyPuck year = season start
PRIOR_SEASON = 2025
MIN_ICE_SECONDS = 120
HEADERS = {"User-Agent": "GoalEdgeNHL/1.0 (personal research app)"}

st.set_page_config(
    page_title="GoalEdge NHL",
    page_icon="🏒",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
      /* Phone-friendly defaults */
      .block-container { padding-top: 1rem; padding-bottom: 2rem; }
      [data-testid="stMetricValue"] { font-size: 1.35rem; }
      div[data-testid="stDataFrame"] { font-size: 0.85rem; }
      @media (max-width: 640px) {
        h1 { font-size: 1.45rem !important; }
        h2, h3 { font-size: 1.15rem !important; }
      }
    </style>
    """,
    unsafe_allow_html=True,
)


def _get_csv(url: str) -> pd.DataFrame:
    r = requests.get(url, headers=HEADERS, timeout=60)
    r.raise_for_status()
    return pd.read_csv(io.StringIO(r.text))


def per60(count: pd.Series, ice_seconds: pd.Series) -> pd.Series:
    return count * 3600.0 / ice_seconds.replace(0, np.nan)


def _situation(df: pd.DataFrame, situation: str) -> pd.DataFrame:
    out = df[df["situation"] == situation].copy()
    out["playerId"] = out["playerId"].astype(str)
    return out


def build_skater_table(current: pd.DataFrame, prior: pd.DataFrame) -> pd.DataFrame:
    cur_all = _situation(current, "all")
    cur_pp = _situation(current, "5on4")[["playerId", "icetime"]].rename(columns={"icetime": "pp_ice"})
    pri_all = _situation(prior, "all")

    cur = cur_all.merge(cur_pp, on="playerId", how="left")
    cur["pp_ice"] = cur["pp_ice"].fillna(0.0)
    cur = cur[cur["icetime"] >= MIN_ICE_SECONDS].copy()

    zone = (
        cur["I_F_oZoneShiftStarts"]
        + cur["I_F_dZoneShiftStarts"]
        + cur["I_F_neutralZoneShiftStarts"]
    ).replace(0, np.nan)

    cur["ixG_60"] = per60(cur["I_F_xGoals"], cur["icetime"])
    cur["SOG_60"] = per60(cur["I_F_shotsOnGoal"], cur["icetime"])
    cur["HD_60"] = per60(cur["I_F_highDangerShots"], cur["icetime"])
    cur["G_60"] = per60(cur["I_F_goals"], cur["icetime"])
    cur["P_60"] = per60(cur["I_F_points"], cur["icetime"])
    cur["hits_60"] = per60(cur["I_F_hits"], cur["icetime"])
    cur["blocks_60"] = per60(cur["shotsBlockedByPlayer"], cur["icetime"])
    cur["takeaways_60"] = per60(cur["I_F_takeaways"], cur["icetime"])
    cur["giveaways_60"] = per60(cur["I_F_giveaways"], cur["icetime"])
    cur["rel_xGF"] = (cur["onIce_xGoalsPercentage"] - cur["offIce_xGoalsPercentage"]) * 100
    cur["OZS_pct"] = 100 * cur["I_F_oZoneShiftStarts"] / zone
    cur["PP_share"] = 100 * cur["pp_ice"] / cur["icetime"]
    cur["G_minus_xG"] = cur["I_F_goals"] - cur["I_F_xGoals"]
    cur["toi_pg"] = cur["icetime"] / 60.0 / cur["games_played"].clip(lower=1)

    pri = pri_all[pri_all["icetime"] >= 200 * 60].copy()
    pri["ixG_60_prior"] = per60(pri["I_F_xGoals"], pri["icetime"])
    pri["SOG_60_prior"] = per60(pri["I_F_shotsOnGoal"], pri["icetime"])
    pri["HD_60_prior"] = per60(pri["I_F_highDangerShots"], pri["icetime"])
    pri["G_60_prior"] = per60(pri["I_F_goals"], pri["icetime"])
    pri["P_60_prior"] = per60(pri["I_F_points"], pri["icetime"])
    pri["rel_xGF_prior"] = (pri["onIce_xGoalsPercentage"] - pri["offIce_xGoalsPercentage"]) * 100
    pri = pri[
        [
            "playerId",
            "ixG_60_prior",
            "SOG_60_prior",
            "HD_60_prior",
            "G_60_prior",
            "P_60_prior",
            "rel_xGF_prior",
            "games_played",
        ]
    ].rename(columns={"games_played": "gp_prior"})

    out = cur.merge(pri, on="playerId", how="left")
    # Trust current season more as GP grows. At 4 GP, prior still carries ~70%.
    w = (out["games_played"] / (out["games_played"] + 10)).clip(0.15, 0.85)
    w = w.where(out["ixG_60_prior"].notna(), 1.0)

    def blend(now, old):
        return w * now + (1 - w) * old.fillna(now)

    out["ixG_60_L10"] = blend(out["ixG_60"], out["ixG_60_prior"])
    out["SOG_60"] = blend(out["SOG_60"], out["SOG_60_prior"])
    out["HD_60"] = blend(out["HD_60"], out["HD_60_prior"])
    out["G_60_L5"] = blend(out["G_60"], out["G_60_prior"])
    out["P_60_L10"] = blend(out["P_60"], out["P_60_prior"])
    out["rel_xGF"] = blend(out["rel_xGF"], out["rel_xGF_prior"])
    out["form_weight"] = w.round(2)
    out["player"] = out["name"]
    out["pos"] = out["position"]
    out["gp"] = out["games_played"]
    return out


def calculate_player_rating(df: pd.DataFrame) -> pd.DataFrame:
    """Composite 0–100. Weights match the original GoalEdge design."""
    work = df.copy()
    cols = [
        "ixG_60_L10",
        "SOG_60",
        "HD_60",
        "G_60_L5",
        "P_60_L10",
        "rel_xGF",
        "hits_60",
        "blocks_60",
        "takeaways_60",
        "giveaways_60",
        "OZS_pct",
        "PP_share",
        "G_minus_xG",
    ]
    for col in cols:
        if col not in work.columns:
            work[col] = 0.0
        work[col] = work[col].fillna(work[col].median() if work[col].notna().any() else 0.0)

    def normalize(s: pd.Series) -> pd.Series:
        if s.std(skipna=True) == 0 or s.isna().all():
            return pd.Series(0.5, index=s.index)
        return (s - s.min()) / (s.max() - s.min() + 1e-9)

    # Rank within position so defenders are not punished for lower shot volume.
    parts = []
    for _, grp in work.groupby(work["pos"].where(work["pos"].isin(["D", "C", "L", "R", "LW", "RW"]), "F")):
        idx = grp.index
        shot_q = normalize(0.6 * grp["ixG_60_L10"] + 0.3 * grp["SOG_60"] + 0.1 * grp["HD_60"])
        recent = normalize(0.5 * grp["G_60_L5"] + 0.3 * grp["P_60_L10"] + 0.2 * grp["G_minus_xG"])
        on_ice = normalize(grp["rel_xGF"])
        effort = normalize(grp["hits_60"] + grp["blocks_60"] + grp["takeaways_60"] - grp["giveaways_60"])
        deploy = normalize(0.5 * grp["OZS_pct"] + 0.5 * grp["PP_share"])
        finish = normalize(grp["G_minus_xG"].clip(-1.5, 1.5))
        score = (
            0.35 * shot_q
            + 0.20 * recent
            + 0.15 * on_ice
            + 0.15 * effort
            + 0.10 * deploy
            + 0.05 * finish
        ) * 100
        parts.append(score.rename("rating"))
    work["rating"] = pd.concat(parts).reindex(work.index).round(1)
    return work


def build_teams(current: pd.DataFrame, prior: pd.DataFrame) -> pd.DataFrame:
    def rates(df: pd.DataFrame, suffix: str) -> pd.DataFrame:
        all_ = df[df["situation"] == "all"].copy()
        five = df[df["situation"] == "5on5"][["team", "xGoalsFor", "xGoalsAgainst", "iceTime"]].rename(
            columns={
                "xGoalsFor": "xgf5",
                "xGoalsAgainst": "xga5",
                "iceTime": "ice5",
            }
        )
        m = all_.merge(five, on="team", how="left")
        m[f"xgf60{suffix}"] = m["xGoalsFor"] * 3600 / m["iceTime"]
        m[f"xga60{suffix}"] = m["xGoalsAgainst"] * 3600 / m["iceTime"]
        m[f"gf{suffix}"] = m["goalsFor"]
        m[f"ga{suffix}"] = m["goalsAgainst"]
        m[f"gp{suffix}"] = m["games_played"]
        return m[["team", "name", f"xgf60{suffix}", f"xga60{suffix}", f"gf{suffix}", f"ga{suffix}", f"gp{suffix}"]]

    cur = rates(current, "")
    pri = rates(prior, "_prior").drop(columns=["name"])
    out = cur.merge(pri, on="team", how="left")
    w = (out["gp"] / (out["gp"] + 15)).clip(0.2, 0.8)
    out["xgf60"] = w * out["xgf60"] + (1 - w) * out["xgf60_prior"].fillna(out["xgf60"])
    out["xga60"] = w * out["xga60"] + (1 - w) * out["xga60_prior"].fillna(out["xga60"])
    return out


@st.cache_data(ttl=3600)
def load_data():
    sk_c = _get_csv(MP_SKATERS.format(year=CURRENT_SEASON))
    sk_p = _get_csv(MP_SKATERS.format(year=PRIOR_SEASON))
    tm_c = _get_csv(MP_TEAMS.format(year=CURRENT_SEASON))
    tm_p = _get_csv(MP_TEAMS.format(year=PRIOR_SEASON))
    players = calculate_player_rating(build_skater_table(sk_c, sk_p))
    teams = build_teams(tm_c, tm_p)
    fetched = datetime.now().strftime("%Y-%m-%d %H:%M")
    return players, teams, fetched


def fetch_slate(date_str: str) -> pd.DataFrame:
    try:
        r = requests.get(NHL_SCHEDULE.format(date=date_str), headers=HEADERS, timeout=30)
        r.raise_for_status()
        payload = r.json()
    except Exception:
        return pd.DataFrame(columns=["date", "away", "home", "state"])
    rows = []
    for day in payload.get("gameWeek", []):
        if day.get("date") != date_str:
            continue
        for g in day.get("games", []):
            rows.append(
                {
                    "date": date_str,
                    "away": g.get("awayTeam", {}).get("abbrev"),
                    "home": g.get("homeTeam", {}).get("abbrev"),
                    "state": g.get("gameState"),
                }
            )
    return pd.DataFrame(rows)


def project_game(teams: pd.DataFrame, home: str, away: str) -> tuple[float, float]:
    lookup = teams.set_index("team")
    if home not in lookup.index or away not in lookup.index:
        return float("nan"), float("nan")
    h, a = lookup.loc[home], lookup.loc[away]
    # Average of own attack and opponent defense, plus a small home bump.
    proj_home = (float(h["xgf60"]) + float(a["xga60"])) / 2 * 1.03
    proj_away = (float(a["xgf60"]) + float(h["xga60"])) / 2 * 0.97
    return proj_home, proj_away


def poisson_matrix(lh: float, la: float, n: int = 8) -> pd.DataFrame:
    from math import factorial

    ks = np.arange(0, n + 1)
    ph = np.array([np.exp(-lh) * lh**k / factorial(int(k)) for k in ks])
    pa = np.array([np.exp(-la) * la**k / factorial(int(k)) for k in ks])
    return pd.DataFrame(np.outer(ph, pa), index=[f"H{k}" for k in ks], columns=[f"A{k}" for k in ks])


players, teams, fetched = load_data()

st.title("🏒 GoalEdge NHL — Player Ratings & Goal Projection")
st.caption(
    f"Loaded {fetched}  ·  MoneyPuck {CURRENT_SEASON}-{str(CURRENT_SEASON+1)[-2:]} blended with {PRIOR_SEASON}-{str(PRIOR_SEASON+1)[-2:]}  ·  Credit: MoneyPuck.com"
)

st.sidebar.header("Filters")
pos_sel = st.sidebar.multiselect("Position", sorted(players["pos"].dropna().unique()), default=sorted(players["pos"].dropna().unique()))
team_sel = st.sidebar.multiselect("Team", sorted(players["team"].dropna().unique()))
min_gp = st.sidebar.slider("Minimum games this season", 1, int(players["gp"].max()), 1)
min_rating = st.sidebar.slider("Minimum rating", 0.0, 100.0, 40.0)
min_toi = st.sidebar.slider("Minimum TOI/game (min)", 0.0, 25.0, 8.0)

filtered = players[
    players["pos"].isin(pos_sel)
    & (players["rating"] >= min_rating)
    & (players["gp"] >= min_gp)
    & (players["toi_pg"] >= min_toi)
].copy()
if team_sel:
    filtered = filtered[filtered["team"].isin(team_sel)]

tab1, tab2, tab3 = st.tabs(["Player ratings", "Matchup predictor", "Compare"])

with tab1:
    st.subheader("Ratings — shot quality, form, on-ice impact, effort, deployment")
    st.caption("Rating is normalized inside position group. Early-season rates are shrunk toward last season.")
    show = filtered.sort_values("rating", ascending=False)[
        ["player", "team", "pos", "gp", "rating", "ixG_60_L10", "SOG_60", "rel_xGF", "PP_share", "form_weight", "toi_pg"]
    ].rename(
        columns={
            "ixG_60_L10": "ixG/60",
            "SOG_60": "SOG/60",
            "rel_xGF": "Rel xGF",
            "PP_share": "PP share %",
            "form_weight": "Current weight",
            "toi_pg": "TOI/G",
        }
    )
    st.dataframe(show, use_container_width=True, height=520, hide_index=True)
    if not filtered.empty:
        selected = st.selectbox("Player detail", filtered.sort_values("rating", ascending=False)["player"].tolist())
        p = filtered[filtered["player"] == selected].iloc[0]
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Rating", f"{p['rating']:.1f}")
        c2.metric("ixG/60", f"{p['ixG_60_L10']:.2f}")
        c3.metric("SOG/60", f"{p['SOG_60']:.1f}")
        c4.metric("Rel xGF", f"{p['rel_xGF']:+.1f}")
        c5.metric("Goals − xG", f"{p['G_minus_xG']:+.2f}")

with tab2:
    st.subheader("Team goal projection")
    st.caption("Projected goals = average of team xGF/60 and opponent xGA/60, with a 3% home bump. Rates are shrunk toward last season.")
    slate_date = st.date_input("Slate date", value=datetime.now().date())
    slate = fetch_slate(slate_date.strftime("%Y-%m-%d"))
    team_names = sorted(teams["team"].unique())
    if not slate.empty:
        st.write(f"{len(slate)} games on the NHL slate")
        pick = st.selectbox(
            "Load from slate",
            ["Manual"] + [f"{r.away} @ {r.home}" for r in slate.itertuples()],
        )
    else:
        pick = "Manual"
        st.info("No NHL slate returned for that date. Pick teams manually.")

    default_away, default_home = team_names[0], team_names[min(1, len(team_names) - 1)]
    if pick != "Manual":
        default_away, default_home = pick.split(" @ ")
    c1, c2 = st.columns(2)
    away = c1.selectbox("Away", team_names, index=team_names.index(default_away) if default_away in team_names else 0)
    home = c2.selectbox("Home", team_names, index=team_names.index(default_home) if default_home in team_names else 0)
    proj_home, proj_away = project_game(teams, home, away)
    st.markdown(f"### Projected score: **{home} {proj_home:.2f} – {proj_away:.2f} {away}**")
    if np.isfinite(proj_home):
        mat = poisson_matrix(proj_home, proj_away)
        home_win = float(np.tril(mat.values, -1).sum())
        away_win = float(np.triu(mat.values, 1).sum())
        tie = float(np.trace(mat.values))
        m1, m2, m3 = st.columns(3)
        m1.metric("Home win (reg)", f"{100 * home_win:.1f}%")
        m2.metric("Away win (reg)", f"{100 * away_win:.1f}%")
        m3.metric("Tie after 60", f"{100 * tie:.1f}%")
        st.caption("Independent Poisson on projected goals. Not a betting model. Special teams and goalies are only inside the MoneyPuck team rates.")

with tab3:
    st.subheader("Head-to-head")
    names = players.sort_values("rating", ascending=False)["player"].tolist()
    a = st.selectbox("Player A", names, index=0, key="p1")
    b = st.selectbox("Player B", names, index=min(1, len(names) - 1), key="p2")
    cols = ["player", "team", "pos", "gp", "rating", "ixG_60_L10", "SOG_60", "P_60_L10", "rel_xGF", "PP_share"]
    st.dataframe(players[players["player"].isin([a, b])][cols], use_container_width=True, hide_index=True)

st.markdown("---")
st.markdown(
    "Stats from [MoneyPuck](https://moneypuck.com/data.htm) season summaries. "
    "Schedule from the public NHL web API. Non-commercial use; credit MoneyPuck if you republish figures."
)
