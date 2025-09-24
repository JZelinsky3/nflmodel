#!/usr/bin/env python3
"""
NFL win probability model with market baseline plus residual features.
Compatible with Python 3.12.

Features included
- Market baseline from closing spread and moneyline fallback
- Rest days for each team and rest differential
- Rolling form based on last 3 games net point differential and differential
- Home flag
Model
- GLM Binomial with logit link using statsmodels
- Uses logit(p_market) as an offset so features only tilt the market prior
Outputs
- Evaluation metrics on completed games
- Upcoming games with predicted win probabilities for both teams

Run
  python model_with_features.py
"""
import math
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Optional

# data sources
try:
    from nflreadpy import nflreadr as nfl
    HAS_NFLREADPY = True
except Exception:
    HAS_NFLREADPY = False

# modeling
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler

K_SPREAD = 0.13  # spread to win curve parameter


def spread_to_prob_home(spread_points: float, k: float = K_SPREAD) -> float:
    """Convert point spread in points to home win probability.
    Positive means home favored by that many points."""
    return 1.0 / (1.0 + math.exp(-k * spread_points))


def moneyline_to_prob(ml: float) -> float:
    """American odds to implied probability without vig removal."""
    if ml is None or pd.isna(ml):
        return np.nan
    ml = float(ml)
    if ml > 0:
        return 100.0 / (ml + 100.0)
    else:
        return (-ml) / ((-ml) + 100.0)


def load_games_and_lines(years: List[int]) -> pd.DataFrame:
    if not HAS_NFLREADPY:
        raise RuntimeError("nflreadpy is required. Please install per requirements.txt")
    sched = nfl.load_schedules(years=years)
    lines = nfl.load_betting_lines(years=years)

    # Basic schedule columns
    sched_small = sched[[
        "game_id","season","week","gameday","home_team","away_team",
        "home_score","away_score","game_type"
    ]].drop_duplicates("game_id")

    # Betting lines columns with safe defaults
    for col in ["spread_line_close","total_line_close","home_moneyline_close","away_moneyline_close"]:
        if col not in lines.columns:
            lines[col] = np.nan
    line_small = lines[[
        "game_id","spread_line_close","total_line_close",
        "home_moneyline_close","away_moneyline_close"
    ]].drop_duplicates("game_id")

    df = sched_small.merge(line_small, on="game_id", how="left")
    # Parse dates
    df["gameday"] = pd.to_datetime(df["gameday"])

    # Define outcome and market priors
    df["spread_home"] = df["spread_line_close"]
    df["home_win"] = (df["home_score"] > df["away_score"]).astype("Int64")
    df["played"] = df["home_score"].notna() & df["away_score"].notna()

    # Spread based probability
    df["p_spread"] = df["spread_home"].apply(lambda s: np.nan if pd.isna(s) else spread_to_prob_home(float(s)))
    # Moneyline based probability pair normalized when available
    p_home_ml = df["home_moneyline_close"].apply(moneyline_to_prob)
    p_away_ml = df["away_moneyline_close"].apply(moneyline_to_prob)
    both = p_home_ml.notna() & p_away_ml.notna()
    p_ml_norm = pd.Series(np.nan, index=df.index, dtype="float")
    p_ml_norm.loc[both] = p_home_ml.loc[both] / (p_home_ml.loc[both] + p_away_ml.loc[both])

    # Prefer spread, fallback to moneyline
    df["p_market"] = df["p_spread"].where(df["p_spread"].notna(), p_ml_norm)

    return df


def add_rest_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute rest days for each team before the game and differential."""
    df = df.sort_values("gameday").copy()

    def team_rest(team_col: str, opp_col: str, prefix: str) -> pd.DataFrame:
        tmp = df[[
            "game_id","gameday", team_col, opp_col, "home_score","away_score"
        ]].copy()
        tmp = tmp.rename(columns={team_col:"team", opp_col:"opp"})
        # from team perspective, signed margin
        tmp["team_pts"] = np.where(tmp["team"] == df["home_team"], df["home_score"], df["away_score"])
        tmp["opp_pts"] = np.where(tmp["team"] == df["home_team"], df["away_score"], df["home_score"])
        tmp["mov_signed"] = tmp["team_pts"] - tmp["opp_pts"]

        # Compute previous game date per team
        tmp = tmp.sort_values(["team","gameday"])
        tmp["prev_date"] = tmp.groupby("team")["gameday"].shift(1)
        tmp["rest_days"] = (tmp["gameday"] - tmp["prev_date"]).dt.days
        tmp["roll3_mov"] = tmp.groupby("team")["mov_signed"].shift(1).rolling(3, min_periods=1).mean().reset_index(level=0, drop=True)
        return tmp[["game_id","rest_days","roll3_mov"]].rename(columns={
            "rest_days": f"{prefix}_rest_days",
            "roll3_mov": f"{prefix}_form3_mov"
        })

    home_feats = team_rest("home_team","away_team","home")
    away_feats = team_rest("away_team","home_team","away")

    out = df.merge(home_feats, on="game_id", how="left").merge(away_feats, on="game_id", how="left")
    # Rest differential and form differential
    out["rest_diff"] = out["home_rest_days"].fillna(7) - out["away_rest_days"].fillna(7)
    out["form3_diff"] = out["home_form3_mov"].fillna(0.0) - out["away_form3_mov"].fillna(0.0)
    # Home flag
    out["home_flag"] = 1.0
    return out


@dataclass
class FitResult:
    model: object
    scaler: Optional[StandardScaler]
    features: List[str]


def fit_residual_glm(df: pd.DataFrame, feature_cols: List[str]) -> FitResult:
    """Fit GLM Binomial with logit link and offset logit(p_market)."""
    train = df[df["played"] & df["p_market"].notna()].copy()
    # Clip extreme probabilities
    train["p_market"] = train["p_market"].clip(1e-6, 1 - 1e-6)
    # Offset
    train["offset"] = np.log(train["p_market"] / (1 - train["p_market"]))
    # Features
    X = train[feature_cols].astype(float).values
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    Xs = sm.add_constant(Xs, has_constant="add")
    y = train["home_win"].astype(int).values
    # Fit
    model = sm.GLM(y, Xs, family=sm.families.Binomial(), offset=train["offset"].values).fit()
    return FitResult(model=model, scaler=scaler, features=["const"] + feature_cols)


def predict_with_model(fit: FitResult, df: pd.DataFrame) -> pd.Series:
    """Predict home win probability with offset market prior."""
    use = df.copy()
    use["p_market"] = use["p_market"].clip(lower=1e-6, upper=1 - 1e-6)
    use["offset"] = np.log(use["p_market"] / (1 - use["p_market"]))
    X = use[fit.features[1:]].astype(float).values
    Xs = fit.scaler.transform(X)
    Xs = sm.add_constant(Xs, has_constant="add")
    # Linear predictor is offset plus X beta
    lin = use["offset"].values + np.dot(Xs, fit.model.params)
    p = 1.0 / (1.0 + np.exp(-lin))
    return pd.Series(p, index=use.index, name="p_home_pred")


def evaluate(df: pd.DataFrame, p_col: str = "p_home_pred") -> dict:
    """Compute log loss, Brier, and simple calibration for completed games."""
    from sklearn.metrics import log_loss, brier_score_loss
    mask = df["played"] & df[p_col].notna()
    if mask.sum() == 0:
        return {}
    y = df.loc[mask, "home_win"].astype(int).values
    p = df.loc[mask, p_col].clip(1e-6, 1 - 1e-6).values
    ll = log_loss(y, p)
    br = brier_score_loss(y, p)
    # calibration by decile
    cuts = pd.cut(p, bins=np.linspace(0,1,11), include_lowest=True)
    calib = df.loc[mask].groupby(cuts).apply(
        lambda g: pd.Series({"n": len(g), "avg_pred": g[p_col].mean(), "empirical": g["home_win"].mean()})
    ).reset_index(names="bucket")
    return {"log_loss": ll, "brier": br, "calibration": calib}


def upcoming_board(df: pd.DataFrame, p_col: str = "p_home_pred") -> pd.DataFrame:
    up = df[~df["played"]].copy()
    cols = [
        "season","week","gameday","away_team","home_team",
        "spread_home","total_line_close","home_moneyline_close","away_moneyline_close",
        "p_market"
    ]
    for c in cols:
        if c not in up.columns:
            up[c] = np.nan
    if p_col in up.columns:
        up["home_win_prob_model"] = up[p_col]
        up["away_win_prob_model"] = 1.0 - up[p_col]
    up["home_win_prob_market"] = up["p_market"]
    up["away_win_prob_market"] = 1.0 - up["p_market"]
    up = up[cols + ["home_win_prob_market","away_win_prob_market"] + (
        ["home_win_prob_model","away_win_prob_model"] if p_col in up.columns else []
    )]
    return up.sort_values(["season","week","gameday","home_team"])


def main():
    years = list(range(2015, 2026))
    df = load_games_and_lines(years)
    df = add_rest_features(df)

    # Choose features for residual tilt
    feature_cols = ["rest_diff","form3_diff","home_flag"]

    # Fit residual model
    fit = fit_residual_glm(df, feature_cols=feature_cols)

    # Predict on all rows
    df["p_home_pred"] = predict_with_model(fit, df)

    # Evaluate
    scores = evaluate(df, p_col="p_home_pred")
    if scores:
        print("Residual model evaluation on completed games")
        print(f"log loss {scores['log_loss']:.4f}")
        print(f"brier {scores['brier']:.4f}")
        print("calibration by predicted bucket")
        print(scores["calibration"].to_string(index=False))

    # Export upcoming board
    up = upcoming_board(df, p_col="p_home_pred")
    up.to_csv("upcoming_with_features.csv", index=False)
    print("\nSaved upcoming_with_features.csv")
    played = df[df["played"]].copy()
    played[["season","week","gameday","away_team","home_team","home_win","p_market","p_home_pred"]].to_csv(
        "historical_with_features.csv", index=False
    )
    print("Saved historical_with_features.csv")

if __name__ == "__main__":
    main()