import math
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, brier_score_loss
import nfl_data_py as nfl

# logistic spread to win parameter
K_SPREAD = 0.13

def spread_to_prob_home(spread_points: float, k: float = K_SPREAD) -> float:
    """Convert point spread in points to home win probability.
    Positive spread means home favored by that many points."""
    return 1.0 / (1.0 + math.exp(-k * spread_points))

def moneyline_to_prob(ml: float) -> float:
    """American odds to implied probability without vig removal."""
    if ml is None or np.isnan(ml):
        return np.nan
    if ml > 0:
        return 100.0 / (ml + 100.0)
    else:
        return (-ml) / ((-ml) + 100.0)

def load_games_and_lines(years):
    """Load schedules and betting lines. Returns merged game level frame."""
    # schedules has scores and participants
    sched = nfl.import_schedules(years)
    # betting lines has open and close for spread and total and moneyline
    lines = nfl.import_betting_lines(years)

    # select closing numbers when available
    cols_keep = [
        "game_id","season","week","gameday","home_team","away_team","home_score","away_score",
        "result","result_margin","home_away","location","game_type"
    ]
    sched_small = sched[cols_keep].drop_duplicates("game_id")

    # closing line columns vary slightly by provider, use consensus if present
    line_cols = [
        "game_id","season","week","gameday",
        "spread_line_close","total_line_close",
        "home_moneyline_close","away_moneyline_close"
    ]
    # if consensus close not present, fallback to close from any book averaged
    for c in line_cols:
        if c not in lines.columns:
            # build a fallback from available columns
            pass

    # Try to prefer consensus close if present, else compute simple average of closes across books
    # nfl_data_py supplies one row per game with consensus columns in recent versions
    line_small = lines.copy()
    for col in ["spread_line_close","total_line_close","home_moneyline_close","away_moneyline_close"]:
        if col not in line_small.columns:
            line_small[col] = np.nan

    merged = sched_small.merge(
        line_small[["game_id","spread_line_close","total_line_close","home_moneyline_close","away_moneyline_close"]],
        on="game_id",
        how="left"
    )
    return merged

def build_training_frame(df):
    """Create modeling columns.
       Home favored by S points means spread_line_close is positive for home."""
    df = df.copy()

    # Some feeds store spread relative to home already. We assume positive favors home.
    # If your feed defines spread from away perspective, flip the sign here.
    df["spread_home"] = df["spread_line_close"]

    # outcome from home perspective
    df["home_win"] = (df["home_score"] > df["away_score"]).astype(int)
    df["played"] = df["home_score"].notna() & df["away_score"].notna()

    # market baseline probability from spread
    df["p_market"] = df["spread_home"].apply(lambda s: np.nan if pd.isna(s) else spread_to_prob_home(float(s)))

    # implied from moneylines as a cross check
    df["p_ml_home"] = moneyline_to_prob(df["home_moneyline_close"])
    df["p_ml_away"] = moneyline_to_prob(df["away_moneyline_close"])
    # normalize the moneyline pair if both present
    both = df["p_ml_home"].notna() & df["p_ml_away"].notna()
    df.loc[both, "p_market_ml"] = df.loc[both, "p_ml_home"] / (df.loc[both, "p_ml_home"] + df.loc[both, "p_ml_away"])

    # choose p_market_pref as spread based, fall back to moneyline based
    df["p_market_pref"] = df["p_market"]
    need_ml = df["p_market_pref"].isna() & df["p_market_ml"].notna()
    df.loc[need_ml, "p_market_pref"] = df.loc[need_ml, "p_market_ml"]

    return df

def score_baseline(df):
    """Evaluate baseline on games with results and a market probability."""
    played = df["played"] & df["p_market_pref"].notna()
    if played.sum() == 0:
        return None
    y_true = df.loc[played, "home_win"].values
    p_pred = df.loc[played, "p_market_pref"].clip(1e-6, 1 - 1e-6).values
    ll = log_loss(y_true, p_pred)
    br = brier_score_loss(y_true, p_pred)
    calib = pd.cut(p_pred, bins=[0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0], include_lowest=True)
    calib_df = df.loc[played].groupby(calib).apply(
        lambda g: pd.Series({
            "n": len(g),
            "avg_pred": g["p_market_pref"].mean(),
            "empirical": g["home_win"].mean()
        })
    ).reset_index(names="bucket")
    return {"log_loss": ll, "brier": br, "calibration": calib_df}

def upcoming_board(df):
    """Return a tidy board for unplayed games with market baseline probabilities."""
    up = df[~df["played"]].copy()
    cols = [
        "season","week","gameday","away_team","home_team",
        "spread_home","total_line_close","home_moneyline_close","away_moneyline_close","p_market_pref"
    ]
    up = up[cols].sort_values(["season","week","gameday","home_team"])
    # present both sides
    up["away_win_prob_market"] = 1.0 - up["p_market_pref"]
    up.rename(columns={"p_market_pref":"home_win_prob_market"}, inplace=True)
    return up

def main():
    # choose seasons to build history and the current season for the live board
    seasons_hist = list(range(2015, 2026))
    df = load_games_and_lines(seasons_hist)
    df = build_training_frame(df)

    scores = score_baseline(df)
    if scores is not None:
        print("Baseline evaluation on games with results")
        print(f"log loss {scores['log_loss']:.4f}")
        print(f"brier {scores['brier']:.4f}")
        print("calibration by predicted bucket")
        print(scores["calibration"].to_string(index=False))

    up = upcoming_board(df)
    print("\nUpcoming games market baseline")
    print(up.head(20).to_string(index=False))

    # optionally save boards
    up.to_csv("upcoming_market_baseline.csv", index=False)
    played = df[df["played"] & df["p_market_pref"].notna()].copy()
    played[["season","week","gameday","away_team","home_team","spread_home","home_win","p_market_pref"]].to_csv(
        "historical_market_baseline.csv", index=False
    )

if __name__ == "__main__":
    main()
