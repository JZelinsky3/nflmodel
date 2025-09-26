# pick_board.py
import pandas as pd
import numpy as np

EDGE_THRESHOLD = 0.02  # 2 percent edge cutoff
TOP_N = None

def prob_to_american(p: float) -> int:
    """Convert probability to fair American odds without vig."""
    p = max(1e-6, min(1 - 1e-6, float(p)))
    dec = 1.0 / p
    if dec >= 2.0:
        return int(round((dec - 1.0) * 100))          # underdog
    else:
        return int(round(-100.0 / (dec - 1.0)))       # favorite

def to_01(series: pd.Series) -> pd.Series:
    """
    Normalize a probability series to 0..1.
    Accepts numbers like 58 or "58%" and converts to 0.58.
    Clips to [0,1]. Leaves NaN if input is missing.
    """
    s = series.astype(str).str.replace("%", "", regex=False)
    s = pd.to_numeric(s, errors="coerce")
    if pd.api.types.is_numeric_dtype(s) and np.nanmax(s.values) > 1.0 + 1e-9:
        s = s / 100.0
    return s.clip(lower=0.0, upper=1.0)

def main():
    df = pd.read_csv("upcoming_with_features.csv")

    # parse dates
    if "gameday" in df.columns:
        df["gameday"] = pd.to_datetime(df["gameday"], errors="coerce")
    else:
        df["gameday"] = pd.NaT

    # choose target week
    now = pd.Timestamp.now(tz=None)
    future = df[df["gameday"].notna() & (df["gameday"] >= now)]
    if future.empty:
        target_week = int(df["week"].dropna().max())
    else:
        target_week = int(future["week"].dropna().min())

    wk = df[df["week"] == target_week].copy()

    # choose model column
    if "home_win_prob_model_cal" in wk.columns:
        model_col_name = "home_win_prob_model_cal"
    elif "home_win_prob_model" in wk.columns:
        model_col_name = "home_win_prob_model"
    else:
        model_col_name = "home_win_prob_market"

    # normalize to 0..1
    wk[model_col_name] = to_01(wk[model_col_name])

    # market probs home and away normalized and filled
    if "home_win_prob_market" in wk.columns:
        wk["home_win_prob_market"] = to_01(wk["home_win_prob_market"])
    else:
        wk["home_win_prob_market"] = np.nan
    if "away_win_prob_market" in wk.columns:
        wk["away_win_prob_market"] = to_01(wk["away_win_prob_market"])
    else:
        wk["away_win_prob_market"] = np.nan

    # if either market prob missing, fall back to 0.5 each
    both_missing = wk["home_win_prob_market"].isna() | wk["away_win_prob_market"].isna()
    wk.loc[both_missing, "home_win_prob_market"] = 0.5
    wk.loc[both_missing, "away_win_prob_market"] = 0.5

    p_model_home = wk[model_col_name].fillna(0.5)
    p_market_home = wk["home_win_prob_market"].fillna(0.5)

    # edges
    wk["edge_home"] = p_model_home - p_market_home
    wk["edge_away"] = -wk["edge_home"]

    # pick side
    wk["pick_side"] = np.where(wk["edge_home"] >= 0.0, "HOME", "AWAY")

    # chosen probabilities for picked side
    wk["p_model_pick"] = np.where(wk["pick_side"] == "HOME", p_model_home, 1.0 - p_model_home)
    wk["p_market_pick"] = np.where(wk["pick_side"] == "HOME", p_market_home, 1.0 - p_market_home)

    # fair price
    wk["fair_price_am"] = wk["p_model_pick"].apply(prob_to_american)

    # labels
    wk["matchup"] = wk["away_team"].astype(str) + " @ " + wk["home_team"].astype(str)
    wk["kickoff"] = wk["gameday"].dt.strftime("%Y-%m-%d %H:%M")
    wk["model_col_used"] = model_col_name

    # board
    board = wk.sort_values("edge_home", ascending=False).copy()
    board["edge"] = np.where(board["pick_side"] == "HOME", board["edge_home"], board["edge_away"])

    # filter by edge threshold
    board = board[board["edge"].abs() >= EDGE_THRESHOLD]
    if TOP_N:
        board = board.head(TOP_N)

    # debug checks before percent scaling
    print("model min max", float(board[model_col_name].min()), float(board[model_col_name].max()))
    print("market min max", float(board["home_win_prob_market"].min()), float(board["home_win_prob_market"].max()))

    # percent display rounding
    pct_cols = [
        "home_win_prob_market",
        "away_win_prob_market",
        model_col_name,
        "p_model_pick",
        "p_market_pick",
        "edge",
    ]
    for c in pct_cols:
        if c in board.columns:
            board[c] = (board[c] * 100.0).round(1)

    out_cols = [
        "season","week","kickoff","matchup","spread_home",
        "home_moneyline_close","away_moneyline_close",
        "home_win_prob_market","away_win_prob_market",
        "pick_side","edge","p_model_pick","p_market_pick","fair_price_am","model_col_used"
    ]
    board[out_cols].to_csv(f"pick_board_week_{target_week}.csv", index=False)

    # pretty print
    print(f"\n=== Week {target_week} pick board edge >= {EDGE_THRESHOLD*100:.0f} percent ===")
    show = board[out_cols].rename(columns={
        "home_win_prob_market":"Mkt% H",
        "away_win_prob_market":"Mkt% A",
        "p_model_pick":"Model% Pick",
        "p_market_pick":"Mkt% Pick",
        "spread_home":"Home Spread",
        "fair_price_am":"Fair ML",
        "edge":"Edge %"
    })
    print(show.to_string(index=False))

if __name__ == "__main__":
    main()
