# model_with_features.py
#!/usr/bin/env python3
"""
NFL win probability model with market baseline + residual features (Python 3.12)

- Data: nflreadpy only (schedules, play-by-play)
- Final scores from PBP (robust to schema)
- Features: rest diff, rolling MOV (3g), rolling EPA off/def (3g) + differentials
- Market: prefers spread; falls back to ML; if neither → CSV fallback; else neutral 0.5
- Model: Binomial GLM (logit), offset = logit(p_market); optional isotonic calibration
"""

import os
import math
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Optional

# nflreadpy
try:
    import nflreadpy as nfl
    HAS_NFLREADPY = True
except Exception:
    nfl = None
    HAS_NFLREADPY = False

# modeling
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from sklearn.isotonic import IsotonicRegression

K_SPREAD = 0.13
ROLL_N = 3

# -------------------- utils --------------------
def ensure_pandas(df):
    """Return a pandas.DataFrame; handle Polars without needing pyarrow."""
    if isinstance(df, pd.DataFrame):
        return df

    # Polars fast path
    try:
        import polars as pl  # type: ignore
        if isinstance(df, (pl.DataFrame, pl.LazyFrame)):
            # Try native conversion first
            try:
                return df.to_pandas()
            except Exception:
                # Pyarrow not installed? Fallback: pure-Python conversion.
                # This avoids heavy deps at the cost of a bit of speed.
                if isinstance(df, pl.LazyFrame):
                    df = df.collect()
                return pd.DataFrame(df.to_dicts())
    except Exception:
        pass

    # Generic objects with .to_pandas()
    if hasattr(df, "to_pandas"):
        try:
            return df.to_pandas()
        except Exception:
            pass

    # Common coercions
    if isinstance(df, list) and (len(df) == 0 or isinstance(df[0], dict)):
        return pd.DataFrame(df)
    if isinstance(df, dict):
        return pd.DataFrame(df)

    return df

def spread_to_prob_home(spread_points: float, k: float = K_SPREAD) -> float:
    return 1.0 / (1.0 + math.exp(-k * spread_points))

def moneyline_to_prob(ml) -> float:
    if ml is None or (isinstance(ml, float) and np.isnan(ml)): return np.nan
    try:
        s = str(ml).strip()
        if s.startswith("+"): s = s[1:]
        m = float(s)
    except Exception:
        return np.nan
    return 100.0 / (m + 100.0) if m > 0 else (-m) / ((-m) + 100.0)

def try_callables(obj, names: List[str], *args, **kwargs):
    if obj is None: return None
    for n in names:
        func = getattr(obj, n, None)
        if callable(func):
            try: return func(*args, **kwargs)
            except Exception: continue
    return None

def _maybe_flip_spread_sign(df):
    """
    Many sources use negative numbers for home favorites.
    Our logistic expects 'home favorite = positive'.
    Heuristic: if most non-null spreads are negative, flip sign.
    """
    s = df["spread_line_close"]
    sn = s.dropna().astype(float)
    if len(sn) >= 50 and (sn < 0).mean() > 0.6:
        df["spread_line_close"] = -s
        print("normalized spreads: detected 'negative = home favorite'; flipped sign so home favorite is positive.")
    return df

# -------------------- loaders --------------------
def load_schedules(years: List[int]) -> pd.DataFrame:
    if not HAS_NFLREADPY: raise RuntimeError("nflreadpy not available")
    got = []
    for y in years:
        try:
            df_y = nfl.load_schedules([y])
            print(f"schedules {y} type {type(df_y)}")
            got.append(df_y)
        except Exception as e:
            print(f"schedules {y} failed {e!r}")
    if not got: raise RuntimeError(f"no schedules for {years}")

    got_pd = [ensure_pandas(g) if not isinstance(g, pd.DataFrame) else g for g in got]
    got_pd = [ensure_pandas(g) for g in got]

    # If any item is still not a pandas.DataFrame (e.g., Polars slipped through),
    # coerce it via .to_dicts() fallback.
    
    try:
        import polars as pl  # type: ignore
        fixed = []
        for g in got_pd:
            if isinstance(g, pd.DataFrame):
                fixed.append(g)
            elif isinstance(g, (pl.DataFrame, pl.LazyFrame)):
                if isinstance(g, pl.LazyFrame):
                    g = g.collect()
                fixed.append(pd.DataFrame(g.to_dicts()))
            else:
                fixed.append(ensure_pandas(g))
        got_pd = fixed
    except Exception:
        pass

    out = pd.concat(got_pd, ignore_index=True)

    for col in ["game_id","season","week","gameday","home_team","away_team"]:
        if col not in out.columns: out[col] = pd.NA
    out["gameday"] = pd.to_datetime(out["gameday"], errors="coerce")
    out["season"] = pd.to_numeric(out["season"], errors="coerce").astype("Int64")
    out["week"]   = pd.to_numeric(out["week"],  errors="coerce").astype("Int64")

    if "game_id" not in out.columns or out["game_id"].isna().all():
        out["game_id"] = (
            out["season"].astype(str) + "_" +
            out["week"].astype(str) + "_" +
            out["away_team"].astype(str) + "_at_" + out["home_team"].astype(str)
        )
    return out

def load_scores_from_pbp(years: List[int]) -> pd.DataFrame:
    pbp = try_callables(nfl, ["load_pbp"], years)
    if pbp is None: raise RuntimeError("nflreadpy.load_pbp not available")
    pbp = ensure_pandas(pbp)

    for need in ["game_id","home_score","away_score"]:
        if need not in pbp.columns: pbp[need] = pd.NA
    if pbp["home_score"].isna().all() or pbp["away_score"].isna().all():
        raise RuntimeError("PBP missing home_score/away_score — cannot aggregate final scores")

    agg = (
        pbp.groupby("game_id", as_index=False)
        .agg(home_score=("home_score","max"), away_score=("away_score","max"))
        .astype({"home_score":"Int64","away_score":"Int64"})
    )
    return agg

def load_epa_team_game(years: List[int]) -> pd.DataFrame:
    pbp = try_callables(nfl, ["load_pbp"], years)
    if pbp is None:
        return pd.DataFrame(columns=["game_id","team","off_epa_game","def_epa_allowed_game"])
    pbp = ensure_pandas(pbp)

    for c in ["game_id","posteam","defteam","epa"]:
        if c not in pbp.columns: pbp[c] = pd.NA
    pbp = pbp[pbp["epa"].notna()].copy()

    off = pbp.groupby(["game_id","posteam"], as_index=False)["epa"].mean().rename(
        columns={"posteam":"team","epa":"off_epa_game"}
    )
    deff = pbp.groupby(["game_id","defteam"], as_index=False)["epa"].mean().rename(
        columns={"defteam":"team","epa":"def_epa_allowed_game"}
    )
    deff["def_epa_allowed_game"] = -deff["def_epa_allowed_game"]

    return pd.merge(off, deff, on=["game_id","team"], how="outer")

def _normalize_line_columns(lines: pd.DataFrame) -> pd.DataFrame:
    lines = lines.copy()
    remap = {
        "spread_line": "spread_line_close",
        "spread_close": "spread_line_close",
        "total_line": "total_line_close",
        "total_close": "total_line_close",
        "home_moneyline": "home_moneyline_close",
        "away_moneyline": "away_moneyline_close",
    }
    for src, dst in remap.items():
        if src in lines.columns and dst not in lines.columns:
            lines[dst] = lines[src]
    return lines

def load_lines_any(years: List[int]) -> pd.DataFrame:
    # 1) nflreadpy variants (rarely present)
    if HAS_NFLREADPY:
        for cand in ["load_lines", "load_vegas", "load_weekly_lines", "import_betting_lines"]:
            got = []
            for y in years:
                try:
                    df = try_callables(nfl, [cand], [y])
                    if df is not None: got.append(df)
                except Exception:
                    continue
            if got:
                got_pd = [ensure_pandas(g) if not isinstance(g, pd.DataFrame) else g for g in got]
                out = pd.concat(got_pd, ignore_index=True)
                out = _normalize_line_columns(out)
                if "game_id" in out.columns:
                    print(f"loaded lines via nflreadpy {cand} rows {len(out)}")
                    return out

    # 2) Local CSV fallback(s)
    for path in ["betting_lines_2018_2025.csv","betting_lines.csv", os.path.join("data","betting_lines.csv")]:
        if os.path.exists(path):
            df = pd.read_csv(path)
            df = _normalize_line_columns(df)
            print(f"loaded lines from local CSV {path} rows {len(df)}")
            return df

    print("warning: no betting lines found — model will use neutral 0.5 prior where needed")
    return pd.DataFrame(columns=["game_id","spread_line_close","total_line_close","home_moneyline_close","away_moneyline_close"])

# -------------------- assemble --------------------
def load_games_and_lines(years: List[int]) -> pd.DataFrame:
    sched = load_schedules(years)
    scores = load_scores_from_pbp(years)
    epa = load_epa_team_game(years)
    lines = load_lines_any(years)

    base = sched[["game_id","season","week","gameday","home_team","away_team"]].drop_duplicates("game_id")
    base = base.merge(scores, on="game_id", how="left")

    for col in ["spread_line_close","total_line_close","home_moneyline_close","away_moneyline_close"]:
        if col not in lines.columns: lines[col] = np.nan
    line_small = lines[["game_id","spread_line_close","total_line_close","home_moneyline_close","away_moneyline_close"]].drop_duplicates("game_id")
    df = base.merge(line_small, on="game_id", how="left")

    # outcomes & priors
    df["played"] = df["home_score"].notna() & df["away_score"].notna()
    df["home_win"] = (df["home_score"] > df["away_score"]).astype("Int64")

    df["spread_home"] = df["spread_line_close"]
    df["p_spread"] = df["spread_home"].apply(lambda s: np.nan if pd.isna(s) else spread_to_prob_home(float(s)))

    p_home_ml = df["home_moneyline_close"].apply(moneyline_to_prob)
    p_away_ml = df["away_moneyline_close"].apply(moneyline_to_prob)
    both = p_home_ml.notna() & p_away_ml.notna()
    p_ml_norm = pd.Series(np.nan, index=df.index, dtype="float")
    p_ml_norm.loc[both] = p_home_ml.loc[both] / (p_home_ml.loc[both] + p_away_ml.loc[both])

    df["p_market"] = df["p_spread"].where(df["p_spread"].notna(), p_ml_norm).fillna(0.5)

    # attach per-game EPA (home/away)
    h = epa.rename(columns={"team":"home_team"}).add_prefix("home_").rename(columns={"home_game_id":"game_id","home_home_team":"home_team"})
    a = epa.rename(columns={"team":"away_team"}).add_prefix("away_").rename(columns={"away_game_id":"game_id","away_away_team":"away_team"})
    df = df.merge(h[["game_id","home_off_epa_game","home_def_epa_allowed_game"]], on="game_id", how="left")
    df = df.merge(a[["game_id","away_off_epa_game","away_def_epa_allowed_game"]], on="game_id", how="left")
    return df

# -------------------- engineered features --------------------
def add_rest_and_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("gameday").copy()

    def team_view(side: str) -> pd.DataFrame:
        is_home = side == "home"
        cols = ["game_id","gameday",f"{side}_team","home_score","away_score",
                f"{side}_off_epa_game",f"{side}_def_epa_allowed_game"]
        tmp = df[cols].copy()
        tmp = tmp.rename(columns={f"{side}_team":"team"})
        tmp["team_pts"] = tmp["home_score"] if is_home else tmp["away_score"]
        tmp["opp_pts"]  = tmp["away_score"] if is_home else tmp["home_score"]
        tmp["mov_signed"] = tmp["team_pts"] - tmp["opp_pts"]
        tmp["off_epa_game"] = tmp[f"{side}_off_epa_game"]
        tmp["def_epa_allowed_game"] = tmp[f"{side}_def_epa_allowed_game"]
        return tmp[["game_id","gameday","team","mov_signed","off_epa_game","def_epa_allowed_game"]]

    tv = pd.concat([team_view("home"), team_view("away")], ignore_index=True).sort_values(["team","gameday"])
    tv["prev_date"] = tv.groupby("team")["gameday"].shift(1)
    tv["rest_days"] = (tv["gameday"] - tv["prev_date"]).dt.days

    def roll_mean(s: pd.Series) -> pd.Series:
        return s.shift(1).rolling(ROLL_N, min_periods=1).mean()

    tv["form3_mov"]     = tv.groupby("team")["mov_signed"].apply(roll_mean).reset_index(level=0, drop=True)
    tv["off_epa_roll"]  = tv.groupby("team")["off_epa_game"].apply(roll_mean).reset_index(level=0, drop=True)
    tv["def_epa_roll"]  = tv.groupby("team")["def_epa_allowed_game"].apply(roll_mean).reset_index(level=0, drop=True)

    home_roll = tv.merge(df[["game_id","home_team"]], left_on=["game_id","team"], right_on=["game_id","home_team"], how="inner")
    away_roll = tv.merge(df[["game_id","away_team"]], left_on=["game_id","team"], right_on=["game_id","away_team"], how="inner")

    out = df.merge(
        home_roll[["game_id","rest_days","form3_mov","off_epa_roll","def_epa_roll"]].rename(
            columns={"rest_days":"home_rest_days","form3_mov":"home_form3_mov","off_epa_roll":"home_off_epa_roll","def_epa_roll":"home_def_epa_roll"}
        ),
        on="game_id", how="left"
    ).merge(
        away_roll[["game_id","rest_days","form3_mov","off_epa_roll","def_epa_roll"]].rename(
            columns={"rest_days":"away_rest_days","form3_mov":"away_form3_mov","off_epa_roll":"away_off_epa_roll","def_epa_roll":"away_def_epa_roll"}
        ),
        on="game_id", how="left"
    )

    fills = {
        "home_rest_days":7, "away_rest_days":7,
        "home_form3_mov":0.0, "away_form3_mov":0.0,
        "home_off_epa_roll":0.0, "away_off_epa_roll":0.0,
        "home_def_epa_roll":0.0, "away_def_epa_roll":0.0,
    }
    for c,v in fills.items(): out[c] = out[c].fillna(v)

    out["rest_diff"]   = out["home_rest_days"] - out["away_rest_days"]
    out["form3_diff"]  = out["home_form3_mov"] - out["away_form3_mov"]
    out["off_epa_diff"]= out["home_off_epa_roll"] - out["away_off_epa_roll"]
    out["def_epa_diff"]= out["home_def_epa_roll"] - out["away_def_epa_roll"]

    return out

# -------------------- model --------------------
@dataclass
class FitResult:
    model: Optional[object]
    scaler: Optional[StandardScaler]
    features: List[str]
    iso: Optional[IsotonicRegression]

def _select_nonconstant_features(df: pd.DataFrame, feature_cols: List[str]) -> List[str]:
    present = [c for c in feature_cols if c in df.columns]
    if not present: return []
    stds = df[present].astype(float).std(axis=0, ddof=0)
    nonconst = [c for c in present if not (stds.get(c,0.0)==0.0 or np.isnan(stds.get(c,0.0)))]
    if len(nonconst) < len(present):
        print(f"dropping constant or nan features {set(present)-set(nonconst)}")
    return nonconst

def fit_residual_glm(df: pd.DataFrame, feature_cols: List[str], do_calibrate: bool=True) -> FitResult:
    train = df[df["played"] & df["p_market"].notna()].copy()
    if len(train)==0:
        print("no completed games with market data; skipping fit")
        return FitResult(None,None,[],None)

    train["p_market"] = train["p_market"].clip(1e-6, 1-1e-6)
    train["offset"] = np.log(train["p_market"]/(1-train["p_market"]))

    used = _select_nonconstant_features(train, feature_cols)
    if not used:
        print("no usable residual features found — returning market-only fit")
        return FitResult(None,None,[],None)

    X = train[used].astype(float).values
    scaler = StandardScaler(); Xs = scaler.fit_transform(X)
    Xs = sm.add_constant(Xs, has_constant="add")
    y = train["home_win"].astype(int).values

    model = sm.GLM(y, Xs, family=sm.families.Binomial(), offset=train["offset"].values).fit()

    iso = None
    if do_calibrate:
        p_hat = predict_with_model(FitResult(model,scaler,used,None), train).clip(1e-6,1-1e-6)
        iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(p_hat.values, y)

    return FitResult(model, scaler, used, iso)

def predict_with_model(fit: FitResult, df: pd.DataFrame) -> pd.Series:
    use = df.copy()
    use["p_market"] = use["p_market"].clip(1e-6,1-1e-6)
    use["offset"] = np.log(use["p_market"]/(1-use["p_market"]))
    if fit.model is None or not fit.features:
        return pd.Series(use["p_market"].values, index=use.index, name="p_home_pred")
    for c in fit.features:
        if c not in use.columns: use[c] = 0.0
    X = use[fit.features].astype(float).values
    Xs = fit.scaler.transform(X); Xs = sm.add_constant(Xs, has_constant="add")
    lin = use["offset"].values + np.dot(Xs, fit.model.params)
    return pd.Series(1/(1+np.exp(-lin)), index=use.index, name="p_home_pred")

# -------------------- eval & outputs --------------------
def evaluate(df: pd.DataFrame, p_col: str="p_home_pred") -> dict:
    from sklearn.metrics import log_loss, brier_score_loss
    mask = df["played"] & df[p_col].notna()
    if mask.sum()==0: return {}
    y = df.loc[mask,"home_win"].astype(int).values
    p = df.loc[mask,p_col].clip(1e-6,1-1e-6).values
    ll = log_loss(y,p); br = brier_score_loss(y,p)
    cuts = pd.cut(p, bins=np.linspace(0,1,11), include_lowest=True)
    calib = df.loc[mask].groupby(cuts).apply(
        lambda g: pd.Series({"n":len(g),"avg_pred":g[p_col].mean(),"empirical":g["home_win"].mean()})
    ).reset_index(names="bucket")
    return {"log_loss":ll,"brier":br,"calibration":calib}

def upcoming_board(df: pd.DataFrame, p_col: str, p_cal_col: Optional[str]) -> pd.DataFrame:
    up = df[~df["played"]].copy()
    cols = ["season","week","gameday","away_team","home_team","spread_home",
            "total_line_close","home_moneyline_close","away_moneyline_close","p_market"]
    for c in cols:
        if c not in up.columns: up[c] = np.nan
    up["home_win_prob_market"] = up["p_market"]
    up["away_win_prob_market"] = 1 - up["p_market"]
    if p_col in up.columns:
        up["home_win_prob_model"] = up[p_col]
        up["away_win_prob_model"] = 1 - up[p_col]
    if p_cal_col and p_cal_col in up.columns:
        up["home_win_prob_model_cal"] = up[p_cal_col]
        up["away_win_prob_model_cal"] = 1 - up[p_cal_col]
    order = cols + ["home_win_prob_market","away_win_prob_market"]
    if "home_win_prob_model" in up.columns: order += ["home_win_prob_model","away_win_prob_model"]
    if "home_win_prob_model_cal" in up.columns: order += ["home_win_prob_model_cal","away_win_prob_model_cal"]
    return up[order].sort_values(["season","week","gameday","home_team"])

# -------------------- main --------------------
def main():
    years = list(range(2018, 2026))

    df = load_games_and_lines(years)
    print(f"schedules combined: seasons {df['season'].min()} → {df['season'].max()}  rows {len(df)}")

    if df[["spread_line_close","home_moneyline_close","away_moneyline_close"]].isna().all(axis=None):
        print("WARNING: no betting lines found; p_market will be neutral 0.5 (add betting_lines_2018_2025.csv to improve)")

    df = add_rest_and_rolling_features(df)

    feature_cols = ["rest_diff","form3_diff","off_epa_diff","def_epa_diff"]
    fit = fit_residual_glm(df, feature_cols=feature_cols, do_calibrate=True)

    df["p_home_pred"] = predict_with_model(fit, df)
    if fit.iso is not None:
        df["p_home_pred_cal"] = pd.Series(fit.iso.predict(df["p_home_pred"].clip(1e-6,1-1e-6).values), index=df.index)

    scores = evaluate(df, p_col="p_home_pred")
    if scores:
        print("Residual model evaluation on completed games")
        print(f"log loss {scores['log_loss']:.4f}")
        print(f"brier     {scores['brier']:.4f}")
        print("calibration by predicted bucket")
        print(scores["calibration"].to_string(index=False))

    up = upcoming_board(df, p_col="p_home_pred", p_cal_col="p_home_pred_cal" if "p_home_pred_cal" in df.columns else None)
    up.to_csv("upcoming_with_features.csv", index=False); print("Saved upcoming_with_features.csv")

    hist_cols = ["season","week","gameday","away_team","home_team",
                 "home_win","p_market","p_home_pred","p_home_pred_cal" if "p_home_pred_cal" in df.columns else "p_home_pred"]
    df[df["played"]][hist_cols].to_csv("historical_with_features.csv", index=False); print("Saved historical_with_features.csv")

if __name__ == "__main__":
    main()