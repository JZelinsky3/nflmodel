import pandas as pd

import nflreadpy as nfl

WEEK = 4     # <-- change when needed
SEASON = 2025

def ensure_pandas(df):
    try:
        import polars as pl
        if isinstance(df, (pl.DataFrame, pl.LazyFrame)):
            return df.to_pandas()
    except Exception:
        pass
    return df

def main():
    sched = ensure_pandas(nfl.load_schedules([SEASON]))
    wk = sched[(sched["season"]==SEASON) & (sched["week"]==WEEK)].copy()

    # minimal columns you’ll fill
    out = wk[["game_id","away_team","home_team","gameday"]].sort_values("gameday").copy()
    out["spread_line_close"] = ""          # positive = home favorite (if you follow this convention)
    out["total_line_close"] = ""
    out["home_moneyline_close"] = ""
    out["away_moneyline_close"] = ""

    # save a hand-editable CSV
    fname = f"week{WEEK}_lines_template.csv"
    out.to_csv(fname, index=False)
    print("Wrote", fname, "rows:", len(out))

if __name__ == "__main__":
    main()
