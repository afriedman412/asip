# Cleaned and consolidated version of metrics_prep.py

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from config import EPSILON, TOPICS, EMOTIONS, ALL_CHARS
from data_helpers import normalizer, parse_col


# ---------------------------------------------------------------------------
#  Character-season aggregation
# ---------------------------------------------------------------------------

def build_char_season_from_counts(df, metrics=None, char_count_cols=ALL_CHARS):
    """
    Build (SHOW, SEASON, CHAR) weighted mean metrics.
    df rows represent 8-line chunks (SP) or single lines (other shows).
    """
    if metrics is None:
        metrics = make_topic_emotion_metrics()

    long = df.melt(
        id_vars=["SHOW", "SEASON"] + metrics,
        value_vars=char_count_cols,
        var_name="CHAR",
        value_name="COUNT_CHAR"
    )

    long = long[long["COUNT_CHAR"] > 0].copy()
    long["w"] = long["COUNT_CHAR"].astype(float)

    # weighted sums
    for m in metrics:
        long[f"{m}_WSUM"] = long[m] * long["w"]

    agg = (
        long.groupby(["SHOW", "SEASON", "CHAR"], as_index=False)
            .agg({"w": "sum", **{f"{m}_WSUM": "sum" for m in metrics}})
    )

    for m in metrics:
        agg[m] = agg[f"{m}_WSUM"] / agg["w"]
        agg.drop(columns=[f"{m}_WSUM"], inplace=True)

    agg.rename(columns={"w": "N_LINES"}, inplace=True)
    return agg


# ---------------------------------------------------------------------------
#  Movement / Drift / Chaos
# ---------------------------------------------------------------------------

def compute_character_movement_per_metric(char_season_df, metric_cols):
    out_rows = []
    for (show, char), sub in char_season_df.groupby(["SHOW", "CHAR"]):
        sub = sub.sort_values("SEASON")
        vecs = sub[metric_cols].to_numpy()

        if len(vecs) < 2:
            continue

        diffs = np.abs(vecs[1:] - vecs[:-1])
        movement_per_metric = diffs.sum(axis=0)

        row = {"SHOW": show, "CHAR": char}
        for m, v in zip(metric_cols, movement_per_metric):
            row[f"MOVEMENT_{m}"] = v
        out_rows.append(row)

    return pd.DataFrame(out_rows)


def compute_movement_drift_chaos(char_season_df, metric_cols, name=None):
    results = []
    for (show, char), sub in char_season_df.groupby(["SHOW", "CHAR"]):
        sub = sub.sort_values("SEASON")
        V = sub[metric_cols].to_numpy()
        if V.shape[0] < 2:
            continue

        deltas = V[1:] - V[:-1]
        movement = np.linalg.norm(deltas, axis=1).sum()

        drift_vec = V[-1] - V[0]
        drift_mag = np.linalg.norm(drift_vec)
        chaos = movement / (drift_mag + EPSILON)

        N_STEPS = V.shape[0] - 1
        prefix = f"{name}_" if name else ""

        results.append({
            "SHOW": show,
            "CHAR": char,
            "N_STEPS": N_STEPS,
            f"{prefix}MOVEMENT": movement,
            f"{prefix}DRIFT": drift_mag,
            f"{prefix}CHAOS": chaos,
            f"{prefix}MOVEMENT_PER_STEP": movement / N_STEPS,
            f"{prefix}DRIFT_PER_STEP": drift_mag / N_STEPS,
            f"{prefix}CHAOS_PER_STEP": chaos / N_STEPS,
        })

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
#  Topic–emotion interactions
# ---------------------------------------------------------------------------

def make_topic_emotion_metrics(df=None):
    names = [f"{t}__{e}" for t in TOPICS for e in EMOTIONS]
    if df is None:
        return names

    # Build all new columns in a dict first
    new_cols = {}
    for t in TOPICS:
        for e in EMOTIONS:
            col_name = f"{t}__{e}"
            new_cols[col_name] = df[t] * df[e]

    # Add them all at once – avoids fragmentation
    df = df.assign(**new_cols).copy()
    return df


def make_top_col_map():
    return {t: [f"{t}__{e}" for e in EMOTIONS] for t in TOPICS}


# ---------------------------------------------------------------------------
#  Movement wrappers
# ---------------------------------------------------------------------------

def make_movement(df):
    movement_dfs = []
    metrics = make_topic_emotion_metrics()

    for t in TOPICS:
        mask = df['TOP_TOPIC'] == t
        char_season = build_char_season_from_counts(
            df[mask], metrics, ALL_CHARS)

        topic_metrics = [f"{t}__{e}" for e in EMOTIONS]
        movement_df = compute_character_movement_per_metric(
            char_season, topic_metrics)
        movement_dfs.append(movement_df.set_index(["SHOW", "CHAR"]))

    return pd.concat(movement_dfs, axis=1)


def make_per_topic_emotion_movement(char_season):
    dfs = []
    for t, cols in make_top_col_map().items():
        df_t = compute_movement_drift_chaos(char_season, cols, name=t)
        dfs.append(df_t.set_index(["SHOW", "CHAR", "N_STEPS"]))
    per_topic_mov = pd.concat(dfs, axis=1)

    dfs = []
    for t in TOPICS:
        for e in EMOTIONS:
            col = f"{t}__{e}"
            df_te = compute_movement_drift_chaos(char_season, [col], name=col)
            dfs.append(df_te.set_index(["SHOW", "CHAR", "N_STEPS"]))
    per_topic_emotion_mov = pd.concat(dfs, axis=1)

    return per_topic_mov.join(per_topic_emotion_mov).reset_index()


# ---------------------------------------------------------------------------
#  Tidy + normalization
# ---------------------------------------------------------------------------

def tidy_movement_df(all_movement_df):
    long = all_movement_df.reset_index().melt(
        id_vars=["SHOW", "CHAR", "N_STEPS"],
        var_name="METRIC",
        value_name="VALUE"
    )

    meta = pd.json_normalize(long['METRIC'].map(parse_col))
    tidy = pd.concat([long, meta], axis=1)[
        ["SHOW", "CHAR", "N_STEPS", "TOPIC", "EMOTION", "MEASURE", "VALUE"]
    ]
    tidy["EMOTION"] = tidy["EMOTION"].fillna("TOTAL")
    wide = tidy.pivot(
        index=["SHOW", "CHAR", "TOPIC", "EMOTION", "N_STEPS"],
        columns="MEASURE",
        values="VALUE"
    ).reset_index()

    return tidy, adjust_and_normalize(wide)


def adjust_and_normalize(df, col="CHAOS"):
    assert col in df.columns and "N_STEPS" in df.columns

    df[f"LOG_{col}"] = np.log1p(df[col])
    df[f"LOG_{col}_PER_STEP"] = np.log1p(df[col] / df["N_STEPS"])

    for base in [f"LOG_{col}", f"LOG_{col}_PER_STEP"]:
        df[f"{base}_CHAR_NORM"] = normalizer(df, "CHAR", base)
        df[f"{base}_TOPIC_NORM"] = normalizer(df, "TOPIC", base)

        model = smf.ols(f"{base} ~ C(CHAR) + C(TOPIC)", data=df).fit()
        df[f"{base}_BOTH_RESID"] = model.resid

    return df
