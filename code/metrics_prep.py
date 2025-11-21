import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from .config import EPSILON, TOPICS, EMOTIONS, ALL_CHARS
from .data_helpers import normalizer, parse_col


def build_char_season_from_counts(df, metrics=None, char_count_cols=ALL_CHARS):
    """
    df rows:
        - SP: one row per 8-line chunk, each row has count columns
        (cartman=3, kyle=2, etc.)
        - Other shows: one row per line, one-hot counts (speaker=1, others=0)

    char_count_cols: list of columns with integer counts per row.
        (this can be ALL_CHARS but go back to data for more characters)
    metrics: list of emotion/topic/toxic columns.

    Output:
        (show, season, char) table with weighted metric means
    """

    if metrics is None:
        metrics = make_topic_emotion_metrics()

    # reshape to long format
    long = df.melt(
        id_vars=["show", "season"] + metrics,
        value_vars=char_count_cols,
        var_name="char",
        value_name="count_char"
    )

    # keep only characters that appear in that row
    long = long[long["count_char"] > 0].copy()

    # effective weight for this (row, char)
    long["w"] = long["count_char"].astype(float)

    # weighted sums
    for m in metrics:
        long[f"{m}_wsum"] = long[m] * long["w"]

    agg = (
        long.groupby(["show", "season", "char"], as_index=False)
            .agg({"w": "sum", **{f"{m}_wsum": "sum" for m in metrics}})
    )

    # convert weighted sums to weighted means
    for m in metrics:
        agg[m] = agg[f"{m}_wsum"] / agg["w"]
        agg.drop(columns=[f"{m}_wsum"], inplace=True)

    # rename w → total count of lines spoken
    agg.rename(columns={"w": "n_lines"}, inplace=True)

    return agg


def compute_character_movement_per_metric(char_season_df, metric_cols):
    """
    For each (show, char), compute movement for EACH metric separately.
    movement_metric = sum over seasons of absolute differences.
    Returns: long-form DataFrame
    """
    out_rows = []

    for (show, char), sub in char_season_df.groupby(["show", "char"]):
        sub = sub.sort_values("season")
        vecs = sub[metric_cols].to_numpy()

        # absolute difference per metric (season-to-season)
        diffs = np.abs(vecs[1:] - vecs[:-1])

        # sum per metric
        movement_per_metric = diffs.sum(axis=0)

        row = {"show": show, "char": char}
        for m, v in zip(metric_cols, movement_per_metric):
            row[f"movement_{m}"] = v

        out_rows.append(row)

    return pd.DataFrame(out_rows)


def compute_movement_drift_chaos(
        char_season_df, metric_cols, name=None
):
    """
    For each (show, char), compute:
      - movement: total path length across seasons
      - drift: straight-line distance from first to last season
      - chaos: movement / drift
      - n_steps: number of season-to-season jumps
      - movement_per_step: movement / n_steps
      - drift_per_step: drift / n_steps
      - chaos_per_step: chaos / n_steps
    """
    results = []

    for (show, char), sub in char_season_df.groupby(["show", "char"]):
        sub = sub.sort_values("season")
        V = sub[metric_cols].to_numpy()

        # Need at least 2 seasons to define a step
        if V.shape[0] < 2:
            continue

        # Season-to-season deltas
        deltas = V[1:] - V[:-1]

        movement = np.linalg.norm(deltas, axis=1).sum()   # total path length
        # straight-line vector
        drift_vec = V[-1] - V[0]
        # straight-line distance
        drift_mag = np.linalg.norm(drift_vec)

        # path / straight line
        chaos = movement / (drift_mag + EPSILON)
        n_steps = V.shape[0] - 1                          # number of jumps

        # Per-step versions (normalize by how many jumps we observed)
        movement_per_step = movement / n_steps
        drift_per_step = drift_mag / n_steps
        chaos_per_step = chaos / n_steps

        prefix = f"{name}_" if name is not None else ""
        row = {
            "show": show,
            "char": char,
            "n_steps": n_steps,
            f"{prefix}movement": movement,
            f"{prefix}drift": drift_mag,
            f"{prefix}chaos": chaos,
            f"{prefix}movement_per_step": movement_per_step,
            f"{prefix}drift_per_step": drift_per_step,
            f"{prefix}chaos_per_step": chaos_per_step,
        }

        results.append(row)

    return pd.DataFrame(results)


def make_topic_emotion_metrics(df=None):
    topic_emotion_metrics = []
    for t in TOPICS:
        for e in EMOTIONS:
            topic_emotion_metrics.append(f"{t}__{e}")
            if df is not None:
                df[f"{t}__{e}"] = df[t] * df[e]
    if df is not None:
        return df
    else:
        return topic_emotion_metrics


def make_top_col_map():
    topic_to_cols = {
        t: [f"{t}__{e}" for e in EMOTIONS]
        for t in TOPICS
    }
    return topic_to_cols


def make_movement(df):
    movement_dfs = []
    topic_emotion_metrics = make_topic_emotion_metrics()

    for t in TOPICS:
        mask = df['top_topic'] == t

        char_season = build_char_season_from_counts(
            df[mask],
            topic_emotion_metrics,
            ALL_CHARS
        )

        movement_df = compute_character_movement_per_metric(
            char_season,
            # ONLY the interactions for THIS topic
            [f"{t}__{e}" for e in EMOTIONS]
        ).rename(columns={"movement": f"movement_{t}"})

        movement_dfs.append(movement_df.set_index(['show', 'char']))

    movement_df = pd.concat(movement_dfs, axis=1)
    return movement_df


def make_per_topic_emotion_movement(char_season):
    """
    Combines 2 functions that probably don't even need to be 2 loops
    """
    dfs = []
    topic_to_cols = make_top_col_map()

    for t, cols in topic_to_cols.items():
        df_t = compute_movement_drift_chaos(
            char_season,
            metric_cols=cols,
            name=t  # will prefix movement/drift/chaos with topic name
        )
        dfs.append(df_t.set_index(["show", "char", "n_steps"]))

    per_topic_mov = pd.concat(dfs, axis=1).reset_index()

    dfs = []
    for t in TOPICS:
        for e in EMOTIONS:
            col = f"{t}__{e}"
            df_te = compute_movement_drift_chaos(
                char_season,
                metric_cols=[col],
                name=f"{t}__{e}"
            )
            dfs.append(df_te.set_index(["show", "char", "n_steps"]))

    per_topic_emotion_mov = pd.concat(dfs, axis=1).reset_index()

    for p_ in [per_topic_mov, per_topic_emotion_mov]:
        p_.set_index(['show', 'char', 'n_steps'], inplace=True)

    all_movement_df = per_topic_mov.join(per_topic_emotion_mov)
    return all_movement_df


def tidy_movement_df(all_movement_df):
    long = all_movement_df.reset_index().melt(
        id_vars=["show", "char", "n_steps"],
        var_name="metric",
        value_name="value"
    )
    tidy_df = long.merge(
        pd.json_normalize(long['metric'].map(parse_col)),
        left_index=True,
        right_index=True
    )[['show', 'char', 'n_steps', 'topic', 'emotion', 'measure', 'value']]

    # tidy_df['emotion'] = tidy_df['emotion'].fillna("total")

    tidy2 = tidy_df.pivot(index=["show", "char", "topic", "emotion", 'n_steps'],
                          columns="measure",
                          values="value").reset_index()

    tidy_adjusted = adjust_and_normalize(tidy2)
    return tidy_df, tidy_adjusted


def adjust_and_normalize(df, col="chaos"):
    """
    Feature prep on chaos
    """
    assert col in df.columns, f"Missing {col} column"
    assert "n_steps" in df.columns, "Missing n_steps column"

    log_ = f'log_{col}'
    per_step = f'log_{col}_per_step'

    df[log_] = np.log1p(df[col])
    df[per_step] = np.log1p(df[col]/df['n_steps'])

    for t in [log_, per_step]:
        df[f"{t}_char_norm"] = normalizer(df, 'char', t)
        df[f"{t}_topic_norm"] = normalizer(df, 'topic', t)

        # two-way residualization
        model = smf.ols(f"{t} ~ C(char) + C(topic)", data=df).fit()
        df[f"{t}_both_resid"] = model.resid
    return df
