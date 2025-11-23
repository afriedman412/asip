import numpy as np
import pandas as pd
from config import TOPICS, EMOTIONS


def add_top_topics(df, topics=TOPICS):
    df["TOP_TOPIC"] = (
        df[topics]
        .astype(float)
        .fillna(-np.inf)     # so NaNs don't win
        .idxmax(axis=1)
    )

    scores = df[topics].astype(float).where(~df[topics].isna(), -np.inf)

    vals = scores.to_numpy()
    top_idx = vals.argmax(axis=1)
    df["TOP_TOPIC_SCORE"] = vals[np.arange(len(df)), top_idx]
    return df


def average_emotions(df, emotions=EMOTIONS, drop=True):
    e_cols_ = []
    for e in emotions:
        e_cols = [c for c in df.columns if c.startswith(e)]
        df[e] = df[e_cols].mean(axis=1)
        if drop:
            e_cols_ += e_cols
    if drop:
        df.drop(columns=e_cols_, inplace=True)
    return df


def consolidate_dialogue(df: pd.DataFrame) -> pd.DataFrame:
    """
    Consolidate consecutive DIALOGUE lines by the same character
    within each (ep_name, scene_num) group.
    Keeps non-dialogue rows unchanged.

    This helps transcript cleanup!
    """

    df = df.copy()

    # Make sure data are properly ordered
    # df = df.sort_values(["ep_name", "scene_num", "page_num"]).reset_index(drop=True)

    # Identify where a new dialogue segment begins
    new_group = (
        # non-dialogue rows
        (df["label"] != "DIALOGUE") |
        # different speaker
        (df["char"].ne(df["char"].shift())) |
        (df["scene_num"].ne(df["scene_num"].shift())) |           # new scene
        (df["ep_name"].ne(df["ep_name"].shift()))                 # new episode
    ).astype(int).cumsum()

    # Group and collapse text within same dialogue runs
    grouped = (
        df.groupby(new_group, as_index=False)
        .agg({
            "ep_name": "first",
            "scene_num": "first",
            "season": "first",
            # "page_num": "first",     # you can also use "min"
            "label": "first",
            "char": "first",
            "text": " ".join        # concatenate text lines
        })
    )

    return grouped


def parse_col(col):
    parts = col.split("__")

    # Topic-only metric
    if len(parts) == 1:
        topic = parts[0].split("_")[0]
        rest = "_".join(parts[0].split("_")[1:])
        return {
            "TOPIC": topic,
            "EMOTION": "TOTAL",
            "MEASURE": rest
        }

    # Topic + emotion metric
    else:
        topic = parts[0]
        emo_and_meas = parts[1].split("_")
        emotion = emo_and_meas[0]
        measure = "_".join(emo_and_meas[1:])
        return {
            "TOPIC": topic,
            "EMOTION": emotion,
            "MEASURE": measure
        }


def normalizer(df, g, t):
    normalized = df.groupby(g)[t].transform(
        lambda x: (x - x.mean()) / x.std())
    return normalized
