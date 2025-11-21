import numpy as np
import pandas as pd
from .config import TOPICS, EMOTIONS


def add_top_topics(df):
    df["top_topic"] = (
        df[TOPICS]
        .astype(float)
        .fillna(-np.inf)     # so NaNs don't win
        .idxmax(axis=1)
    )

    scores = df[TOPICS].astype(float).where(~df[TOPICS].isna(), -np.inf)

    vals = scores.to_numpy()
    top_idx = vals.argmax(axis=1)
    df["top_topic_score"] = vals[np.arange(len(df)), top_idx]
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


def parse_col(c):
    dicto = dict.fromkeys(['topic', 'emotion', 'measure'])
    for c_ in TOPICS:
        if c_ in c:
            dicto['topic'] = c_
    for c_ in EMOTIONS:
        if c_ in c:
            dicto['emotion'] = c_
    for c_ in ['chaos', 'movement', 'drift']:
        if c_ in c:
            dicto['measure'] = c_
    if "per_step" in c:
        dicto['measure'] += "_per_step"
    return dicto


def normalizer(df, g, t):
    normalized = df.groupby(g)[t].transform(
        lambda x: (x - x.mean()) / x.std())
    return normalized
