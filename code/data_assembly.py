"""
piecemeal functions that are probably outdated
get_sp_chars controls # of characters!
"""
import pandas as pd
from .data_helpers import add_top_topics

from .config import (
    SP_DF, SP_DF_TOPICS, SP_LINES, PREPRO,
    EMOTIONS, TOPICS
)


def chunk_sp(df):
    """
    Loosely the process to chunk SP lines into ~scenes.
    """
    CHUNK_SIZE = 8
    OVERLAP = 3
    STEP = CHUNK_SIZE - OVERLAP

    df = df.copy()
    # within-episode running line numbers
    df["LineNumber"] = df.groupby(["Season", "Episode"]).cumcount() + 1

    def chunk_episode(g: pd.DataFrame):
        chunks = []
        n = len(g)
        for n_, start in enumerate(range(0, n, STEP)):
            end = min(start + CHUNK_SIZE, n)
            sub = g.iloc[start:end]

            # character counts
            counts = sub["Character"].value_counts().to_dict()

            # space-concatenated text; robust to NaNs and non-strings
            if "Line" in sub.columns:
                text = " ".join(sub["Line"].astype(str).fillna("").tolist())
                # optional: normalize whitespace
                text = " ".join(text.split())
            else:
                text = None

            chunks.append({
                "Season": g["Season"].iloc[0],
                "Episode": g["Episode"].iloc[0],
                "chunk_start_line": int(sub["LineNumber"].iloc[0]),
                "chunk_end_line": int(sub["LineNumber"].iloc[-1]),
                "chunk_idx": n_,
                "n_lines": int(len(sub)),
                "CharacterCounts": counts,
                "TextConcat": text,
            })
        return chunks

    out = []
    for _, g in df.sort_values(
        ["Season", "Episode", "LineNumber"]
    ).groupby(["Season", "Episode"]):
        out.extend(chunk_episode(g))

    chunk_df = pd.DataFrame(out)
    return chunk_df


def get_sp_chars(n):
    """
    n = number of top chars to pull
    """
    sp_lines = pd.read_csv(SP_LINES)
    sp_lines.columns = sp_lines.columns.str.lower()
    sp_lines.drop(columns='kenny', errors="ignore", inplace=True)
    more_sp_chars = sp_lines.sum().sort_values(
        ascending=False).head(n).index.str.lower()
    sp_lines = sp_lines[more_sp_chars].fillna(0)
    return sp_lines


def prepro(df=None):
    if df is None:
        df = pd.read_csv(PREPRO).set_index(
            ['show', 'season', 'episode', 'scene']
        )
        col_map = {
            c: c+"_toxic" for c in [
                'identity_hate', 'insult', 'obscene', 'threat', 'toxic'
            ]}
        df.rename(columns=col_map, inplace=True)
        df.reset_index(drop=False, inplace=True)

        # drop low frequncy office chars
        df = df[~df['char'].isin([
            'phyllis', 'kelly', 'toby', 'jan', 'stanley',
            'meredith', 'holly', 'nellie', 'gabe'
        ])]
        df = pd.concat(
            [df, pd.get_dummies(df['char']).astype(int)], axis=1)
        df.drop(columns="char", inplace=True)
        return df


def sp_prepro(n=20):
    sp_df = pd.read_csv(SP_DF)
    sp_chars = get_sp_chars(n)
    sp_df = pd.concat([sp_df, sp_chars], axis=1)
    sp_df_topics = pd.read_csv(SP_DF_TOPICS)

    sp_df[['race and ethnicity',
           'LGBTQ issues', 'gun policy and firearms', 'religion and faith',
           'electoral politics and government',
           'socioeconomic class and inequality',
           'illegal drugs and substance policy',]
          ] = sp_df_topics[
        ['race and ethnicity', 'LGBTQ issues', 'gun policy and firearms',
            'religion and faith', 'electoral politics and government',
            'socioeconomic class and inequality',
            'illegal drugs and substance policy',]
    ]

    sp_df.rename(
        columns={'severe_toxic_toxic': 'severe_toxic',
                 'scene_chunk': 'scene'},
        errors="ignore", inplace=True)
    for c in [
        'anger_jieli', 'disgust_jieli', 'fear_jieli',
        'joy_jieli', 'neutral_jieli',
        'sadness_jieli', 'surprise_jieli'
    ]:
        c_ = c.replace("jieli", "mjm")
        sp_df[c_] = sp_df[c]
        sp_df.drop(columns=[c], inplace=True)
        sp_df['show'] = "southpark"
        return sp_df


def get_all_data(n_sp_chars=10):
    df = prepro()
    sp_df = sp_prepro(n_sp_chars)
    df = pd.concat([df, sp_df])
    for e in EMOTIONS:
        df[e] = df[[c for c in df if c.startswith(e)]].mean(axis=1)
    df = add_top_topics(df)
    df.drop(columns=['LABEL0_mrm', 'LABEL1_mrm'], inplace=True)
    df.fillna(0, inplace=True)
    return df
