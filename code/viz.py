import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from redux import sil
from config import (EMOTIONS, TOPICS, SHOWS, TOPIC_PALETTE, EMOTION_PALETTE,
                    LEGEND_LABELS, TOPIC_ORDER)


def plot_chaos_bar(tidy_df, ax=None, title=None, legend=True):
    """
    Plot log chaos per step for a tidy df on a given axis.

    Expects columns:
        - 'EMOTION'
        - 'TOPIC'
        - 'LOG_CHAOS_PER_STEP_BOTH_RESID'
    """
    df = tidy_df.copy()

    created_fig = None
    if ax is None:
        created_fig, ax = plt.subplots(figsize=(7, 5))

    # Set emotion/topic ordering
    df['EMOTION'] = pd.Categorical(df['EMOTION'],
                                   categories=EMOTIONS,
                                   ordered=True)

    # Restrict topics to those actually present, but in canonical order
    topics_present = [t for t in TOPICS if t in df['TOPIC'].unique()]
    df['TOPIC'] = pd.Categorical(df['TOPIC'],
                                 categories=topics_present,
                                 ordered=True)

    sns.barplot(
        data=df,
        x="EMOTION",
        y="LOG_CHAOS_PER_STEP_BOTH_RESID",
        hue="TOPIC",
        hue_order=topics_present,
        palette={k: v for k, v in TOPIC_PALETTE.items() if k in topics_present},
        edgecolor="white",
        linewidth=1.0,
        ax=ax,
    )

    # Spines
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    for z in ['bottom', 'left', 'top', 'right']:
        ax.spines[z].set_linewidth(2)
        ax.spines[z].set_color('black')
        ax.spines[z].set_visible(True)

    # Title & labels
    if title is not None:
        ax.set_title(title, pad=6, y=1.02)
    ax.set_xlabel("")
    ax.set_ylabel("")

    # X tick labels
    ax.tick_params(axis="x", labelrotation=0, labelsize=10)
    ax.set_xticks(range(len(EMOTIONS)))
    ax.set_xticklabels(
        [e.title() for e in EMOTIONS], fontsize=8, rotation=30)

    # Y ticks
    ax.tick_params(axis="y", labelsize=10)

    # 0 reference line
    ax.axhline(0, color="0.7", linewidth=1)

    # Vertical separators between emotions
    for i in range(len(EMOTIONS) - 1):
        ax.axvline(i + 0.5, color="0.9", linewidth=0.6, zorder=0)
    if legend:
        # Legend with short labels
        leg = ax.legend(loc='best', fontsize=8)
        if leg is not None:
            for text in leg.get_texts():
                orig = text.get_text()
                if orig in LEGEND_LABELS:
                    text.set_text(LEGEND_LABELS[orig])
    else:
        ax.get_legend().remove()

    # Turn off horizontal grid
    ax.grid(axis='y', visible=False)

    return ax


def tri_plot(tidy2, i, j):
    """
    i is the plot columns, j is the boxes
    """
    # prepro steps to ensure consistency
    # tidy2['emotion'] = tidy2['emotion'].fillna("total")
    tidy2['EMOTION'] = pd.Categorical(
        tidy2['EMOTION'], categories=EMOTIONS, ordered=True)
    tidy2['SHOW'] = pd.Categorical(
        tidy2['SHOW'], categories=SHOWS, ordered=True)
    tidy2['TOPIC'] = pd.Categorical(
        tidy2['TOPIC'], categories=TOPIC_ORDER, ordered=True)

    agg = (
        tidy2[tidy2['TOPIC'].isin(TOPIC_ORDER)]
        .groupby(['SHOW', 'TOPIC', 'EMOTION'], observed=True)['LOG_CHAOS_PER_STEP_BOTH_RESID']
        .mean()
        .reset_index()
    )

    agg = agg.query("TOPIC in @TOPIC_ORDER")

    fig, axes = plt.subplots(1, 3, figsize=(12, 5), sharey=True)
    axes = axes.flatten()

    for ax, i_ in zip(axes, tidy2[i].cat.categories):
        df_ = agg[agg[i] == i_]

        sns.barplot(
            data=df_,
            x=j,
            y="LOG_CHAOS_PER_STEP_BOTH_RESID",
            hue="EMOTION",
            hue_order=tidy2['EMOTION'].cat.categories,
            palette=EMOTION_PALETTE,
            ax=ax
        )
        ax.tick_params(axis='x', labelrotation=90)
        ax.set_title(i_.upper())
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.get_legend().remove()

    # final legend outside
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        title="Emotion",
        bbox_to_anchor=(0.98, .95)
    )

    # fig.suptitle("Mean Residual Log Chaos Per Step by Show × Topic × Emotion", y=0.98)
    plt.tight_layout(rect=[0, 0, 0.85, 0.97])
    # plt.tight_layout()
    plt.show()


LABEL_MAP = {
    "asip": "Always Sunny",
    "office": "The Office",
    "southpark": "South Park",
}


def plot_umap(
    char_season,
    category="SHOW",
    ax=None,
    sil_=None,
    title=None,
    legend=True,
    **kwargs
):
    for c in char_season[category].unique():
        mask = char_season[category] == c
        ax.scatter(
            char_season.loc[mask, 'UMAP1'],
            char_season.loc[mask, 'UMAP2'],
            label=LABEL_MAP.get(c, c),
            alpha=0.3)
        ax.set_title(title)
    if sil_ is not None:
        sil(char_season, category, sil_)
    if legend:
        ax.legend()
