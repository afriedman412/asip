import numpy as np
import matplotlib.pyplot as plt
from umap import UMAP
from config import ALL_METRICS
from sklearn.metrics import silhouette_score

# this should import cuml if gpu is available
from gpu import StandardScaler, PCA


def sil(char_season, category="show"):
  print(silhouette_score(char_season[["umap1", "umap2"]],
                           char_season[category]))


def do_umap(char_season, metrics=ALL_METRICS, PCA_=None):
    um = UMAP(
        n_neighbors=10,
        min_dist=0.1,
        metric='euclidean',
        n_jobs=-1
    )

    scaler = StandardScaler()

    X = char_season[metrics].to_numpy()
    X_scaled = scaler.fit_transform(X)

    w = char_season['n_lines'].to_numpy()
    w = w / w.mean()

    # Weight the *scaled* features
    Xw = X_scaled * np.sqrt(w[:, None])

    if PCA_ is not None:
        X_umap = PCA(n_components=PCA_).fit_transform(Xw)

    else:
        X_umap = um.fit_transform(Xw)

    char_season['umap1'] = X_umap[:, 0]
    char_season['umap2'] = X_umap[:, 1]
    return char_season


def make_char_season(df, metrics=ALL_METRICS):
    """old version pre-SP char distributions"""
    char_season = (
        df.groupby(['show', 'season', 'char'])[metrics]
          .mean()
          .reset_index()
    )

    char_season['n_lines'] = (
        df
        .groupby(['show', 'season', 'char'])
        .size()
        .reset_index(name='n_lines')
    )['n_lines']
    return char_season


def plot_umap(
        char_season, category=None,
        title="UMAP (topics + emotion + toxicity)",
        sil=True,
        ax=None):
    plt.figure(figsize=(7, 5))
    plt.title(title)
    if category is not None:
        for c in char_season[category].unique():
            mask = char_season[category] == c
            plt.scatter(char_season.loc[mask, 'umap1'],
                        char_season.loc[mask, 'umap2'], label=c, alpha=0.3)
        plt.legend()
    else:
        plt.scatter(char_season['umap1'],
                    char_season['umap2'], label=category, alpha=0.3)
    plt.show()
    if sil:
        sil_ = sil(char_season[["umap1", "umap2"]],
                               char_season[category])
        print("Silhouette Score:", sil)


