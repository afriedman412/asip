import numpy as np
from gpu import UMAP
from config import ALL_METRICS
from sklearn.metrics import silhouette_score

# this should import cuml if gpu is available
from gpu import StandardScaler, PCA

LABEL_MAP = {
    "asip": "Always Sunny",
    "office": "The Office",
    "southpark": "South Park",
}


def sil(char_season, category="SHOW", text_=None):
    if text_ is not None:
        print(f"*** {text_}:")
    sil_ = silhouette_score(
        char_season[["UMAP1", "UMAP2"]],
        char_season[category])
    print(sil_)
    return sil_


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

    w = char_season['N_LINES'].to_numpy()
    w = w / w.mean()

    # Weight the *scaled* features
    Xw = X_scaled * np.sqrt(w[:, None])

    if PCA_ is not None:
        X_umap = PCA(n_components=PCA_).fit_transform(Xw)

    else:
        X_umap = um.fit_transform(Xw)

    char_season['UMAP1'] = X_umap[:, 0]
    char_season['UMAP2'] = X_umap[:, 1]
    return char_season


def make_char_season(df, metrics=ALL_METRICS):
    """old version pre-SP char distributions"""
    char_season = (
        df.groupby(['SHOW', 'SEASON', 'CHAR'])[metrics]
          .mean()
          .reset_index()
    )

    char_season['N_LINES'] = (
        df
        .groupby(['SHOW', 'SEASON', 'CHAR'])
        .size()
        .reset_index(name='N_LINES')
    )['N_LINES']
    return char_season


def do_umap_unweighted(df, metrics=ALL_METRICS, PCA_=None, **umap_kwargs):
    # Start with defaults
    umap_params = dict(
        n_neighbors=10,
        min_dist=0.1,
        metric='euclidean',
    )
    # Override with any user-specified kwargs
    umap_params.update(umap_kwargs)

    um = UMAP(**umap_params)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[metrics].to_numpy())

    if PCA_:
        X_scaled = PCA(n_components=PCA_).fit_transform(X_scaled)

    X_umap = um.fit_transform(X_scaled)

    df_ = df.copy()
    df_['UMAP1'] = X_umap[:, 0]
    df_['UMAP2'] = X_umap[:, 1]
    return df_
