"""Fit a shape model to already transformed camera histories."""
from sklearn.cluster import KMeans


def fit_camera_shapes(features, k, seed=42, n_init=30):
    return KMeans(n_clusters=k, random_state=seed, n_init=n_init).fit(features)
