import numpy as np
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from viasckde import viasckde_score

def test_viasckde_basic():
    X, _ = make_blobs(n_samples=100, centers=3, random_state=42)
    labels = KMeans(n_clusters=3, random_state=42).fit_predict(X)
    val = viasckde_score(X, labels)
    assert isinstance(val, float)
