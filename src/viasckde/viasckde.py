
from typing import Optional
import numpy as np
from sklearn.neighbors import KernelDensity
from sklearn.metrics import pairwise_distances
from ._utils import minmax_normalize

_EPS = 1e-12

class VIASCKDE:
    def __init__(self, bandwidth=0.05, kernel="gaussian", treat_noise="ignore"):
        self.bandwidth = float(bandwidth)
        self.kernel = kernel
        self.treat_noise = treat_noise

    def score(self, X, labels):
        X = np.asarray(X)
        labels = np.asarray(labels)
        if self.treat_noise == "ignore":
            mask = labels != -1
            X = X[mask]
            labels = labels[mask]
        if X.size == 0:
            return 0.0

        unique_labels = np.unique(labels)
        if unique_labels.size <= 1:
            return 0.0

        kde = KernelDensity(bandwidth=self.bandwidth, kernel=self.kernel).fit(X)
        logdens = kde.score_samples(X)
        dens = np.exp(logdens)

        dists = pairwise_distances(X)
        np.fill_diagonal(dists, np.inf)

        a = np.zeros(len(X))
        b = np.zeros(len(X))
        for i in range(len(X)):
            li = labels[i]
            same = labels == li
            same[i] = False
            a[i] = np.min(dists[i, same]) if np.any(same) else np.inf
            other = labels != li
            b[i] = np.min(dists[i, other]) if np.any(other) else np.inf

        wkde = np.zeros(len(X))
        for lab in unique_labels:
            idx = np.where(labels == lab)[0]
            wkde[idx] = minmax_normalize(dens[idx])

        codesd = wkde * ((b - a) / np.maximum(np.maximum(a, b), _EPS))

        cosec = {}
        for lab in unique_labels:
            idx = np.where(labels == lab)[0]
            cosec[lab] = float(np.mean(codesd[idx]))

        total = sum(len(np.where(labels == lab)[0]) * cosec[lab] for lab in unique_labels)
        total_n = len(X)
        return total / total_n

def viasckde_score(X, labels, bandwidth=0.05, kernel="gaussian", treat_noise="ignore"):
    return VIASCKDE(bandwidth, kernel, treat_noise).score(X, labels)
