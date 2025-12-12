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

        # Ignore noise points
        if self.treat_noise == "ignore":
            mask = labels != -1
            X = X[mask]
            labels = labels[mask]

        if X.size == 0:
            return 0.0

        unique_labels = np.unique(labels)
        if unique_labels.size <= 1:
            return 0.0

        # KDE density estimation
        kde = KernelDensity(bandwidth=self.bandwidth, kernel=self.kernel).fit(X)
        logdens = kde.score_samples(X)
        dens = np.exp(logdens)

        # Pairwise distances
        dists = pairwise_distances(X)
        np.fill_diagonal(dists, np.inf)

        n = len(X)
        a = np.zeros(n)
        b = np.zeros(n)

        # Compute a(i) and b(i)
        for i in range(n):
            li = labels[i]

            same = labels == li
            same[i] = False

            if np.any(same):
                a[i] = np.min(dists[i, same])
            else:
                a[i] = np.inf

            other = labels != li
            if np.any(other):
                b[i] = np.min(dists[i, other])
            else:
                b[i] = np.inf

        # stabilize a and b (remove inf, nan) 
        a = np.nan_to_num(a, posinf=1e12, neginf=0.0)
        b = np.nan_to_num(b, posinf=1e12, neginf=0.0)
        denom = np.maximum(np.maximum(a, b), _EPS)


        # Cluster-wise KDE normalization
        wkde = np.zeros(n)
        for lab in unique_labels:
            idx = np.where(labels == lab)[0]
            wkde[idx] = minmax_normalize(dens[idx])

        # Compute codesd safely
        codesd = wkde * ((b - a) / denom)

        # Cluster averages
        cosec = {}
        for lab in unique_labels:
            idx = np.where(labels == lab)[0]
            cosec[lab] = float(np.mean(codesd[idx]))

        # Weighted average score
        total = sum(len(np.where(labels == lab)[0]) * cosec[lab] for lab in unique_labels)
        total_n = len(X)
        return total / total_n


def viasckde_score(X, labels, bandwidth=0.05, kernel="gaussian", treat_noise="ignore"):
    return VIASCKDE(bandwidth, kernel, treat_noise).score(X, labels)
