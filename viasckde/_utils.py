import numpy as np

def minmax_normalize(x, eps=1e-12):
    x = np.asarray(x).astype(float)
    lo, hi = np.min(x), np.max(x)
    if hi - lo < eps:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)
