# VIASCKDE Index

**VIASCKDE** is a novel internal cluster validity index for arbitrary-shaped clusters based on Kernel Density Estimation (KDE).

---

## Motivation

The VIASCKDE Index was proposed in:

*"VIASCKDE Index: A Novel Internal Cluster Validity Index for Arbitrary Shaped Clusters Based on Kernel Density Estimation"*  
by **Ali Şenol**

The index evaluates clustering quality regardless of cluster shape by computing **compactness** and **separation** at the *point level* instead of relying on cluster centroids. This makes it robust for non-spherical and arbitrarily shaped clusters.

---

## Installation

```bash
pip install viasckde
```

## Usage 

```bash
from viasckde import viasckde_score


score = viasckde_score(X, labels)
print("VIASCKDE Score:", score)

VIASCKDE index needs four parameters (two are optional) that are:
    # X: your data array (NumPy-like)
	# labels: predicted cluster labels
    # kernel (optional): selected kernel method, krnl='gaussian' is default kernel. 
But it could be 'tophat', 'epanechnikov', 'exponential', 'linear', or 'cosine'.
    # bandwidth(optional): the bandwidth value of kernel density estimation. b_width=0.05 
is the default value. But it could be changed.
```

## Concept

In non-spherical clusters, the distance from a point to the nearest neighbor in the same cluster is often more meaningful than the distance to the cluster centroid.
VIASCKDE computes:

Compactness: distance to the closest point in the same cluster

Separation: distance to the closest point in a different cluster

This point-level computation ensures realistic evaluation of clusters regardless of their shape.


## Output Range

```bash
VIASCKDE returns a score in [-1, +1]:

+1: best clustering

-1: worst clustering
```

## Citation

```bash
Ali Şenol, "VIASCKDE Index: A Novel Internal Cluster Validity Index for Arbitrary-Shaped 
Clusters Based on the Kernel Density Estimation", Computational Intelligence and 
Neuroscience, vol. 2022, Article ID 4059302, 20 pages, 2022. 
https://doi.org/10.1155/2022/4059302
```

## BibTeX

```bash
@article{csenol2022viasckde,
  title={VIASCKDE Index: A Novel Internal Cluster Validity Index for Arbitrary-Shaped Clusters Based on the Kernel Density Estimation},
  author={{\c{S}}enol, Ali},
  journal={Computational Intelligence and Neuroscience},
  volume={2022},
  number={1},
  pages={4059302},
  year={2022},
  publisher={Wiley Online Library}
}
```

## License & Author

```bash
Author: Assoc. Prof. Dr. Ali Şenol
Computer Engineering Department, Tarsus University

License: MIT
```
