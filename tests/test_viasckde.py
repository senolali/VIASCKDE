from sklearn.cluster import DBSCAN
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score
from viasckde import viasckde_score
import matplotlib.pyplot as plt

# 1. Arbitrary-shaped dataset (moons)
X, y_true = make_moons(n_samples=10000, noise=0.07, random_state=42)
X = StandardScaler().fit_transform(X)

# Clustering process with DBSCAN
db = DBSCAN(eps=0.1, min_samples=5)
labels = db.fit_predict(X)

# VIASCKDE Score
viasckde = viasckde_score(X, labels)

ari = adjusted_rand_score(y_true, labels)


print("VIASCKDE Score:", viasckde)
print("ARI Score:", ari)

plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap="viridis", s=12)
plt.title(f"Best DBSCAN Clusters (eps=0.1, min_samples=5)\n"
          f"VIASCKDE={viasckde:.4f}, ARI={ari:.4f}")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.grid(True)
plt.show()
