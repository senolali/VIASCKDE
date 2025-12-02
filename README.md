# VIASCKDE

Python implementation of the VIASCKDE internal cluster validity index.

## Installation

```bash
pip install viasckde
```

## Usage
```python
from viasckde import viasckde_score

# X: your data array
# labels: cluster labels
score = viasckde_score(X, labels, bandwidth=0.05)
```
