"""Clustering for the Failure Atlas.

HDBSCAN when it is installed, a deterministic numpy k-means otherwise. Both behind one
interface, because the selection policy downstream must not care which ran, and because
the tests must run without a compiled dependency.

The reason clusters matter at all: "3,000 false positives" tells you nothing actionable.
"Cluster 17: citation-heavy academic prose with long paragraphs, 32 percent of all false
positives" tells you exactly what data to generate next. Clusters are the input to
targeted generation, so a bug here silently misdirects the entire flywheel.

Note on the k-means fallback: HDBSCAN finds clusters of varying density and marks
outliers as noise, which is genuinely better for this problem, because failure modes are
not equal-sized blobs. k-means forces every point into a cluster including one-off
oddities. That difference is recorded in the result so a report cannot silently present
k-means output as if it were density-based.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

NOISE = -1


@dataclass
class ClusterResult:
    labels: np.ndarray          # cluster id per point, NOISE for outliers
    method: str
    n_clusters: int
    n_noise: int
    params: dict = field(default_factory=dict)

    def sizes(self) -> dict[int, int]:
        return {
            int(c): int((self.labels == c).sum())
            for c in sorted(set(self.labels.tolist()))
            if c != NOISE
        }

    def as_dict(self) -> dict:
        return {
            "method": self.method, "n_clusters": self.n_clusters,
            "n_noise": self.n_noise, "sizes": self.sizes(), "params": self.params,
        }


def _kmeanspp_init(x: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    centres = [x[rng.integers(len(x))]]
    for _ in range(1, k):
        d2 = np.min(((x[:, None, :] - np.array(centres)[None, :, :]) ** 2).sum(-1), axis=1)
        total = d2.sum()
        probs = d2 / total if total > 0 else np.full(len(x), 1 / len(x))
        centres.append(x[rng.choice(len(x), p=probs)])
    return np.array(centres)


def kmeans(x: np.ndarray, k: int, iters: int = 100, seed: int = 42) -> ClusterResult:
    """Deterministic k-means++ . Seeded so a rebuild produces the identical atlas."""
    if len(x) < k:
        raise ValueError(f"cannot form {k} clusters from {len(x)} points")
    rng = np.random.default_rng(seed)
    centres = _kmeanspp_init(x, k, rng)
    labels = np.zeros(len(x), dtype=int)
    for _ in range(iters):
        d = ((x[:, None, :] - centres[None, :, :]) ** 2).sum(-1)
        new = d.argmin(1)
        if np.array_equal(new, labels):
            break
        labels = new
        for c in range(k):
            m = labels == c
            if m.any():
                centres[c] = x[m].mean(0)
    return ClusterResult(labels, "kmeans", k, 0, {"k": k, "seed": seed, "iters": iters})


def hdbscan_cluster(x: np.ndarray, min_cluster_size: int = 25) -> ClusterResult:  # pragma: no cover
    import hdbscan as _h

    m = _h.HDBSCAN(min_cluster_size=min_cluster_size, metric="euclidean")
    labels = m.fit_predict(x)
    uniq = {int(c) for c in labels if c != NOISE}
    return ClusterResult(
        np.asarray(labels), "hdbscan", len(uniq), int((labels == NOISE).sum()),
        {"min_cluster_size": min_cluster_size},
    )


def cluster(
    x: np.ndarray,
    method: str = "auto",
    min_cluster_size: int = 25,
    k: int | None = None,
    seed: int = 42,
) -> ClusterResult:
    """method: 'auto' prefers hdbscan and falls back to kmeans, recording which ran."""
    if method in ("auto", "hdbscan"):
        try:
            return hdbscan_cluster(x, min_cluster_size)
        except ImportError:
            if method == "hdbscan":
                raise
    if k is None:
        # A defensible default when hdbscan is unavailable: enough clusters that the
        # average one is roughly min_cluster_size, bounded so tiny sets stay usable.
        k = max(2, min(len(x) // max(min_cluster_size, 1), 50))
    return kmeans(x, k, seed=seed)


def silhouette(x: np.ndarray, labels: np.ndarray, sample: int = 2000, seed: int = 0) -> float:
    """Mean silhouette over non-noise points. Reported so a clustering that found
    nothing real is visible as a number rather than as a confident-looking chart."""
    mask = labels != NOISE
    x, labels = x[mask], labels[mask]
    if len(set(labels.tolist())) < 2:
        return float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(x), size=min(sample, len(x)), replace=False)
    d = np.sqrt(((x[idx][:, None, :] - x[None, :, :]) ** 2).sum(-1))
    scores = []
    for row, i in enumerate(idx):
        same = labels == labels[i]
        same_others = same.copy()
        same_others[i] = False
        if not same_others.any():
            continue
        a = d[row][same_others].mean()
        b = min(d[row][labels == c].mean() for c in set(labels.tolist()) if c != labels[i])
        scores.append((b - a) / max(a, b))
    return float(np.mean(scores)) if scores else float("nan")
