"""Cluster-aware bootstrap utilities for repeated-seed experiments."""

from __future__ import annotations

import numpy as np

from .io import stable_int


def hierarchical_mean_ci(
    frame,
    value: str,
    *,
    samples: int,
    confidence: float,
    seed_key: str,
    seed_column: str = "seed",
    cluster_column: str = "source_id",
) -> dict[str, float]:
    clean = frame[[seed_column, cluster_column, value]].dropna()
    if clean.empty:
        return {
            "mean": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "p_positive": float("nan"),
        }
    point = float(clean[value].mean())
    rng = np.random.default_rng(stable_int(seed_key) % (2**32))
    seeds = clean[seed_column].unique()
    per_seed = []
    for seed in seeds:
        values = (
            clean.loc[clean[seed_column] == seed]
            .groupby(cluster_column)[value]
            .mean()
            .to_numpy(dtype=np.float64)
        )
        draws = rng.integers(0, len(values), size=(samples, len(values)))
        per_seed.append(values[draws].mean(axis=1))
    per_seed = np.stack(per_seed)
    sampled_seeds = rng.integers(0, len(seeds), size=(samples, len(seeds)))
    sample_rows = np.arange(samples)[:, None]
    estimates = per_seed[sampled_seeds, sample_rows].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    return {
        "mean": point,
        "ci_low": float(np.quantile(estimates, alpha)),
        "ci_high": float(np.quantile(estimates, 1.0 - alpha)),
        "p_positive": float((np.count_nonzero(estimates <= 0.0) + 1) / (samples + 1)),
    }


def hierarchical_stratified_mean_ci(
    frame,
    value: str,
    *,
    stratum_column: str,
    samples: int,
    confidence: float,
    seed_key: str,
    seed_column: str = "seed",
    cluster_column: str = "source_id",
) -> dict[str, float]:
    """Bootstrap seeds and examples while giving every semantic stratum equal weight."""
    columns = [seed_column, stratum_column, cluster_column, value]
    clean = frame[columns].dropna()
    if clean.empty:
        return {
            "mean": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "p_positive": float("nan"),
        }
    seed_stratum_means = clean.groupby([seed_column, stratum_column])[value].mean()
    point = float(seed_stratum_means.groupby(level=seed_column).mean().mean())
    rng = np.random.default_rng(stable_int(seed_key) % (2**32))
    seeds = clean[seed_column].unique()
    per_seed = []
    for seed in seeds:
        seed_frame = clean.loc[clean[seed_column] == seed]
        stratum_estimates = []
        for _, stratum in seed_frame.groupby(stratum_column):
            values = stratum.groupby(cluster_column)[value].mean().to_numpy(dtype=np.float64)
            draws = rng.integers(0, len(values), size=(samples, len(values)))
            stratum_estimates.append(values[draws].mean(axis=1))
        per_seed.append(np.stack(stratum_estimates).mean(axis=0))
    per_seed = np.stack(per_seed)
    sampled_seeds = rng.integers(0, len(seeds), size=(samples, len(seeds)))
    sample_rows = np.arange(samples)[:, None]
    estimates = per_seed[sampled_seeds, sample_rows].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    return {
        "mean": point,
        "ci_low": float(np.quantile(estimates, alpha)),
        "ci_high": float(np.quantile(estimates, 1.0 - alpha)),
        "p_positive": float((np.count_nonzero(estimates <= 0.0) + 1) / (samples + 1)),
    }


def clustered_mean_ci(
    frame,
    value: str,
    *,
    cluster_column: str,
    samples: int,
    confidence: float,
    seed_key: str,
) -> dict[str, float]:
    clean = frame[[cluster_column, value]].dropna()
    if clean.empty:
        return {"mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}
    means = clean.groupby(cluster_column)[value].mean()
    values = means.to_numpy(dtype=np.float64)
    rng = np.random.default_rng(stable_int(seed_key) % (2**32))
    indices = rng.integers(0, len(values), size=(samples, len(values)))
    estimates = values[indices].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    return {
        "mean": float(values.mean()),
        "ci_low": float(np.quantile(estimates, alpha)),
        "ci_high": float(np.quantile(estimates, 1.0 - alpha)),
    }


def fdr_bh(p_values: list[float]) -> list[float]:
    from statsmodels.stats.multitest import multipletests

    values = np.asarray(p_values, dtype=np.float64)
    valid = np.isfinite(values)
    adjusted = np.full_like(values, np.nan)
    if valid.any():
        adjusted[valid] = multipletests(values[valid], method="fdr_bh")[1]
    return adjusted.tolist()
