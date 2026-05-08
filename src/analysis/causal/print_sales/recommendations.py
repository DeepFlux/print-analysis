"""Recommendation engine for next print buys.

Consumes per-dimension Incremental Sales breakdowns (publication, size, position, edition,
region, product) and surfaces:
  - prioritised list of cuts to scale up (positive Incremental Sales, statistically credible)
  - cuts to deprioritise (negative Incremental Sales that is statistically credible)

The signal used is the Incremental Sales itself (incremental outcome per ₹ of spend);
candidates are ranked by Incremental Sales descending. Significance is treated as a soft
gate via the ``p_threshold`` parameter — rows with p ≥ threshold are dropped
from the "scale up" list because the effect could be noise.
"""
from __future__ import annotations

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)

_DEFAULT_TOP_N: int = 5
_DEFAULT_P_THRESHOLD: float = 0.10
_ATE_SCALE: int = 1_000_000


def _rank_dimension(
    breakdown: pd.DataFrame,
    dimension_label: str,
    top_n: int,
    p_threshold: float,
) -> pd.DataFrame:
    """Return the top-N positive-Incremental Sales rows for a single dimension.

    Args:
        breakdown: Output of one of the run_*_breakdown functions.
        dimension_label: Human-readable dimension name (e.g. "Publication").
        top_n: Maximum rows to return.
        p_threshold: Drop rows with p_value >= this value.

    Returns:
        DataFrame with columns: dimension, group, ate_per_10L, p_value,
        n_obs, total_spend_inr. Empty if no significant positive rows.
    """
    if breakdown is None or breakdown.empty:
        return pd.DataFrame()

    df = breakdown.copy()
    df = df[(df["ate"] > 0) & (df["p_value"] < p_threshold)]
    if df.empty:
        return pd.DataFrame()

    df = df.sort_values("ate", ascending=False).head(top_n)
    out = pd.DataFrame({
        "dimension": dimension_label,
        "group": df["group"].astype(str).to_numpy(),
        "ate_per_10L": (df["ate"] * _ATE_SCALE).round(3).to_numpy(),
        "p_value": df["p_value"].round(3).to_numpy(),
        "n_obs": df["n_obs"].to_numpy(),
        "total_spend_inr": df.get("total_spend_inr", pd.Series([0.0] * len(df))).to_numpy(),
    })
    return out


def _rank_avoid(
    breakdown: pd.DataFrame,
    dimension_label: str,
    top_n: int,
    p_threshold: float,
) -> pd.DataFrame:
    """Return rows where Incremental Sales is credibly negative — candidates to deprioritise."""
    if breakdown is None or breakdown.empty:
        return pd.DataFrame()

    df = breakdown.copy()
    df = df[(df["ate"] < 0) & (df["p_value"] < p_threshold)]
    if df.empty:
        return pd.DataFrame()

    df = df.sort_values("ate", ascending=True).head(top_n)
    out = pd.DataFrame({
        "dimension": dimension_label,
        "group": df["group"].astype(str).to_numpy(),
        "ate_per_10L": (df["ate"] * _ATE_SCALE).round(3).to_numpy(),
        "p_value": df["p_value"].round(3).to_numpy(),
        "n_obs": df["n_obs"].to_numpy(),
        "total_spend_inr": df.get("total_spend_inr", pd.Series([0.0] * len(df))).to_numpy(),
    })
    return out


def build_recommendations(
    publication_breakdown: pd.DataFrame,
    size_breakdown: pd.DataFrame,
    position_breakdown: pd.DataFrame,
    edition_breakdown: pd.DataFrame,
    region_breakdown: pd.DataFrame,
    product_breakdown: pd.DataFrame,
    top_n: int = _DEFAULT_TOP_N,
    p_threshold: float = _DEFAULT_P_THRESHOLD,
) -> dict:
    """Build a recommendation bundle from the per-dimension breakdowns.

    Args:
        publication_breakdown: Output of run_publication_breakdown.
        size_breakdown: Output of run_size_breakdown.
        position_breakdown: Output of run_position_breakdown.
        edition_breakdown: Output of run_edition_breakdown.
        region_breakdown: Output of run_region_breakdown.
        product_breakdown: Output of run_product_breakdown.
        top_n: Maximum rows per dimension. Default 5.
        p_threshold: Drop rows with p_value >= this. Default 0.10.

    Returns:
        Dict with keys:
            - per_dimension: dict[str, pd.DataFrame] — top rows per dimension
            - combined_scale_up: pd.DataFrame — all top rows union, ranked by Incremental Sales
            - combined_avoid: pd.DataFrame — all credibly negative rows
            - p_threshold: float — gate that was applied
            - top_n: int — cap that was applied
    """
    dimensions: list[tuple[str, pd.DataFrame]] = [
        ("Publication", publication_breakdown),
        ("Size", size_breakdown),
        ("Position", position_breakdown),
        ("Edition", edition_breakdown),
        ("Region", region_breakdown),
        ("Product", product_breakdown),
    ]

    per_dimension: dict[str, pd.DataFrame] = {}
    scale_up_frames: list[pd.DataFrame] = []
    avoid_frames: list[pd.DataFrame] = []

    for label, breakdown in dimensions:
        scale_up = _rank_dimension(breakdown, label, top_n, p_threshold)
        per_dimension[label] = scale_up
        if not scale_up.empty:
            scale_up_frames.append(scale_up)
        avoid = _rank_avoid(breakdown, label, top_n, p_threshold)
        if not avoid.empty:
            avoid_frames.append(avoid)

    combined_scale_up = (
        pd.concat(scale_up_frames, ignore_index=True).sort_values("ate_per_10L", ascending=False)
        if scale_up_frames
        else pd.DataFrame(columns=["dimension", "group", "ate_per_10L", "p_value", "n_obs", "total_spend_inr"])
    )
    combined_avoid = (
        pd.concat(avoid_frames, ignore_index=True).sort_values("ate_per_10L", ascending=True)
        if avoid_frames
        else pd.DataFrame(columns=["dimension", "group", "ate_per_10L", "p_value", "n_obs", "total_spend_inr"])
    )

    logger.info(
        "Recommendations: %d scale-up rows, %d avoid rows (p_threshold=%.2f)",
        len(combined_scale_up), len(combined_avoid), p_threshold,
    )

    return {
        "per_dimension": per_dimension,
        "combined_scale_up": combined_scale_up.reset_index(drop=True),
        "combined_avoid": combined_avoid.reset_index(drop=True),
        "p_threshold": p_threshold,
        "top_n": top_n,
    }
