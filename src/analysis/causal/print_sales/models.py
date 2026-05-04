from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from src.analysis.causal.models import CausalResult


@dataclass
class PrintCausalResult(CausalResult):
    """Causal analysis result for the Print Spend → Sales module.

    Extends CausalResult with adstock decay details, sub-group breakdowns,
    and print-specific metadata.

    Args:
        ate_pct_impact: ATE expressed as a percentage of mean baseline sales.
        best_decay_theta: Winning adstock decay rate θ selected from sweep.
        refutation_details: Per-test breakdown of refutation results.
        decay_sweep: DataFrame with columns: theta, ate, ate_lower, ate_upper,
            p_value, r_squared — one row per decay value tested.
        region_breakdown: DataFrame with columns: group, ate, ate_lower,
            ate_upper, p_value, n_obs — one row per region.
        edition_breakdown: Same schema as region_breakdown, one row per edition.
        size_breakdown: Same schema, one row per ad size.
        position_breakdown: Same schema, one row per placement position.
        publication_breakdown: Same schema, one row per publication.
        product_breakdown: Same schema, one row per product.
        total_spend_inr: Total print spend (INR) across all regions in the analysis window.
        date_range: (start_date_str, end_date_str) of the analysis window.
        n_observations: Total panel rows used in the primary model.
        regions_analysed: List of region names included in the analysis.
        assumptions: Stated model assumptions surfaced to the user in the UI.
    """

    ate_pct_impact: float = 0.0
    best_decay_theta: float = 0.0
    refutation_details: dict = field(default_factory=dict)
    decay_sweep: pd.DataFrame = field(default_factory=pd.DataFrame)
    region_breakdown: pd.DataFrame = field(default_factory=pd.DataFrame)
    edition_breakdown: pd.DataFrame = field(default_factory=pd.DataFrame)
    size_breakdown: pd.DataFrame = field(default_factory=pd.DataFrame)
    position_breakdown: pd.DataFrame = field(default_factory=pd.DataFrame)
    publication_breakdown: pd.DataFrame = field(default_factory=pd.DataFrame)
    product_breakdown: pd.DataFrame = field(default_factory=pd.DataFrame)
    total_spend_inr: float = 0.0
    date_range: tuple[str, str] = ("", "")
    n_observations: int = 0
    regions_analysed: list[str] = field(default_factory=list)
    assumptions: list[str] = field(default_factory=list)
