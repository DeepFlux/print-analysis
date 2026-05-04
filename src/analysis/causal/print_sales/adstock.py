from __future__ import annotations

import numpy as np
import pandas as pd

DECAY_VALUES: list[float] = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def apply_adstock(
    spend_series: pd.Series,
    decay: float,
    max_lag: int = 7,
) -> pd.Series:
    """Apply geometric adstock transformation to a spend series.

    adstock_t = spend_t + decay * adstock_(t-1)

    Carry-over is capped at max_lag days: contributions older than max_lag
    days are zeroed out via a sliding-window implementation.

    Args:
        spend_series: Daily spend values, ordered chronologically.
        decay: Decay rate θ in [0.0, 1.0]. 0.0 = no carry-over (raw spend).
        max_lag: Maximum number of days of carry-over. Default 7.

    Returns:
        Series of adstock-transformed values with the same index as spend_series.
    """
    if not 0.0 <= decay <= 1.0:
        raise ValueError(f"decay must be in [0.0, 1.0], got {decay}")

    values = spend_series.to_numpy(dtype=float)
    n = len(values)
    adstock = np.zeros(n)

    for t in range(n):
        adstock[t] = values[t]
        for lag in range(1, min(t, max_lag) + 1):
            adstock[t] += (decay**lag) * values[t - lag]

    return pd.Series(adstock, index=spend_series.index, name=spend_series.name)


def build_adstock_columns(
    panel: pd.DataFrame,
    spend_col: str,
    decay_values: list[float] = DECAY_VALUES,
    max_lag: int = 7,
) -> pd.DataFrame:
    """Add one adstock column per decay value to the panel.

    Column names follow the pattern ``adstock_theta_{decay}`` where the
    decimal point is replaced with underscore (e.g. ``adstock_theta_0_5``).

    Args:
        panel: DataFrame containing spend_col and a DatetimeIndex or a
            ``date`` column sorted chronologically within each region.
        spend_col: Name of the raw spend column.
        decay_values: List of θ values to sweep. Defaults to DECAY_VALUES.
        max_lag: Maximum carry-over days. Default 7.

    Returns:
        Copy of panel with adstock columns appended.
    """
    result = panel.copy()
    group_cols = [c for c in ["region", "product"] if c in result.columns]
    for decay in decay_values:
        col_name = f"adstock_theta_{str(decay).replace('.', '_')}"
        if group_cols:
            result[col_name] = (
                result.groupby(group_cols)[spend_col]
                .transform(lambda s: apply_adstock(s, decay, max_lag))
            )
        else:
            result[col_name] = apply_adstock(result[spend_col], decay, max_lag)
    return result
