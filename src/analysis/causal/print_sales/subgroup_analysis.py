from __future__ import annotations

import warnings

import pandas as pd

from src.analysis.causal.print_sales.adstock import apply_adstock
from src.analysis.causal.print_sales.data_processor import aggregate_daily_spend
from src.utils.logger import get_logger

logger = get_logger(__name__)

_BREAKDOWN_COLS: list[str] = [
    "group", "ate", "ate_lower", "ate_upper", "p_value", "n_obs", "total_spend_inr"
]
_MIN_OBS: int = 30
_DEFAULT_OUTCOME_COL: str = "enquiries"


def _adstock_col(decay: float) -> str:
    return f"adstock_theta_{str(decay).replace('.', '_')}"


def _fit_ols_single(
    data: pd.DataFrame,
    treatment_col: str,
    outcome_col: str,
    confounder_cols: list[str],
) -> tuple[float, float, float, float] | None:
    """Fit OLS and return (ate, ate_lower, ate_upper, p_value) or None on failure."""
    import statsmodels.formula.api as smf

    available_confounders = [c for c in confounder_cols if c in data.columns]
    predictors = " + ".join([treatment_col] + available_confounders)
    formula = f"{outcome_col} ~ {predictors}"

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = smf.ols(formula, data=data).fit()
        ate = float(result.params[treatment_col])
        ci = result.conf_int().loc[treatment_col]
        return ate, float(ci.iloc[0]), float(ci.iloc[1]), float(result.pvalues[treatment_col])
    except Exception as exc:
        logger.warning("OLS failed for treatment=%s: %s", treatment_col, exc)
        return None


def _build_subgroup_panel(
    print_spend: pd.DataFrame,
    sales: pd.DataFrame,
    dimension: str,
    group_value: str,
    best_decay: float,
    confounder_cols: list[str],
    max_lag: int,
    outcome_col: str = _DEFAULT_OUTCOME_COL,
) -> pd.DataFrame | None:
    """Build a sub-group panel filtered to a single dimension value.

    Filters print_spend to rows where dimension == group_value, re-aggregates
    to daily × region, left-joins with sales, applies adstock at best_decay,
    and adds the confounder columns needed by the OLS model.

    Args:
        print_spend: Original validated insertion-level spend DataFrame.
        sales: Validated sales DataFrame.
        dimension: Column name to filter on (e.g. "edition").
        group_value: Value to filter to (e.g. "The Times").
        best_decay: Adstock θ to apply.
        confounder_cols: Required confounder column names.
        max_lag: Max adstock carry-over days.

    Returns:
        Sub-group panel DataFrame, or None if too few observations.
    """
    from sklearn.preprocessing import LabelEncoder

    filtered_spend = print_spend[print_spend[dimension] == group_value].copy()
    if filtered_spend.empty:
        return None

    daily = aggregate_daily_spend(filtered_spend)

    merge_cols = ["date", "region", "product"]
    panel = sales.merge(daily, on=merge_cols, how="left")
    panel["total_spend_inr"] = panel["total_spend_inr"].fillna(0.0)

    treatment_col = _adstock_col(best_decay)
    panel[treatment_col] = (
        panel.sort_values(["region", "product", "date"])
        .groupby(["region", "product"])["total_spend_inr"]
        .transform(lambda s: apply_adstock(s, best_decay, max_lag))
    )

    panel["day_of_week"] = panel["date"].dt.dayofweek
    panel["week_of_year"] = panel["date"].dt.isocalendar().week.astype(int)
    panel["region_id"] = LabelEncoder().fit_transform(panel["region"])
    panel["product_id"] = LabelEncoder().fit_transform(panel["product"])

    panel = panel.sort_values(["region", "product", "date"])
    panel["lagged_outcome_1d"] = panel.groupby(["region", "product"])[outcome_col].shift(1)
    panel = panel.dropna(subset=["lagged_outcome_1d"])

    if len(panel) < _MIN_OBS:
        logger.warning(
            "Sub-group %s=%s has only %d observations — skipping", dimension, group_value, len(panel)
        )
        return None

    return panel


def _run_dimension_breakdown(
    print_spend: pd.DataFrame,
    sales: pd.DataFrame,
    dimension: str,
    best_decay: float,
    confounder_cols: list[str],
    max_lag: int,
    outcome_col: str = _DEFAULT_OUTCOME_COL,
) -> pd.DataFrame:
    """Run OLS for each unique value of a print dimension.

    Args:
        print_spend: Insertion-level spend DataFrame.
        sales: Sales DataFrame.
        dimension: Column name (edition, size, or position).
        best_decay: Best-fit adstock θ.
        confounder_cols: Confounder column names.
        max_lag: Max adstock carry-over days.
        outcome_col: Outcome column to model.

    Returns:
        DataFrame with columns: group, ate, ate_lower, ate_upper, p_value,
        n_obs, total_spend_inr.
    """
    treatment_col = _adstock_col(best_decay)
    rows = []
    for value in sorted(print_spend[dimension].unique()):
        panel = _build_subgroup_panel(
            print_spend, sales, dimension, value, best_decay, confounder_cols,
            max_lag, outcome_col,
        )
        if panel is None:
            continue
        fit = _fit_ols_single(panel, treatment_col, outcome_col, confounder_cols)
        if fit is None:
            continue
        ate, ate_lower, ate_upper, p_value = fit
        spend_total = float(
            print_spend.loc[print_spend[dimension] == value, "spend_in_inr"].sum()
        )
        rows.append({
            "group": value,
            "ate": ate,
            "ate_lower": ate_lower,
            "ate_upper": ate_upper,
            "p_value": p_value,
            "n_obs": len(panel),
            "total_spend_inr": spend_total,
        })
        logger.info("%s=%s  Incremental Sales=%.4f  p=%.4f  n=%d", dimension, value, ate, p_value, len(panel))

    if not rows:
        return pd.DataFrame(columns=_BREAKDOWN_COLS)

    return pd.DataFrame(rows).sort_values("ate", ascending=False).reset_index(drop=True)


def run_region_breakdown(
    panel: pd.DataFrame,
    best_decay: float,
    confounder_cols: list[str],
    outcome_col: str = _DEFAULT_OUTCOME_COL,
) -> pd.DataFrame:
    """Estimate Incremental Sales per region by filtering the primary panel.

    Args:
        panel: Full analytical panel with all adstock columns.
        best_decay: Best-fit adstock θ.
        confounder_cols: Confounder column names.
        outcome_col: Outcome column to model.

    Returns:
        DataFrame with columns: group, ate, ate_lower, ate_upper, p_value,
        n_obs, total_spend_inr.
    """
    treatment_col = _adstock_col(best_decay)
    rows = []
    for region in sorted(panel["region"].unique()):
        sub = panel[panel["region"] == region].copy()
        if len(sub) < _MIN_OBS:
            logger.warning("Region %s has only %d rows — skipping", region, len(sub))
            continue
        fit = _fit_ols_single(sub, treatment_col, outcome_col, confounder_cols)
        if fit is None:
            continue
        ate, ate_lower, ate_upper, p_value = fit
        rows.append({
            "group": region,
            "ate": ate,
            "ate_lower": ate_lower,
            "ate_upper": ate_upper,
            "p_value": p_value,
            "n_obs": len(sub),
            "total_spend_inr": float(sub["total_spend_inr"].sum()),
        })
        logger.info("Region=%s  Incremental Sales=%.4f  p=%.4f  n=%d", region, ate, p_value, len(sub))

    if not rows:
        return pd.DataFrame(columns=_BREAKDOWN_COLS)

    return pd.DataFrame(rows).sort_values("ate", ascending=False).reset_index(drop=True)


def run_edition_breakdown(
    print_spend: pd.DataFrame,
    sales: pd.DataFrame,
    best_decay: float,
    confounder_cols: list[str],
    max_lag: int = 7,
    outcome_col: str = _DEFAULT_OUTCOME_COL,
) -> pd.DataFrame:
    """Estimate Incremental Sales per edition."""
    logger.info("Running edition breakdown...")
    return _run_dimension_breakdown(
        print_spend, sales, "edition", best_decay, confounder_cols, max_lag, outcome_col
    )


def run_size_breakdown(
    print_spend: pd.DataFrame,
    sales: pd.DataFrame,
    best_decay: float,
    confounder_cols: list[str],
    max_lag: int = 7,
    outcome_col: str = _DEFAULT_OUTCOME_COL,
) -> pd.DataFrame:
    """Estimate Incremental Sales per ad size (full_page, half_page, etc.)."""
    logger.info("Running size breakdown...")
    return _run_dimension_breakdown(
        print_spend, sales, "size", best_decay, confounder_cols, max_lag, outcome_col
    )


def run_position_breakdown(
    print_spend: pd.DataFrame,
    sales: pd.DataFrame,
    best_decay: float,
    confounder_cols: list[str],
    max_lag: int = 7,
    outcome_col: str = _DEFAULT_OUTCOME_COL,
) -> pd.DataFrame:
    """Estimate Incremental Sales per placement position (front_page, back_page, etc.)."""
    logger.info("Running position breakdown...")
    return _run_dimension_breakdown(
        print_spend, sales, "position", best_decay, confounder_cols, max_lag, outcome_col
    )


def run_publication_breakdown(
    print_spend: pd.DataFrame,
    sales: pd.DataFrame,
    best_decay: float,
    confounder_cols: list[str],
    max_lag: int = 7,
    outcome_col: str = _DEFAULT_OUTCOME_COL,
) -> pd.DataFrame:
    """Estimate Incremental Sales per publication."""
    logger.info("Running publication breakdown...")
    return _run_dimension_breakdown(
        print_spend, sales, "publication", best_decay, confounder_cols, max_lag, outcome_col
    )


def run_product_breakdown(
    panel: pd.DataFrame,
    best_decay: float,
    confounder_cols: list[str],
    outcome_col: str = _DEFAULT_OUTCOME_COL,
) -> pd.DataFrame:
    """Estimate Incremental Sales per product by filtering the primary panel.

    Unlike edition/size/position, product is in the primary panel directly
    (it is part of the join key), so no panel rebuild is needed.

    Args:
        panel: Full analytical panel (date × region × product).
        best_decay: Best-fit adstock θ.
        confounder_cols: Confounder column names.
        outcome_col: Outcome column to model.

    Returns:
        DataFrame with columns: group, ate, ate_lower, ate_upper, p_value,
        n_obs, total_spend_inr.
    """
    treatment_col = _adstock_col(best_decay)
    rows = []
    for product in sorted(panel["product"].unique()):
        sub = panel[panel["product"] == product].copy()
        if len(sub) < _MIN_OBS:
            logger.warning("Product %s has only %d rows — skipping", product, len(sub))
            continue
        # product_id is constant within a product slice — exclude to avoid collinearity
        available_confounders = [c for c in confounder_cols if c != "product_id" and c in sub.columns]
        fit = _fit_ols_single(sub, treatment_col, outcome_col, available_confounders)
        if fit is None:
            continue
        ate, ate_lower, ate_upper, p_value = fit
        rows.append({
            "group": product,
            "ate": ate,
            "ate_lower": ate_lower,
            "ate_upper": ate_upper,
            "p_value": p_value,
            "n_obs": len(sub),
            "total_spend_inr": float(sub["total_spend_inr"].sum()),
        })
        logger.info("Product=%s  Incremental Sales=%.4f  p=%.4f  n=%d", product, ate, p_value, len(sub))

    if not rows:
        return pd.DataFrame(columns=_BREAKDOWN_COLS)

    return pd.DataFrame(rows).sort_values("ate", ascending=False).reset_index(drop=True)
