from __future__ import annotations

import pandas as pd
from sklearn.preprocessing import LabelEncoder

from src.analysis.causal.print_sales.adstock import DECAY_VALUES, build_adstock_columns
from src.utils.logger import get_logger

logger = get_logger(__name__)

PRINT_SPEND_REQUIRED_COLS: list[str] = [
    "date", "region", "edition", "publication", "size", "position", "spend_in_inr"
]
SALES_REQUIRED_COLS: list[str] = ["date", "region", "product", "Units Sold"]

VALID_SIZES: set[str] = {"full_page", "half_page", "quarter_page", "strip"}
VALID_POSITIONS: set[str] = {
    "Front Page", "Inside Page", "Jacket Page 1", "Page 3"
}


class DataValidationError(Exception):
    """Raised when input data fails schema, dtype, or consistency checks."""


def validate_print_spend(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and coerce the print_spend DataFrame.

    Checks column presence, key-column nullability, spend non-negativity,
    and that size/position values are within the expected controlled vocabulary.

    Args:
        df: Raw print spend data (insertion-level).

    Returns:
        Validated copy of df with ``date`` cast to datetime.

    Raises:
        DataValidationError: If any validation check fails.
    """
    result = df.copy()
    missing = set(PRINT_SPEND_REQUIRED_COLS) - set(result.columns)
    if missing:
        raise DataValidationError(
            f"print_spend is missing required columns: {sorted(missing)}"
        )

    for col in ["date", "region", "edition", "publication", "size", "position"]:
        null_count = result[col].isna().sum()
        if null_count:
            raise DataValidationError(
                f"print_spend.{col} has {null_count} null value(s) — all key columns must be non-null"
            )

    try:
        result["date"] = pd.to_datetime(result["date"], format="mixed", dayfirst=True)
    except Exception as exc:
        raise DataValidationError(f"print_spend.date could not be parsed as datetime: {exc}") from exc

    result["spend_in_inr"] = pd.to_numeric(result["spend_in_inr"], errors="coerce")
    if result["spend_in_inr"].isna().any():
        raise DataValidationError(
            "print_spend.spend_in_inr contains non-numeric values — must be a numeric spend amount"
        )
    if (result["spend_in_inr"] < 0).any():
        raise DataValidationError("print_spend.spend_in_inr contains negative values")

    invalid_sizes = set(result["size"].unique()) - VALID_SIZES
    if invalid_sizes:
        raise DataValidationError(
            f"print_spend.size contains unexpected values: {invalid_sizes}. "
            f"Expected: {VALID_SIZES}"
        )

    invalid_positions = set(result["position"].unique()) - VALID_POSITIONS
    if invalid_positions:
        raise DataValidationError(
            f"print_spend.position contains unexpected values: {invalid_positions}. "
            f"Expected: {VALID_POSITIONS}"
        )

    logger.info("print_spend validated: %d rows, date range %s to %s",
                len(result), result["date"].min().date(), result["date"].max().date())
    return result


def validate_sales(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and coerce the sales DataFrame.

    Args:
        df: Raw daily regional sales data.

    Returns:
        Validated copy with ``date`` cast to datetime.

    Raises:
        DataValidationError: If any validation check fails.
    """
    result = df.copy()
    missing = set(SALES_REQUIRED_COLS) - set(result.columns)
    if missing:
        raise DataValidationError(
            f"sales is missing required columns: {sorted(missing)}"
        )

    for col in ["date", "region", "product"]:
        null_count = result[col].isna().sum()
        if null_count:
            raise DataValidationError(
                f"sales.{col} has {null_count} null value(s)"
            )

    try:
        result["date"] = pd.to_datetime(result["date"], format="mixed", dayfirst=True)
    except Exception as exc:
        raise DataValidationError(f"sales.date could not be parsed as datetime: {exc}") from exc

    result = result.rename(columns={"Units Sold": "sales_units"})
    result["sales_units"] = pd.to_numeric(result["sales_units"], errors="coerce")
    if result["sales_units"].isna().any():
        raise DataValidationError(
            "sales.'Units Sold' contains non-numeric or null values — must be a numeric unit count"
        )

    logger.info("sales validated: %d rows, date range %s to %s",
                len(result), result["date"].min().date(), result["date"].max().date())
    return result


def aggregate_daily_spend(print_spend: pd.DataFrame) -> pd.DataFrame:
    """Aggregate insertion-level print spend to daily × region.

    Args:
        print_spend: Validated insertion-level print spend DataFrame.

    Returns:
        DataFrame with columns: date, region, product, total_spend_inr, n_insertions,
        has_full_page, has_front_page.
    """
    logger.info("Aggregating print spend: %d insertion rows", len(print_spend))

    daily = (
        print_spend
        .groupby(["date", "region", "product"], sort=True)
        .agg(
            total_spend_inr=("spend_in_inr", "sum"),
            n_insertions=("spend_in_inr", "count"),
            has_full_page=("size", lambda x: (x == "full_page").any()),
            has_front_page=("position", lambda x: (x == "front_page").any()),
        )
        .reset_index()
    )

    logger.info("Aggregated to %d daily×region rows", len(daily))
    return daily


def build_analytical_panel(
    daily_spend: pd.DataFrame,
    sales: pd.DataFrame,
    decay_values: list[float] = DECAY_VALUES,
    max_lag: int = 7,
) -> pd.DataFrame:
    """Build the balanced analytical panel for causal modelling.

    Joins aggregated daily spend with sales on date × region, adds adstock
    columns for every decay value, and attaches confounder features:
    day_of_week, week_of_year, region_id (label-encoded), is_zero_spend,
    and lagged_sales_1d (first row per region is dropped).

    Regions present in spend but absent in sales raise DataValidationError.

    Args:
        daily_spend: Aggregated daily × region spend (from aggregate_daily_spend).
        sales: Validated sales DataFrame.
        decay_values: θ values to sweep. Default is DECAY_VALUES.
        max_lag: Maximum adstock carry-over days. Default 7.

    Returns:
        Panel DataFrame ready for causal modelling.

    Raises:
        DataValidationError: If spend regions are not present in sales.
    """
    spend_regions = set(daily_spend["region"].unique())
    sales_regions = set(sales["region"].unique())
    unmatched = spend_regions - sales_regions
    if unmatched:
        logger.warning(
            "Dropping %d spend region(s) with no matching sales data: %s",
            len(unmatched), sorted(unmatched),
        )
        daily_spend = daily_spend[~daily_spend["region"].isin(unmatched)].copy()
        spend_regions -= unmatched

    if not spend_regions:
        raise DataValidationError(
            "No overlapping regions between print_spend and sales — cannot build panel."
        )

    logger.info(
        "Building panel — spend rows: %d, sales rows: %d, regions: %s",
        len(daily_spend), len(sales), sorted(spend_regions),
    )

    panel = sales.merge(daily_spend, on=["date", "region", "product"], how="left")
    panel["total_spend_inr"] = panel["total_spend_inr"].fillna(0.0)
    panel["n_insertions"] = panel["n_insertions"].fillna(0).astype(int)
    panel["has_full_page"] = panel["has_full_page"].fillna(False)
    panel["has_front_page"] = panel["has_front_page"].fillna(False)

    logger.info("After left join: %d rows", len(panel))

    panel["day_of_week"] = panel["date"].dt.dayofweek
    panel["week_of_year"] = panel["date"].dt.isocalendar().week.astype(int)
    panel["is_zero_spend"] = (panel["total_spend_inr"] == 0.0).astype(int)

    encoder = LabelEncoder()
    panel["region_id"] = encoder.fit_transform(panel["region"])
    panel["product_id"] = LabelEncoder().fit_transform(panel["product"])

    panel = panel.sort_values(["region", "product", "date"]).reset_index(drop=True)
    panel["lagged_sales_1d"] = panel.groupby(["region", "product"])["sales_units"].shift(1)

    rows_before = len(panel)
    panel = panel.dropna(subset=["lagged_sales_1d"])
    dropped = rows_before - len(panel)
    if dropped:
        logger.info("Dropped %d rows with null lagged_sales_1d (first day per region)", dropped)

    panel = build_adstock_columns(panel, "total_spend_inr", decay_values, max_lag)
    logger.info("Panel built: %d rows, %d columns", len(panel), len(panel.columns))

    return panel
