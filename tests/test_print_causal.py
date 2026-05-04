"""Tests for the print_sales causal analysis module.

All fixtures use synthetic data — no real client data is used.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.analysis.causal.print_sales.adstock import DECAY_VALUES, apply_adstock, build_adstock_columns
from src.analysis.causal.print_sales.data_processor import (
    DataValidationError,
    aggregate_daily_spend,
    build_analytical_panel,
    validate_print_spend,
    validate_sales,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_print_spend(
    n_days: int = 30,
    regions: list[str] | None = None,
    products: list[str] | None = None,
    spend_values: list[float] | None = None,
) -> pd.DataFrame:
    """Create synthetic insertion-level print spend DataFrame."""
    if regions is None:
        regions = ["North", "South", "Midlands"]
    if products is None:
        products = ["ProductA", "ProductB"]
    rng = np.random.default_rng(42)
    rows = []
    dates = pd.date_range("2024-01-01", periods=n_days, freq="D")
    sizes = ["full_page", "half_page", "quarter_page", "strip"]
    positions = ["Front Page", "Inside Page", "Jacket Page 1", "Page 3"]
    for date in dates:
        for region in regions:
            for product in products:
                spend = spend_values[0] if spend_values else float(rng.integers(100, 5000))
                rows.append({
                    "date": date,
                    "region": region,
                    "product": product,
                    "edition": f"{region} Daily",
                    "publication": f"{region} Times",
                    "size": rng.choice(sizes),
                    "position": rng.choice(positions),
                    "spend_in_inr": spend,
                })
    return pd.DataFrame(rows)


def _make_sales(
    n_days: int = 30,
    regions: list[str] | None = None,
    products: list[str] | None = None,
    base_sales: float = 500.0,
) -> pd.DataFrame:
    """Create synthetic daily regional sales DataFrame."""
    if regions is None:
        regions = ["North", "South", "Midlands"]
    if products is None:
        products = ["ProductA", "ProductB"]
    rng = np.random.default_rng(99)
    rows = []
    dates = pd.date_range("2024-01-01", periods=n_days, freq="D")
    for date in dates:
        for region in regions:
            for product in products:
                rows.append({
                    "date": date,
                    "region": region,
                    "product": product,
                    "Units Sold": base_sales + rng.normal(0, 50),
                })
    return pd.DataFrame(rows)


# ── Adstock tests ─────────────────────────────────────────────────────────────

class TestAdstock:
    def test_adstock_zero_decay(self):
        """With decay=0.0, adstock should equal raw spend (no carry-over)."""
        spend = pd.Series([100.0, 200.0, 150.0, 0.0, 300.0])
        result = apply_adstock(spend, decay=0.0)
        pd.testing.assert_series_equal(result, spend, check_names=False)

    def test_adstock_full_decay(self):
        """With decay=1.0, adstock is the cumulative sum of spend."""
        spend = pd.Series([100.0, 200.0, 50.0])
        result = apply_adstock(spend, decay=1.0, max_lag=10)
        expected = pd.Series([100.0, 300.0, 350.0])
        pd.testing.assert_series_equal(result, expected, check_names=False, atol=1e-6)

    def test_adstock_max_lag(self):
        """Carry-over must not extend beyond max_lag days."""
        n = 10
        spend = pd.Series([1000.0] + [0.0] * (n - 1))
        result = apply_adstock(spend, decay=0.9, max_lag=7)
        # Day 8 onward (index 8+): no carry-over from day 0 spend
        assert result.iloc[8] == pytest.approx(0.0, abs=1e-6)
        assert result.iloc[9] == pytest.approx(0.0, abs=1e-6)
        # Days 1–7 should have non-zero carry-over
        for i in range(1, 8):
            assert result.iloc[i] > 0.0

    def test_adstock_invalid_decay_raises(self):
        spend = pd.Series([100.0, 200.0])
        with pytest.raises(ValueError, match="decay must be in"):
            apply_adstock(spend, decay=1.5)

    def test_build_adstock_columns_names(self):
        """build_adstock_columns should add one column per decay value."""
        panel = pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=5),
            "region": ["North"] * 5,
            "total_spend_inr": [100.0, 200.0, 0.0, 150.0, 300.0],
        })
        result = build_adstock_columns(panel, "total_spend_inr", [0.0, 0.5])
        assert "adstock_theta_0_0" in result.columns
        assert "adstock_theta_0_5" in result.columns
        assert len(result) == 5


# ── Data validation tests ──────────────────────────────────────────────────────

class TestDataValidation:
    def test_validate_print_spend_happy_path(self):
        df = _make_print_spend(n_days=5)
        validated = validate_print_spend(df)
        assert pd.api.types.is_datetime64_any_dtype(validated["date"])

    def test_validate_print_spend_missing_column(self):
        df = _make_print_spend(n_days=5).drop(columns=["spend_in_inr"])
        with pytest.raises(DataValidationError, match="spend_in_inr"):
            validate_print_spend(df)

    def test_validate_sales_happy_path(self):
        df = _make_sales(n_days=5)
        validated = validate_sales(df)
        assert pd.api.types.is_datetime64_any_dtype(validated["date"])

    def test_validate_sales_missing_column(self):
        df = _make_sales(n_days=5).drop(columns=["Units Sold"])
        with pytest.raises(DataValidationError, match="Units Sold"):
            validate_sales(df)


# ── Panel build tests ──────────────────────────────────────────────────────────

class TestPanelBuild:
    def test_panel_build_region_mismatch_partial(self):
        """Spend regions absent in sales are dropped with a warning — no error if overlap exists."""
        spend = _make_print_spend(regions=["North", "Wales"])
        sales = _make_sales(regions=["North", "South"])
        spend = validate_print_spend(spend)
        sales = validate_sales(sales)
        daily = aggregate_daily_spend(spend)
        panel = build_analytical_panel(daily, sales)
        assert "Wales" not in panel["region"].unique()
        assert "North" in panel["region"].unique()

    def test_panel_build_region_mismatch_total(self):
        """DataValidationError raised when there is zero region overlap."""
        spend = _make_print_spend(regions=["Wales"])
        sales = _make_sales(regions=["North", "South"])
        spend = validate_print_spend(spend)
        sales = validate_sales(sales)
        daily = aggregate_daily_spend(spend)
        with pytest.raises(DataValidationError, match="No overlapping"):
            build_analytical_panel(daily, sales)

    def test_panel_build_adds_adstock_columns(self):
        spend = validate_print_spend(_make_print_spend())
        sales = validate_sales(_make_sales())
        daily = aggregate_daily_spend(spend)
        panel = build_analytical_panel(daily, sales, decay_values=[0.0, 0.5])
        assert "adstock_theta_0_0" in panel.columns
        assert "adstock_theta_0_5" in panel.columns

    def test_panel_build_confounders_present(self):
        spend = validate_print_spend(_make_print_spend())
        sales = validate_sales(_make_sales())
        daily = aggregate_daily_spend(spend)
        panel = build_analytical_panel(daily, sales)
        for col in ["day_of_week", "week_of_year", "region_id", "product_id", "lagged_sales_1d"]:
            assert col in panel.columns

    def test_panel_build_no_lagged_nulls(self):
        spend = validate_print_spend(_make_print_spend())
        sales = validate_sales(_make_sales())
        daily = aggregate_daily_spend(spend)
        panel = build_analytical_panel(daily, sales)
        assert panel["lagged_sales_1d"].isna().sum() == 0


# ── Decay sweep test ───────────────────────────────────────────────────────────

class TestDecaySweep:
    def test_decay_sweep_returns_all_thetas(self):
        """Decay sweep should return one row per θ value in DECAY_VALUES (10 total)."""
        from src.analysis.causal.print_sales.causal_model import (
            CONFOUNDER_COLS,
            OUTCOME_COL,
            run_decay_sweep,
        )

        rng = np.random.default_rng(0)
        n = 200
        spend = validate_print_spend(_make_print_spend(n_days=n, regions=["North"]))
        sales = validate_sales(_make_sales(n_days=n, regions=["North"]))
        daily = aggregate_daily_spend(spend)
        panel = build_analytical_panel(daily, sales, decay_values=DECAY_VALUES)

        result = run_decay_sweep(panel, OUTCOME_COL, CONFOUNDER_COLS, DECAY_VALUES)
        assert len(result) == len(DECAY_VALUES)
        assert set(DECAY_VALUES) == set(result["theta"].tolist())


# ── Zero-spend test ────────────────────────────────────────────────────────────

class TestZeroSpend:
    def test_ate_zero_spend_warning(self):
        """When all spend is zero, a warning should be added to the result."""
        from src.analysis.causal.print_sales.causal_model import run_print_causal_analysis

        spend = _make_print_spend(n_days=60, spend_values=[0.0])
        sales = _make_sales(n_days=60)
        result = run_print_causal_analysis(spend, sales, max_lag=3)
        warning_texts = " ".join(result.warnings).lower()
        assert "zero" in warning_texts or "spend" in warning_texts


# ── Refutation placebo test ────────────────────────────────────────────────────

class TestRefutation:
    def test_refutation_placebo_passes_on_zero_effect_data(self):
        """On synthetic data where spend has no causal effect, placebo should pass."""
        from src.analysis.causal.print_sales.causal_model import (
            CONFOUNDER_COLS,
            OUTCOME_COL,
            _adstock_col,
            _build_dowhy_model,
            _run_refutations,
        )
        from src.analysis.causal.print_sales.causal_model import _fit_ols

        rng = np.random.default_rng(7)
        # Two regions × two products × 200 days = 800 rows; varying region_id and
        # product_id avoids constant-column collinearity issues in OLS/DoWhy.
        n_days = 200
        dates = pd.date_range("2024-01-01", periods=n_days, freq="D")
        rows = []
        for region_idx, region in enumerate(["North", "South"]):
            for prod_idx, product in enumerate(["ProductA", "ProductB"]):
                rows.extend({
                    "date": d,
                    "region": region,
                    "product": product,
                    "sales_units": rng.normal(500, 50),
                    "total_spend_inr": rng.uniform(0, 100),
                    "day_of_week": d.dayofweek,
                    "week_of_year": d.isocalendar().week,
                    "region_id": region_idx,
                    "product_id": prod_idx,
                    "lagged_sales_1d": rng.normal(500, 50),
                } for d in dates)
        panel = pd.DataFrame(rows)
        treatment_col = _adstock_col(0.0)
        panel[treatment_col] = panel["total_spend_inr"]

        ate, _, _, _, _ = _fit_ols(panel, treatment_col, OUTCOME_COL, CONFOUNDER_COLS)
        causal_model, identified_estimand, causal_estimate = _build_dowhy_model(
            panel, treatment_col, OUTCOME_COL, CONFOUNDER_COLS
        )
        passed, details = _run_refutations(
            causal_model, identified_estimand, causal_estimate, ate
        )
        assert details["placebo"]["passed"] is True


# ── Sub-group coverage test ────────────────────────────────────────────────────

class TestSubgroupCoverage:
    def test_subgroup_region_coverage(self):
        """region_breakdown should have one row per region in the input (if data is sufficient)."""
        from src.analysis.causal.print_sales.subgroup_analysis import run_region_breakdown
        from src.analysis.causal.print_sales.causal_model import CONFOUNDER_COLS

        regions = ["North", "South", "Midlands"]
        spend = validate_print_spend(_make_print_spend(n_days=90, regions=regions))
        sales = validate_sales(_make_sales(n_days=90, regions=regions))
        daily = aggregate_daily_spend(spend)
        panel = build_analytical_panel(daily, sales)

        breakdown = run_region_breakdown(panel, best_decay=0.3, confounder_cols=CONFOUNDER_COLS)
        assert len(breakdown) == len(regions)
        assert set(breakdown["group"].tolist()) == set(regions)

    def test_subgroup_product_coverage(self):
        """product_breakdown should have one row per product in the input."""
        from src.analysis.causal.print_sales.subgroup_analysis import run_product_breakdown
        from src.analysis.causal.print_sales.causal_model import CONFOUNDER_COLS

        products = ["ProductA", "ProductB", "ProductC"]
        spend = validate_print_spend(_make_print_spend(n_days=90, products=products))
        sales = validate_sales(_make_sales(n_days=90, products=products))
        daily = aggregate_daily_spend(spend)
        panel = build_analytical_panel(daily, sales)

        breakdown = run_product_breakdown(panel, best_decay=0.3, confounder_cols=CONFOUNDER_COLS)
        assert len(breakdown) == len(products)
        assert set(breakdown["group"].tolist()) == set(products)
