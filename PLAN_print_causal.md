# Plan: Print Spend → Sales Causal Analysis

## Context
Havas Martech Analytics Platform needs a causal analysis module estimating the Average Treatment Effect (ATE) of Print media spend on daily regional sales. The module uses geometric adstock decay (θ sweep 0.0–0.9), DoWhy with Linear Regression, mandatory refutation tests, and dimension-level sub-group breakdowns (Region, Edition, Size, Position). Output is a Streamlit page styled in Havas Black & Red theme.

---

## 1. File Structure (all files to create)

```
src/
├── analysis/
│   └── causal/
│       ├── models.py                        ← Base CausalResult dataclass
│       └── print_sales/
│           ├── __init__.py
│           ├── models.py                    ← PrintCausalResult(CausalResult)
│           ├── adstock.py                   ← apply_adstock, decay sweep
│           ├── data_processor.py            ← Load, validate, aggregate, panel build
│           ├── causal_model.py              ← DoWhy pipeline + refutations
│           └── subgroup_analysis.py         ← Region/Edition/Size/Position breakdowns
├── pages/
│   └── 03_print_causal_analysis.py          ← Streamlit UI
├── components/
│   └── causal_charts.py                     ← Reusable Plotly chart functions
└── utils/
    └── logger.py                            ← Structured logging utility
tests/
└── test_print_causal.py                     ← All 8 required test cases
PLAN_print_causal.md                         ← Copy of this plan (project root, per CLAUDE.md)
requirements.txt                             ← All pip dependencies
```

---

## 2. Dependencies (`requirements.txt`)

```
dowhy>=0.11
econml>=0.15
causalml>=0.15
statsmodels>=0.14
scipy>=1.11
pingouin>=0.5
plotly>=5.0
openpyxl>=3.1
pandas>=2.0
numpy>=1.25
streamlit>=1.30
pydantic>=2.0
python-dotenv>=1.0
pytest>=7.4
pytest-cov>=4.1
```

---

## 3. Function Signatures

### `src/analysis/causal/models.py`
```python
from dataclasses import dataclass

@dataclass
class CausalResult:
    treatment: str
    outcome: str
    ate: float
    ate_lower: float          # 95% CI lower
    ate_upper: float          # 95% CI upper
    p_value: float
    method: str               # e.g. "DoWhy-LinearRegression"
    refutation_passed: bool
    interpretation: str       # plain-English summary
    warnings: list[str]
```

### `src/analysis/causal/print_sales/models.py`
```python
from dataclasses import dataclass, field
import pandas as pd
from src.analysis.causal.models import CausalResult

@dataclass
class PrintCausalResult(CausalResult):
    ate_pct_impact: float             # ATE as % of mean baseline sales
    best_decay_theta: float           # Winning θ
    refutation_details: dict          # per-test breakdown
    decay_sweep: pd.DataFrame         # theta, ATE, p_value, r_squared
    region_breakdown: pd.DataFrame    # region, ATE, ate_lower, ate_upper, p_value, n_obs
    edition_breakdown: pd.DataFrame
    size_breakdown: pd.DataFrame
    position_breakdown: pd.DataFrame
    date_range: tuple[str, str]
    n_observations: int
    regions_analysed: list[str]
    assumptions: list[str]            # surfaced to UI
```

### `src/analysis/causal/print_sales/adstock.py`
```python
DECAY_VALUES: list[float] = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

def apply_adstock(
    spend_series: pd.Series,
    decay: float,
    max_lag: int = 7,
) -> pd.Series:
    """Geometric adstock: adstock_t = spend_t + decay * adstock_(t-1), capped at max_lag."""

def build_adstock_columns(
    panel: pd.DataFrame,
    spend_col: str,
    decay_values: list[float],
    max_lag: int = 7,
) -> pd.DataFrame:
    """Add adstock_θ{decay} columns to panel for each decay. Returns copy."""
```

### `src/analysis/causal/print_sales/data_processor.py`
```python
class DataValidationError(Exception): ...

PRINT_SPEND_REQUIRED_COLS: list[str] = ["date", "region", "edition", "size", "position", "spend_gbp"]
SALES_REQUIRED_COLS: list[str] = ["date", "region", "sales_units"]

def validate_print_spend(df: pd.DataFrame) -> pd.DataFrame:
    """Check columns, dtypes, no nulls in key cols. Raises DataValidationError."""

def validate_sales(df: pd.DataFrame) -> pd.DataFrame:
    """Check columns, dtypes. Raises DataValidationError."""

def aggregate_daily_spend(print_spend: pd.DataFrame) -> pd.DataFrame:
    """Aggregate insertion-level print_spend to date × region.
    Returns: date, region, total_spend_gbp, n_insertions, has_full_page, has_front_page."""

def build_analytical_panel(
    daily_spend: pd.DataFrame,
    sales: pd.DataFrame,
    decay_values: list[float],
    max_lag: int = 7,
) -> pd.DataFrame:
    """Join daily_spend + sales on date+region. Add adstock columns, day_of_week,
    week_of_year, region_id (label-encoded), is_zero_spend, lagged_sales_1d."""
```

### `src/analysis/causal/print_sales/causal_model.py`
```python
from src.analysis.causal.print_sales.models import PrintCausalResult

CONFOUNDER_COLS: list[str] = ["day_of_week", "week_of_year", "region_id", "lagged_sales_1d"]
ASSUMPTIONS: list[str] = [...]   # 6 assumptions from Spec Section 11

def _fit_dowhy(
    panel: pd.DataFrame,
    treatment_col: str,
    outcome_col: str,
    confounder_cols: list[str],
) -> tuple[float, float, float, float]:
    """Run DoWhy LinearRegression. Returns (ate, ate_lower, ate_upper, p_value).
    CIs derived from OLS standard errors via statsmodels."""

def _run_refutations(
    panel: pd.DataFrame,
    treatment_col: str,
    outcome_col: str,
    confounder_cols: list[str],
    original_ate: float,
) -> tuple[bool, dict]:
    """Run placebo, random_common_cause, data_subset refuters.
    Pass condition: placebo ATE p>0.05; others change <20% and no sign flip."""

def run_decay_sweep(
    panel: pd.DataFrame,
    outcome_col: str,
    confounder_cols: list[str],
) -> pd.DataFrame:
    """Fit DoWhy for each θ. Returns DataFrame: theta, ate, ate_lower, ate_upper, p_value, r_squared."""

def select_best_decay(decay_sweep: pd.DataFrame) -> float:
    """Select θ with lowest p_value; ties broken by highest r_squared."""

def build_interpretation(result_data: dict) -> str:
    """Generate plain-English summary from result fields."""

def run_print_causal_analysis(
    print_spend: pd.DataFrame,
    sales: pd.DataFrame,
    max_lag: int = 7,
    regions: list[str] | None = None,
    date_range: tuple[str, str] | None = None,
) -> PrintCausalResult:
    """Top-level entry point: validate → aggregate → panel → decay sweep → best model
    → refutations → sub-group breakdowns → return PrintCausalResult."""
```

### `src/analysis/causal/print_sales/subgroup_analysis.py`
```python
def _build_subgroup_panel(
    print_spend: pd.DataFrame,
    sales: pd.DataFrame,
    dimension: str,
    group_value: str,
    best_decay: float,
    max_lag: int,
) -> pd.DataFrame:
    """Filter print_spend to rows where dimension==group_value,
    aggregate to daily×region, join with sales, apply adstock at best_decay."""

def run_region_breakdown(
    panel: pd.DataFrame,
    best_decay: float,
    confounder_cols: list[str],
) -> pd.DataFrame:
    """Filter full panel by region. Returns: region, ate, ate_lower, ate_upper, p_value, n_obs."""

def run_edition_breakdown(
    print_spend: pd.DataFrame,
    sales: pd.DataFrame,
    best_decay: float,
    confounder_cols: list[str],
    max_lag: int = 7,
) -> pd.DataFrame:
    """Per-edition sub-panel from original print_spend. Returns same schema as region_breakdown."""

def run_size_breakdown(
    print_spend: pd.DataFrame,
    sales: pd.DataFrame,
    best_decay: float,
    confounder_cols: list[str],
    max_lag: int = 7,
) -> pd.DataFrame:
    """Per-size (full_page, half_page, etc.) sub-panel. Same return schema."""

def run_position_breakdown(
    print_spend: pd.DataFrame,
    sales: pd.DataFrame,
    best_decay: float,
    confounder_cols: list[str],
    max_lag: int = 7,
) -> pd.DataFrame:
    """Per-position (front_page, back_page, etc.) sub-panel. Same return schema."""
```

### `src/components/causal_charts.py`
```python
import plotly.graph_objects as go

def plot_decay_sweep(decay_sweep: pd.DataFrame) -> go.Figure:
    """Line chart: x=θ, y=ATE with CI bands. Havas Red primary series."""

def plot_ate_by_region(region_breakdown: pd.DataFrame) -> go.Figure:
    """Horizontal bar chart sorted by ATE. Havas Red bars."""

def plot_subgroup_breakdown(breakdown_df: pd.DataFrame, dimension_label: str) -> go.Figure:
    """Horizontal bar of ATEs for a given dimension. Error bars for CIs."""
```

### `src/utils/logger.py`
```python
import logging

def get_logger(name: str) -> logging.Logger:
    """Return configured logger with consistent format."""
```

---

## 4. Streamlit Page: `src/pages/03_print_causal_analysis.py`

Layout per Spec Section 8:
- `st.set_page_config(layout="wide")` (mandatory)
- Havas theme CSS injected via `st.markdown(..., unsafe_allow_html=True)`
- Sidebar: region multi-select, date range, max adstock lag slider (1–7)
- Row 1: 4 metric cards — ATE, Best θ, p-value badge, Refutation badge
- Row 2: Decay sweep chart | ATE by region bar chart
- Row 3: Sub-group tabs (Region / Edition / Size / Position), each with table + chart
- Row 4: Expandable assumptions & warnings + interpretation paragraph
- Row 5: Download button (Excel export via openpyxl)
- Computation gated behind "Run Analysis" button; result stored in `st.session_state`

---

## 5. Test Cases (`tests/test_print_causal.py`)

| Test | What it tests |
|---|---|
| `test_adstock_zero_decay` | apply_adstock(decay=0.0) → equals raw spend |
| `test_adstock_full_decay` | apply_adstock(decay=1.0) → cumulative sum |
| `test_adstock_max_lag` | 10-day series, max_lag=7 → no carry-over after day 7 |
| `test_ate_zero_spend` | All spend=0 → ATE=0, warning in result |
| `test_panel_build_region_mismatch` | Regions don't match → DataValidationError |
| `test_decay_sweep_returns_all_thetas` | Normal input → 10 rows in decay_sweep |
| `test_refutation_placebo` | Synthetic zero-effect data → refutation_passed=True |
| `test_subgroup_region_coverage` | 3-region input → 3 rows in region_breakdown |

All tests use synthetic fixtures; no real client data.

---

## 6. Key Architectural Decisions

1. **Edition/Size/Position sub-group panels**: These dimensions exist only at insertion level. Sub-group functions receive the original `print_spend` DataFrame, filter to the dimension value, re-aggregate to daily × region, and re-join with sales — then apply adstock at best_decay. The full panel (daily × region) is only used for Region breakdown.

2. **CI bounds**: Extracted from OLS standard errors via statsmodels `OLS.fit()` result `.conf_int()` at 95%, not bootstrap (too slow for UI). Same model as DoWhy's LinearRegression estimator.

3. **`region_id` encoding**: Label-encoded integer (not one-hot) passed as confounder to DoWhy. One-hot encoding is handled internally by DoWhy's LinearRegression estimator.

4. **`lagged_sales_1d`**: Computed during `build_analytical_panel` as `sales_units.shift(1)` per region. Rows with NaN lag (first day per region) are dropped and logged.

5. **`refutation_passed` logic**:
   - Placebo: passes if placebo ATE is not significant (p > 0.05 on placebo estimate)
   - Random common cause: passes if |new_ATE - original_ATE| / |original_ATE| < 0.20 and no sign flip
   - Data subset: same threshold as random common cause

---

## 7. Ambiguities Noted in Spec

1. `n_insertions` appears in Step 2 aggregation snippet but not in the Step 4 panel columns list — included in `daily_spend` but not passed to panel (minor inconsistency; handled by keeping it in `daily_spend` for reference).

2. The spec's `PrintCausalResult` in Section 7 lists fields directly, while CLAUDE.md shows it inheriting from `CausalResult`. CLAUDE.md takes precedence — use inheritance.

3. `PLAN_print_causal.md` per CLAUDE.md should also be created in the project root during implementation.

---

## 8. Implementation Order

1. `requirements.txt`
2. `src/utils/logger.py`
3. `src/analysis/causal/models.py` (base class)
4. `src/analysis/causal/print_sales/models.py`
5. `src/analysis/causal/print_sales/adstock.py`
6. `src/analysis/causal/print_sales/data_processor.py`
7. `src/analysis/causal/print_sales/causal_model.py`
8. `src/analysis/causal/print_sales/subgroup_analysis.py`
9. `src/analysis/causal/print_sales/__init__.py`
10. `src/components/causal_charts.py`
11. `src/pages/03_print_causal_analysis.py`
12. `tests/test_print_causal.py`
13. `PLAN_print_causal.md` (project root copy)

---

## 9. Verification

1. `pytest tests/test_print_causal.py -v` — all 8 tests pass
2. `pytest --cov=src/analysis tests/` — coverage ≥ 80%
3. `streamlit run src/pages/03_print_causal_analysis.py` — UI renders, Run Analysis completes with synthetic data, all charts appear, Excel download works
4. Manually verify: metric cards, decay sweep chart, sub-group tabs, assumptions section, PASS/FAIL badge
