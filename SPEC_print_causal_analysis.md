# SPEC: Print Spend → Sales Causal Analysis

**Module**: `src/analysis/causal/print_sales/`
**Status**: Ready for implementation
**Last updated**: May 2026

---

## 1. Problem Statement

We want to estimate the **Average Treatment Effect (Incremental Sales)** of Print media spend on daily regional sales.

Print spend is described by four dimensions:
- **Region** — geographic area of the edition
- **Edition** — specific newspaper/publication
- **Size** — ad size (e.g. full page, half page, quarter page)
- **Position** — placement in publication (e.g. front page, back page, inside)

Print advertising can drive sales **same-day OR with a carry-over lag of up to 5–7 days** (adstock). We need to model both and find the best-fit decay.

There are **no other media channels** in this dataset — Print is the sole treatment variable, which simplifies confounder identification.

---

## 2. Causal Question

> *"What is the average causal effect of a unit increase in Print spend (by region, edition, size, and position) on daily regional sales, accounting for adstock decay of up to 7 days?"*

---

## 3. Data Inputs

### 3a. Print Spend Data (`print_spend`)

| Column | Type | Description |
|---|---|---|
| `date` | `datetime` | Date of publication |
| `region` | `str` | Geographic region (e.g. "North", "South", "Midlands") |
| `edition` | `str` | Publication name |
| `size` | `str` | Ad size — `full_page`, `half_page`, `quarter_page`, `strip` |
| `position` | `str` | Placement — `front_page`, `back_page`, `inside_rhs`, `inside_lhs`, `supplement` |
| `spend_gbp` | `float` | Spend in GBP for this insertion |

### 3b. Sales Data (`sales`)

| Column | Type | Description |
|---|---|---|
| `date` | `datetime` | Sales date |
| `region` | `str` | Geographic region (must match `print_spend.region`) |
| `sales_units` | `float` | Units sold |
| `revenue_gbp` | `float` | Revenue in GBP (optional, secondary outcome) |

### 3c. Join Key
- `date` + `region` — sales and spend are matched at daily-regional granularity

---

## 4. Data Processing Pipeline

### Step 1 — Load & Validate
- Validate column names, dtypes, and date ranges on both datasets
- Check that all regions in `print_spend` exist in `sales`
- Log row counts at every step
- Raise `DataValidationError` (custom exception) with clear message if anything fails

### Step 2 — Aggregate Spend to Daily × Region
Print spend input is at insertion level (one row per ad placement). Aggregate:

```python
daily_spend = (
    print_spend
    .groupby(["date", "region"])
    .agg(
        total_spend_gbp=("spend_gbp", "sum"),
        n_insertions=("spend_gbp", "count"),
        # Weighted presence flags per dimension
        has_full_page=("size", lambda x: (x == "full_page").any()),
        has_front_page=("position", lambda x: (x == "front_page").any()),
    )
    .reset_index()
)
```

Also keep dimension-level breakdowns available for sub-group analysis.

### Step 3 — Build Adstock Transformations
For each candidate decay rate `θ ∈ {0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9}`:

```python
def apply_adstock(spend_series: pd.Series, decay: float, max_lag: int = 7) -> pd.Series:
    """
    Geometric adstock transformation.
    adstock_t = spend_t + decay * adstock_(t-1)
    Capped at max_lag days of carry-over.
    """
```

This produces one adstock column per decay value per region.

### Step 4 — Build Analytical Panel
Create a balanced daily × region panel:

```python
panel = pd.DataFrame with columns:
    date, region,
    sales_units,            # outcome
    spend_gbp,              # raw spend (treatment — no adstock)
    adstock_θ0.0 ... adstock_θ0.9,   # adstock-transformed treatments
    is_zero_spend,          # flag: no print on this day in this region
    day_of_week,            # confounder: seasonality
    week_of_year,           # confounder: seasonality
    region_id,              # confounder: fixed effect
```

**Note**: With only Print in the dataset, the main confounders are:
- Day-of-week effects (weekend vs weekday sales patterns)
- Seasonal trends (week of year)
- Region-level baseline differences (regional fixed effects)
- Autocorrelation in sales (lagged sales as covariate)

---

## 5. Causal Model

### 5a. DAG Definition
```
day_of_week ──────────────────┐
week_of_year ─────────────────┤
region_fixed_effect ──────────┼──► sales_units
                              │
print_spend_adstock(θ) ───────┘
```

No unobserved confounders assumed (stated assumption — must be surfaced to user in output).

### 5b. Identification Strategy
- **Method**: DoWhy with Linear Regression estimator
- **Treatment**: `adstock_spend` (continuous — GBP)
- **Outcome**: `sales_units`
- **Confounders**: `day_of_week`, `week_of_year`, `region_id`, `lagged_sales_1d`
- **Assumption**: No other media channels active (verified by user input — log this)

### 5c. Adstock Decay Selection
Run the causal model for **each decay value θ**. Select best θ by:
1. Highest Incremental Sales statistical significance (lowest p-value)
2. As a tiebreaker: highest model R² on the outcome regression

Report the full decay sweep as a chart so the analyst can see the sensitivity.

### 5d. Refutation Tests (mandatory)
For the best-fit θ, run all three:

| Test | What it checks |
|---|---|
| Placebo treatment | Replace real spend with random noise → Incremental Sales should collapse to ~0 |
| Random common cause | Add a random confounder → Incremental Sales should be stable |
| Data subset refuter | Estimate on 80% random subset → Incremental Sales should be similar |

Flag `refutation_passed = False` if any test materially changes the Incremental Sales (>20% change or flips sign).

---

## 6. Sub-Group Analysis (Dimension Breakdown)

After the primary Incremental Sales, run secondary analyses:

| Dimension | Question |
|---|---|
| Region | Which regions show the highest sales lift per £ spent? |
| Edition | Which publications are most effective? |
| Size | Does full-page outperform half-page causally? |
| Position | Does front-page placement drive more sales? |

Each sub-group analysis uses the same DoWhy pipeline but filters the panel to that group.
Return a ranked table of ATEs with CIs.

---

## 7. Output Data Model

```python
@dataclass
class PrintCausalResult:
    # Primary result
    ate: float                        # Average Treatment Effect (sales units per £1 spend)
    ate_lower: float                  # 95% CI lower bound
    ate_upper: float                  # 95% CI upper bound
    ate_pct_impact: float             # Incremental Sales as % of mean baseline sales
    p_value: float
    best_decay_theta: float           # Winning adstock decay rate
    method: str                       # "DoWhy-LinearRegression"
    refutation_passed: bool
    refutation_details: dict          # Per-test results

    # Decay sweep
    decay_sweep: pd.DataFrame         # theta, Incremental Sales, p_value, r_squared per decay

    # Sub-group breakdowns
    region_breakdown: pd.DataFrame    # region, Incremental Sales, CI, p_value, n_days
    edition_breakdown: pd.DataFrame
    size_breakdown: pd.DataFrame
    position_breakdown: pd.DataFrame

    # Metadata
    date_range: tuple[str, str]
    n_observations: int
    regions_analysed: list[str]
    assumptions: list[str]            # Stated assumptions surfaced to user
    warnings: list[str]
    interpretation: str               # Plain English summary paragraph
```

---

## 8. Streamlit UI — Page: `03_print_causal_analysis.py`

### Layout
```
[Sidebar: region filter, date range, max adstock lag slider (1–7)]

[Main area]
  Row 1: 4 metric cards
    ├── Incremental Sales (sales units per £1000 spend)
    ├── Best Adstock Decay (θ)
    ├── Statistical Significance (p-value badge: green <0.05, amber <0.1, red ≥0.1)
    └── Refutation Status (PASS / FAIL badge)

  Row 2: [Adstock Decay Sweep Chart]  |  [Incremental Sales by Region Bar Chart]
    (line chart: x=θ, y=Incremental Sales with CI bands)    (horizontal bar, sorted by Incremental Sales)

  Row 3: [Sub-group Breakdown Tabs]
    Tabs: Region | Edition | Size | Position
    Each tab: sortable table (Incremental Sales, CI lower, CI upper, p-value, n_obs)
              + small bar chart of ATEs

  Row 4: [Causal Assumptions & Warnings]
    Expandable section listing all stated assumptions + any warnings
    Plain-English interpretation paragraph

  Row 5: [Download Results Button]
    Exports PrintCausalResult to Excel (one sheet per breakdown)
```

### Havas Theme Application
- Page header: dark bar (`#1A1A1A`) with white text and Havas Red left border
- Metric cards: white background, Havas Red value text for primary Incremental Sales card
- Charts: Havas Red (`#CC0000`) primary series, dark grey secondary
- PASS badge: green background; FAIL badge: red (`#CC0000`) background
- Tabs: active tab underline in Havas Red

---

## 9. File Structure for This Feature

```
src/
├── analysis/
│   └── causal/
│       └── print_sales/
│           ├── __init__.py
│           ├── data_processor.py     ← Steps 1–4 above
│           ├── adstock.py            ← apply_adstock(), decay sweep
│           ├── causal_model.py       ← DoWhy pipeline, refutations
│           ├── subgroup_analysis.py  ← dimension-level breakdowns
│           └── models.py             ← PrintCausalResult dataclass
├── pages/
│   └── 03_print_causal_analysis.py  ← Streamlit UI
└── components/
    └── causal_charts.py              ← Reusable chart functions for this page
tests/
└── test_print_causal.py              ← Tests for all analysis functions
```

---

## 10. Test Cases Required

| Test | Input | Expected |
|---|---|---|
| `test_adstock_zero_decay` | decay=0.0 | adstock equals raw spend |
| `test_adstock_full_decay` | decay=1.0 | cumulative sum of spend |
| `test_adstock_max_lag` | 10-day series, max_lag=7 | No carry-over beyond day 7 |
| `test_ate_zero_spend` | All spend = 0 | Incremental Sales = 0, warning raised |
| `test_panel_build_region_mismatch` | Regions in spend ≠ regions in sales | `DataValidationError` raised |
| `test_decay_sweep_returns_all_thetas` | Normal input | 10 rows in decay_sweep df |
| `test_refutation_placebo` | Synthetic data with known Incremental Sales=0 | refutation_passed = True |
| `test_subgroup_region_coverage` | 3-region input | 3 rows in region_breakdown |

---

## 11. Assumptions to Surface to User (in UI)

1. No other media channels active during the analysis period
2. Regional sales differences are fully captured by the region fixed effect
3. Adstock decay is geometric (exponential) — not S-curve or diminishing returns
4. No lagged effects beyond 7 days
5. Spend → Sales relationship is linear (Linear Regression estimator)
6. No reverse causality (sales performance does not cause Print budget allocation in same period)

---

## 12. Implementation Prompt for Claude Code

Use this prompt to kick off implementation in Claude Code:

```
Read SPEC_print_causal_analysis.md and CLAUDE.md carefully.

Enter Plan Mode. Do the following before writing any code:
1. Confirm the full file structure from Section 9
2. List all pip dependencies needed (dowhy, econml, causalml, statsmodels, plotly, openpyxl)
3. Write function signatures with type hints for every function in every file
4. Identify any ambiguities in the spec and list them
5. Save the plan to PLAN_print_causal.md

Do not write implementation code until I approve the plan.
```

---

*Spec status: COMPLETE — ready to hand to Claude Code*
