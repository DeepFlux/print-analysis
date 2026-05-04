# CLAUDE.md — Havas Martech Analytics Platform

## Project Overview

This project is a **Martech Analytics Platform** built for Havas Group. The primary use case is **Causal Analysis** between marketing input variables (spend, impressions, GRPs, reach) and output/business variables (sales, revenue, brand metrics, conversions). Future modules will include Market Mix Modelling (MMM) and Campaign Attribution.

**Current active module**: Print Spend → Sales Causal Analysis
**Spec file**: `SPEC_print_causal_analysis.md`

---

## Brand & UI Guidelines

### Colour Theme: Havas (Black & Red)
- **Primary**: `#CC0000` (Havas Red)
- **Secondary**: `#1A1A1A` (Near Black)
- **Background**: `#FFFFFF` (White) / `#F5F5F5` (Light Grey surface)
- **Accent**: `#E8E8E8` (Borders, dividers)
- **Text Primary**: `#1A1A1A`
- **Text Secondary**: `#555555`
- **Success**: `#2E7D32`
- **Warning**: `#F57C00`
- **Error**: `#CC0000`

### UI Style Rules
- Clean, corporate, data-dense layouts — no decorative elements
- Use Havas Red ONLY for primary CTAs, active states, key metric highlights, and brand headers
- Tables must have alternating row shading using `#F5F5F5`
- Charts: use Red (`#CC0000`) as primary series, Black/Dark Grey for secondary series
- All charts must have axis labels, legends, and titles
- Sidebar navigation: dark background (`#1A1A1A`), white text
- Page headers: dark bar (`#1A1A1A`) with white text and a Havas Red left border accent
- Status badges: green background for PASS, `#CC0000` background for FAIL/ERROR

### Streamlit Theme CSS (apply on every page)
```python
st.markdown("""
<style>
    [data-testid="stSidebar"] { background-color: #1A1A1A; }
    [data-testid="stSidebar"] * { color: #FFFFFF; }
    .metric-card { background: #FFFFFF; border-left: 4px solid #CC0000; padding: 1rem; }
    .stTabs [data-baseweb="tab-highlight"] { background-color: #CC0000; }
    .stTabs [aria-selected="true"] { color: #CC0000; font-weight: 500; }
</style>
""", unsafe_allow_html=True)
```

---

## Tech Stack

### Frontend
- **Primary**: Streamlit
- All pages go in `src/pages/`
- Shared components go in `src/components/`
- Use `st.set_page_config(layout="wide")` on every page — no exceptions
- Keep `src/styles/havas_theme.css` for all reusable custom CSS

### Backend
- **Language**: Python 3.11+
- **Data**: `pandas`, `numpy`
- **Causal Analysis**: `dowhy>=0.11`, `econml>=0.15`, `causalml>=0.15`, `statsmodels>=0.14`
- **Statistical Tests**: `scipy`, `pingouin`
- **Visualisation**: `plotly>=5.0` (preferred), `matplotlib` as fallback only
- **Data Models**: `pydantic>=2.0`
- **Config / Secrets**: `python-dotenv`
- **Export**: `openpyxl` (Excel downloads)
- **Testing**: `pytest`, `pytest-cov`

### No JavaScript frameworks unless absolutely required for a Streamlit component

---

## Project Structure

```
project-root/
├── CLAUDE.md                            ← You are here
├── SPEC_print_causal_analysis.md        ← Active feature spec
├── PLAN_print_causal.md                 ← Claude Code plan (auto-generated, do not edit)
├── README.md
├── requirements.txt
├── .env.example
├── src/
│   ├── app.py                           ← Streamlit entry point
│   ├── pages/
│   │   ├── 01_data_upload.py
│   │   ├── 02_eda.py
│   │   ├── 03_print_causal_analysis.py  ← ACTIVE: Print → Sales
│   │   └── 04_results.py
│   ├── components/
│   │   ├── sidebar.py
│   │   ├── charts.py
│   │   ├── causal_charts.py             ← Reusable chart functions for causal pages
│   │   └── data_table.py
│   ├── styles/
│   │   └── havas_theme.css
│   ├── analysis/
│   │   ├── causal/
│   │   │   └── print_sales/             ← ACTIVE MODULE
│   │   │       ├── __init__.py
│   │   │       ├── models.py            ← PrintCausalResult dataclass
│   │   │       ├── adstock.py           ← Adstock transforms + decay sweep
│   │   │       ├── data_processor.py    ← Load, validate, aggregate, build panel
│   │   │       ├── causal_model.py      ← DoWhy pipeline + refutation tests
│   │   │       └── subgroup_analysis.py ← Region/Edition/Size/Position breakdowns
│   │   ├── mmm/                         ← Future: Market Mix Modelling
│   │   └── attribution/                 ← Future: Campaign Attribution
│   ├── data/
│   │   ├── loaders.py
│   │   └── validators.py
│   └── utils/
│       ├── logger.py
│       └── helpers.py
├── tests/
│   ├── test_print_causal.py             ← Tests for print_sales module
│   ├── test_causal.py
│   └── test_data.py
└── notebooks/                           ← Exploratory only, never production code
```

---

## Domain Knowledge — Martech Context

### Input Variables (Marketing)
- **Paid Media Spend**: TV, Digital, OOH, Radio, Print — always in GBP
- **Print-specific dimensions**: Region, Edition, Size, Position (see below)
- **Impressions / GRPs**: Gross Rating Points, reach, frequency
- **Campaign Variables**: flight dates, creative type, channel mix
- **Pricing**: product price, promotions, discounts

### Output Variables (Business)
- **Sales / Revenue**: units sold, revenue (£) — primary outcomes
- **Brand Metrics**: awareness, consideration, preference (%)
- **Conversions**: leads, sign-ups, purchases
- **Engagement**: website traffic, CTR, engagement rate

### Print Media Dimensions (critical for current module)
```
Region    → geographic area where the edition is distributed
Edition   → specific newspaper or publication title
Size      → full_page | half_page | quarter_page | strip
Position  → front_page | back_page | inside_rhs | inside_lhs | supplement
```
All four dimensions affect the causal effect of spend — always include in sub-group analysis.

### Key Causal Relationships
- Print spend → Sales with geometric adstock decay (carry-over up to 7 days) — **current focus**
- Spend → Sales direct path (future: all channels)
- Impressions → Brand Awareness → Sales (mediated — future)
- Price × Promotion → Conversion Rate (interaction — future)

---

## Causal Analysis Module — Coding Standards

### Methodology Priority
1. **DoWhy** — causal graph definition and identification (primary)
2. **EconML** — heterogeneous treatment effects (HTE)
3. **CausalML** — uplift modelling
4. **Statsmodels OLS/IV** — simpler regression-based inference

### DAG — Mandatory Before Every Model
Always define and document the DAG. For Print → Sales:
```
day_of_week ──────────────────┐
week_of_year ─────────────────┤
region_fixed_effect ──────────┼──► sales_units
lagged_sales_1d ──────────────┤
                              │
print_adstock_spend ──────────┘
```
No unobserved confounders assumed — **this assumption must be displayed in the UI**.

### Adstock Rules
- Always use geometric decay: `adstock_t = spend_t + θ × adstock_(t-1)`
- Always sweep θ ∈ {0.0, 0.1, 0.2 … 0.9} — select best by lowest p-value
- Always cap carry-over at `max_lag` days (default 7, user-configurable)
- Never pass raw spend as treatment without also testing θ=0.0 as a baseline

### Refutation Tests — Mandatory, Run All Three
| Test | Pass Condition |
|---|---|
| Placebo treatment | ATE collapses to ~0 |
| Random common cause | ATE changes <20% and does not flip sign |
| Data subset refuter (80%) | ATE changes <20% and does not flip sign |

Set `refutation_passed = False` if any test fails. Show this prominently in the UI.

### Required Output Dataclasses
Never return raw dicts or loose variables from analysis functions — always use typed dataclasses:

```python
# Base class — src/analysis/causal/models.py
@dataclass
class CausalResult:
    treatment: str
    outcome: str
    ate: float                   # Average Treatment Effect
    ate_lower: float             # 95% CI lower bound
    ate_upper: float             # 95% CI upper bound
    p_value: float
    method: str                  # e.g. "DoWhy-LinearRegression"
    refutation_passed: bool
    interpretation: str          # Plain English summary — always required
    warnings: list[str]

# Print module extension — src/analysis/causal/print_sales/models.py
@dataclass
class PrintCausalResult(CausalResult):
    ate_pct_impact: float            # ATE as % of mean baseline sales
    best_decay_theta: float          # Winning adstock decay rate (θ)
    refutation_details: dict         # Per-test breakdown
    decay_sweep: pd.DataFrame        # theta, ATE, p_value, r_squared per decay
    region_breakdown: pd.DataFrame   # region, ATE, CI, p_value, n_days
    edition_breakdown: pd.DataFrame
    size_breakdown: pd.DataFrame
    position_breakdown: pd.DataFrame
    date_range: tuple[str, str]
    n_observations: int
    regions_analysed: list[str]
    assumptions: list[str]           # Stated assumptions — always surface to user
```

### Variable Naming Convention
```python
treatment_var: str        # e.g. "print_adstock_spend_gbp"
outcome_var: str          # e.g. "daily_sales_units"
confounder_vars: list     # e.g. ["day_of_week", "week_of_year", "region_id", "lagged_sales_1d"]
instrument_vars: list     # only if Instrumental Variable method is used
```

---

## General Coding Standards

### Python Style
- Follow PEP 8 strictly
- Type hints on ALL functions — no exceptions
- Google-style docstrings on all public functions
- Max line length: 100 characters
- No hardcoded values — use constants or config files

### Error Handling
- Never use bare `except:` — catch specific exceptions only
- Log errors with context via `src/utils/logger.py`
- Surface user-friendly messages in Streamlit via `st.error()` / `st.warning()`
- Raise `DataValidationError` (custom) for all data input failures

### Data Handling
- Never mutate input DataFrames — always `.copy()` before any transformation
- Validate column names, dtypes, and value ranges in `validators.py` before analysis
- Never silently drop rows — handle missing values explicitly and log what was removed
- Log row counts before and after every filter, join, or aggregation step

### Streamlit Patterns
```python
# Cache data loading and heavy computation
@st.cache_data
def load_data(file_path: str) -> pd.DataFrame: ...

@st.cache_resource
def build_causal_model(config: dict): ...

# Cross-page state via session_state
if "print_causal_result" not in st.session_state:
    st.session_state.print_causal_result = None

# Gate computation behind a button — never rerun on every render
if st.button("Run Analysis"):
    with st.spinner("Running causal model..."):
        result = run_print_causal_analysis(config)
        st.session_state.print_causal_result = result
```

---

## Testing Requirements

- Every function in `src/analysis/` must have a corresponding test
- Cover: happy path, missing data, wrong dtypes, edge cases, region mismatches
- Use synthetic Martech fixtures — never use real client data in tests
- Minimum coverage target: **80%** for `src/analysis/`
- Run `pytest tests/` before every commit

### Key Test Cases for Print Module
| Test | Expected Outcome |
|---|---|
| `test_adstock_zero_decay` | adstock equals raw spend |
| `test_adstock_full_decay` | cumulative sum of spend |
| `test_adstock_max_lag` | no carry-over beyond day 7 |
| `test_ate_zero_spend` | ATE = 0, warning raised |
| `test_region_mismatch` | `DataValidationError` raised |
| `test_decay_sweep_all_thetas` | 10 rows in decay_sweep DataFrame |
| `test_refutation_placebo` | `refutation_passed = True` on synthetic zero-effect data |
| `test_subgroup_region_coverage` | output rows = number of input regions |

---

## What NOT to Do

- Do NOT use `scikit-learn` for causal inference — it is for prediction, not causation
- Do NOT interpret a high R² as evidence of causation
- Do NOT drop confounders to simplify the model
- Do NOT use raw spend as treatment without testing adstock transformations
- Do NOT skip the Region/Edition/Size/Position sub-group breakdowns
- Do NOT hardcode client names, data paths, or file locations
- Do NOT commit `.env` files or credentials of any kind

---

## Starter Prompt Template (for new features)

```
Read CLAUDE.md and [SPEC_filename.md] carefully.

Enter Plan Mode. Before writing any code:
1. Confirm the full file and folder structure to be created
2. List every pip dependency needed with version constraints
3. Write every function signature with type hints for every file
4. List ambiguities or assumptions about the data
5. Save the complete plan to PLAN_[feature].md

Do not write any implementation code until I approve the plan.
```

---

*Last updated: May 2026 | Project: Havas Martech Analytics Platform*
*Active spec: SPEC_print_causal_analysis.md*
