"""Streamlit page: Print Media Optimization Suite."""
from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Print Media Optimization Suite | Havas Martech",
    page_icon="📊",
    layout="wide",
)

_OUTCOME_OPTIONS: dict[str, str] = {
    "Enquiries (Leads)": "enquiries",
    "Dealer Visits": "dealer_visits",
    "Sales": "sales",
}
_OUTCOME_UNIT_LABEL: dict[str, str] = {
    "enquiries": "enquiries",
    "dealer_visits": "dealer visits",
    "sales": "sales units",
}

# ── Havas theme ───────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
        [data-testid="stSidebar"] { background-color: #1A1A1A; }
        [data-testid="stSidebar"] * { color: #FFFFFF !important; }
        [data-testid="stSidebar"] .stSelectbox label,
        [data-testid="stSidebar"] .stMultiSelect label,
        [data-testid="stSidebar"] .stSlider label,
        [data-testid="stSidebar"] .stDateInput label { color: #FFFFFF !important; }
        .metric-card {
            background: #FFFFFF;
            border-left: 4px solid #CC0000;
            padding: 1rem 1.25rem;
            border-radius: 4px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }
        .metric-label { font-size: 0.8rem; color: #555555; font-weight: 500; }
        .metric-value { font-size: 1.6rem; font-weight: 700; color: #CC0000; }
        .metric-value-neutral { font-size: 1.6rem; font-weight: 700; color: #1A1A1A; }
        .badge-pass {
            background: #2E7D32; color: #FFFFFF;
            padding: 0.3rem 0.8rem; border-radius: 4px;
            font-size: 0.85rem; font-weight: 600;
        }
        .badge-fail {
            background: #CC0000; color: #FFFFFF;
            padding: 0.3rem 0.8rem; border-radius: 4px;
            font-size: 0.85rem; font-weight: 600;
        }
        .badge-sig { background: #2E7D32; color: #FFFFFF; padding: 0.2rem 0.6rem; border-radius: 4px; }
        .badge-marginal { background: #F57C00; color: #FFFFFF; padding: 0.2rem 0.6rem; border-radius: 4px; }
        .badge-insig { background: #CC0000; color: #FFFFFF; padding: 0.2rem 0.6rem; border-radius: 4px; }
        .stTabs [data-baseweb="tab-highlight"] { background-color: #CC0000; }
        .stTabs [aria-selected="true"] { color: #CC0000 !important; font-weight: 500; }
        .page-header {
            background: #1A1A1A;
            border-left: 6px solid #CC0000;
            padding: 1rem 1.5rem;
            border-radius: 2px;
        }
        .page-header h1 { color: #FFFFFF; margin: 0; font-size: 1.5rem; }
        .page-header p { color: #AAAAAA; margin: 0.25rem 0 0 0; font-size: 0.85rem; }
        table { width: 100%; }
        tr:nth-child(even) { background-color: #F5F5F5; }
    </style>
    """,
    unsafe_allow_html=True,
)

_LOGO_PATH = Path(__file__).parent.parent.parent / "assets" / "CSA.jpeg"
_ATE_SCALE = 1_000_000


def _fmt_inr(value: float) -> str:
    """Format a number in Indian numbering (crores/lakhs)."""
    abs_val = abs(value)
    if abs_val >= 1e7:
        return f"₹{value / 1e7:,.2f} Cr"
    if abs_val >= 1e5:
        return f"₹{value / 1e5:,.2f} L"
    return f"₹{value:,.0f}"


def _p_badge(p_value: float) -> str:
    if p_value < 0.05:
        return f'<span class="badge-sig">p={p_value:.3f} ✓</span>'
    if p_value < 0.10:
        return f'<span class="badge-marginal">p={p_value:.3f} ~</span>'
    return f'<span class="badge-insig">p={p_value:.3f} ✗</span>'


def _refutation_badge(passed: bool) -> str:
    return '<span class="badge-pass">PASS</span>' if passed else '<span class="badge-fail">FAIL</span>'


def _prepare_breakdown_display(df: pd.DataFrame) -> pd.DataFrame:
    """Scale ATE columns to per ₹10,00,000 and round p_value to 3 dp."""
    out = df.copy()
    for col in ["ate", "ate_lower", "ate_upper"]:
        if col in out.columns:
            out[col] = (out[col] * _ATE_SCALE).round(3)
    if "p_value" in out.columns:
        out["p_value"] = out["p_value"].round(3)
    rename = {
        "group": "Group",
        "ate": "ATE (units / ₹10L)",
        "ate_lower": "CI Lower",
        "ate_upper": "CI Upper",
        "p_value": "p-value",
        "n_obs": "Observations",
    }
    out = out.rename(columns={k: v for k, v in rename.items() if k in out.columns})
    return out


def _display_breakdown_tab(df: pd.DataFrame, dimension_label: str) -> None:
    from src.components.causal_charts import plot_subgroup_breakdown

    if df.empty:
        st.info(f"No {dimension_label} breakdown available.")
        return

    st.dataframe(_prepare_breakdown_display(df), use_container_width=True, hide_index=True)
    st.plotly_chart(
        plot_subgroup_breakdown(df, dimension_label),
        use_container_width=True,
        key=f"chart_{dimension_label.lower().replace(' ', '_')}",
    )


def _to_excel(result) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        summary = pd.DataFrame([{
            "ate_per_10L_spend": round(result.ate * _ATE_SCALE, 3),
            "ate_lower": round(result.ate_lower * _ATE_SCALE, 3),
            "ate_upper": round(result.ate_upper * _ATE_SCALE, 3),
            "p_value": round(result.p_value, 3),
            "best_decay_theta": result.best_decay_theta,
            "ate_pct_impact": result.ate_pct_impact,
            "total_spend_inr": result.total_spend_inr,
            "total_incremental_sales_units": round(result.ate * result.total_spend_inr, 0),
            "refutation_passed": result.refutation_passed,
            "method": result.method,
            "date_range_start": result.date_range[0],
            "date_range_end": result.date_range[1],
            "n_observations": result.n_observations,
        }])
        summary.to_excel(writer, sheet_name="Summary", index=False)
        result.decay_sweep.to_excel(writer, sheet_name="Decay Sweep", index=False)
        result.region_breakdown.to_excel(writer, sheet_name="Region", index=False)
        result.edition_breakdown.to_excel(writer, sheet_name="Edition", index=False)
        result.size_breakdown.to_excel(writer, sheet_name="Size", index=False)
        result.position_breakdown.to_excel(writer, sheet_name="Position", index=False)
        result.publication_breakdown.to_excel(writer, sheet_name="Publication", index=False)
        result.product_breakdown.to_excel(writer, sheet_name="Product", index=False)

        recs = getattr(result, "recommendations", {}) or {}
        scale_up = recs.get("combined_scale_up")
        if scale_up is not None and not scale_up.empty:
            scale_up.to_excel(writer, sheet_name="Recs - Scale Up", index=False)
        avoid = recs.get("combined_avoid")
        if avoid is not None and not avoid.empty:
            avoid.to_excel(writer, sheet_name="Recs - Avoid", index=False)

        assumptions_df = pd.DataFrame({"assumption": result.assumptions})
        assumptions_df.to_excel(writer, sheet_name="Assumptions", index=False)
    return buf.getvalue()


# ── Top bar: logo + page header ───────────────────────────────────────────────
logo_col, header_col = st.columns([1, 8])

with logo_col:
    if _LOGO_PATH.exists():
        st.image(str(_LOGO_PATH), width=110)

with header_col:
    st.markdown(
        """
        <div class="page-header">
            <h1>Print Media Optimization Suite</h1>
            <p>Estimate the incremental impact of Print spend on enquiries, dealer visits, or sales — and surface the publications, sizes, and positions to scale up next.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Data Upload")
    spend_file = st.file_uploader("Print Spend CSV", type=["csv"])
    sales_file = st.file_uploader("Sales CSV", type=["csv"])

    st.markdown("---")
    st.markdown("### Analysis Config")
    outcome_label = st.selectbox(
        "Outcome metric",
        options=list(_OUTCOME_OPTIONS.keys()),
        index=0,
        help="Funnel metric to model as the dependent variable.",
    )
    outcome_col = _OUTCOME_OPTIONS[outcome_label]
    max_lag = st.slider("Max Adstock Lag (days)", min_value=1, max_value=7, value=7)

    region_filter: list[str] = []
    date_start = date_end = None

    if spend_file and sales_file:
        try:
            _spend_preview = pd.read_csv(spend_file)
            spend_file.seek(0)
            _sales_preview = pd.read_csv(sales_file)
            sales_file.seek(0)

            all_regions = sorted(
                set(_spend_preview["region"].dropna().unique())
                & set(_sales_preview["region"].dropna().unique())
            )
            region_filter = st.multiselect("Regions", options=all_regions, default=all_regions)

            all_dates = pd.to_datetime(_sales_preview["date"].dropna(), format="mixed", dayfirst=True)
            date_start = st.date_input("From", value=all_dates.min().date())
            date_end = st.date_input("To", value=all_dates.max().date())
        except Exception:
            st.warning("Upload both files to configure filters.")

# ── Main area ─────────────────────────────────────────────────────────────────
if not spend_file or not sales_file:
    st.info(
        "Upload **Print Spend CSV** and **Sales Funnel CSV** in the sidebar to begin.\n\n"
        "Required columns:\n"
        "- **Print Spend**: date, region, product, edition, publication, size, position, spend_in_inr\n"
        "- **Sales Funnel**: date, region, product, and at least one of "
        "`enquiries` / `dealer_visits` / `sales` (selectable as the outcome metric)"
    )
    st.stop()

if st.button("▶  Run Optimization Analysis", type="primary"):
    from src.analysis.causal.print_sales.causal_model import run_print_causal_analysis
    from src.analysis.causal.print_sales.data_processor import DataValidationError

    try:
        print_spend_df = pd.read_csv(spend_file)
        sales_df = pd.read_csv(sales_file)

        date_range_arg = (str(date_start), str(date_end)) if date_start and date_end else None
        regions_arg = region_filter if region_filter else None

        with st.spinner(f"Running model on {outcome_label}… this may take a minute."):
            result = run_print_causal_analysis(
                print_spend=print_spend_df,
                sales=sales_df,
                max_lag=max_lag,
                regions=regions_arg,
                date_range=date_range_arg,
                outcome_col=outcome_col,
            )
        st.session_state["print_causal_result"] = result
        st.success(f"Analysis complete on {outcome_label}.")

    except DataValidationError as exc:
        st.error(f"Data validation failed: {exc}")
        st.stop()
    except Exception as exc:
        st.error(f"Unexpected error during analysis: {exc}")
        raise

# ── Results ───────────────────────────────────────────────────────────────────
result = st.session_state.get("print_causal_result")
if result is None:
    st.info("Configure your inputs and click **Run Optimization Analysis** to see results.")
    st.stop()

_outcome_unit = _OUTCOME_UNIT_LABEL.get(result.outcome, result.outcome)

from src.components.causal_charts import plot_ate_by_region, plot_decay_sweep

st.markdown("<br>", unsafe_allow_html=True)

# Row 1 — Primary metric cards
col1, col2, col3, col4 = st.columns(4)

with col1:
    ate_per_m = result.ate * _ATE_SCALE
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">ATE ({_outcome_unit} per ₹10,00,000 spend)</div>
            <div class="metric-value">{ate_per_m:+.3f}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col2:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Best Adstock Decay (θ)</div>
            <div class="metric-value-neutral">{result.best_decay_theta:.1f}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col3:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Statistical Significance</div>
            <div style="margin-top:0.5rem">{_p_badge(result.p_value)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col4:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Refutation Status</div>
            <div style="margin-top:0.5rem">{_refutation_badge(result.refutation_passed)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)

# Row 2 — Total spend & total impact summary cards
total_impact_units = result.ate * result.total_spend_inr
scol1, scol2 = st.columns(2)

with scol1:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Total Print Spend (Analysis Period)</div>
            <div class="metric-value-neutral">{_fmt_inr(result.total_spend_inr)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with scol2:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Total Incremental {_outcome_unit.title()} (ATE × Total Spend)</div>
            <div class="metric-value">{total_impact_units:+,.0f}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)

# Row 3 — Decay sweep | ATE by region
col_left, col_right = st.columns(2)

with col_left:
    st.plotly_chart(plot_decay_sweep(result.decay_sweep), use_container_width=True, key="chart_decay_sweep")

with col_right:
    if not result.region_breakdown.empty:
        st.plotly_chart(plot_ate_by_region(result.region_breakdown), use_container_width=True, key="chart_ate_region")
    else:
        st.info("Region breakdown not available.")

# Row 4 — Sub-group tabs
st.markdown("### Sub-Group Breakdowns")
tab_region, tab_product, tab_edition, tab_publication, tab_size, tab_position = st.tabs(
    ["Region", "Product", "Edition", "Publication", "Size", "Position"]
)

with tab_region:
    _display_breakdown_tab(result.region_breakdown, "Region")
with tab_product:
    _display_breakdown_tab(result.product_breakdown, "Product")
with tab_edition:
    _display_breakdown_tab(result.edition_breakdown, "Edition")
with tab_publication:
    _display_breakdown_tab(result.publication_breakdown, "Publication")
with tab_size:
    _display_breakdown_tab(result.size_breakdown, "Ad Size")
with tab_position:
    _display_breakdown_tab(result.position_breakdown, "Position")

# Row 4.5 — Recommendations for next print buys
st.markdown("### Next-Buy Recommendations")
recs = getattr(result, "recommendations", {}) or {}
if not recs:
    st.info("No recommendations available — run the analysis to generate.")
else:
    p_thr = recs.get("p_threshold", 0.10)
    st.caption(
        f"Cuts where the estimated incremental {_outcome_unit} per ₹ of Print spend is "
        f"credibly positive (p < {p_thr:.2f}). Use this to prioritise the next "
        f"buying cycle. ATE values shown per ₹10,00,000 of spend."
    )

    rec_cols = ["dimension", "group", "ate_per_10L", "p_value", "n_obs", "total_spend_inr"]
    rec_rename = {
        "dimension": "Dimension",
        "group": "Group",
        "ate_per_10L": f"ATE ({_outcome_unit} / ₹10L)",
        "p_value": "p-value",
        "n_obs": "Observations",
        "total_spend_inr": "Spend in window (₹)",
    }

    scale_up = recs.get("combined_scale_up", pd.DataFrame())
    avoid = recs.get("combined_avoid", pd.DataFrame())

    rec_tab_scale, rec_tab_avoid, rec_tab_per_dim = st.tabs(
        ["⬆ Scale Up", "⬇ Deprioritise", "Per Dimension"]
    )

    with rec_tab_scale:
        if scale_up.empty:
            st.info("No statistically credible positive cuts at the current p-threshold.")
        else:
            display = scale_up[[c for c in rec_cols if c in scale_up.columns]].rename(
                columns=rec_rename
            )
            st.dataframe(display, use_container_width=True, hide_index=True)

    with rec_tab_avoid:
        if avoid.empty:
            st.info("No statistically credible negative cuts — nothing to deprioritise.")
        else:
            display = avoid[[c for c in rec_cols if c in avoid.columns]].rename(
                columns=rec_rename
            )
            st.dataframe(display, use_container_width=True, hide_index=True)

    with rec_tab_per_dim:
        from src.components.causal_charts import plot_recommendation_column_chart

        per_dim = recs.get("per_dimension", {})
        any_shown = False
        for dim_label, df in per_dim.items():
            if df is None or df.empty:
                continue
            any_shown = True
            st.markdown(f"**{dim_label}**")
            st.plotly_chart(
                plot_recommendation_column_chart(df, dim_label),
                use_container_width=True,
                key=f"rec_chart_{dim_label.lower().replace(' ', '_')}",
            )
            display = df[[c for c in rec_cols if c in df.columns]].rename(columns=rec_rename)
            st.dataframe(display, use_container_width=True, hide_index=True)
            st.markdown("---")
        if not any_shown:
            st.info("No dimension-level recommendations available.")

# Row 5 — Assumptions & warnings
with st.expander("Modelling Assumptions & Warnings", expanded=False):
    st.markdown("**Stated Assumptions**")
    for i, assumption in enumerate(result.assumptions, 1):
        st.markdown(f"{i}. {assumption}")

    if result.warnings:
        st.markdown("---")
        st.markdown("**Warnings**")
        for warning in result.warnings:
            st.warning(warning)

    st.markdown("---")
    st.markdown("**Interpretation**")
    st.info(result.interpretation)

    if result.refutation_details:
        st.markdown("**Refutation Test Details**")
        for test_name, details in result.refutation_details.items():
            status = "✅ Passed" if details.get("passed") else "❌ Failed"
            st.markdown(
                f"**{test_name.replace('_', ' ').title()}** — {status}  \n"
                f"{details.get('description', '')}  \n"
                f"Original ATE: `{details.get('original_effect', 0):.4f}` → "
                f"New ATE: `{details.get('new_effect', 0):.4f}`"
            )

# Row 6 — Download
st.markdown("### Download Results")
st.download_button(
    label="⬇  Download Results (Excel)",
    data=_to_excel(result),
    file_name="print_optimization_results.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)
