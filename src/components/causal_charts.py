from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

HAVAS_RED = "#CC0000"
HAVAS_DARK = "#1A1A1A"
HAVAS_GREY = "#555555"
HAVAS_LIGHT = "#F5F5F5"

_ATE_SCALE = 1_000_000
_ATE_AXIS_LABEL = "ATE (sales units per ₹10,00,000 spend)"


def _scale_ate(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of df with ate, ate_lower, ate_upper scaled to per ₹10,00,000."""
    out = df.copy()
    for col in ["ate", "ate_lower", "ate_upper"]:
        if col in out.columns:
            out[col] = out[col] * _ATE_SCALE
    return out


def plot_decay_sweep(decay_sweep: pd.DataFrame) -> go.Figure:
    """Line chart of ATE vs adstock decay θ with 95% CI bands.

    Args:
        decay_sweep: DataFrame with columns: theta, ate, ate_lower, ate_upper.

    Returns:
        Plotly Figure with ATE scaled to per ₹10,00,000 spend.
    """
    df = _scale_ate(decay_sweep)
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df["theta"],
            y=df["ate_upper"],
            mode="lines",
            line={"width": 0},
            showlegend=False,
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df["theta"],
            y=df["ate_lower"],
            mode="lines",
            fill="tonexty",
            fillcolor="rgba(204, 0, 0, 0.15)",
            line={"width": 0},
            name="95% CI",
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df["theta"],
            y=df["ate"],
            mode="lines+markers",
            name="ATE",
            line={"color": HAVAS_RED, "width": 2},
            marker={"color": HAVAS_RED, "size": 8},
            hovertemplate="θ=%{x:.1f}<br>ATE=%{y:.3f}<extra></extra>",
        )
    )

    fig.update_layout(
        title={"text": "Adstock Decay Sweep — ATE by θ", "font": {"color": HAVAS_DARK}},
        xaxis={
            "title": "Adstock Decay (θ)",
            "tickvals": df["theta"].tolist(),
            "gridcolor": "#E8E8E8",
        },
        yaxis={"title": _ATE_AXIS_LABEL, "gridcolor": "#E8E8E8"},
        legend={"orientation": "h", "y": -0.2},
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        font={"color": HAVAS_DARK},
        height=350,
    )
    return fig


def plot_ate_by_region(region_breakdown: pd.DataFrame) -> go.Figure:
    """Horizontal bar chart of ATE by region, sorted descending.

    Args:
        region_breakdown: DataFrame with columns: group, ate, ate_lower, ate_upper.

    Returns:
        Plotly Figure with ATE scaled to per ₹10,00,000 spend.
    """
    df = _scale_ate(region_breakdown).sort_values("ate", ascending=True)
    error_minus = (df["ate"] - df["ate_lower"]).tolist()
    error_plus = (df["ate_upper"] - df["ate"]).tolist()

    fig = go.Figure(
        go.Bar(
            x=df["ate"],
            y=df["group"],
            orientation="h",
            marker_color=HAVAS_RED,
            error_x={
                "type": "data",
                "symmetric": False,
                "array": error_plus,
                "arrayminus": error_minus,
                "color": HAVAS_GREY,
                "thickness": 1.5,
            },
            hovertemplate="<b>%{y}</b><br>ATE=%{x:.3f}<extra></extra>",
        )
    )
    fig.update_layout(
        title={"text": "ATE by Region", "font": {"color": HAVAS_DARK}},
        xaxis={"title": _ATE_AXIS_LABEL, "gridcolor": "#E8E8E8"},
        yaxis={"title": "Region"},
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        font={"color": HAVAS_DARK},
        height=350,
    )
    return fig


def plot_subgroup_breakdown(breakdown_df: pd.DataFrame, dimension_label: str) -> go.Figure:
    """Horizontal bar chart for a dimension sub-group breakdown with CI error bars.

    Args:
        breakdown_df: DataFrame with columns: group, ate, ate_lower, ate_upper.
        dimension_label: Human-readable dimension name (e.g. "Edition").

    Returns:
        Plotly Figure with ATE scaled to per ₹10,00,000 spend.
    """
    df = _scale_ate(breakdown_df).sort_values("ate", ascending=True)
    error_minus = (df["ate"] - df["ate_lower"]).tolist()
    error_plus = (df["ate_upper"] - df["ate"]).tolist()

    fig = go.Figure(
        go.Bar(
            x=df["ate"],
            y=df["group"],
            orientation="h",
            marker_color=HAVAS_RED,
            error_x={
                "type": "data",
                "symmetric": False,
                "array": error_plus,
                "arrayminus": error_minus,
                "color": HAVAS_GREY,
                "thickness": 1.5,
            },
            hovertemplate="<b>%{y}</b><br>ATE=%{x:.3f}<extra></extra>",
        )
    )
    fig.update_layout(
        title={"text": f"ATE by {dimension_label}", "font": {"color": HAVAS_DARK}},
        xaxis={"title": _ATE_AXIS_LABEL, "gridcolor": "#E8E8E8"},
        yaxis={"title": dimension_label},
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        font={"color": HAVAS_DARK},
        height=max(300, 60 * len(df)),
    )
    return fig
