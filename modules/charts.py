"""Plotly chart builders shared by the dashboard pages.

Charts take DataFrames and return Plotly figures. They contain no Streamlit
code, so they can be reused in reports (Phase 7) and tested on their own.
"""

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from modules.config import SEVERITY_COLORS, STATUS_COLORS
from modules.route_engine import CRITERIA

HIGHLIGHT = "#2a78d6"
CONTEXT_GREY = "#B8BCC2"
THRESHOLD_COLOR = "#B42318"
REPLAY_COLOR = "#52514e"

BASE_LAYOUT = {
    "margin": {"l": 10, "r": 10, "t": 40, "b": 10},
    "plot_bgcolor": "rgba(0,0,0,0)",
    "paper_bgcolor": "rgba(0,0,0,0)",
    "hovermode": "x unified",
    "font": {"size": 12},
}


def _style_axes(fig: go.Figure) -> None:
    fig.update_xaxes(showgrid=False, linecolor="#D0D4DA")
    fig.update_yaxes(gridcolor="rgba(128,128,128,0.18)", zeroline=False)


def trend_chart(
    readings: pd.DataFrame, column: str, title: str, unit: str,
    highlight_section: str | None = None, replay_time: pd.Timestamp | None = None,
) -> go.Figure:
    """All sections over time: the chosen section highlighted, others as grey context."""
    fig = go.Figure()
    first_context = True
    for sid, df in readings.groupby("section_id", sort=True):
        if sid == highlight_section:
            continue  # drawn last so it sits on top
        fig.add_trace(go.Scatter(
            x=df["timestamp"], y=df[column], mode="lines", name="Other sections",
            legendgroup="context", showlegend=first_context,
            line={"color": CONTEXT_GREY, "width": 1}, opacity=0.8,
            hovertemplate=f"{sid}: %{{y:,.1f}} {unit}<extra></extra>",
        ))
        first_context = False
    if highlight_section is not None:
        df = readings[readings["section_id"] == highlight_section]
        fig.add_trace(go.Scatter(
            x=df["timestamp"], y=df[column], mode="lines", name=f"{highlight_section} (selected)",
            line={"color": HIGHLIGHT, "width": 2.5},
            hovertemplate=f"<b>{highlight_section}</b>: %{{y:,.1f}} {unit}<extra></extra>",
        ))
    if replay_time is not None:
        fig.add_vline(x=replay_time, line={"color": REPLAY_COLOR, "width": 1, "dash": "dot"})
    fig.update_layout(**BASE_LAYOUT, title={"text": title, "x": 0}, height=300, showlegend=True,
                      legend={"orientation": "h", "y": -0.18}, yaxis_title=unit)
    _style_axes(fig)
    return fig


def section_detail_chart(
    section_readings: pd.DataFrame, section_events: pd.DataFrame, expected_pressure: float,
    pressure_high_pct: float, pressure_low_pct: float, nominal_temperature: float,
    temperature_deviation: float, flow_unit: str, replay_time: pd.Timestamp | None = None,
) -> go.Figure:
    """Four stacked panels for one section, with thresholds and shaded event periods."""
    df = section_readings.sort_values("timestamp")
    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.06,
        subplot_titles=("Pressure (kPa)", "Temperature (°C)", f"Flow ({flow_unit})", "Sensor health (0-100)"),
    )
    line = {"color": HIGHLIGHT, "width": 2}

    fig.add_trace(go.Scatter(x=df["timestamp"], y=df["pressure_kpa"], name="Pressure", line=line), 1, 1)
    for value, label in [
        (expected_pressure * (1 + pressure_high_pct / 100), "High limit"),
        (expected_pressure * (1 - pressure_low_pct / 100), "Low limit"),
    ]:
        fig.add_hline(y=value, line={"color": THRESHOLD_COLOR, "width": 1, "dash": "dash"},
                      annotation_text=label, annotation_position="top left", row=1, col=1)

    fig.add_trace(go.Scatter(x=df["timestamp"], y=df["temperature_c"], name="Temperature", line=line), 2, 1)
    for value in (nominal_temperature + temperature_deviation, nominal_temperature - temperature_deviation):
        fig.add_hline(y=value, line={"color": THRESHOLD_COLOR, "width": 1, "dash": "dash"}, row=2, col=1)

    fig.add_trace(go.Scatter(x=df["timestamp"], y=df["flow_in"], name="Inlet flow", line=line), 3, 1)
    fig.add_trace(go.Scatter(x=df["timestamp"], y=df["flow_out"], name="Outlet flow",
                             line={"color": "#eb6834", "width": 2, "dash": "dot"}), 3, 1)

    fig.add_trace(go.Scatter(x=df["timestamp"], y=df["sensor_health"], name="Sensor health", line=line), 4, 1)

    # Shade each event period in its severity colour (label in hover via legend name).
    for _, ev in section_events.iterrows():
        end = ev["end"] if ev["end"] > ev["start"] else ev["start"] + pd.Timedelta(minutes=15)
        fig.add_vrect(x0=ev["start"], x1=end, fillcolor=SEVERITY_COLORS[ev["severity"]], opacity=0.12,
                      line_width=0)
    if replay_time is not None:
        fig.add_vline(x=replay_time, line={"color": REPLAY_COLOR, "width": 1, "dash": "dot"})

    fig.update_layout(**BASE_LAYOUT, height=720, showlegend=True, legend={"orientation": "h", "y": -0.06})
    _style_axes(fig)
    return fig


def status_strip(sections: pd.DataFrame, statuses: pd.DataFrame) -> go.Figure:
    """A horizontal strip along the pipeline (KP 0 to end) coloured by section status."""
    merged = sections.merge(statuses, on="section_id")
    fig = go.Figure()
    for status in STATUS_COLORS:
        part = merged[merged["status"] == status]
        if part.empty:
            continue
        fig.add_trace(go.Bar(
            y=["Pipeline"] * len(part), x=part["end_km"] - part["start_km"], base=part["start_km"],
            orientation="h", name=status, marker={"color": STATUS_COLORS[status], "line": {"color": "white", "width": 2}},
            text=part["section_id"], textposition="inside", insidetextanchor="middle",
            customdata=part[["section_id", "reasons", "start_km", "end_km"]],
            hovertemplate="<b>%{customdata[0]}</b> · " + status
                          + "<br>KP %{customdata[2]:.0f}–%{customdata[3]:.0f} km<br>%{customdata[1]}<extra></extra>",
        ))
    fig.update_layout(**{**BASE_LAYOUT, "hovermode": "closest"}, barmode="overlay", height=150,
                      xaxis_title="Distance along pipeline, KP (km)",
                      legend={"orientation": "h", "y": 1.35, "x": 0}, yaxis={"visible": False})
    _style_axes(fig)
    return fig


def route_score_chart(scored: pd.DataFrame) -> go.Figure:
    """Horizontal bars of the weighted score; routes with hard constraints are hatched."""
    df = scored.sort_values("score")
    fig = go.Figure(go.Bar(
        x=df["score"], y=df["name"], orientation="h",
        marker={
            "color": [CONTEXT_GREY if h else HIGHLIGHT for h in df["has_hard_constraint"]],
            "pattern": {"shape": ["/" if h else "" for h in df["has_hard_constraint"]]},
            "cornerradius": 4,
        },
        text=[f"{s:.0f}" + (" · not ranked" if h else "")
              for s, h in zip(df["score"], df["has_hard_constraint"])],
        textposition="outside",
        hovertemplate="%{y}<br>Weighted score %{x:.0f}/100<extra></extra>",
    ))
    fig.update_layout(**{**BASE_LAYOUT, "hovermode": "closest"}, height=260,
                      xaxis={"range": [0, 125], "title": "Weighted score (relative to these alternatives)"},
                      title={"text": "Multi-criteria score", "x": 0})
    _style_axes(fig)
    return fig


def route_tradeoff_heatmap(scored: pd.DataFrame) -> go.Figure:
    """Each route's relative performance on each criterion (1 = best of the set, 0 = worst)."""
    z = [[float(r[f"norm_{c.key}"]) for c in CRITERIA] for _, r in scored.iterrows()]
    raw = [[r[c.key] for c in CRITERIA] for _, r in scored.iterrows()]
    fig = go.Figure(go.Heatmap(
        z=z, x=[c.label for c in CRITERIA], y=scored["name"],
        colorscale=[[0, "#f3f4f6"], [1, "#1c5aa6"]], zmin=0, zmax=1,
        xgap=2, ygap=2, customdata=raw,
        colorbar={"title": "Relative<br>performance", "tickvals": [0, 1], "ticktext": ["Worst", "Best"]},
        hovertemplate="%{y}<br>%{x}: %{customdata:,.1f}<br>Relative performance %{z:.2f}<extra></extra>",
    ))
    fig.update_layout(**{**BASE_LAYOUT, "hovermode": "closest"}, height=300,
                      title={"text": "Trade-off profile (darker = better among these alternatives)", "x": 0})
    return fig


def elevation_profile_chart(profiles: dict[str, pd.DataFrame], names: dict[str, str],
                            colors: dict[str, str], length_unit: str = "km",
                            terrain_label: str = "synthetic terrain") -> go.Figure:
    """Ground elevation along each route (synthetic terrain in this phase)."""
    factor = 1 / 1.609344 if length_unit == "mi" else 1.0
    fig = go.Figure()
    for rid, prof in profiles.items():
        fig.add_trace(go.Scatter(
            x=prof["chainage_km"] * factor, y=prof["elevation_m"], mode="lines", name=names[rid],
            line={"color": colors[rid], "width": 1.5 if rid == "REF" else 2, "dash": "dash" if rid == "REF" else None},
            customdata=prof["grade_pct"],
            hovertemplate=f"{names[rid]}<br>%{{x:.1f}} {length_unit} · %{{y:.0f}} m · grade %{{customdata:.1f}}%<extra></extra>",
        ))
    fig.update_layout(**{**BASE_LAYOUT, "hovermode": "closest"}, height=320,
                      title={"text": f"Elevation profiles ({terrain_label})", "x": 0},
                      xaxis_title=f"Distance along route ({length_unit})", yaxis_title="Elevation (m)",
                      legend={"orientation": "h", "y": -0.25})
    _style_axes(fig)
    return fig


# Fixed order for cost groups (validated categorical slots 1-6).
COST_GROUP_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#6250d6"]


def cost_breakdown_chart(breakdowns: dict[str, pd.Series], currency: str, factor: float = 1.0) -> go.Figure:
    """Stacked horizontal bars: each route's CAPEX split into cost groups (in millions)."""
    names = list(breakdowns)
    groups = list(next(iter(breakdowns.values())).index)
    fig = go.Figure()
    for i, group in enumerate(groups):
        values = [breakdowns[n][group] * factor / 1e6 for n in names]
        fig.add_trace(go.Bar(
            y=names, x=values, name=group, orientation="h",
            marker={"color": COST_GROUP_COLORS[i % len(COST_GROUP_COLORS)], "line": {"color": "white", "width": 2}},
            hovertemplate=f"%{{y}}<br>{group}: %{{x:,.1f}} {currency} M<extra></extra>",
        ))
    fig.update_layout(**{**BASE_LAYOUT, "hovermode": "closest"}, barmode="stack", height=120 + 55 * len(names),
                      title={"text": f"CAPEX by cost group ({currency} millions)", "x": 0},
                      legend={"orientation": "h", "y": -0.2, "traceorder": "normal"}, xaxis_title=f"{currency} M")
    _style_axes(fig)
    fig.update_yaxes(autorange="reversed", gridcolor="rgba(0,0,0,0)")
    return fig


def tornado_chart(sens: pd.DataFrame, metric_label: str, currency: str, factor: float = 1.0,
                  top: int = 10) -> go.Figure:
    """Change in the result when each assumption is moved down (blue) or up (orange)."""
    df = sens.head(top).iloc[::-1]
    fig = go.Figure()
    for side, color, label in (("low", HIGHLIGHT, "Assumption lower"), ("high", "#eb6834", "Assumption higher")):
        fig.add_trace(go.Bar(
            y=df["assumption"], x=df[f"change_{side}"] * factor / 1e6, orientation="h", name=label,
            marker={"color": color}, customdata=df[f"{side}_value"],
            hovertemplate=f"%{{y}} = %{{customdata:,.4g}}<br>{metric_label} change: %{{x:+,.1f}} {currency} M"
                          "<extra></extra>",
        ))
    fig.update_layout(**{**BASE_LAYOUT, "hovermode": "closest"}, barmode="overlay", height=90 + 34 * len(df),
                      title={"text": f"What moves {metric_label.lower()} most", "x": 0},
                      xaxis_title=f"Change from base ({currency} M)", legend={"orientation": "h", "y": -0.15})
    fig.add_vline(x=0, line={"color": REPLAY_COLOR, "width": 1})
    _style_axes(fig)
    return fig


def cumulative_cost_chart(series: dict[str, pd.DataFrame], colors: dict[str, str], currency: str,
                          factor: float = 1.0, discounted: bool = False) -> go.Figure:
    """Cumulative spending over the design life for each route."""
    fig = go.Figure()
    for name, df in series.items():
        fig.add_trace(go.Scatter(
            x=df["year"], y=df["cumulative_cad"] * factor / 1e6, mode="lines", name=name,
            line={"color": colors[name], "width": 2},
            hovertemplate=f"{name}<br>Year %{{x}}: %{{y:,.0f}} {currency} M<extra></extra>",
        ))
    kind = "discounted (present value)" if discounted else "undiscounted"
    fig.update_layout(**{**BASE_LAYOUT, "hovermode": "x unified"}, height=330,
                      title={"text": f"Cumulative lifecycle cost, {kind}", "x": 0},
                      xaxis_title="Year of operation", yaxis_title=f"{currency} M",
                      legend={"orientation": "h", "y": -0.25})
    _style_axes(fig)
    return fig


def ml_score_chart(section_scores: pd.DataFrame, threshold: float, baseline_end: pd.Timestamp, section_id: str,
                   replay_time: pd.Timestamp | None = None) -> go.Figure:
    """ML anomaly score over time for one section, with the learned threshold and baseline period."""
    df = section_scores.sort_values("timestamp")
    fig = go.Figure(go.Scatter(x=df["timestamp"], y=df["score"], mode="lines", name=f"{section_id} anomaly score",
                               line={"color": HIGHLIGHT, "width": 2},
                               hovertemplate="%{x|%b %d %H:%M}<br>Score %{y:.2f}<extra></extra>"))
    fig.add_hline(y=threshold, line={"color": THRESHOLD_COLOR, "width": 1, "dash": "dash"},
                  annotation_text="Alert threshold", annotation_position="top left")
    fig.add_vrect(x0=df["timestamp"].min(), x1=baseline_end, fillcolor="#8A8F98", opacity=0.10, line_width=0,
                  annotation_text="Baseline (learned as normal)", annotation_position="top left")
    if replay_time is not None:
        fig.add_vline(x=replay_time, line={"color": REPLAY_COLOR, "width": 1, "dash": "dot"})
    fig.update_layout(**BASE_LAYOUT, height=300, showlegend=False, yaxis_title="Anomaly score",
                      title={"text": f"Experimental ML anomaly score: {section_id}", "x": 0})
    _style_axes(fig)
    return fig


def risk_matrix_chart(grid: pd.DataFrame, title: str) -> go.Figure:
    """5 x 5 likelihood x consequence matrix; cell colour = rating level, number = count of open risks."""
    from modules.risk_register import CONSEQUENCE_LABELS, LEVEL_COLORS, LEVELS, LIKELIHOOD_LABELS, level

    lk, ck = list(grid.index), list(grid.columns)
    z = [[LEVELS.index(level(l * c)) for c in ck] for l in lk]
    text = [[f"<b>{grid.loc[l, c]}</b>" if grid.loc[l, c] else "" for c in ck] for l in lk]
    hover = [[f"Likelihood {l} ({LIKELIHOOD_LABELS[l]}) × consequence {c} ({CONSEQUENCE_LABELS[c]})<br>"
              f"Rating {l * c}: {level(l * c)}<br>{grid.loc[l, c]} open risk(s)" for c in ck] for l in lk]
    scale = []
    for i, name in enumerate(LEVELS):  # discrete colour bands
        scale += [[i / len(LEVELS), LEVEL_COLORS[name]], [(i + 1) / len(LEVELS), LEVEL_COLORS[name]]]
    fig = go.Figure(go.Heatmap(
        z=z, x=[f"{c} {CONSEQUENCE_LABELS[c]}" for c in ck], y=[f"{l} {LIKELIHOOD_LABELS[l]}" for l in lk],
        zmin=-0.5, zmax=len(LEVELS) - 0.5, colorscale=scale, showscale=False, xgap=3, ygap=3,
        text=text, texttemplate="%{text}", textfont={"color": "white", "size": 16},
        customdata=hover, hovertemplate="%{customdata}<extra></extra>",
    ))
    fig.update_layout(**{**BASE_LAYOUT, "hovermode": "closest"}, height=320, title={"text": title, "x": 0},
                      xaxis={"title": "Consequence", "side": "bottom"}, yaxis={"title": "Likelihood"})
    return fig
