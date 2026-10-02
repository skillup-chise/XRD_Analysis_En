"""Plotly figures for measured scans, reference lines, and screening scores."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import plotly.graph_objects as go

from crystallography import Reflection
from xrd_processing import DetectedPeak, PhaseScore

MEASURED_COLOR = "#1F4E79"
BACKGROUND_COLOR = "#A67C52"
PEAK_COLOR = "#9B2C2C"
REFERENCE_COLOR = "#1B7F5A"
CANDIDATE_COLORS = ("#C45C26", "#2E5A88", "#6C3483", "#0E6655", "#922B21", "#1A5276")
ASSESSMENT_COLORS = {
    "Consistent": "#1B7F5A",
    "Possible": "#B86E00",
    "Unlikely": "#8A8680",
}

_FONT = "Arial, Helvetica, sans-serif"


def build_pattern_figure(
    material_name: str,
    wavelength_label: str,
    wavelength: float,
    two_theta: np.ndarray | None = None,
    intensity: np.ndarray | None = None,
    background: np.ndarray | None = None,
    peaks: Sequence[DetectedPeak] | None = None,
    reference: Sequence[Reflection] | None = None,
    overlays: Sequence[tuple[str, Sequence[Reflection]]] | None = None,
    show_net: bool = False,
    net: np.ndarray | None = None,
) -> go.Figure:
    """Overlay a measured scan, detected peaks, and calculated stick patterns."""

    figure = go.Figure()
    scale = 1.0
    if intensity is not None and len(intensity):
        maximum = float(np.max(intensity))
        scale = 100.0 / maximum if maximum > 0 else 1.0
        figure.add_trace(
            go.Scatter(
                x=two_theta,
                y=np.asarray(intensity) * scale,
                mode="lines",
                name="Measured",
                line={"color": MEASURED_COLOR, "width": 1.6},
                hovertemplate="2θ = %{x:.3f}°<br>Relative intensity = %{y:.1f}<extra>Measured</extra>",
            )
        )
    if background is not None and two_theta is not None:
        figure.add_trace(
            go.Scatter(
                x=two_theta,
                y=np.asarray(background) * scale,
                mode="lines",
                name="Background",
                line={"color": BACKGROUND_COLOR, "width": 1.4, "dash": "dash"},
                hovertemplate="2θ = %{x:.3f}°<br>Background = %{y:.1f}<extra>Background</extra>",
            )
        )
    if show_net and net is not None and two_theta is not None:
        figure.add_trace(
            go.Scatter(
                x=two_theta,
                y=np.asarray(net) * scale,
                mode="lines",
                name="Net intensity",
                line={"color": "#5D6D7E", "width": 1.2},
                hovertemplate="2θ = %{x:.3f}°<br>Net = %{y:.1f}<extra>Net</extra>",
            )
        )
    if peaks and intensity is not None:
        peak_y = [peak.raw_intensity * scale for peak in peaks]
        figure.add_trace(
            go.Scatter(
                x=[peak.two_theta for peak in peaks],
                y=peak_y,
                mode="markers",
                name="Detected peaks",
                marker={"color": PEAK_COLOR, "size": 8, "symbol": "diamond"},
                text=[f"d = {peak.d_spacing:.4f} Å" for peak in peaks],
                hovertemplate=(
                    "2θ = %{x:.3f}°<br>%{text}<br>Relative intensity = %{y:.1f}"
                    "<extra>Detected peak</extra>"
                ),
            )
        )
    if reference:
        _add_sticks(
            figure,
            reference,
            f"{material_name} reference",
            REFERENCE_COLOR,
        )
    for index, (name, sticks) in enumerate(overlays or ()):
        _add_sticks(
            figure,
            sticks,
            f"{name} candidate",
            CANDIDATE_COLORS[index % len(CANDIDATE_COLORS)],
        )

    figure.update_layout(
        title={
            "text": f"{material_name} X-ray diffraction pattern",
            "subtitle": {"text": f"{wavelength_label}, {wavelength:.5f} Å"},
            "x": 0,
            "xanchor": "left",
        },
        font={"family": _FONT, "size": 13, "color": "#1C2833"},
        template="plotly_white",
        height=560,
        margin={"l": 64, "r": 24, "t": 88, "b": 56},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
        hovermode="closest",
        xaxis={
            "title": "2θ (°)",
            "showgrid": True,
            "zeroline": False,
            "ticks": "outside",
        },
        yaxis={
            "title": "Relative intensity",
            "rangemode": "tozero",
            "showgrid": True,
            "zeroline": False,
        },
    )
    return figure


def build_score_figure(scores: Sequence[PhaseScore], material_name: str) -> go.Figure:
    """Horizontal score chart for the primary phase and impurity candidates."""

    ordered = list(scores)
    figure = go.Figure(
        go.Bar(
            x=[score.score for score in ordered],
            y=[score.name for score in ordered],
            orientation="h",
            marker={"color": [ASSESSMENT_COLORS[score.assessment] for score in ordered]},
            text=[f"{score.score:.0f} · {score.assessment}" for score in ordered],
            textposition="outside",
            hovertemplate=(
                "%{y}<br>Score = %{x:.1f}<br>%{customdata}<extra></extra>"
            ),
            customdata=[score.assessment for score in ordered],
            cliponaxis=False,
        )
    )
    figure.update_layout(
        title={"text": f"Phase screening for {material_name}", "x": 0, "xanchor": "left"},
        font={"family": _FONT, "size": 13, "color": "#1C2833"},
        template="plotly_white",
        height=max(280, 64 * max(1, len(ordered)) + 80),
        margin={"l": 120, "r": 80, "t": 60, "b": 48},
        xaxis={"title": "Match score", "range": [0, 110], "dtick": 20},
        yaxis={"autorange": "reversed", "title": ""},
        showlegend=False,
    )
    return figure


def _add_sticks(
    figure: go.Figure,
    reflections: Sequence[Reflection],
    name: str,
    color: str,
) -> None:
    if not reflections:
        return
    x_values: list[float | None] = []
    y_values: list[float | None] = []
    for reflection in reflections:
        x_values.extend([reflection.two_theta, reflection.two_theta, None])
        y_values.extend([0.0, reflection.intensity, None])
    figure.add_trace(
        go.Scatter(
            x=x_values,
            y=y_values,
            mode="lines",
            name=name,
            line={"color": color, "width": 1.8},
            hoverinfo="skip",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=[reflection.two_theta for reflection in reflections],
            y=[reflection.intensity for reflection in reflections],
            mode="markers",
            name=name,
            marker={"color": color, "size": 7, "symbol": "line-ns-open"},
            showlegend=False,
            text=[
                (
                    f"{name}<br>{reflection.hkl_label}"
                    f"<br>d = {reflection.d_spacing:.4f} Å"
                    f"<br>multiplicity {reflection.multiplicity}"
                )
                for reflection in reflections
            ],
            hovertemplate="2θ = %{x:.3f}°<br>I = %{y:.1f}<br>%{text}<extra></extra>",
        )
    )
