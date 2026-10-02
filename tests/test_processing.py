"""Tests for scan parsing, background removal, and phase screening."""

from __future__ import annotations

import numpy as np
import pytest

from crystallography import CU_KA1, calculate_pattern, constrained_lattice
from materials import PHASES, PRESETS
from plotting import build_pattern_figure, build_score_figure
from xrd_processing import (
    build_ti2aln_example,
    claimed_peak_indices,
    parse_xrd_text,
    process_scan,
    score_phase,
    snip_background,
)


def _pattern(key: str, minimum: float = 2.0):
    phase = PHASES[key]
    lattice = constrained_lattice(
        phase.crystal_system, phase.a, phase.b, phase.c, phase.alpha, phase.beta, phase.gamma
    )
    return calculate_pattern(
        lattice,
        CU_KA1,
        (5, 90),
        phase.atoms,
        phase.space_group,
        b_iso=0.4,
        min_relative_intensity=minimum,
    )


def test_parse_commented_xy_and_csv_header():
    xy = """
# comment
5.0 10
6.0 12
7.0 11
"""
    # Fewer than 10 points is rejected.
    with pytest.raises(ValueError):
        parse_xrd_text(xy, "short.xy")

    rows = "\n".join(f"{angle:.2f} {100 + angle:.2f}" for angle in np.linspace(10, 40, 30))
    text = "# scan\n" + rows + "\n"
    angles, intensity = parse_xrd_text(text, "scan.xy")
    assert angles[0] == pytest.approx(10)
    assert len(angles) == 30

    csv = "two_theta,intensity\n" + "\n".join(
        f"{angle:.2f},{50 + index}" for index, angle in enumerate(np.linspace(5, 30, 20))
    )
    angles, intensity = parse_xrd_text(csv, "scan.csv")
    assert intensity[0] == pytest.approx(50)
    assert angles[-1] == pytest.approx(30)

    european = "\n".join(
        f"{str(angle).replace('.', ',')};{100 + index}"
        for index, angle in enumerate(np.linspace(5, 25, 15))
    )
    angles, intensity = parse_xrd_text(european, "scan.txt")
    assert angles[0] == pytest.approx(5)
    assert intensity[-1] == pytest.approx(114)


def test_snip_preserves_a_flat_baseline_and_removes_a_peak():
    baseline = np.full(200, 25.0)
    assert snip_background(baseline, 20) == pytest.approx(baseline)
    signal = baseline.copy()
    signal[80:100] += 80
    background = snip_background(signal, 30)
    assert background[90] < signal[90] - 40
    assert background[10] == pytest.approx(25, abs=1)


def test_example_scan_identifies_ti2aln_and_tin():
    angles, intensity = build_ti2aln_example()
    processed = process_scan(angles, intensity, wavelength=CU_KA1, prominence_percent=3)
    assert len(processed.peaks) >= 8
    primary = score_phase(
        processed.peaks,
        _pattern("Ti2AlN").reflections,
        name="Ti2AlN",
        role="Primary",
        tolerance=0.15,
    )
    claimed = claimed_peak_indices(primary, processed.peaks)
    tin = score_phase(
        processed.peaks,
        _pattern("TiN").reflections,
        name="TiN",
        role="Candidate",
        tolerance=0.15,
        claimed=claimed,
    )
    aluminum = score_phase(
        processed.peaks,
        _pattern("Al").reflections,
        name="Al",
        role="Candidate",
        tolerance=0.15,
        claimed=claimed,
    )
    assert primary.assessment == "Consistent"
    assert primary.score > tin.score > aluminum.score
    assert tin.score >= 40
    assert aluminum.assessment == "Unlikely"
    figure = build_pattern_figure(
        "Ti2AlN",
        "Cu Kα1",
        CU_KA1,
        processed.two_theta,
        processed.intensity,
        processed.background,
        processed.peaks,
        _pattern("Ti2AlN").reflections,
    )
    assert "Ti2AlN" in figure.layout.title.text
    score_figure = build_score_figure([primary, tin, aluminum], "Ti2AlN")
    assert "Ti2AlN" in score_figure.layout.title.text


def test_preset_impurity_lists_match_the_request():
    assert PRESETS["Ti2AlN"].impurity_keys[:3] == ("TiN", "Ti", "Al")
    assert "NbC" in PRESETS["Nb2AlC"].impurity_keys
    assert "Nb" in PRESETS["Nb2AlC"].impurity_keys
    assert "Al" in PRESETS["Nb2AlC"].impurity_keys


def test_app_module_exposes_main():
    import app

    assert callable(app.main)
