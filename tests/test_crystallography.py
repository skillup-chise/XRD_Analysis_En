"""Tests for lattice geometry, intensities, and CIF parsing."""

from __future__ import annotations

import math

import numpy as np
import pytest

from crystallography import (
    CU_KA1,
    AtomSite,
    bragg_two_theta,
    calculate_pattern,
    constrained_lattice,
    d_spacing,
    infer_crystal_system,
    is_systematically_absent,
    laue_operations,
    normalize_space_group,
    parse_cif,
    powder_multiplicity,
    scattering_factor,
)
from materials import PHASES


def _reflection(pattern, h: int, k: int, l: int):
    for reflection in pattern.reflections:
        if (h, k, l) in reflection.hkls:
            return reflection
    raise AssertionError(f"{(h, k, l)} was not found")


def test_laue_group_orders():
    assert len(laue_operations("Cubic")) == 48
    assert len(laue_operations("Hexagonal")) == 24
    assert len(laue_operations("Trigonal")) == 12
    assert len(laue_operations("Tetragonal")) == 16
    assert len(laue_operations("Orthorhombic")) == 8
    assert len(laue_operations("Monoclinic")) == 4
    assert len(laue_operations("Triclinic")) == 2
    assert len(laue_operations("Rhombohedral")) == 12


def test_powder_multiplicities():
    assert powder_multiplicity(1, 0, 0, "Cubic") == 6
    assert powder_multiplicity(1, 1, 0, "Cubic") == 12
    assert powder_multiplicity(1, 1, 1, "Cubic") == 8
    assert powder_multiplicity(3, 2, 1, "Cubic") == 48
    assert powder_multiplicity(1, 0, 0, "Hexagonal") == 6
    assert powder_multiplicity(1, 1, 0, "Hexagonal") == 6
    assert powder_multiplicity(1, 0, 1, "Hexagonal") == 12
    assert powder_multiplicity(0, 0, 2, "Hexagonal") == 2
    assert powder_multiplicity(1, 2, 3, "Hexagonal") == 24
    assert powder_multiplicity(1, 0, 0, "Tetragonal") == 4
    assert powder_multiplicity(0, 0, 1, "Tetragonal") == 2
    assert powder_multiplicity(1, 2, 3, "Orthorhombic") == 8
    assert powder_multiplicity(1, 0, 2, "Monoclinic") == 2
    assert powder_multiplicity(1, 2, 3, "Triclinic") == 2
    assert powder_multiplicity(1, 1, 1, "Rhombohedral") == 2
    assert powder_multiplicity(1, 2, 3, "Rhombohedral") == 12
    assert powder_multiplicity(1, 0, 0, "Trigonal") == 6
    assert powder_multiplicity(0, 0, 1, "Trigonal") == 2
    assert powder_multiplicity(1, 2, 3, "Trigonal") == 12


def test_hexagonal_and_cubic_d_spacings():
    hexagonal = constrained_lattice("Hexagonal", 2.989, c=13.614)
    assert d_spacing(hexagonal, 0, 0, 2) == pytest.approx(13.614 / 2)
    assert d_spacing(hexagonal, 1, 0, 0) == pytest.approx(2.989 * math.sqrt(3) / 2)
    cubic = constrained_lattice("Cubic", 4.241)
    assert d_spacing(cubic, 1, 1, 1) == pytest.approx(4.241 / math.sqrt(3))
    assert d_spacing(cubic, 2, 0, 0) == pytest.approx(4.241 / 2)


def test_monoclinic_d_spacing_matches_the_standard_formula():
    lattice = constrained_lattice("Monoclinic", 5.0, b=6.0, c=7.0, beta=100.0)
    beta = math.radians(100.0)
    inv_d2 = (1.0 / math.sin(beta) ** 2) * (
        1.0 / 25.0 + (math.sin(beta) ** 2) / 36.0 + 1.0 / 49.0 - 2.0 * math.cos(beta) / (5.0 * 7.0)
    )
    assert d_spacing(lattice, 1, 1, 1) == pytest.approx(1.0 / math.sqrt(inv_d2))


def test_bragg_angle_for_ti2aln_002():
    lattice = constrained_lattice("Hexagonal", 2.989, c=13.614)
    expected = bragg_two_theta(d_spacing(lattice, 0, 0, 2), CU_KA1)
    assert expected == pytest.approx(12.995, abs=0.02)


def test_systematic_absences():
    assert is_systematically_absent(1, 0, 0, "Fm-3m", "Cubic")
    assert not is_systematically_absent(1, 1, 1, "Fm-3m", "Cubic")
    assert not is_systematically_absent(2, 0, 0, "Fm-3m", "Cubic")
    assert is_systematically_absent(1, 0, 0, "Im-3m", "Cubic")
    assert not is_systematically_absent(1, 1, 0, "Im-3m", "Cubic")
    assert is_systematically_absent(0, 0, 1, "P63/mmc", "Hexagonal")
    assert not is_systematically_absent(0, 0, 2, "P63/mmc", "Hexagonal")
    assert is_systematically_absent(1, 1, 1, "P63/mmc", "Hexagonal")
    assert not is_systematically_absent(1, 0, 3, "P63/mmc", "Hexagonal")
    assert is_systematically_absent(1, 0, 0, "R-3m", "Hexagonal")
    assert not is_systematically_absent(1, 1, 0, "R-3m", "Hexagonal")
    assert not is_systematically_absent(1, 0, 0, "R-3m", "Rhombohedral")


def test_scattering_factor_limits():
    assert scattering_factor("Al", 0.0) == pytest.approx(13.0)
    assert scattering_factor("Ti", 0.0) == pytest.approx(22.0)
    assert scattering_factor("Nb", 0.0) == pytest.approx(41.0)
    assert scattering_factor("Al", 0.5) < scattering_factor("Al", 0.0)
    assert scattering_factor("Al", 0.5) == pytest.approx(5.7, abs=0.4)


def test_tin_reflections_and_ti2aln_absences():
    tin = PHASES["TiN"]
    lattice = constrained_lattice(tin.crystal_system, tin.a, tin.b, tin.c, tin.alpha, tin.beta, tin.gamma)
    pattern = calculate_pattern(lattice, CU_KA1, (20, 80), tin.atoms, tin.space_group, b_iso=0.3)
    labels = {reflection.hkl_label for reflection in pattern.reflections}
    assert not any("(1 0 0)" in label or "(1 1 0)" in label for label in labels)
    line_111 = _reflection(pattern, 1, 1, 1)
    line_200 = _reflection(pattern, 2, 0, 0)
    assert line_111.two_theta == pytest.approx(36.66, abs=0.08)
    assert line_200.two_theta == pytest.approx(42.61, abs=0.08)
    assert line_111.multiplicity == 8
    assert line_200.intensity > 20

    phase = PHASES["Ti2AlN"]
    max_lattice = constrained_lattice(
        phase.crystal_system, phase.a, phase.b, phase.c, phase.alpha, phase.beta, phase.gamma
    )
    max_pattern = calculate_pattern(
        max_lattice, CU_KA1, (5, 90), phase.atoms, phase.space_group, b_iso=0.4
    )
    assert all((0, 0, 1) not in reflection.hkls for reflection in max_pattern.reflections)
    line_002 = _reflection(max_pattern, 0, 0, 2)
    line_100 = _reflection(max_pattern, 1, 0, 0)
    line_103 = _reflection(max_pattern, 1, 0, 3)
    assert line_002.two_theta == pytest.approx(12.995, abs=0.03)
    assert line_100.two_theta == pytest.approx(34.64, abs=0.05)
    assert line_002.multiplicity == 2
    assert line_100.multiplicity == 6
    assert line_103.intensity > 20
    assert max(reflection.intensity for reflection in max_pattern.reflections) == pytest.approx(100)


def test_geometry_only_pattern_uses_multiplicity():
    lattice = constrained_lattice("Cubic", 4.0)
    pattern = calculate_pattern(
        lattice,
        CU_KA1,
        (20, 90),
        atoms=(),
        space_group="Pm-3m",
        b_iso=0.0,
        apply_lorentz_polarization=False,
        min_relative_intensity=0,
    )
    line_100 = _reflection(pattern, 1, 0, 0)
    line_111 = _reflection(pattern, 1, 1, 1)
    assert line_100.multiplicity == 6
    assert line_111.multiplicity == 8


def test_crystal_system_inference_and_constraints():
    assert infer_crystal_system(4, 4, 4, 90, 90, 90) == "Cubic"
    assert infer_crystal_system(3, 3, 12, 90, 90, 120) == "Hexagonal"
    assert infer_crystal_system(5, 5, 5, 55, 55, 55) == "Rhombohedral"
    assert infer_crystal_system(5, 6, 7, 90, 99, 90) == "Monoclinic"
    cubic = constrained_lattice("Cubic", 3.5, b=9, c=8, alpha=70, beta=80, gamma=100)
    assert (cubic.a, cubic.b, cubic.c, cubic.gamma) == (3.5, 3.5, 3.5, 90.0)
    with pytest.raises(ValueError):
        constrained_lattice("Cubic", -1)


def test_normalize_space_group():
    assert normalize_space_group("P 6_3/m m c") == "P63/mmc"
    assert normalize_space_group("F m -3 m") == "Fm-3m"
    assert normalize_space_group("P 1 21/c 1") == "P21/c"


def test_cif_parser_reads_cell_and_expands_symmetry():
    text = """
data_Demo
_cell_length_a 4.241(1)
_cell_length_b 4.241
_cell_length_c 4.241
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
_space_group_name_H-M_alt 'F m -3 m'
_space_group_IT_number 225
_chemical_formula_sum 'Ti1 N1'
loop_
_symmetry_equiv_pos_as_xyz
x,y,z
x,y+1/2,z+1/2
x+1/2,y,z+1/2
x+1/2,y+1/2,z
loop_
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
_atom_site_occupancy
Ti 0 0 0 1
N 0.5 0.5 0.5 1
"""
    parsed = parse_cif(text)
    assert parsed.space_group == "Fm-3m"
    assert parsed.space_group_number == 225
    assert parsed.crystal_system == "Cubic"
    assert parsed.a == pytest.approx(4.241)
    assert parsed.formula == "Ti1 N1"
    assert parsed.asymmetric_site_count == 2
    assert len(parsed.atoms) == 8
    elements = sorted(atom.element for atom in parsed.atoms)
    assert elements.count("Ti") == 4
    assert elements.count("N") == 4


def test_diamond_200_is_absent_when_atoms_are_supplied():
    fcc = ((0, 0, 0), (0.5, 0.5, 0), (0.5, 0, 0.5), (0, 0.5, 0.5))
    basis = ((0, 0, 0), (0.25, 0.25, 0.25))
    atoms = [
        AtomSite("Si", (origin[0] + shift[0]) % 1, (origin[1] + shift[1]) % 1, (origin[2] + shift[2]) % 1)
        for origin in fcc
        for shift in basis
    ]
    lattice = constrained_lattice("Cubic", 5.431)
    pattern = calculate_pattern(lattice, CU_KA1, (20, 80), atoms, "Fd-3m", b_iso=0.4)
    assert all((2, 0, 0) not in reflection.hkls for reflection in pattern.reflections)
    assert _reflection(pattern, 1, 1, 1).two_theta == pytest.approx(28.44, abs=0.05)


def test_catalog_phases_have_finite_patterns():
    for phase in PHASES.values():
        lattice = constrained_lattice(
            phase.crystal_system, phase.a, phase.b, phase.c, phase.alpha, phase.beta, phase.gamma
        )
        pattern = calculate_pattern(
            lattice, CU_KA1, (5, 90), phase.atoms, phase.space_group, min_relative_intensity=2
        )
        assert pattern.reflections, phase.name
        assert max(item.intensity for item in pattern.reflections) == pytest.approx(100)
        assert np.isfinite([item.two_theta for item in pattern.reflections]).all()
