"""Crystallographic calculations for powder X-ray diffraction.

Peak positions come from the reciprocal metric tensor and Bragg's law.
Relative intensities use atomic scattering factors, the structure factor,
reflection multiplicity, and the Lorentz–polarization correction.

Scattering coefficients are the four-term parameters distributed with
pymatgen (MIT License). The form factor follows De Graef and McHenry,
*Structure of Materials*:

    f(s) = Z - 41.78214 * s^2 * sum_i a_i * exp(-b_i * s^2)

with s = sin(theta) / lambda in inverse angstroms.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

CU_KA1 = 1.54056

RADIATIONS: tuple[tuple[str, float], ...] = (
    ("Cu Kα1", 1.54056),
    ("Cu Kα2", 1.54439),
    ("Cu Kα (weighted average)", 1.54184),
    ("Co Kα1", 1.78896),
    ("Fe Kα1", 1.93604),
    ("Cr Kα1", 2.28970),
    ("Mo Kα1", 0.70930),
)

CRYSTAL_SYSTEMS: tuple[str, ...] = (
    "Cubic",
    "Tetragonal",
    "Orthorhombic",
    "Hexagonal",
    "Trigonal",
    "Rhombohedral",
    "Monoclinic",
    "Triclinic",
)

# Form-factor constant in f(s) = Z - CONSTANT * s^2 * sum(a_i exp(-b_i s^2)).
_SCATTERING_CONSTANT = 41.78214
_MAX_INDEX = 30
_COINCIDENT_TOLERANCE = 0.01

_ATOMIC_NUMBERS: dict[str, int] = {
    "H": 1, "He": 2, "Li": 3, "Be": 4, "B": 5, "C": 6, "N": 7, "O": 8, "F": 9,
    "Ne": 10, "Na": 11, "Mg": 12, "Al": 13, "Si": 14, "P": 15, "S": 16, "Cl": 17,
    "Ar": 18, "K": 19, "Ca": 20, "Sc": 21, "Ti": 22, "V": 23, "Cr": 24, "Mn": 25,
    "Fe": 26, "Co": 27, "Ni": 28, "Cu": 29, "Zn": 30, "Ga": 31, "Ge": 32, "As": 33,
    "Se": 34, "Br": 35, "Kr": 36, "Rb": 37, "Sr": 38, "Y": 39, "Zr": 40, "Nb": 41,
    "Mo": 42, "Tc": 43, "Ru": 44, "Rh": 45, "Pd": 46, "Ag": 47, "Cd": 48, "In": 49,
    "Sn": 50, "Sb": 51, "Te": 52, "I": 53, "Xe": 54, "Cs": 55, "Ba": 56, "La": 57,
    "Ce": 58, "Pr": 59, "Nd": 60, "Pm": 61, "Sm": 62, "Eu": 63, "Gd": 64, "Tb": 65,
    "Dy": 66, "Ho": 67, "Er": 68, "Tm": 69, "Yb": 70, "Lu": 71, "Hf": 72, "Ta": 73,
    "W": 74, "Re": 75, "Os": 76, "Ir": 77, "Pt": 78, "Au": 79, "Hg": 80, "Tl": 81,
    "Pb": 82, "Bi": 83, "Po": 84, "At": 85, "Rn": 86, "Fr": 87, "Ra": 88, "Ac": 89,
    "Th": 90, "Pa": 91, "U": 92, "Np": 93, "Pu": 94, "Am": 95, "Cm": 96, "Bk": 97,
    "Cf": 98, "D": 1,
}

_ELEMENT_RE = re.compile(r"^([A-Z][a-z]?)")
_CIF_NUMBER_RE = re.compile(
    r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
)


@dataclass(frozen=True)
class SystemSpec:
    """Free lattice parameters for one crystal system."""

    lengths: tuple[str, ...]
    angles: tuple[str, ...]
    description: str


SYSTEM_SPECS: dict[str, SystemSpec] = {
    "Cubic": SystemSpec(("a",), (), "a = b = c, and alpha = beta = gamma = 90°."),
    "Tetragonal": SystemSpec(("a", "c"), (), "a = b, and alpha = beta = gamma = 90°."),
    "Orthorhombic": SystemSpec(
        ("a", "b", "c"), (), "alpha = beta = gamma = 90°."
    ),
    "Hexagonal": SystemSpec(
        ("a", "c"), (), "a = b, alpha = beta = 90°, and gamma = 120°."
    ),
    "Trigonal": SystemSpec(
        ("a", "c"),
        (),
        "Hexagonal setting: a = b, alpha = beta = 90°, and gamma = 120°.",
    ),
    "Rhombohedral": SystemSpec(
        ("a",), ("alpha",), "a = b = c, and alpha = beta = gamma."
    ),
    "Monoclinic": SystemSpec(
        ("a", "b", "c"), ("beta",), "alpha = gamma = 90°, with beta free."
    ),
    "Triclinic": SystemSpec(
        ("a", "b", "c"),
        ("alpha", "beta", "gamma"),
        "All six lattice parameters are independent.",
    ),
}


@dataclass(frozen=True)
class AtomSite:
    """One atom in fractional coordinates."""

    element: str
    x: float
    y: float
    z: float
    occupancy: float = 1.0
    b_iso: float = 0.0


@dataclass(frozen=True)
class Lattice:
    """Direct-space unit cell. Lengths are in angstroms and angles in degrees."""

    a: float
    b: float
    c: float
    alpha: float
    beta: float
    gamma: float
    crystal_system: str

    def metric_tensor(self) -> np.ndarray:
        return metric_tensor(self.a, self.b, self.c, self.alpha, self.beta, self.gamma)

    def d_spacing(self, h: int, k: int, l: int) -> float:
        return d_spacing(self, h, k, l)


@dataclass(frozen=True)
class Reflection:
    """One powder line, possibly combining coincident families."""

    two_theta: float
    d_spacing: float
    intensity: float
    multiplicity: int
    hkls: tuple[tuple[int, int, int], ...]

    @property
    def hkl_label(self) -> str:
        parts = [format_hkl(h, k, l) for h, k, l in self.hkls[:4]]
        text = ", ".join(parts)
        extra = len(self.hkls) - len(parts)
        if extra > 0:
            text += f" +{extra} more"
        return text

    @property
    def primary_hkl(self) -> tuple[int, int, int]:
        return self.hkls[0]


@dataclass(frozen=True)
class CalculatedPattern:
    """Calculated powder pattern for one phase."""

    reflections: tuple[Reflection, ...]
    wavelength: float
    has_structure_factors: bool
    index_limit: int
    index_truncated: bool


@dataclass(frozen=True)
class ParsedCIF:
    """Cell, symmetry, and atoms read from a CIF."""

    data_name: str
    formula: str
    space_group: str
    space_group_number: int | None
    crystal_system: str
    a: float
    b: float
    c: float
    alpha: float
    beta: float
    gamma: float
    atoms: tuple[AtomSite, ...]
    asymmetric_site_count: int
    symmetry_operation_count: int
    warnings: tuple[str, ...]


def format_hkl(h: int, k: int, l: int) -> str:
    return f"({h} {k} {l})"


def format_miller_bravais(h: int, k: int, l: int) -> str:
    return f"({h} {k} {-h - k} {l})"


def metric_tensor(
    a: float, b: float, c: float, alpha: float, beta: float, gamma: float
) -> np.ndarray:
    """Real-space metric tensor G, with angles in degrees."""

    alpha_r = math.radians(alpha)
    beta_r = math.radians(beta)
    gamma_r = math.radians(gamma)
    return np.array(
        [
            [a * a, a * b * math.cos(gamma_r), a * c * math.cos(beta_r)],
            [a * b * math.cos(gamma_r), b * b, b * c * math.cos(alpha_r)],
            [a * c * math.cos(beta_r), b * c * math.cos(alpha_r), c * c],
        ],
        dtype=float,
    )


def d_spacing(lattice: Lattice, h: int, k: int, l: int) -> float:
    """Interplanar spacing from 1/d^2 = h_i (G^-1)_ij h_j."""

    reciprocal = np.linalg.inv(lattice.metric_tensor())
    vector = np.array([h, k, l], dtype=float)
    inv_d2 = float(vector @ reciprocal @ vector)
    if inv_d2 <= 0:
        raise ValueError(f"Reflection {(h, k, l)} is not defined for this cell.")
    return 1.0 / math.sqrt(inv_d2)


def bragg_two_theta(d: float, wavelength: float) -> float:
    """Return 2θ in degrees for a d-spacing in angstroms."""

    argument = wavelength / (2.0 * d)
    if argument <= 0 or argument > 1:
        raise ValueError("This d-spacing cannot diffract at the selected wavelength.")
    return math.degrees(2.0 * math.asin(argument))


def bragg_d_spacing(two_theta: float, wavelength: float) -> float:
    """Return d in angstroms from a measured 2θ in degrees."""

    sine = math.sin(math.radians(two_theta) / 2.0)
    if sine <= 0:
        raise ValueError("2θ must be greater than 0 to compute a d-spacing.")
    return wavelength / (2.0 * sine)


def constrained_lattice(
    crystal_system: str,
    a: float,
    b: float | None = None,
    c: float | None = None,
    alpha: float | None = None,
    beta: float | None = None,
    gamma: float | None = None,
) -> Lattice:
    """Apply the constraints of a crystal system and return a full cell."""

    if crystal_system not in SYSTEM_SPECS:
        known = ", ".join(CRYSTAL_SYSTEMS)
        raise ValueError(f"Unknown crystal system '{crystal_system}'. Choose one of: {known}.")

    def _positive(value: float | None, fallback: float, label: str) -> float:
        chosen = fallback if value is None else value
        if chosen <= 0:
            raise ValueError(f"Lattice length {label} must be positive.")
        return float(chosen)

    def _angle(value: float | None, fallback: float, label: str) -> float:
        chosen = fallback if value is None else value
        if not 0 < chosen < 180:
            raise ValueError(f"Lattice angle {label} must lie between 0° and 180°.")
        return float(chosen)

    aa = _positive(a, a, "a")
    if crystal_system == "Cubic":
        cell = Lattice(aa, aa, aa, 90.0, 90.0, 90.0, crystal_system)
    elif crystal_system == "Tetragonal":
        cc = _positive(c, aa, "c")
        cell = Lattice(aa, aa, cc, 90.0, 90.0, 90.0, crystal_system)
    elif crystal_system == "Orthorhombic":
        cell = Lattice(
            aa,
            _positive(b, aa, "b"),
            _positive(c, aa, "c"),
            90.0,
            90.0,
            90.0,
            crystal_system,
        )
    elif crystal_system in {"Hexagonal", "Trigonal"}:
        cell = Lattice(aa, aa, _positive(c, aa, "c"), 90.0, 90.0, 120.0, crystal_system)
    elif crystal_system == "Rhombohedral":
        angle = _angle(alpha, 60.0, "alpha")
        cell = Lattice(aa, aa, aa, angle, angle, angle, crystal_system)
    elif crystal_system == "Monoclinic":
        cell = Lattice(
            aa,
            _positive(b, aa, "b"),
            _positive(c, aa, "c"),
            90.0,
            _angle(beta, 90.0, "beta"),
            90.0,
            crystal_system,
        )
    else:
        cell = Lattice(
            aa,
            _positive(b, aa, "b"),
            _positive(c, aa, "c"),
            _angle(alpha, 90.0, "alpha"),
            _angle(beta, 90.0, "beta"),
            _angle(gamma, 90.0, "gamma"),
            crystal_system,
        )

    determinant = float(np.linalg.det(cell.metric_tensor()))
    if determinant <= 1e-8:
        raise ValueError("The unit cell is degenerate. Check the lattice lengths and angles.")
    return cell


def infer_crystal_system(
    a: float, b: float, c: float, alpha: float, beta: float, gamma: float
) -> str:
    """Infer a crystal system from cell edges and angles."""

    length_tol = max(1e-3, 0.002 * max(a, b, c))
    angle_tol = 0.05

    def close(left: float, right: float, tol: float) -> bool:
        return abs(left - right) <= tol

    right_angles = (
        close(alpha, 90.0, angle_tol)
        and close(beta, 90.0, angle_tol)
        and close(gamma, 90.0, angle_tol)
    )
    edges_equal = close(a, b, length_tol) and close(b, c, length_tol)
    angles_equal = close(alpha, beta, angle_tol) and close(beta, gamma, angle_tol)

    if edges_equal and right_angles:
        return "Cubic"
    if edges_equal and angles_equal and not close(alpha, 90.0, angle_tol):
        return "Rhombohedral"
    if close(a, b, length_tol) and close(alpha, 90.0, angle_tol) and close(beta, 90.0, angle_tol):
        if close(gamma, 120.0, angle_tol):
            return "Hexagonal"
        if close(gamma, 90.0, angle_tol) and not close(a, c, length_tol):
            return "Tetragonal"
    if right_angles:
        if close(a, b, length_tol) and not close(a, c, length_tol):
            return "Tetragonal"
        return "Orthorhombic"
    if close(alpha, 90.0, angle_tol) and close(gamma, 90.0, angle_tol):
        return "Monoclinic"
    return "Triclinic"


def crystal_system_from_space_group(
    cell_system: str, space_group: str = "", number: int | None = None
) -> str:
    """Prefer the space-group number when it distinguishes trigonal cells."""

    if number is not None:
        if 1 <= number <= 2:
            system = "Triclinic"
        elif 3 <= number <= 15:
            system = "Monoclinic"
        elif 16 <= number <= 74:
            system = "Orthorhombic"
        elif 75 <= number <= 142:
            system = "Tetragonal"
        elif 143 <= number <= 167:
            system = "Rhombohedral" if cell_system == "Rhombohedral" else "Trigonal"
        elif 168 <= number <= 194:
            system = "Hexagonal"
        elif 195 <= number <= 230:
            system = "Cubic"
        else:
            system = cell_system
        return system

    symbol = normalize_space_group(space_group)
    if cell_system == "Hexagonal" and (
        symbol.startswith("R") or re.match(r"P-?3", symbol)
    ):
        return "Trigonal"
    return cell_system


def normalize_space_group(symbol: str) -> str:
    """Compact a Hermann–Mauguin symbol from CIF spacing and underscores."""

    text = symbol.strip().strip("'\"").replace(" ", "")
    text = re.sub(r"_(\d)", r"\1", text)
    text = text.replace("_", "")
    match = re.fullmatch(r"([A-Z])1(\d+/.+?)1", text)
    if match:
        return f"{match.group(1)}{match.group(2)}"
    return text


def centering_letter(space_group: str) -> str:
    symbol = normalize_space_group(space_group)
    if not symbol:
        return "P"
    letter = symbol[0].upper()
    return letter if letter in set("PIFABCR") else "P"


def is_systematically_absent(
    h: int,
    k: int,
    l: int,
    space_group: str,
    crystal_system: str,
    *,
    apply_glide_rules: bool = True,
) -> bool:
    """Return True when a reflection is forbidden by centering or a known screw axis."""

    if h == 0 and k == 0 and l == 0:
        return True
    letter = centering_letter(space_group)
    if letter == "I" and (h + k + l) % 2 != 0:
        return True
    if letter == "F" and not (h % 2 == k % 2 == l % 2):
        return True
    if letter == "A" and (k + l) % 2 != 0:
        return True
    if letter == "B" and (h + l) % 2 != 0:
        return True
    if letter == "C" and (h + k) % 2 != 0:
        return True
    if letter == "R" and crystal_system in {"Hexagonal", "Trigonal"}:
        if (-h + k + l) % 3 != 0:
            return True
    if not apply_glide_rules:
        return False
    symbol = normalize_space_group(space_group)
    if symbol in {"P63/mmc", "P63mc"}:
        if h == 0 and k == 0 and l % 2 != 0:
            return True
        if h == k and l % 2 != 0:
            return True
    return False


def powder_multiplicity(h: int, k: int, l: int, crystal_system: str) -> int:
    """Number of Laue-equivalent planes, including Friedel mates."""

    if (h, k, l) == (0, 0, 0):
        return 0
    operations = laue_operations(crystal_system)
    return len({_apply_miller(op, h, k, l) for op in operations})


def laue_operations(crystal_system: str) -> tuple[tuple[tuple[int, int, int], ...], ...]:
    """Integer matrices that act on Miller indices for the Laue class."""

    try:
        return _LAUE_OPERATIONS[crystal_system]
    except KeyError as exc:
        known = ", ".join(CRYSTAL_SYSTEMS)
        raise ValueError(
            f"Unknown crystal system '{crystal_system}'. Choose one of: {known}."
        ) from exc


def element_symbol(label: str) -> str:
    """Extract an element symbol from a CIF label such as 'Ti1' or 'Fe2+'."""

    match = _ELEMENT_RE.match(label.strip())
    if match is None:
        raise ValueError(f"Could not read an element symbol from '{label}'.")
    symbol = match.group(1)
    if symbol == "D":
        return "D"
    if symbol not in _ATOMIC_NUMBERS:
        raise ValueError(f"Element '{symbol}' is not recognized.")
    return symbol


def scattering_factor(element: str, s: float) -> float:
    """Atomic scattering factor f(s) for s = sin(theta) / lambda."""

    symbol = element_symbol(element)
    coefficients = _scattering_coefficients().get(symbol)
    if coefficients is None:
        raise ValueError(
            f"No X-ray scattering coefficients are available for {symbol}."
        )
    s2 = s * s
    total = 0.0
    for amplitude, breadth in coefficients:
        total += amplitude * math.exp(-breadth * s2)
    return _ATOMIC_NUMBERS[symbol] - _SCATTERING_CONSTANT * s2 * total


def calculate_pattern(
    lattice: Lattice,
    wavelength: float,
    two_theta_range: tuple[float, float] = (5.0, 90.0),
    atoms: tuple[AtomSite, ...] | list[AtomSite] | None = None,
    space_group: str = "P1",
    b_iso: float = 0.4,
    min_relative_intensity: float = 1.0,
    apply_lorentz_polarization: bool = True,
) -> CalculatedPattern:
    """Calculate a powder pattern.

    When atomic positions are supplied, intensities are |F|^2 multiplied by
    the Lorentz–polarization factor and summed over coincident planes.
    Without positions, each allowed plane contributes equally before that
    correction, so the result tracks multiplicity rather than chemistry.
    """

    if wavelength <= 0:
        raise ValueError("Wavelength must be positive.")
    two_theta_min, two_theta_max = two_theta_range
    if not 0 <= two_theta_min < two_theta_max <= 180:
        raise ValueError("The 2θ range must satisfy 0 ≤ minimum < maximum ≤ 180.")
    if min_relative_intensity < 0:
        raise ValueError("The minimum relative intensity cannot be negative.")
    if b_iso < 0:
        raise ValueError("The isotropic B factor cannot be negative.")

    sin_max = math.sin(math.radians(two_theta_max / 2.0))
    if sin_max <= 0:
        raise ValueError("The maximum 2θ does not produce a usable d-spacing limit.")
    d_min = wavelength / (2.0 * sin_max)
    longest = max(lattice.a, lattice.b, lattice.c)
    # Hexagonal Laue images such as (h, k) -> (h + k, -h) need a wider index box
    # than the shortest in-range d-spacing alone.
    index_factor = 2 if lattice.crystal_system in {"Hexagonal", "Trigonal"} else 1
    raw_limit = index_factor * (int(math.ceil(longest / d_min)) + 1)
    index_limit = max(1, min(raw_limit, _MAX_INDEX))
    truncated = raw_limit > _MAX_INDEX

    indices = np.arange(-index_limit, index_limit + 1)
    grid_h, grid_k, grid_l = np.meshgrid(indices, indices, indices, indexing="ij")
    h = grid_h.ravel().astype(int)
    k = grid_k.ravel().astype(int)
    l = grid_l.ravel().astype(int)
    keep = ~((h == 0) & (k == 0) & (l == 0))
    keep &= ~_absence_mask(h, k, l, space_group, lattice.crystal_system)
    h, k, l = h[keep], k[keep], l[keep]
    if len(h) == 0:
        return CalculatedPattern((), wavelength, bool(atoms), index_limit, truncated)

    reciprocal = np.linalg.inv(lattice.metric_tensor())
    vectors = np.column_stack([h, k, l]).astype(float)
    inv_d2 = np.einsum("ni,ij,nj->n", vectors, reciprocal, vectors)
    positive = inv_d2 > 1e-12
    h, k, l = h[positive], k[positive], l[positive]
    inv_d2 = inv_d2[positive]
    d_values = 1.0 / np.sqrt(inv_d2)
    sin_theta = wavelength / (2.0 * d_values)
    in_range = (d_values >= d_min - 1e-8) & (sin_theta > 0) & (sin_theta < 1.0)
    h, k, l = h[in_range], k[in_range], l[in_range]
    d_values = d_values[in_range]
    sin_theta = np.clip(sin_theta[in_range], 0.0, 1.0)
    theta = np.arcsin(sin_theta)
    two_theta = np.degrees(2.0 * theta)
    visible = (two_theta >= two_theta_min) & (two_theta <= two_theta_max)
    h, k, l = h[visible], k[visible], l[visible]
    d_values = d_values[visible]
    theta = theta[visible]
    two_theta = two_theta[visible]
    if len(h) == 0:
        return CalculatedPattern((), wavelength, bool(atoms), index_limit, truncated)

    s = np.sin(theta) / wavelength
    s2 = s * s
    if apply_lorentz_polarization:
        lorentz = (1.0 + np.cos(2.0 * theta) ** 2) / (
            np.sin(theta) ** 2 * np.cos(theta)
        )
    else:
        lorentz = np.ones_like(two_theta)

    site_list = tuple(atoms or ())
    if site_list:
        amplitude2 = _structure_factor_intensity(h, k, l, s2, site_list, b_iso)
    else:
        amplitude2 = np.ones(len(h), dtype=float)

    finite = np.isfinite(amplitude2) & np.isfinite(lorentz) & (amplitude2 > 0)
    if not np.any(finite):
        return CalculatedPattern((), wavelength, bool(site_list), index_limit, truncated)
    max_amplitude = float(np.max(amplitude2[finite]))
    significant = finite & (amplitude2 >= max_amplitude * 1e-8)
    h, k, l = h[significant], k[significant], l[significant]
    d_values = d_values[significant]
    two_theta = two_theta[significant]
    intensity = amplitude2[significant] * lorentz[significant]

    families = _group_laue_families(
        h, k, l, d_values, two_theta, intensity, lattice.crystal_system
    )
    merged = _merge_coincident(families, _COINCIDENT_TOLERANCE)
    if not merged:
        return CalculatedPattern((), wavelength, bool(site_list), index_limit, truncated)

    maximum = max(item.intensity for item in merged)
    if maximum <= 0:
        return CalculatedPattern((), wavelength, bool(site_list), index_limit, truncated)
    scaled = []
    for item in merged:
        relative = 100.0 * item.intensity / maximum
        if relative >= min_relative_intensity:
            scaled.append(
                Reflection(
                    two_theta=item.two_theta,
                    d_spacing=item.d_spacing,
                    intensity=relative,
                    multiplicity=item.multiplicity,
                    hkls=item.hkls,
                )
            )
    scaled.sort(key=lambda item: (item.two_theta, item.hkl_label))
    return CalculatedPattern(
        tuple(scaled),
        wavelength,
        bool(site_list),
        index_limit,
        truncated,
    )


def parse_cif(text: str) -> ParsedCIF:
    """Parse cell, space group, formula, and atoms from the first CIF data block."""

    tags, loops, data_name = _parse_cif_blocks(text)
    warnings: list[str] = []

    def require(name: str) -> float:
        raw = tags.get(name)
        if raw is None:
            raise ValueError(
                f"The CIF is missing {name}. A unit cell and its angles are required."
            )
        value = _parse_cif_float(raw)
        if value is None:
            raise ValueError(f"The CIF value for {name} is not a number.")
        return value

    a = require("_cell_length_a")
    b = require("_cell_length_b")
    c = require("_cell_length_c")
    alpha = require("_cell_angle_alpha")
    beta = require("_cell_angle_beta")
    gamma = require("_cell_angle_gamma")

    space_group = ""
    for key in (
        "_space_group_name_h-m_alt",
        "_symmetry_space_group_name_h-m",
        "_space_group_name_h-m",
    ):
        if tags.get(key):
            space_group = normalize_space_group(tags[key])
            break
    number = None
    for key in ("_space_group_it_number", "_symmetry_int_tables_number"):
        if tags.get(key):
            parsed = _parse_cif_float(tags[key])
            if parsed is not None:
                number = int(round(parsed))
            break
    if not space_group and number is None:
        warnings.append(
            "The CIF has no space-group symbol. Centering is treated as primitive."
        )
        space_group = "P1"
    elif not space_group and number is not None:
        space_group = f"No. {number}"
        warnings.append(
            "The CIF gives a space-group number without a Hermann–Mauguin symbol. "
            "Lattice centering could not be inferred."
        )

    formula = _clean_formula(
        tags.get("_chemical_formula_sum")
        or tags.get("_chemical_formula_structural")
        or ""
    )
    cell_system = infer_crystal_system(a, b, c, alpha, beta, gamma)
    crystal_system = crystal_system_from_space_group(cell_system, space_group, number)

    asym_atoms = _atoms_from_loops(loops)
    operations = _symmetry_operations(loops)
    if operations:
        atoms = _expand_sites(asym_atoms, operations)
    else:
        atoms = asym_atoms
        if asym_atoms:
            warnings.append(
                "No symmetry operations were listed, so atomic positions are used exactly as written."
            )

    return ParsedCIF(
        data_name=data_name,
        formula=formula,
        space_group=space_group,
        space_group_number=number,
        crystal_system=crystal_system,
        a=a,
        b=b,
        c=c,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        atoms=atoms,
        asymmetric_site_count=len(asym_atoms),
        symmetry_operation_count=len(operations),
        warnings=tuple(warnings),
    )


def _matmul(
    left: tuple[tuple[int, int, int], ...], right: tuple[tuple[int, int, int], ...]
) -> tuple[tuple[int, int, int], ...]:
    return tuple(
        tuple(sum(left[i][k] * right[k][j] for k in range(3)) for j in range(3))
        for i in range(3)
    )


def _closure(
    generators: list[tuple[tuple[int, int, int], ...]],
) -> tuple[tuple[tuple[int, int, int], ...], ...]:
    identity = ((1, 0, 0), (0, 1, 0), (0, 0, 1))
    group = {identity}
    changed = True
    while changed:
        changed = False
        snapshot = list(group)
        for current in snapshot:
            for generator in generators:
                product = _matmul(current, generator)
                if product not in group:
                    group.add(product)
                    changed = True
    return tuple(sorted(group))


def _build_laue_groups() -> dict[str, tuple[tuple[tuple[int, int, int], ...], ...]]:
    inversion = ((-1, 0, 0), (0, -1, 0), (0, 0, -1))
    four_fold = ((0, -1, 0), (1, 0, 0), (0, 0, 1))
    three_fold_cubic = ((0, 1, 0), (0, 0, 1), (1, 0, 0))
    six_fold = ((0, -1, 0), (1, 1, 0), (0, 0, 1))
    swap_hk = ((0, 1, 0), (1, 0, 0), (0, 0, 1))
    flip_l = ((1, 0, 0), (0, 1, 0), (0, 0, -1))
    mirror_x = ((-1, 0, 0), (0, 1, 0), (0, 0, 1))
    mirror_y = ((1, 0, 0), (0, -1, 0), (0, 0, 1))
    two_fold_b = ((-1, 0, 0), (0, 1, 0), (0, 0, -1))
    trigonal_three = ((-1, -1, 0), (1, 0, 0), (0, 0, 1))
    trigonal_mirror = ((0, -1, 0), (-1, 0, 0), (0, 0, 1))
    return {
        "Cubic": _closure([four_fold, three_fold_cubic, inversion]),
        "Tetragonal": _closure([four_fold, swap_hk, inversion]),
        "Orthorhombic": _closure([mirror_x, mirror_y, flip_l]),
        "Hexagonal": _closure([six_fold, swap_hk, flip_l]),
        "Trigonal": _closure([trigonal_three, trigonal_mirror, inversion]),
        "Rhombohedral": _closure([three_fold_cubic, swap_hk, inversion]),
        "Monoclinic": _closure([two_fold_b, inversion]),
        "Triclinic": _closure([inversion]),
    }


_LAUE_OPERATIONS = _build_laue_groups()


def _apply_miller(
    operation: tuple[tuple[int, int, int], ...], h: int, k: int, l: int
) -> tuple[int, int, int]:
    return (
        operation[0][0] * h + operation[0][1] * k + operation[0][2] * l,
        operation[1][0] * h + operation[1][1] * k + operation[1][2] * l,
        operation[2][0] * h + operation[2][1] * k + operation[2][2] * l,
    )


def _absence_mask(
    h: np.ndarray,
    k: np.ndarray,
    l: np.ndarray,
    space_group: str,
    crystal_system: str,
) -> np.ndarray:
    absent = np.zeros(h.shape, dtype=bool)
    letter = centering_letter(space_group)
    if letter == "I":
        absent |= (h + k + l) % 2 != 0
    elif letter == "F":
        absent |= ~((h % 2 == k % 2) & (k % 2 == l % 2))
    elif letter == "A":
        absent |= (k + l) % 2 != 0
    elif letter == "B":
        absent |= (h + l) % 2 != 0
    elif letter == "C":
        absent |= (h + k) % 2 != 0
    elif letter == "R" and crystal_system in {"Hexagonal", "Trigonal"}:
        absent |= (-h + k + l) % 3 != 0
    symbol = normalize_space_group(space_group)
    if symbol in {"P63/mmc", "P63mc"}:
        absent |= (h == 0) & (k == 0) & (l % 2 != 0)
        absent |= (h == k) & (l % 2 != 0)
    return absent


def _scattering_coefficients() -> dict[str, list[list[float]]]:
    global _SCATTERING_CACHE
    if _SCATTERING_CACHE is None:
        path = Path(__file__).with_name("scattering_factors.json")
        with path.open(encoding="utf-8") as handle:
            _SCATTERING_CACHE = json.load(handle)
    return _SCATTERING_CACHE


_SCATTERING_CACHE: dict[str, list[list[float]]] | None = None


def _structure_factor_intensity(
    h: np.ndarray,
    k: np.ndarray,
    l: np.ndarray,
    s2: np.ndarray,
    atoms: tuple[AtomSite, ...],
    default_b: float,
) -> np.ndarray:
    symbols = []
    coordinates = []
    occupancies = []
    b_factors = []
    for atom in atoms:
        if atom.occupancy == 0:
            continue
        symbols.append(element_symbol(atom.element))
        coordinates.append((atom.x, atom.y, atom.z))
        occupancies.append(atom.occupancy)
        b_factors.append(atom.b_iso if atom.b_iso > 0 else default_b)
    if not symbols:
        return np.zeros(len(h), dtype=float)

    coefficients = _scattering_coefficients()
    missing = sorted({symbol for symbol in symbols if symbol not in coefficients})
    if missing:
        names = ", ".join(missing)
        raise ValueError(f"No X-ray scattering coefficients are available for {names}.")

    coeff = np.array([coefficients[symbol] for symbol in symbols], dtype=float)
    atomic_numbers = np.array([_ATOMIC_NUMBERS[symbol] for symbol in symbols], dtype=float)
    coords = np.array(coordinates, dtype=float)
    occupancy = np.array(occupancies, dtype=float)
    thermal = np.array(b_factors, dtype=float)

    # coeff shape is (atoms, 4, 2): amplitude, breadth.
    decay = np.exp(-coeff[None, :, :, 1] * s2[:, None, None])
    summed = np.sum(coeff[None, :, :, 0] * decay, axis=2)
    form = atomic_numbers[None, :] - _SCATTERING_CONSTANT * s2[:, None] * summed
    form *= np.exp(-thermal[None, :] * s2[:, None])
    form *= occupancy[None, :]

    miller = np.column_stack([h, k, l]).astype(float)
    phase = np.exp(2j * np.pi * (miller @ coords.T))
    structure = np.sum(form * phase, axis=1)
    return np.abs(structure) ** 2


def _display_better(
    h1: np.ndarray,
    k1: np.ndarray,
    l1: np.ndarray,
    h2: np.ndarray,
    k2: np.ndarray,
    l2: np.ndarray,
) -> np.ndarray:
    """True where (h2, k2, l2) is the preferred display index."""

    keys1 = ((h1 >= 0), (k1 >= 0), (l1 >= 0), h1, k1, l1)
    keys2 = ((h2 >= 0), (k2 >= 0), (l2 >= 0), h2, k2, l2)
    better = np.zeros(h1.shape, dtype=bool)
    undecided = np.ones(h1.shape, dtype=bool)
    for left, right in zip(keys1, keys2):
        left_i = left.astype(np.int64)
        right_i = right.astype(np.int64)
        better |= undecided & (right_i > left_i)
        undecided &= left_i == right_i
    return better


def _group_laue_families(
    h: np.ndarray,
    k: np.ndarray,
    l: np.ndarray,
    d_values: np.ndarray,
    two_theta: np.ndarray,
    intensity: np.ndarray,
    crystal_system: str,
) -> list[Reflection]:
    rep_h = h.copy()
    rep_k = k.copy()
    rep_l = l.copy()
    for operation in laue_operations(crystal_system):
        matrix = np.array(operation, dtype=int)
        stacked = np.column_stack([h, k, l]) @ matrix.T
        cand_h = stacked[:, 0]
        cand_k = stacked[:, 1]
        cand_l = stacked[:, 2]
        replace = _display_better(rep_h, rep_k, rep_l, cand_h, cand_k, cand_l)
        rep_h = np.where(replace, cand_h, rep_h)
        rep_k = np.where(replace, cand_k, rep_k)
        rep_l = np.where(replace, cand_l, rep_l)

    grouped: dict[tuple[int, int, int], list[int]] = {}
    for index, key in enumerate(zip(rep_h.tolist(), rep_k.tolist(), rep_l.tolist())):
        grouped.setdefault(key, []).append(index)

    families: list[Reflection] = []
    for key, members in grouped.items():
        member_index = np.array(members)
        weight = intensity[member_index]
        total = float(np.sum(weight))
        if total <= 0 or not math.isfinite(total):
            continue
        families.append(
            Reflection(
                two_theta=float(np.average(two_theta[member_index], weights=weight)),
                d_spacing=float(np.average(d_values[member_index], weights=weight)),
                intensity=total,
                multiplicity=len(members),
                hkls=(key,),
            )
        )
    return families


def _merge_coincident(reflections: list[Reflection], tolerance: float) -> list[Reflection]:
    ordered = sorted(reflections, key=lambda item: item.two_theta)
    merged: list[Reflection] = []
    for item in ordered:
        if merged and abs(item.two_theta - merged[-1].two_theta) <= tolerance:
            previous = merged[-1]
            total = previous.intensity + item.intensity
            merged[-1] = Reflection(
                two_theta=(
                    previous.two_theta * previous.intensity
                    + item.two_theta * item.intensity
                )
                / total,
                d_spacing=(
                    previous.d_spacing * previous.intensity
                    + item.d_spacing * item.intensity
                )
                / total,
                intensity=total,
                multiplicity=previous.multiplicity + item.multiplicity,
                hkls=previous.hkls + item.hkls,
            )
        else:
            merged.append(item)
    return merged


def _parse_cif_blocks(
    text: str,
) -> tuple[dict[str, str], list[tuple[list[str], list[list[str]]]], str]:
    tokens = _cif_tokens(text)
    tags: dict[str, str] = {}
    loops: list[tuple[list[str], list[list[str]]]] = []
    data_name = ""
    seen_data = False
    index = 0
    while index < len(tokens):
        token = tokens[index]
        lowered = token.lower()
        if lowered.startswith("data_"):
            if seen_data:
                break
            seen_data = True
            data_name = token[5:]
            index += 1
            continue
        if not seen_data:
            index += 1
            continue
        if lowered == "loop_":
            index += 1
            columns: list[str] = []
            while index < len(tokens) and tokens[index].startswith("_"):
                columns.append(tokens[index].lower())
                index += 1
            rows: list[list[str]] = []
            while index < len(tokens):
                current = tokens[index]
                current_lower = current.lower()
                if (
                    current.startswith("_")
                    or current_lower == "loop_"
                    or current_lower.startswith("data_")
                ):
                    break
                row = tokens[index : index + len(columns)]
                if len(row) < len(columns):
                    break
                rows.append(row)
                index += len(columns)
            if columns:
                loops.append((columns, rows))
            continue
        if token.startswith("_"):
            index += 1
            if index < len(tokens):
                tags[lowered] = tokens[index]
                index += 1
            continue
        index += 1
    if not seen_data:
        raise ValueError("The file does not contain a CIF data block.")
    return tags, loops, data_name


def _cif_tokens(text: str) -> list[str]:
    source = text.replace("\r\n", "\n").replace("\r", "\n")
    tokens: list[str] = []
    length = len(source)
    index = 0
    while index < length:
        char = source[index]
        if char in " \t\n":
            index += 1
            continue
        if char == "#":
            newline = source.find("\n", index)
            index = length if newline < 0 else newline + 1
            continue
        if char in "'\"":
            quote = char
            index += 1
            start = index
            while index < length and source[index] != quote:
                index += 1
            tokens.append(source[start:index])
            index = min(length, index + 1)
            continue
        if char == ";":
            line_start = source.rfind("\n", 0, index) + 1
            if source[line_start:index].strip() == "":
                index += 1
                if index < length and source[index] == "\n":
                    index += 1
                end = index
                while end < length:
                    newline = source.find("\n", end)
                    if newline < 0:
                        tokens.append(source[index:].strip())
                        index = length
                        break
                    if source[end:newline].strip() == ";":
                        tokens.append(source[index:end].strip())
                        index = newline + 1
                        break
                    end = newline + 1
                else:
                    tokens.append(source[index:].strip())
                    index = length
                continue
        start = index
        while index < length and source[index] not in " \t\n":
            index += 1
        tokens.append(source[start:index])
    return tokens


def _parse_cif_float(token: str) -> float | None:
    text = token.strip().strip("'\"")
    if text in {".", "?"}:
        return None
    base = re.sub(r"\(.*\)$", "", text)
    if _CIF_NUMBER_RE.fullmatch(base) is None:
        return None
    return float(base)


def _clean_formula(raw: str) -> str:
    text = raw.strip().strip("'\"")
    text = text.replace("~", "")
    return re.sub(r"\s+", " ", text).strip()


def _atoms_from_loops(
    loops: list[tuple[list[str], list[list[str]]]],
) -> tuple[AtomSite, ...]:
    for columns, rows in loops:
        if "_atom_site_fract_x" not in columns:
            continue
        x_index = columns.index("_atom_site_fract_x")
        y_index = columns.index("_atom_site_fract_y")
        z_index = columns.index("_atom_site_fract_z")
        symbol_index = _column(columns, "_atom_site_type_symbol")
        label_index = _column(columns, "_atom_site_label")
        occ_index = _column(columns, "_atom_site_occupancy")
        b_index = _column(columns, "_atom_site_b_iso_or_equiv")
        u_index = _column(columns, "_atom_site_u_iso_or_equiv")
        atoms: list[AtomSite] = []
        for row in rows:
            symbol_token = ""
            if symbol_index is not None:
                symbol_token = row[symbol_index]
            elif label_index is not None:
                symbol_token = row[label_index]
            if not symbol_token:
                continue
            x = _parse_cif_float(row[x_index])
            y = _parse_cif_float(row[y_index])
            z = _parse_cif_float(row[z_index])
            if x is None or y is None or z is None:
                continue
            occupancy = 1.0
            if occ_index is not None:
                parsed = _parse_cif_float(row[occ_index])
                if parsed is not None:
                    occupancy = parsed
            b_iso = 0.0
            if b_index is not None:
                parsed = _parse_cif_float(row[b_index])
                if parsed is not None:
                    b_iso = parsed
            elif u_index is not None:
                parsed = _parse_cif_float(row[u_index])
                if parsed is not None:
                    b_iso = 8.0 * math.pi**2 * parsed
            atoms.append(
                AtomSite(element_symbol(symbol_token), x % 1.0, y % 1.0, z % 1.0, occupancy, b_iso)
            )
        return tuple(atoms)
    return ()


def _column(columns: list[str], name: str) -> int | None:
    return columns.index(name) if name in columns else None


def _symmetry_operations(
    loops: list[tuple[list[str], list[list[str]]]],
) -> list[tuple[np.ndarray, np.ndarray]]:
    names = ("_symmetry_equiv_pos_as_xyz", "_space_group_symop_operation_xyz")
    for columns, rows in loops:
        index = next((columns.index(name) for name in names if name in columns), None)
        if index is None:
            continue
        return [_parse_symmetry_operation(row[index]) for row in rows]
    return []


def _parse_symmetry_operation(expression: str) -> tuple[np.ndarray, np.ndarray]:
    components = [part.strip() for part in expression.split(",")]
    if len(components) != 3:
        raise ValueError(f"Could not read the symmetry operation '{expression}'.")
    rotation = np.zeros((3, 3), dtype=float)
    translation = np.zeros(3, dtype=float)
    for axis, component in enumerate(components):
        prepared = component.lower().replace(" ", "")
        prepared = re.sub(
            r"(\d+(?:\.\d+)?)/(\d+(?:\.\d+)?)",
            lambda match: str(float(match.group(1)) / float(match.group(2))),
            prepared,
        )
        for term in re.findall(r"[+-]?[^+-]+", prepared):
            sign = -1.0 if term.startswith("-") else 1.0
            body = term[1:] if term[0] in "+-" else term
            body = body.replace("*", "")
            matched_variable = False
            for variable, column in (("x", 0), ("y", 1), ("z", 2)):
                if variable in body:
                    factor = body.replace(variable, "")
                    rotation[axis, column] += sign * (float(factor) if factor else 1.0)
                    matched_variable = True
                    break
            if not matched_variable and body:
                translation[axis] += sign * float(body)
    return rotation, translation


def _expand_sites(
    atoms: tuple[AtomSite, ...],
    operations: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[AtomSite, ...]:
    expanded: list[AtomSite] = []
    seen: set[tuple[str, float, float, float]] = set()
    for atom in atoms:
        coordinate = np.array([atom.x, atom.y, atom.z], dtype=float)
        for rotation, translation in operations:
            image = (rotation @ coordinate + translation) % 1.0
            image = np.where(np.isclose(image, 1.0), 0.0, image)
            key = (
                atom.element,
                round(float(image[0]), 5),
                round(float(image[1]), 5),
                round(float(image[2]), 5),
            )
            if key in seen:
                continue
            seen.add(key)
            expanded.append(
                AtomSite(atom.element, key[1], key[2], key[3], atom.occupancy, atom.b_iso)
            )
    return tuple(expanded)
