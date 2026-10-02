"""Built-in phases for powder XRD screening.

Ti2AlN lattice constants are the powder values a = 2.989 Å and c = 13.614 Å
(a = 2.989(2) Å, c = 13.614(5) Å). Nb2AlC uses the ambient cell
a = 3.107 Å and c = 13.888 Å reported for material prepared by reactive
hot isostatic pressing. Internal z coordinates are representative 211 MAX
parameters, so calculated intensities are estimates rather than a Rietveld
refinement of a particular sample.

Impurity cells are standard room-temperature structures used as search
candidates. Edit the primary cell in the app when your sample differs.
"""

from __future__ import annotations

from dataclasses import dataclass

from crystallography import AtomSite

CUSTOM_MATERIAL = "Custom (Other)"


@dataclass(frozen=True)
class Phase:
    """A crystalline phase that can be used as the sample or as a candidate."""

    key: str
    name: str
    formula: str
    crystal_system: str
    space_group: str
    space_group_number: int
    a: float
    b: float
    c: float
    alpha: float
    beta: float
    gamma: float
    atoms: tuple[AtomSite, ...]
    notes: str


@dataclass(frozen=True)
class MaterialPreset:
    """A catalog material and the impurity phases commonly checked with it."""

    key: str
    name: str
    description: str
    primary: Phase
    impurity_keys: tuple[str, ...]


def _site(element: str, x: float, y: float, z: float, occupancy: float = 1.0) -> AtomSite:
    return AtomSite(element, x % 1.0, y % 1.0, z % 1.0, occupancy)


def _max_211(metal: str, a_element: str, x_element: str, z: float) -> tuple[AtomSite, ...]:
    """Full P6₃/mmc cell of a 211 MAX phase. X is on 2a, A on 2c, and M on 4f."""

    return (
        _site(metal, 1 / 3, 2 / 3, z),
        _site(metal, 2 / 3, 1 / 3, z + 0.5),
        _site(metal, 2 / 3, 1 / 3, -z),
        _site(metal, 1 / 3, 2 / 3, 0.5 - z),
        _site(a_element, 1 / 3, 2 / 3, 0.25),
        _site(a_element, 2 / 3, 1 / 3, 0.75),
        _site(x_element, 0.0, 0.0, 0.0),
        _site(x_element, 0.0, 0.0, 0.5),
    )


def _fcc(element: str) -> tuple[AtomSite, ...]:
    return tuple(
        _site(element, *point)
        for point in ((0, 0, 0), (0.5, 0.5, 0), (0.5, 0, 0.5), (0, 0.5, 0.5))
    )


def _rocksalt(cation: str, anion: str) -> tuple[AtomSite, ...]:
    cations = ((0, 0, 0), (0.5, 0.5, 0), (0.5, 0, 0.5), (0, 0.5, 0.5))
    anions = ((0.5, 0.5, 0.5), (0, 0, 0.5), (0, 0.5, 0), (0.5, 0, 0))
    return tuple(_site(cation, *point) for point in cations) + tuple(
        _site(anion, *point) for point in anions
    )


def _hcp(element: str) -> tuple[AtomSite, ...]:
    return (
        _site(element, 1 / 3, 2 / 3, 0.25),
        _site(element, 2 / 3, 1 / 3, 0.75),
    )


def _bcc(element: str) -> tuple[AtomSite, ...]:
    return (_site(element, 0, 0, 0), _site(element, 0.5, 0.5, 0.5))


def _wurtzite(cation: str, anion: str, u: float) -> tuple[AtomSite, ...]:
    return (
        _site(cation, 1 / 3, 2 / 3, 0.0),
        _site(cation, 2 / 3, 1 / 3, 0.5),
        _site(anion, 1 / 3, 2 / 3, u),
        _site(anion, 2 / 3, 1 / 3, u + 0.5),
    )


def _l10(first: str, second: str) -> tuple[AtomSite, ...]:
    return (
        _site(first, 0, 0, 0),
        _site(first, 0.5, 0.5, 0),
        _site(second, 0.5, 0, 0.5),
        _site(second, 0, 0.5, 0.5),
    )


def _ti3al() -> tuple[AtomSite, ...]:
    x = 5 / 6
    titanium = (
        (x, 2 * x, 0.25),
        (-2 * x, -x, 0.25),
        (x, -x, 0.25),
        (-x, -2 * x, 0.75),
        (2 * x, x, 0.75),
        (-x, x, 0.75),
    )
    aluminum = ((1 / 3, 2 / 3, 0.25), (2 / 3, 1 / 3, 0.75))
    return tuple(_site("Ti", *point) for point in titanium) + tuple(
        _site("Al", *point) for point in aluminum
    )


def _al3nb() -> tuple[AtomSite, ...]:
    niobium = ((0, 0, 0), (0.5, 0.5, 0.5))
    aluminum = (
        (0, 0, 0.5),
        (0.5, 0.5, 0),
        (0, 0.5, 0.25),
        (0.5, 0, 0.25),
        (0.5, 0, 0.75),
        (0, 0.5, 0.75),
    )
    return tuple(_site("Nb", *point) for point in niobium) + tuple(
        _site("Al", *point) for point in aluminum
    )


def _cubic(a: float) -> tuple[float, float, float, float, float, float]:
    return (a, a, a, 90.0, 90.0, 90.0)


def _hexagonal(a: float, c: float) -> tuple[float, float, float, float, float, float]:
    return (a, a, c, 90.0, 90.0, 120.0)


def _tetragonal(a: float, c: float) -> tuple[float, float, float, float, float, float]:
    return (a, a, c, 90.0, 90.0, 90.0)


def _make(
    key: str,
    formula: str,
    crystal_system: str,
    space_group: str,
    number: int,
    cell: tuple[float, float, float, float, float, float],
    atoms: tuple[AtomSite, ...],
    notes: str,
    name: str | None = None,
) -> Phase:
    a, b, c, alpha, beta, gamma = cell
    return Phase(
        key=key,
        name=name or key,
        formula=formula,
        crystal_system=crystal_system,
        space_group=space_group,
        space_group_number=number,
        a=a,
        b=b,
        c=c,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        atoms=atoms,
        notes=notes,
    )


PHASES: dict[str, Phase] = {
    "Ti2AlN": _make(
        "Ti2AlN",
        "Ti2AlN",
        "Hexagonal",
        "P63/mmc",
        194,
        _hexagonal(2.989, 13.614),
        _max_211("Ti", "Al", "N", 0.085),
        "211 MAX phase. Powder lattice a = 2.989 Å, c = 13.614 Å. "
        "Ti on 4f with a representative z = 0.085.",
    ),
    "Nb2AlC": _make(
        "Nb2AlC",
        "Nb2AlC",
        "Hexagonal",
        "P63/mmc",
        194,
        _hexagonal(3.107, 13.888),
        _max_211("Nb", "Al", "C", 0.089),
        "211 MAX phase. Ambient lattice a = 3.107 Å, c = 13.888 Å. "
        "Nb on 4f with a representative z = 0.089.",
    ),
    "TiN": _make(
        "TiN",
        "TiN",
        "Cubic",
        "Fm-3m",
        225,
        _cubic(4.241),
        _rocksalt("Ti", "N"),
        "Rocksalt titanium nitride, a common secondary phase.",
    ),
    "Ti": _make(
        "Ti",
        "Ti",
        "Hexagonal",
        "P63/mmc",
        194,
        _hexagonal(2.951, 4.686),
        _hcp("Ti"),
        "Unreacted α-Ti.",
        name="Ti",
    ),
    "Al": _make(
        "Al",
        "Al",
        "Cubic",
        "Fm-3m",
        225,
        _cubic(4.0495),
        _fcc("Al"),
        "Unreacted aluminum.",
    ),
    "AlN": _make(
        "AlN",
        "AlN",
        "Hexagonal",
        "P63mc",
        186,
        _hexagonal(3.111, 4.978),
        _wurtzite("Al", "N", 0.382),
        "Wurtzite aluminum nitride.",
    ),
    "TiAl": _make(
        "TiAl",
        "TiAl",
        "Tetragonal",
        "P4/mmm",
        123,
        _tetragonal(4.001, 4.071),
        _l10("Ti", "Al"),
        "γ-TiAl, L1₀, a competing intermetallic.",
    ),
    "Ti3Al": _make(
        "Ti3Al",
        "Ti3Al",
        "Hexagonal",
        "P63/mmc",
        194,
        _hexagonal(5.780, 4.647),
        _ti3al(),
        "α₂-Ti3Al, D0₁₉, with Ti at x = 5/6.",
    ),
    "NbC": _make(
        "NbC",
        "NbC",
        "Cubic",
        "Fm-3m",
        225,
        _cubic(4.470),
        _rocksalt("Nb", "C"),
        "Rocksalt niobium carbide, a common secondary phase.",
    ),
    "Nb": _make(
        "Nb",
        "Nb",
        "Cubic",
        "Im-3m",
        229,
        _cubic(3.300),
        _bcc("Nb"),
        "Unreacted body-centered cubic niobium.",
    ),
    "Al3Nb": _make(
        "Al3Nb",
        "Al3Nb",
        "Tetragonal",
        "I4/mmm",
        139,
        _tetragonal(3.844, 8.605),
        _al3nb(),
        "NbAl3, D0₂₂, a competing phase in the Nb–Al–C system.",
    ),
}


PRESETS: dict[str, MaterialPreset] = {
    "Ti2AlN": MaterialPreset(
        key="Ti2AlN",
        name="Ti2AlN",
        description="Hexagonal MAX phase. Typical impurity checks: TiN, unreacted Ti, and Al.",
        primary=PHASES["Ti2AlN"],
        impurity_keys=("TiN", "Ti", "Al", "AlN", "TiAl", "Ti3Al"),
    ),
    "Nb2AlC": MaterialPreset(
        key="Nb2AlC",
        name="Nb2AlC",
        description="Hexagonal MAX phase. Typical impurity checks: NbC, unreacted Nb, and Al.",
        primary=PHASES["Nb2AlC"],
        impurity_keys=("NbC", "Nb", "Al", "Al3Nb"),
    ),
}


def material_choices() -> list[str]:
    return [*PRESETS.keys(), CUSTOM_MATERIAL]


def get_preset(name: str) -> MaterialPreset | None:
    return PRESETS.get(name)


def get_phase(key: str) -> Phase:
    try:
        return PHASES[key]
    except KeyError as exc:
        raise KeyError(f"No catalog phase is named '{key}'.") from exc


def screening_catalog(primary_key: str | None = None) -> list[Phase]:
    """Catalog phases that can be tested as impurities, primary excluded."""

    ordered: list[Phase] = []
    seen: set[str] = set()
    preset = PRESETS.get(primary_key or "")
    if preset is not None:
        for key in preset.impurity_keys:
            ordered.append(PHASES[key])
            seen.add(key)
    for key, phase in PHASES.items():
        if key == primary_key or key in seen:
            continue
        ordered.append(phase)
    return ordered
