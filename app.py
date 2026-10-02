"""Powder XRD analysis: reference patterns, peak picking, and impurity screening."""

from __future__ import annotations

import re

import pandas as pd
import streamlit as st

from crystallography import (
    CRYSTAL_SYSTEMS,
    RADIATIONS,
    SYSTEM_SPECS,
    AtomSite,
    CalculatedPattern,
    calculate_pattern,
    constrained_lattice,
    format_miller_bravais,
    parse_cif,
)
from materials import (
    CUSTOM_MATERIAL,
    PRESETS,
    Phase,
    get_preset,
    material_choices,
    screening_catalog,
)
from plotting import build_pattern_figure, build_score_figure
from xrd_processing import (
    PhaseScore,
    build_ti2aln_example,
    claimed_peak_indices,
    example_xy_text,
    load_powder_pattern,
    peaks_to_frame,
    process_scan,
    score_phase,
    scores_to_frame,
)

st.set_page_config(
    page_title="XRD Analysis",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _ensure_defaults() -> None:
    if st.session_state.get("ready"):
        return
    st.session_state.ready = True
    st.session_state.material_choice = "Ti2AlN"
    st.session_state.applied_material = "Ti2AlN"
    st.session_state.custom_name = "Custom phase"
    st.session_state.radiation = "Cu Kα1"
    st.session_state.custom_wavelength = 1.54056
    st.session_state.tth_min = 5.0
    st.session_state.tth_max = 90.0
    st.session_state.smooth = True
    st.session_state.smooth_window = 11
    st.session_state.snip_iterations = 40
    st.session_state.prominence = 4.0
    st.session_state.min_sep = 0.15
    st.session_state.tolerance = 0.15
    st.session_state.min_intensity = 2.0
    st.session_state.b_iso = 0.4
    st.session_state.use_lp = True
    st.session_state.custom_candidates = []
    st.session_state.example_active = False
    st.session_state.show_net = False
    st.session_state.show_reference = True
    st.session_state.overlay_candidates = True
    st.session_state.cand_name = ""
    st.session_state.cand_system = "Cubic"
    st.session_state.cand_sg = "Pm-3m"
    st.session_state.cand_a = 4.0
    st.session_state.cand_b = 4.0
    st.session_state.cand_c = 4.0
    st.session_state.cand_alpha = 90.0
    st.session_state.cand_beta = 90.0
    st.session_state.cand_gamma = 90.0
    st.session_state.cif_error = ""
    st.session_state.cif_atoms = None
    st.session_state.cif_warnings = []
    st.session_state.structure_source = "Catalog preset Ti2AlN"
    _apply_material("Ti2AlN")


def _set_cell(a: float, b: float, c: float, alpha: float, beta: float, gamma: float) -> None:
    """Store the full cell, including parameters hidden by the crystal system.

    The canonical keys are the values used for the calculation. The ``w_`` keys
    belong to the number inputs. Streamlit drops a widget key while that input
    is off the page, so the canonical value has to survive on its own.
    """

    values = {
        "lat_a": float(a),
        "lat_b": float(b),
        "lat_c": float(c),
        "lat_alpha": float(alpha),
        "lat_beta": float(beta),
        "lat_gamma": float(gamma),
    }
    for key, value in values.items():
        st.session_state[key] = value
        st.session_state[f"w_{key}"] = value


def _bind_number(label: str, canonical: str, *, container=st, **kwargs) -> None:
    """Show a lattice input bound to a canonical session-state value.

    ``value`` is the number Streamlit sends as the widget default. A key that
    was removed while the input was hidden would otherwise come back as
    ``min_value`` in the browser, even when the canonical cell is unchanged.
    """

    widget_key = f"w_{canonical}"
    current = float(st.session_state[canonical])
    returned = container.number_input(label, key=widget_key, value=current, **kwargs)
    st.session_state[canonical] = float(returned)


def _apply_material(name: str) -> None:
    """Load catalog defaults or a blank custom cell into the structure widgets."""

    if name == CUSTOM_MATERIAL:
        st.session_state.crystal_system = "Cubic"
        st.session_state.space_group = "Pm-3m"
        st.session_state.formula = ""
        _set_cell(4.0, 4.0, 4.0, 90.0, 90.0, 90.0)
        st.session_state.cif_atoms = None
        st.session_state.cif_warnings = []
        st.session_state.selected_impurities = []
        st.session_state.structure_source = "Manual entry"
        if not str(st.session_state.get("custom_name", "")).strip():
            st.session_state.custom_name = "Custom phase"
        return

    phase = PRESETS[name].primary
    st.session_state.crystal_system = phase.crystal_system
    st.session_state.space_group = phase.space_group
    st.session_state.formula = phase.formula
    _set_cell(phase.a, phase.b, phase.c, phase.alpha, phase.beta, phase.gamma)
    st.session_state.cif_atoms = None
    st.session_state.cif_warnings = []
    st.session_state.selected_impurities = list(PRESETS[name].impurity_keys)
    st.session_state.structure_source = f"Catalog preset {phase.name}"


def _apply_cif(parsed, filename: str) -> None:
    st.session_state.crystal_system = parsed.crystal_system
    st.session_state.space_group = parsed.space_group
    st.session_state.formula = parsed.formula
    _set_cell(parsed.a, parsed.b, parsed.c, parsed.alpha, parsed.beta, parsed.gamma)
    st.session_state.cif_atoms = parsed.atoms
    st.session_state.cif_warnings = list(parsed.warnings)
    st.session_state.structure_source = f"CIF file {filename}"
    label = parsed.data_name or parsed.formula or filename
    st.session_state.custom_name = re.sub(r"[_]+", " ", label).strip() or "Custom phase"
    st.session_state.material_choice = CUSTOM_MATERIAL
    st.session_state.applied_material = CUSTOM_MATERIAL
    preset = _preset_for_label(st.session_state.custom_name, parsed.formula)
    st.session_state.selected_impurities = list(preset.impurity_keys) if preset else []


def _preset_for_label(name: str, formula: str):
    compact_name = re.sub(r"[^A-Za-z0-9]", "", name).lower()
    compact_formula = re.sub(r"[^A-Za-z0-9]", "", formula).lower()
    for key, preset in PRESETS.items():
        token = key.lower()
        if compact_name == token or compact_formula == token:
            return preset
    return None


def _consume_requests() -> None:
    if st.session_state.get("load_example_request"):
        st.session_state.load_example_request = False
        st.session_state.example_active = True
        _apply_material("Ti2AlN")
        st.session_state.material_choice = "Ti2AlN"
        st.session_state.applied_material = "Ti2AlN"
        st.session_state.radiation = "Cu Kα1"
        st.session_state.tth_min = 5.0
        st.session_state.tth_max = 90.0
    if st.session_state.get("restore_catalog"):
        st.session_state.restore_catalog = False
        choice = st.session_state.get("material_choice", "Ti2AlN")
        if choice in PRESETS:
            _apply_material(choice)


def _wavelength() -> tuple[str, float]:
    label = st.session_state.radiation
    if label == "Custom wavelength":
        return label, float(st.session_state.custom_wavelength)
    table = dict(RADIATIONS)
    return label, float(table[label])


def _material_name() -> str:
    if st.session_state.material_choice == CUSTOM_MATERIAL:
        name = str(st.session_state.custom_name).strip()
        return name or "Custom phase"
    return str(st.session_state.material_choice)


def _atoms_for_primary() -> tuple[AtomSite, ...]:
    if st.session_state.cif_atoms:
        return tuple(st.session_state.cif_atoms)
    preset = get_preset(st.session_state.material_choice)
    if preset is not None:
        return preset.primary.atoms
    return ()


def _pattern_for_phase(phase: Phase, wavelength: float) -> CalculatedPattern:
    return _pattern_for_cell(
        phase.crystal_system,
        phase.a,
        phase.b,
        phase.c,
        phase.alpha,
        phase.beta,
        phase.gamma,
        phase.space_group,
        phase.atoms,
        wavelength,
    )


def _pattern_for_cell(
    crystal_system: str,
    a: float,
    b: float,
    c: float,
    alpha: float,
    beta: float,
    gamma: float,
    space_group: str,
    atoms: tuple[AtomSite, ...],
    wavelength: float,
) -> CalculatedPattern:
    lattice = constrained_lattice(crystal_system, a, b, c, alpha, beta, gamma)
    return calculate_pattern(
        lattice,
        wavelength,
        (float(st.session_state.tth_min), float(st.session_state.tth_max)),
        atoms,
        space_group,
        b_iso=float(st.session_state.b_iso),
        min_relative_intensity=float(st.session_state.min_intensity),
        apply_lorentz_polarization=bool(st.session_state.use_lp),
    )


def _slug(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", name.strip())
    return cleaned.strip("_") or "xrd"


def _render_sidebar() -> None:
    st.sidebar.title("XRD Analysis")
    st.sidebar.caption("Crystallography, peak picking, and impurity screening.")

    st.sidebar.header("Radiation")
    radiation_labels = [label for label, _value in RADIATIONS] + ["Custom wavelength"]
    st.sidebar.selectbox("Source", radiation_labels, key="radiation")
    if st.session_state.radiation == "Custom wavelength":
        st.sidebar.number_input(
            "Wavelength (Å)",
            min_value=0.1,
            max_value=10.0,
            step=0.00001,
            format="%.5f",
            key="custom_wavelength",
        )
    left, right = st.sidebar.columns(2)
    left.number_input("2θ min (°)", min_value=0.0, max_value=170.0, step=1.0, key="tth_min")
    right.number_input("2θ max (°)", min_value=5.0, max_value=180.0, step=1.0, key="tth_max")

    st.sidebar.header("Material")
    cif_file = st.sidebar.file_uploader(
        "Crystallographic information file",
        type=["cif"],
        help="Reads the unit cell, space group, formula, and atomic positions.",
    )
    if cif_file is not None:
        token = f"{cif_file.name}:{cif_file.size}"
        if token != st.session_state.get("cif_token"):
            try:
                parsed = parse_cif(cif_file.getvalue().decode("utf-8", errors="replace"))
            except Exception as exc:
                st.session_state.cif_token = token
                st.session_state.cif_error = str(exc)
            else:
                _apply_cif(parsed, cif_file.name)
                st.session_state.cif_error = ""
                st.session_state.cif_token = token
                st.rerun()
    if st.session_state.get("cif_error"):
        st.sidebar.error(st.session_state.cif_error)

    choice = st.sidebar.selectbox("Catalog", material_choices(), key="material_choice")
    if choice != st.session_state.applied_material:
        _apply_material(choice)
        st.session_state.applied_material = choice
        st.rerun()

    if choice == CUSTOM_MATERIAL:
        st.sidebar.text_input("Material name", key="custom_name")
    else:
        preset = PRESETS[choice]
        st.sidebar.caption(preset.description)

    st.sidebar.header("Crystal structure")
    st.sidebar.selectbox("Crystal system", CRYSTAL_SYSTEMS, key="crystal_system")
    spec = SYSTEM_SPECS[st.session_state.crystal_system]
    st.sidebar.caption(spec.description)
    length_labels = {"a": "a (Å)", "b": "b (Å)", "c": "c (Å)"}
    angle_labels = {"alpha": "alpha (°)", "beta": "beta (°)", "gamma": "gamma (°)"}
    for name in spec.lengths:
        _bind_number(
            length_labels[name],
            f"lat_{name}",
            container=st.sidebar,
            min_value=0.2,
            max_value=100.0,
            step=0.001,
            format="%.4f",
        )
    for name in spec.angles:
        _bind_number(
            angle_labels[name],
            f"lat_{name}",
            container=st.sidebar,
            min_value=5.0,
            max_value=175.0,
            step=0.01,
            format="%.3f",
        )
    st.sidebar.text_input("Space group", key="space_group")
    st.sidebar.text_input("Formula", key="formula")
    st.sidebar.caption(st.session_state.structure_source)
    if choice in PRESETS:
        if st.sidebar.button("Restore catalog lattice", use_container_width=True):
            st.session_state.restore_catalog = True
            st.rerun()

    st.sidebar.header("Peak detection")
    st.sidebar.checkbox("Smooth with Savitzky–Golay", key="smooth")
    st.sidebar.slider("Smoothing window (points)", 5, 31, step=2, key="smooth_window")
    st.sidebar.slider("SNIP background width (points)", 5, 120, key="snip_iterations")
    st.sidebar.slider("Peak prominence (% of net maximum)", 0.5, 30.0, step=0.5, key="prominence")
    st.sidebar.slider("Minimum peak separation (°)", 0.02, 1.0, step=0.01, key="min_sep")

    with st.sidebar.expander("Advanced calculation"):
        st.slider("Match tolerance (°)", 0.02, 0.50, step=0.01, key="tolerance")
        st.slider("Minimum reference intensity", 0.0, 20.0, step=0.5, key="min_intensity")
        st.slider("Isotropic B (Å²)", 0.0, 3.0, step=0.05, key="b_iso")
        st.checkbox("Lorentz–polarization correction", key="use_lp")
        st.caption(
            "Sites that already carry a B factor from a CIF keep that value. "
            "Other sites use the slider."
        )

    _render_experiment_notes()


def _render_experiment_notes() -> None:
    """Record optional synthesis metadata that is not used in the calculation."""

    st.sidebar.header("Experiment notes")
    st.sidebar.caption(
        "Optional notes for this scan. They are recorded with the results and are not "
        "inputs to the crystallographic calculation. They do not change peak positions, "
        "intensities, or screening scores."
    )
    st.sidebar.number_input(
        "Stirrer size (mm)",
        min_value=0.0,
        max_value=1000.0,
        step=0.1,
        value=None,
        placeholder="Optional",
        key="note_stirrer_mm",
        help="Experiment note only. Not used to calculate the pattern or the screening score.",
    )
    st.sidebar.number_input(
        "Rotation speed (rpm)",
        min_value=0.0,
        max_value=20000.0,
        step=1.0,
        value=None,
        placeholder="Optional",
        key="note_rotation_rpm",
        help="Experiment note only. Not used to calculate the pattern or the screening score.",
    )
    st.sidebar.number_input(
        "Flask size (mL)",
        min_value=0.0,
        max_value=50000.0,
        step=1.0,
        value=None,
        placeholder="Optional",
        key="note_flask_ml",
        help="Experiment note only. Not used to calculate the pattern or the screening score.",
    )
    st.sidebar.text_area(
        "Short note",
        key="note_text",
        placeholder="Optional",
        height=80,
        help="Experiment note only. Not used to calculate the pattern or the screening score.",
    )


def _format_experiment_value(value: float | None, unit: str) -> str:
    if value is None:
        return "—"
    number = float(value)
    text = f"{number:.4f}".rstrip("0").rstrip(".")
    return f"{text} {unit}"


def _render_experiment_note_summary(material_name: str) -> None:
    stirrer = st.session_state.get("note_stirrer_mm")
    speed = st.session_state.get("note_rotation_rpm")
    flask = st.session_state.get("note_flask_ml")
    note = str(st.session_state.get("note_text") or "").strip()
    st.subheader("Experiment notes")
    st.caption(
        f"Recorded for {material_name}. "
        "Stirrer size, rotation speed, and flask size are experiment notes only. "
        "They do not change peak positions, intensities, or screening scores."
    )
    columns = st.columns(3)
    columns[0].metric("Stirrer size", _format_experiment_value(stirrer, "mm"))
    columns[1].metric("Rotation speed", _format_experiment_value(speed, "rpm"))
    columns[2].metric("Flask size", _format_experiment_value(flask, "mL"))
    st.caption(f"Note: {note if note else '—'}")


def _experimental_data(wavelength: float):
    st.subheader("Measured scan")
    upload = st.file_uploader(
        "Powder pattern",
        type=["csv", "txt", "dat", "xy", "xlsx", "xls"],
        help=(
            "Text, CSV, DAT, XY, or Excel .xlsx. The first sheet is used. "
            "A header with 2θ and intensity is used when present; otherwise the first "
            "two numeric columns are used. Legacy .xls files are not read; save them as .xlsx or .csv."
        ),
    )
    action_col, note_col = st.columns([1, 2])
    with action_col:
        if st.button("Load Ti2AlN example", use_container_width=True):
            st.session_state.load_example_request = True
            st.rerun()
    if upload is not None:
        token = f"{upload.name}:{upload.size}"
        if token != st.session_state.get("upload_token"):
            st.session_state.upload_token = token
            st.session_state.example_active = False

    if st.session_state.example_active:
        angles, counts = build_ti2aln_example(wavelength)
        note_col.info(
            "Showing a synthetic Ti2AlN scan (Cu Kα1) with a TiN impurity, "
            "a linear background, and random noise."
        )
        st.download_button(
            "Download example as XY",
            data=example_xy_text(angles, counts),
            file_name="ti2aln_example.xy",
            mime="text/plain",
        )
        return angles, counts, "Ti2AlN example"

    if upload is None:
        note_col.caption("Upload a scan, or load the Ti2AlN example to try the screening.")
        return None

    try:
        angles, counts = load_powder_pattern(upload.getvalue(), upload.name)
    except ValueError as exc:
        st.error(str(exc))
        return None
    return angles, counts, upload.name


def _reference_rows(pattern: CalculatedPattern, crystal_system: str) -> pd.DataFrame:
    rows = []
    for reflection in pattern.reflections:
        row = {
            "2θ (°)": reflection.two_theta,
            "d (Å)": reflection.d_spacing,
            "Relative intensity": reflection.intensity,
            "hkl": reflection.hkl_label,
            "Multiplicity": reflection.multiplicity,
        }
        if crystal_system in {"Hexagonal", "Trigonal"}:
            h, k, l = reflection.primary_hkl
            row["Miller–Bravais"] = format_miller_bravais(h, k, l)
        rows.append(row)
    return pd.DataFrame(rows)


def _line_rows(score: PhaseScore) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "hkl": line.hkl_label,
                "Reference 2θ (°)": line.reference_two_theta,
                "Reference I": line.reference_intensity,
                "Measured 2θ (°)": line.experimental_two_theta,
                "Measured net I": line.experimental_intensity,
                "Δ2θ (°)": line.delta,
                "Status": "Matched" if line.matched else "Missing",
            }
            for line in score.lines
        ]
    )


def main() -> None:
    _ensure_defaults()
    _consume_requests()
    _render_sidebar()

    material_name = _material_name()
    wavelength_label, wavelength = _wavelength()
    st.title(material_name)
    st.caption("Powder X-ray diffraction · reference lines and impurity screening")

    try:
        lattice = constrained_lattice(
            st.session_state.crystal_system,
            float(st.session_state.lat_a),
            float(st.session_state.lat_b),
            float(st.session_state.lat_c),
            float(st.session_state.lat_alpha),
            float(st.session_state.lat_beta),
            float(st.session_state.lat_gamma),
        )
    except ValueError as exc:
        st.error(str(exc))
        return

    formula = str(st.session_state.formula).strip() or "—"
    metric_cols = st.columns(6)
    metric_cols[0].metric("Crystal system", lattice.crystal_system)
    metric_cols[1].metric("Space group", st.session_state.space_group or "—")
    metric_cols[2].metric("Formula", formula)
    metric_cols[3].metric("a (Å)", f"{lattice.a:.4f}")
    metric_cols[4].metric("c (Å)" if lattice.crystal_system != "Cubic" else "b = c (Å)", f"{lattice.c:.4f}")
    metric_cols[5].metric("Wavelength (Å)", f"{wavelength:.5f}")
    st.caption(
        f"Cell used for {material_name}: "
        f"a = {lattice.a:.4f} Å, b = {lattice.b:.4f} Å, c = {lattice.c:.4f} Å, "
        f"alpha = {lattice.alpha:.2f}°, beta = {lattice.beta:.2f}°, gamma = {lattice.gamma:.2f}°."
    )

    for warning in st.session_state.cif_warnings:
        st.warning(warning)

    try:
        primary = _pattern_for_cell(
            lattice.crystal_system,
            lattice.a,
            lattice.b,
            lattice.c,
            lattice.alpha,
            lattice.beta,
            lattice.gamma,
            str(st.session_state.space_group),
            _atoms_for_primary(),
            wavelength,
        )
    except ValueError as exc:
        st.error(str(exc))
        return

    if primary.index_truncated:
        st.warning(
            "The reflection search was capped at Miller indices of ±30. "
            "Narrow the 2θ range or use a smaller cell if high-angle lines are missing."
        )
    if not primary.has_structure_factors:
        st.info(
            f"{material_name} has no atomic positions, so reference intensities follow "
            "multiplicity and the Lorentz–polarization factor only."
        )

    experiment = _experimental_data(wavelength)
    if experiment is not None:
        _render_experiment_note_summary(material_name)
    processed = None
    if experiment is not None:
        angles, counts, _source = experiment
        try:
            processed = process_scan(
                angles,
                counts,
                smooth=bool(st.session_state.smooth),
                smooth_window=int(st.session_state.smooth_window),
                background_iterations=int(st.session_state.snip_iterations),
                prominence_percent=float(st.session_state.prominence),
                minimum_separation=float(st.session_state.min_sep),
                wavelength=wavelength,
            )
        except ValueError as exc:
            st.error(str(exc))
            processed = None
        else:
            for warning in processed.warnings:
                st.warning(warning)

    _render_candidate_picker()
    scores = _screen(material_name, primary, processed, wavelength)
    pattern_tab, screening_tab, reference_tab, method_tab = st.tabs(
        ["Pattern", "Phase screening", "Reference lines", "Method"]
    )

    with pattern_tab:
        _render_pattern(material_name, wavelength_label, wavelength, primary, processed, scores)
    with screening_tab:
        _render_screening(material_name, processed, scores)
    with reference_tab:
        _render_reference(material_name, lattice.crystal_system, primary)
    with method_tab:
        _render_method()


def _candidate_patterns(wavelength: float) -> list[tuple[str, CalculatedPattern, str]]:
    catalog = {phase.name: phase for phase in screening_catalog(st.session_state.material_choice)}
    prepared: list[tuple[str, CalculatedPattern, str]] = []
    for name in st.session_state.selected_impurities:
        phase = catalog.get(name)
        if phase is None:
            continue
        prepared.append((phase.name, _pattern_for_phase(phase, wavelength), phase.notes))
    for candidate in st.session_state.custom_candidates:
        atoms: tuple[AtomSite, ...] = ()
        pattern = _pattern_for_cell(
            candidate["crystal_system"],
            candidate["a"],
            candidate["b"],
            candidate["c"],
            candidate["alpha"],
            candidate["beta"],
            candidate["gamma"],
            candidate["space_group"],
            atoms,
            wavelength,
        )
        note = "Custom candidate. Intensities use multiplicity only."
        prepared.append((candidate["name"], pattern, note))
    return prepared


def _screen(material_name: str, primary: CalculatedPattern, processed, wavelength: float) -> list[PhaseScore]:
    if processed is None:
        return []
    preset = get_preset(st.session_state.material_choice)
    primary_notes = preset.primary.notes if preset and not st.session_state.cif_atoms else ""
    if st.session_state.cif_atoms:
        primary_notes = "Primary cell and positions taken from the uploaded CIF."
    primary_score = score_phase(
        processed.peaks,
        primary.reflections,
        name=material_name,
        role="Primary",
        tolerance=float(st.session_state.tolerance),
        notes=primary_notes,
    )
    claimed = claimed_peak_indices(primary_score, processed.peaks)
    scores = [primary_score]
    for name, pattern, notes in _candidate_patterns(wavelength):
        scores.append(
            score_phase(
                processed.peaks,
                pattern.reflections,
                name=name,
                role="Candidate",
                tolerance=float(st.session_state.tolerance),
                claimed=claimed,
                notes=notes,
            )
        )
    return scores


def _render_pattern(material_name, wavelength_label, wavelength, primary, processed, scores) -> None:
    st.checkbox("Show reference lines", key="show_reference")
    st.checkbox("Show background-subtracted intensity", key="show_net")
    st.checkbox(
        "Overlay consistent or possible candidates",
        key="overlay_candidates",
        help="Each stick pattern is scaled to its own strongest line.",
    )
    overlays = []
    if st.session_state.overlay_candidates and processed is not None:
        ranked = [
            score
            for score in scores
            if score.role == "Candidate" and score.assessment in {"Consistent", "Possible"}
        ]
        ranked = sorted(ranked, key=lambda score: score.score, reverse=True)[:4]
        prepared = {name: pattern for name, pattern, _notes in _candidate_patterns(wavelength)}
        overlays = [(score.name, prepared[score.name].reflections) for score in ranked if score.name in prepared]

    figure = build_pattern_figure(
        material_name,
        wavelength_label,
        wavelength,
        None if processed is None else processed.two_theta,
        None if processed is None else processed.intensity,
        None if processed is None else processed.background,
        None if processed is None else processed.peaks,
        primary.reflections if st.session_state.show_reference else None,
        overlays,
        show_net=bool(st.session_state.show_net),
        net=None if processed is None else processed.net,
    )
    st.plotly_chart(figure, theme=None, key="pattern_chart")
    if processed is not None:
        st.caption(
            f"{len(processed.peaks)} peaks detected in the measured scan. "
            "Markers use the measured intensity. Each stick pattern is scaled to its own strongest line."
        )
        frame = peaks_to_frame(processed.peaks)
        st.dataframe(frame, hide_index=True, width="stretch")
        st.download_button(
            "Download detected peaks",
            data=frame.to_csv(index=False),
            file_name=f"{_slug(material_name)}_peaks.csv",
            mime="text/csv",
        )


def _render_candidate_picker() -> None:
    st.subheader("Impurity candidates")
    catalog_names = [phase.name for phase in screening_catalog(st.session_state.material_choice)]
    st.multiselect(
        "Catalog candidates",
        catalog_names,
        key="selected_impurities",
        help="Preset impurities are selected automatically. Add any other catalog phase.",
    )
    _render_custom_candidate_form()


def _render_screening(material_name: str, processed, scores: list[PhaseScore]) -> None:
    if processed is None:
        st.info(f"Upload a powder pattern to screen {material_name} and the selected candidates.")
        return
    if not scores:
        st.info("No phases were scored.")
        return

    st.plotly_chart(
        build_score_figure(scores, material_name),
        theme=None,
        key="score_chart",
    )
    st.caption(
        "Consistent: score ≥ 70, with at least three matched lines, or two matched lines "
        "that cover at least 60% of the calculated intensity. "
        "Possible: score ≥ 40. "
        "A missing strongest line lowers the score. "
        "This is a peak-position screen, not a phase fraction."
    )
    summary = scores_to_frame(scores)
    st.dataframe(
        summary,
        hide_index=True,
        width="stretch",
        column_config={
            "Score": st.column_config.NumberColumn(format="%.1f"),
            "Coverage": st.column_config.NumberColumn(format="%.2f"),
            "Mean |Δ2θ| (°)": st.column_config.NumberColumn(format="%.3f"),
            "Residual recall": st.column_config.NumberColumn(format="%.2f"),
            "Measured share": st.column_config.NumberColumn(format="%.2f"),
        },
    )
    st.download_button(
        "Download screening summary",
        data=summary.to_csv(index=False),
        file_name=f"{_slug(material_name)}_screening.csv",
        mime="text/csv",
    )

    names = [score.name for score in scores]
    selected = st.selectbox("Inspect lines", names, key="inspect_phase")
    chosen = next(score for score in scores if score.name == selected)
    detail_cols = st.columns(4)
    detail_cols[0].metric("Score", f"{chosen.score:.1f}")
    detail_cols[1].metric("Assessment", chosen.assessment)
    detail_cols[2].metric("Lines matched", f"{chosen.n_matched}/{chosen.n_reference}")
    mean_text = "—" if chosen.mean_abs_delta is None else f"{chosen.mean_abs_delta:.3f}"
    detail_cols[3].metric("Mean |Δ2θ| (°)", mean_text)
    if chosen.notes:
        st.caption(chosen.notes)
    detail = _line_rows(chosen)
    st.dataframe(detail, hide_index=True, width="stretch")
    st.download_button(
        f"Download {chosen.name} line list",
        data=detail.to_csv(index=False),
        file_name=f"{_slug(material_name)}_{_slug(chosen.name)}_lines.csv",
        mime="text/csv",
    )


def _render_custom_candidate_form() -> None:
    with st.expander("Add a custom candidate"):
        st.text_input("Candidate name", key="cand_name")
        st.selectbox("Candidate crystal system", CRYSTAL_SYSTEMS, key="cand_system")
        spec = SYSTEM_SPECS[st.session_state.cand_system]
        st.caption(spec.description)
        length_labels = {"a": "Candidate a (Å)", "b": "Candidate b (Å)", "c": "Candidate c (Å)"}
        angle_labels = {
            "alpha": "Candidate alpha (°)",
            "beta": "Candidate beta (°)",
            "gamma": "Candidate gamma (°)",
        }
        _ensure_candidate_fields()
        for name in spec.lengths:
            _bind_number(
                length_labels[name],
                f"cand_{name}",
                min_value=0.2,
                max_value=100.0,
                step=0.001,
                format="%.4f",
            )
        for name in spec.angles:
            _bind_number(
                angle_labels[name],
                f"cand_{name}",
                min_value=5.0,
                max_value=175.0,
                step=0.01,
                format="%.3f",
            )
        st.text_input("Candidate space group", key="cand_sg")
        if st.button("Add candidate"):
            if _add_custom_candidate():
                st.rerun()

    if not st.session_state.custom_candidates:
        return
    st.markdown("**Custom candidates**")
    for index, candidate in enumerate(list(st.session_state.custom_candidates)):
        cols = st.columns([4, 1])
        cols[0].write(
            f"{candidate['name']} · {candidate['crystal_system']} · "
            f"{candidate['space_group']} · a = {candidate['a']:.4f} Å"
        )
        if cols[1].button("Remove", key=f"remove_candidate_{index}"):
            st.session_state.custom_candidates.pop(index)
            st.rerun()


def _ensure_candidate_fields() -> None:
    st.session_state.setdefault("cand_name", "")
    st.session_state.setdefault("cand_system", "Cubic")
    st.session_state.setdefault("cand_sg", "Pm-3m")
    st.session_state.setdefault("cand_a", 4.0)
    st.session_state.setdefault("cand_b", 4.0)
    st.session_state.setdefault("cand_c", 4.0)
    st.session_state.setdefault("cand_alpha", 90.0)
    st.session_state.setdefault("cand_beta", 90.0)
    st.session_state.setdefault("cand_gamma", 90.0)


def _add_custom_candidate() -> bool:
    name = str(st.session_state.cand_name).strip()
    if not name:
        st.error("Enter a name for the custom candidate.")
        return False
    existing = {item["name"] for item in st.session_state.custom_candidates}
    existing.update(phase.name for phase in screening_catalog(None))
    if name == _material_name() or name in existing:
        st.error(f"'{name}' is already in use. Choose a different candidate name.")
        return False
    try:
        lattice = constrained_lattice(
            st.session_state.cand_system,
            float(st.session_state.cand_a),
            float(st.session_state.cand_b),
            float(st.session_state.cand_c),
            float(st.session_state.cand_alpha),
            float(st.session_state.cand_beta),
            float(st.session_state.cand_gamma),
        )
    except ValueError as exc:
        st.error(str(exc))
        return False
    st.session_state.custom_candidates.append(
        {
            "name": name,
            "crystal_system": lattice.crystal_system,
            "space_group": str(st.session_state.cand_sg).strip() or "P1",
            "a": lattice.a,
            "b": lattice.b,
            "c": lattice.c,
            "alpha": lattice.alpha,
            "beta": lattice.beta,
            "gamma": lattice.gamma,
        }
    )
    st.session_state.cand_name = ""
    return True


def _render_reference(material_name: str, crystal_system: str, primary: CalculatedPattern) -> None:
    st.markdown(f"**Calculated lines for {material_name}**")
    if primary.has_structure_factors:
        st.caption(
            "Relative intensities include the structure factor, multiplicity, and the "
            "selected Lorentz–polarization setting. They describe a random powder."
        )
    frame = _reference_rows(primary, crystal_system)
    if frame.empty:
        st.info("No reference lines fall in the selected 2θ window.")
        return
    st.dataframe(frame, hide_index=True, width="stretch")
    st.download_button(
        "Download reference lines",
        data=frame.to_csv(index=False),
        file_name=f"{_slug(material_name)}_reference.csv",
        mime="text/csv",
    )


def _render_method() -> None:
    st.markdown(
        """
**Peak positions.** d-spacings come from the reciprocal metric tensor,
`1/d² = hᵢ (G⁻¹)ᵢⱼ hⱼ`. Bragg's law then gives `nλ = 2d sinθ` for the
selected wavelength. One wavelength is used, so a Kα1/Kα2 doublet is not split.

**Intensities.** With atomic coordinates, each reflection gets a structure
factor from Cromer–Mann-style scattering factors. Powder intensity is `|F|²`
times the Lorentz–polarization factor `(1 + cos²2θ) / (sin²θ cosθ)`, summed
over planes that land on the same 2θ and scaled to 100. Without coordinates,
every allowed plane contributes equally before that correction.

**Systematic absences.** Lattice centering is read from the space-group symbol.
`P6₃/mmc` and `P6₃mc` also omit `00l` and `hhl` reflections with odd `l`.

**Measured peaks.** An optional Savitzky–Golay smooth is followed by a SNIP
background. Peaks are maxima on the net intensity above the chosen prominence.

**Screening score.** For the primary phase the score is `100 × coverage ×
exp(−mean|Δ2θ| / tolerance)`. Coverage is the share of calculated intensity
that finds a measured peak inside the tolerance. Candidates use the same term
for 65% of the weight and, for the other 35%, the share of residual measured
intensity they explain. Residual peaks are measured peaks not already matched
to the primary phase. If a phase's strongest calculated line is absent, its
score is reduced. The labels Consistent, Possible, and Unlikely are bands on
that score. They are not a quantitative phase analysis.

**Experiment notes.** Stirrer size, rotation speed, flask size, and the short
note are recorded with the scan. They are not inputs to the peak positions,
intensities, or screening scores.
        """
    )


if __name__ == "__main__":
    main()
