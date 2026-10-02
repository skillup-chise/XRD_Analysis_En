"""Load powder scans, remove background, find peaks, and score candidate phases.

The screening score is a peak-position figure of merit. It is not a Rietveld
weight fraction and it does not model texture, crystallite size, or a Kα1/Kα2
doublet.
"""

from __future__ import annotations

import io
import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import find_peaks, savgol_filter

from crystallography import CU_KA1, Reflection, bragg_d_spacing, calculate_pattern
from materials import PHASES

_MAX_PEAKS = 200
_THETA_HEADERS = ("2theta", "2-theta", "twotheta", "theta", "angle", "ttheta", "deg")
_INTENSITY_HEADERS = ("intensity", "counts", "count", "cps", "int", "signal", "yobs")


@dataclass(frozen=True)
class DetectedPeak:
    two_theta: float
    intensity: float
    raw_intensity: float
    d_spacing: float


@dataclass
class ProcessedScan:
    two_theta: np.ndarray
    intensity: np.ndarray
    smoothed: np.ndarray
    background: np.ndarray
    net: np.ndarray
    peaks: list[DetectedPeak]
    warnings: list[str]


@dataclass(frozen=True)
class LineMatch:
    hkl_label: str
    reference_two_theta: float
    reference_intensity: float
    experimental_two_theta: float | None
    experimental_intensity: float | None
    delta: float | None

    @property
    def matched(self) -> bool:
        return self.experimental_two_theta is not None


@dataclass(frozen=True)
class PhaseScore:
    name: str
    role: str
    score: float
    assessment: str
    n_reference: int
    n_matched: int
    coverage: float
    mean_abs_delta: float | None
    residual_recall: float
    measured_share: float
    lines: tuple[LineMatch, ...]
    notes: str


def parse_xrd_text(text: str, source_name: str = "uploaded file") -> tuple[np.ndarray, np.ndarray]:
    """Read 2θ and intensity columns from text, CSV, or two-column XY data."""

    rows: list[list[str]] = []
    delimiter = None
    comma_decimal = False
    for raw_line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        line = raw_line.strip()
        if not line or line[0] in {"#", "!", "%", "*"} or line.startswith("//"):
            continue
        if delimiter is None:
            delimiter, comma_decimal = _detect_delimiter(line)
        rows.append(_split_row(line, delimiter, comma_decimal))
    return _arrays_from_rows(rows, source_name)


def load_powder_pattern(data: bytes, source_name: str = "uploaded file") -> tuple[np.ndarray, np.ndarray]:
    """Read a powder scan from text or from the first sheet of an Excel file.

    ``.xlsx`` is read with openpyxl. Legacy ``.xls`` is rejected so the
    installer does not need xlrd. Column rules match :func:`parse_xrd_text`.
    """

    suffix = Path(source_name).suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        return parse_xrd_workbook(data, source_name)
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        text = data.decode("latin-1")
    return parse_xrd_text(text, source_name)


def parse_xrd_workbook(data: bytes, source_name: str = "uploaded file") -> tuple[np.ndarray, np.ndarray]:
    """Read 2θ and intensity from the first sheet of an .xlsx workbook."""

    if Path(source_name).suffix.lower() == ".xls":
        raise ValueError(
            f"{source_name} is a legacy Excel .xls workbook. "
            "Save it as .xlsx or .csv and upload that file."
        )
    frame = _read_first_sheet(data, source_name)
    return _arrays_from_rows(_rows_from_frame(frame), source_name)


def _arrays_from_rows(rows: list[list[str]], source_name: str) -> tuple[np.ndarray, np.ndarray]:
    if len(rows) < 5:
        raise ValueError(
            f"{source_name} does not contain enough numeric rows. {_column_hint(source_name)}"
        )

    header_map = _header_columns(rows[0])
    start = 1 if header_map is not None else 0
    if header_map is None:
        theta_col, intensity_col = _numeric_columns(rows)
    else:
        theta_col, intensity_col = header_map

    two_theta: list[float] = []
    intensity: list[float] = []
    for row in rows[start:]:
        if max(theta_col, intensity_col) >= len(row):
            continue
        angle = _float_or_none(row[theta_col])
        counts = _float_or_none(row[intensity_col])
        if angle is None or counts is None:
            continue
        two_theta.append(angle)
        intensity.append(counts)
    if len(two_theta) < 10:
        raise ValueError(
            f"Could not find two numeric columns of 2θ and intensity in {source_name}."
        )

    angles = np.asarray(two_theta, dtype=float)
    counts = np.asarray(intensity, dtype=float)
    order = np.argsort(angles, kind="mergesort")
    angles = angles[order]
    counts = counts[order]
    unique_angles, inverse = np.unique(np.round(angles, 6), return_inverse=True)
    if len(unique_angles) != len(angles):
        summed = np.zeros(len(unique_angles), dtype=float)
        tally = np.zeros(len(unique_angles), dtype=float)
        np.add.at(summed, inverse, counts)
        np.add.at(tally, inverse, 1.0)
        angles = unique_angles
        counts = summed / tally
    finite = np.isfinite(angles) & np.isfinite(counts)
    angles = angles[finite]
    counts = counts[finite]
    if len(angles) < 10 or float(angles[-1] - angles[0]) < 5:
        raise ValueError(
            f"{source_name} needs at least 10 points spanning 5° in 2θ."
        )
    return angles, counts


def _column_hint(source_name: str) -> str:
    if Path(source_name).suffix.lower() == ".xlsx":
        return "Use the first sheet with 2θ and intensity columns."
    return "Use a text file with 2θ and intensity columns."


def _read_first_sheet(data: bytes, source_name: str) -> pd.DataFrame:
    if not data:
        raise ValueError(
            f"Could not read the first sheet of {source_name}. "
            "The file is empty or is not a valid .xlsx workbook."
        )
    try:
        frame = pd.read_excel(
            io.BytesIO(data),
            sheet_name=0,
            header=None,
            engine="openpyxl",
        )
    except Exception as exc:
        raise ValueError(
            f"Could not read the first sheet of {source_name}. "
            "Check that the file is a valid .xlsx workbook."
        ) from exc
    if not isinstance(frame, pd.DataFrame) or frame.empty or frame.dropna(how="all").empty:
        raise ValueError(
            f"Could not read the first sheet of {source_name}. "
            "The sheet is empty. Put 2θ and intensity in the first sheet."
        )
    return frame


def _rows_from_frame(frame: pd.DataFrame) -> list[list[str]]:
    rows: list[list[str]] = []
    for record in frame.itertuples(index=False, name=None):
        cells = [_cell_text(value) for value in record]
        if any(cells):
            rows.append(cells)
    return rows


def _cell_text(value: object) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, bool):
        return ""
    if hasattr(value, "item") and not isinstance(value, (bytes, bytearray)):
        try:
            value = value.item()
        except (ValueError, AttributeError):
            pass
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        missing = False
    if missing is True or missing is pd.NA:
        return ""
    if isinstance(value, bool):
        return ""
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            return ""
        return format(value, ".12g")
    return str(value).strip()


def process_scan(
    two_theta: np.ndarray,
    intensity: np.ndarray,
    *,
    smooth: bool = True,
    smooth_window: int = 11,
    background_iterations: int = 40,
    prominence_percent: float = 4.0,
    minimum_separation: float = 0.15,
    wavelength: float = CU_KA1,
) -> ProcessedScan:
    """Smooth, estimate a SNIP background, and pick peaks on the net intensity."""

    angles = np.asarray(two_theta, dtype=float)
    counts = np.asarray(intensity, dtype=float)
    if len(angles) != len(counts) or len(angles) < 10:
        raise ValueError("The scan needs at least 10 paired 2θ and intensity values.")
    warnings: list[str] = []
    if np.nanmin(counts) < 0:
        warnings.append("Negative intensities were clipped to zero before processing.")
        counts = np.clip(counts, 0, None)

    window = _odd_window(smooth_window, len(counts))
    if smooth and window >= 5:
        smoothed = savgol_filter(counts, window_length=window, polyorder=3, mode="interp")
        smoothed = np.clip(smoothed, 0, None)
    else:
        smoothed = counts.copy()

    iterations = int(background_iterations)
    if iterations < 1:
        raise ValueError("Background iterations must be at least 1.")
    background = snip_background(smoothed, iterations)
    net = np.clip(smoothed - background, 0, None)
    maximum = float(np.max(net)) if len(net) else 0.0
    if maximum <= 0:
        return ProcessedScan(angles, counts, smoothed, background, net, [], warnings)

    step = _median_step(angles)
    distance = max(1, int(round(minimum_separation / step))) if step > 0 else 1
    prominence = max(maximum * (prominence_percent / 100.0), 1e-9)
    indices, _properties = find_peaks(net, prominence=prominence, distance=distance)
    peaks: list[DetectedPeak] = []
    for index in indices:
        angle = float(angles[index])
        if angle <= 0:
            continue
        try:
            spacing = bragg_d_spacing(angle, wavelength)
        except ValueError:
            continue
        peaks.append(
            DetectedPeak(
                two_theta=angle,
                intensity=float(net[index]),
                raw_intensity=float(counts[index]),
                d_spacing=spacing,
            )
        )
    peaks.sort(key=lambda peak: peak.intensity, reverse=True)
    if len(peaks) > _MAX_PEAKS:
        warnings.append(
            f"Only the {_MAX_PEAKS} strongest peaks are kept. Raise the prominence to focus the list."
        )
        peaks = peaks[:_MAX_PEAKS]
    peaks.sort(key=lambda peak: peak.two_theta)
    if not peaks:
        warnings.append(
            "No peaks were detected. Lower the prominence or reduce the background width."
        )
    return ProcessedScan(angles, counts, smoothed, background, net, peaks, warnings)


def snip_background(intensity: np.ndarray, iterations: int) -> np.ndarray:
    """Statistics-sensitive non-linear iterative peak clipping (SNIP)."""

    values = np.clip(np.asarray(intensity, dtype=float), 0, None)
    if iterations < 1 or len(values) < 3:
        return np.zeros_like(values)
    iterations = min(iterations, max(1, len(values) // 2 - 1))
    compressed = np.log(np.log(np.sqrt(values + 1.0) + 1.0) + 1.0)
    for width in range(1, iterations + 1):
        previous = compressed.copy()
        averaged = 0.5 * (previous[:-2 * width] + previous[2 * width :])
        compressed[width:-width] = np.minimum(previous[width:-width], averaged)
    background = (np.exp(np.exp(compressed) - 1.0) - 1.0) ** 2 - 1.0
    background = np.where(np.isfinite(background), background, 0.0)
    return np.clip(background, 0, values)


def score_phase(
    experimental: list[DetectedPeak],
    reflections: list[Reflection] | tuple[Reflection, ...],
    *,
    name: str,
    role: str,
    tolerance: float,
    claimed: set[int] | None = None,
    notes: str = "",
) -> PhaseScore:
    """Score one calculated pattern against detected peaks.

    Primary score = 100 × intensity coverage × position factor.
    Impurity score also rewards lines that explain peaks left over by the
    primary phase. A phase whose strongest calculated line is missing is
    penalized. Coverage is the matched share of calculated relative intensity.
    The position factor is exp(−mean|Δ2θ| / tolerance).
    """

    if tolerance <= 0:
        raise ValueError("The matching tolerance must be positive.")
    references = list(reflections)
    total_weight = sum(item.intensity for item in references)
    lines: list[LineMatch] = []
    matched_weight = 0.0
    deltas: list[float] = []
    matched_experimental: set[int] = set()
    claimed_peaks = claimed or set()

    for reflection in references:
        best_index = None
        best_delta = tolerance
        for index, peak in enumerate(experimental):
            delta = abs(peak.two_theta - reflection.two_theta)
            if delta > tolerance + 1e-12:
                continue
            closer = best_index is None or delta < best_delta - 1e-12
            tied = (
                best_index is not None
                and abs(delta - best_delta) <= 1e-12
                and peak.intensity > experimental[best_index].intensity
            )
            if closer or tied:
                best_index = index
                best_delta = delta
        if best_index is None:
            lines.append(
                LineMatch(
                    reflection.hkl_label,
                    reflection.two_theta,
                    reflection.intensity,
                    None,
                    None,
                    None,
                )
            )
            continue
        peak = experimental[best_index]
        matched_weight += reflection.intensity
        deltas.append(best_delta)
        matched_experimental.add(best_index)
        lines.append(
            LineMatch(
                reflection.hkl_label,
                reflection.two_theta,
                reflection.intensity,
                peak.two_theta,
                peak.intensity,
                best_delta,
            )
        )

    coverage = matched_weight / total_weight if total_weight > 0 else 0.0
    mean_delta = float(np.mean(deltas)) if deltas else None
    position_factor = math.exp(-mean_delta / tolerance) if mean_delta is not None else 0.0
    total_experimental = sum(peak.intensity for peak in experimental)
    measured_share = 0.0
    if total_experimental > 0:
        measured_share = (
            sum(experimental[index].intensity for index in matched_experimental)
            / total_experimental
        )

    residual_intensity = sum(
        peak.intensity
        for index, peak in enumerate(experimental)
        if index not in claimed_peaks
    )
    explained_residual = sum(
        experimental[index].intensity
        for index in matched_experimental
        if index not in claimed_peaks
    )
    if role == "Primary":
        residual_recall = 1.0 if residual_intensity <= 0 else explained_residual / residual_intensity
        score = 100.0 * coverage * position_factor
    else:
        residual_recall = (
            explained_residual / residual_intensity if residual_intensity > 0 else 0.0
        )
        score = 100.0 * (0.65 * coverage + 0.35 * residual_recall) * position_factor

    if references:
        strongest = max(references, key=lambda item: item.intensity)
        strongest_found = any(
            line.matched and abs(line.reference_two_theta - strongest.two_theta) < 1e-6
            for line in lines
        )
        if not strongest_found:
            score *= 0.45

    return PhaseScore(
        name=name,
        role=role,
        score=float(np.clip(score, 0.0, 100.0)),
        assessment=_assessment(score, len(deltas), coverage),
        n_reference=len(references),
        n_matched=len(deltas),
        coverage=coverage,
        mean_abs_delta=mean_delta,
        residual_recall=residual_recall,
        measured_share=measured_share,
        lines=tuple(lines),
        notes=notes,
    )


def claimed_peak_indices(score: PhaseScore, experimental: list[DetectedPeak]) -> set[int]:
    """Experimental peaks used by a primary-phase match."""

    claimed: set[int] = set()
    for line in score.lines:
        if line.experimental_two_theta is None:
            continue
        for index, peak in enumerate(experimental):
            if abs(peak.two_theta - line.experimental_two_theta) <= 1e-6:
                claimed.add(index)
                break
    return claimed


def synthesize_scan(
    components: list[tuple[tuple[Reflection, ...] | list[Reflection], float]],
    two_theta: np.ndarray,
    *,
    fwhm: float = 0.18,
    background_intercept: float = 40.0,
    background_slope: float = 0.35,
    noise: float = 6.0,
    seed: int = 7,
) -> np.ndarray:
    """Build a scan from Gaussian powder lines, a linear background, and noise."""

    angles = np.asarray(two_theta, dtype=float)
    intensity = background_intercept + background_slope * angles
    sigma = fwhm / (2.0 * math.sqrt(2.0 * math.log(2.0)))
    for reflections, scale in components:
        for reflection in reflections:
            height = scale * reflection.intensity / 100.0
            intensity += height * np.exp(-0.5 * ((angles - reflection.two_theta) / sigma) ** 2)
    if noise > 0:
        generator = np.random.default_rng(seed)
        intensity += generator.normal(0.0, noise, size=len(angles))
    return np.clip(intensity, 0, None)


def build_ti2aln_example(
    wavelength: float = CU_KA1,
    two_theta_min: float = 5.0,
    two_theta_max: float = 90.0,
    step: float = 0.02,
) -> tuple[np.ndarray, np.ndarray]:
    """Synthetic Cu Kα1 scan of Ti2AlN with a TiN impurity."""

    from crystallography import constrained_lattice

    angles = np.arange(two_theta_min, two_theta_max + step * 0.5, step)
    patterns = []
    for key, scale in (("Ti2AlN", 1200.0), ("TiN", 280.0)):
        phase = PHASES[key]
        lattice = constrained_lattice(
            phase.crystal_system, phase.a, phase.b, phase.c, phase.alpha, phase.beta, phase.gamma
        )
        pattern = calculate_pattern(
            lattice,
            wavelength,
            (two_theta_min, two_theta_max),
            phase.atoms,
            phase.space_group,
            b_iso=0.4,
            min_relative_intensity=1.0,
        )
        patterns.append((pattern.reflections, scale))
    intensity = synthesize_scan(patterns, angles)
    return angles, intensity


def example_xy_text(two_theta: np.ndarray, intensity: np.ndarray) -> str:
    lines = [
        "# Synthetic powder XRD example",
        "# Material: Ti2AlN with a TiN impurity",
        "# Radiation: Cu K-alpha1 (1.54056 Angstrom)",
        "# Columns: two_theta_deg intensity",
    ]
    for angle, counts in zip(two_theta, intensity):
        lines.append(f"{angle:.4f} {counts:.3f}")
    return "\n".join(lines) + "\n"


def _assessment(score: float, n_matched: int, coverage: float) -> str:
    if n_matched == 0:
        return "Unlikely"
    if score >= 70 and (n_matched >= 3 or (n_matched >= 2 and coverage >= 0.6)):
        return "Consistent"
    if score >= 40:
        return "Possible"
    return "Unlikely"


def _odd_window(window: int, n_points: int) -> int:
    chosen = max(5, int(window))
    if chosen % 2 == 0:
        chosen += 1
    upper = n_points if n_points % 2 == 1 else n_points - 1
    return min(chosen, upper)


def _median_step(two_theta: np.ndarray) -> float:
    if len(two_theta) < 2:
        return 0.02
    steps = np.diff(two_theta)
    steps = steps[steps > 0]
    if len(steps) == 0:
        return 0.02
    return float(np.median(steps))


def _detect_delimiter(line: str) -> tuple[str | None, bool]:
    if ";" in line:
        return ";", True
    if "," in line:
        return ",", False
    return None, False


def _split_row(line: str, delimiter: str | None, comma_decimal: bool) -> list[str]:
    if delimiter is None:
        return re.split(r"\s+", line.strip())
    if comma_decimal:
        parts = []
        for part in line.split(";"):
            parts.append(part.strip().replace(",", "."))
        return parts
    return [part.strip() for part in line.split(delimiter)]


def _header_columns(row: list[str]) -> tuple[int, int] | None:
    normalized = [_normalize_header(value) for value in row]
    if not any(any(name in value for name in _THETA_HEADERS) for value in normalized):
        return None
    theta_col = next(
        index
        for index, value in enumerate(normalized)
        if any(name in value for name in _THETA_HEADERS)
    )
    intensity_candidates = [
        index
        for index, value in enumerate(normalized)
        if index != theta_col and any(name in value for name in _INTENSITY_HEADERS)
    ]
    if intensity_candidates:
        return theta_col, intensity_candidates[0]
    numeric = [
        index
        for index, value in enumerate(row)
        if index != theta_col and _float_or_none(value) is not None
    ]
    if numeric:
        return theta_col, numeric[0]
    return None


def _normalize_header(value: str) -> str:
    text = value.lower().replace("θ", "theta").replace("θ".upper(), "theta")
    return re.sub(r"[^a-z0-9]+", "", text)


def _numeric_columns(rows: list[list[str]]) -> tuple[int, int]:
    width = max(len(row) for row in rows)
    scores = []
    for column in range(width):
        parsed = [
            _float_or_none(row[column])
            for row in rows
            if column < len(row)
        ]
        scores.append(sum(value is not None for value in parsed))
    ranked = sorted(range(width), key=lambda column: scores[column], reverse=True)
    if len(ranked) < 2 or scores[ranked[1]] < 5:
        raise ValueError("The file needs at least two numeric columns.")
    return ranked[0], ranked[1]


def _float_or_none(value: str) -> float | None:
    text = value.strip().strip("'\"")
    if not text or text.lower() in {"nan", "none", "."}:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def peaks_to_frame(peaks: list[DetectedPeak]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "2θ (°)": peak.two_theta,
                "d (Å)": peak.d_spacing,
                "Net intensity": peak.intensity,
                "Raw intensity": peak.raw_intensity,
            }
            for peak in peaks
        ]
    )


def scores_to_frame(scores: list[PhaseScore]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Phase": score.name,
                "Role": score.role,
                "Score": score.score,
                "Assessment": score.assessment,
                "Lines matched": f"{score.n_matched}/{score.n_reference}",
                "Coverage": score.coverage,
                "Mean |Δ2θ| (°)": score.mean_abs_delta,
                "Residual recall": score.residual_recall,
                "Measured share": score.measured_share,
                "Notes": score.notes,
            }
            for score in scores
        ]
    )
