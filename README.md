# XRD Analysis

A Streamlit application for powder X-ray diffraction. It calculates reference lines from a crystal structure, reads a measured scan, and screens the pattern against likely impurity phases. There is no model diagnosis and no external API.

## Run

Use CPython 3.10, 3.11, 3.12, 3.13, or 3.14. NumPy, SciPy, pandas, and pyarrow publish binary wheels for those versions. Python 3.9 and 3.15 are outside that set, and an installer will try to compile the libraries instead of downloading a wheel.

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install --only-binary=numpy,scipy,pandas,pyarrow,pillow -r requirements.txt
streamlit run app.py
```

On Windows, activate the environment with `.venv\Scripts\activate` before the `pip` commands. No environment variables are required. `scattering_factors.json` stays next to `crystallography.py`. Sample files live in `examples/` and are optional; the **Load Ti2AlN example** button does not read them.

To replace a broken environment, deactivate it, delete the `.venv` directory, and run the commands above again.

Open the local URL Streamlit prints. The sidebar selects the material, radiation, and peak settings. The main page shows the pattern, the screening scores, and the calculated line list.

To exercise the screen without a file, use **Load Ti2AlN example**. That scan is a synthetic Cu Kα1 pattern of Ti2AlN plus a TiN impurity. `examples/Ti2AlN.cif` is a matching cell for the CIF upload.

## Project layout

| File | Role |
| --- | --- |
| `app.py` | Streamlit interface |
| `materials.py` | Catalog phases, default cells, space groups, and impurity lists |
| `crystallography.py` | Metric tensor, structure factors, Laue multiplicities, CIF parser |
| `xrd_processing.py` | Scan import, SNIP background, peak detection, match scores |
| `plotting.py` | Pattern and score figures |
| `scattering_factors.json` | Atomic form-factor coefficients |

## Catalog

**Ti2AlN** is hexagonal, space group P6₃/mmc (194), with the powder cell a = 2.989 Å and c = 13.614 Å. The default impurity list starts with TiN, unreacted α-Ti, and Al, and also includes AlN, TiAl, and Ti3Al.

**Nb2AlC** is hexagonal, P6₃/mmc, with a = 3.107 Å and c = 13.888 Å. Candidates start with NbC, unreacted Nb, and Al, plus Al3Nb.

**Custom (Other)** accepts any material name, crystal system, space group, and the lattice parameters that system actually uses. A CIF can fill those fields, including atomic positions when they are present.

Internal z coordinates for the MAX phases are representative 211 values. Relative intensities are estimates for a random powder, not a refinement of one sample. Catalog cells can be edited and restored from the sidebar.

## Measured files

Text, CSV, DAT, and XY files are accepted. The loader skips blank lines and comments that start with `#`, `!`, `%`, `*`, or `//`. It uses a header containing 2θ and intensity when one is present; otherwise it uses the first two numeric columns, with 2θ first. Comma, whitespace, tab, and semicolon separators are accepted. Semicolon files may use a comma as the decimal mark.

## How the screen works

d-spacings come from the reciprocal metric tensor, and 2θ from Bragg's law at the selected wavelength. Intensities use a structure factor when atomic coordinates exist, then the Lorentz–polarization factor, summed over planes that fall on the same 2θ and scaled to 100. Without coordinates, each allowed plane contributes equally before that correction.

The primary score is `100 × coverage × exp(−mean|Δ2θ| / tolerance)`. Coverage is the share of calculated intensity that lies within the tolerance of a detected peak. Impurity scores give 65% of that weight to coverage and 35% to how much residual measured intensity they explain. Residual peaks are detected peaks not already matched to the primary phase. A phase whose strongest calculated line is missing is penalized.

**Consistent** means a score of at least 70 with at least three matched lines, or with two matched lines that cover at least 60% of the calculated intensity. **Possible** means a score of at least 40. Anything else is **Unlikely**. These bands are a search aid. They are not phase fractions, and the calculation does not model texture, size, strain, or a Kα1/Kα2 doublet.

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```
