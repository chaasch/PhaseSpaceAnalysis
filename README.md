# IAEA Phase Space Analysis

Reads IAEA format phase space files (`.IAEAheader` + `.IAEAphsp`) and analyzes the particles in them: energy spectra, fluence maps, efficiency, and a comparison of the photon spectrum against [SpekPy](https://bitbucket.org/spekpy/spekpy_release).

## Files

- `PhaseSpaceParser.py`: parses the IAEA header (record layout, byte order, history and particle counts) and unpacks each particle record.
- `ParticleAnalyzer.py`: holds the particles and runs the analyses, writing graphs (`.png`) and `.csv` files.
- `AnalyzePhaseSpaceExample.py`: example script that runs every analysis on one phase space.
- `PhaseSpaceStyle.mplstyle`: matplotlib style used for the graphs.

## Requirements

Python 3.9+ with `numpy` and `matplotlib`. `spekpy` is only needed for the SpekPy comparison.

```bash
pip install numpy matplotlib spekpy
```

## Usage

1. Put the `.IAEAheader` and `.IAEAphsp` files in the project directory (they are not tracked by git).
2. Edit the settings at the top of `AnalyzePhaseSpaceExample.py`: `PHASE_SPACE_NAME`, bin widths, source energy and the tube settings for the SpekPy comparison.
3. Run it:

```bash
python AnalyzePhaseSpaceExample.py
```

Results are written to `Outputs/<PHASE_SPACE_NAME>_Output/`.

## Analyses

| Method | Output |
| --- | --- |
| `analyze_spectra` | Weighted energy spectrum (keV) per particle type |
| `analyze_fluence` | Fluence or energy fluence map in the scoring plane per particle type |
| `analyze_efficiency` | Particles and energy per original history, weight statistics |
| `compare_spekpy` | Photon spectrum, mean energy and first HVL compared against SpekPy |

## Author

Campbell Haasch, University of Wisconsin-Madison, Department of Medical Physics (RED Lab/UWMRRC)
