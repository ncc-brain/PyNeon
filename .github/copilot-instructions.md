# GitHub Copilot Instructions for PyNeon

## Project Overview

PyNeon is a lightweight Python package for reading, preprocessing, and exporting multimodal data from the [Pupil Labs Neon eye-tracking system](https://pupil-labs.com/products/neon). It supports both **native** (companion device) and **Pupil Cloud** data formats.

Key capabilities:
- Reading datasets, recordings, and individual data modalities (gaze, eye states, IMU, events, video)
- Preprocessing: cropping, interpolation, concatenation, window averaging
- Flexible epoch-based trial analysis
- Scene video processing with AprilTag/contour-based surface mapping
- Export to Motion-BIDS and Eye-Tracking-BIDS formats

## Repository Structure

```
pyneon/              # Main package
├── __init__.py      # Public API exports
├── dataset.py       # Dataset class (multi-recording container)
├── recording.py     # Recording class (single recording)
├── stream.py        # Stream class (time-series data)
├── events.py        # Events class (event/annotation data)
├── tabular.py       # BaseTabular (shared base for Stream and Events)
├── epochs.py        # Epochs class + epoch construction helpers
├── preprocess/      # Preprocessing functions (interpolation, concat, etc.)
├── export/          # BIDS and Cloud format exporters
├── video/           # Video class and surface/marker utilities
├── vis/             # Visualization helpers
└── utils/           # Shared utilities, variables, doc decorators
tests/               # pytest test suite
source/              # Sphinx documentation source
```

## Core Classes

| Class | Description |
|---|---|
| `Dataset` | Container for a directory holding multiple recordings |
| `Recording` | Single multimodal recording; exposes `gaze`, `imu`, `eye_states`, `events`, `video`, etc. as lazy properties |
| `Stream` | Time-series data (pandas `DataFrame` indexed by `"timestamp [ns]"`) |
| `Events` | Event/annotation table; subclass of `BaseTabular` |
| `BaseTabular` | Shared validation and dtype-coercion logic for `Stream` and `Events` |
| `Video` | OpenCV-backed scene video with detection and mapping methods |
| `Epochs` | Epoch container built from a `Stream` and an `epochs_info` DataFrame |

## Python Version & Dependencies

- **Python ≥ 3.10** is required; use modern type-hint syntax (`X | Y`, `list[int]`, etc.) instead of `Optional` / `Union` where possible.
- Core runtime dependencies: `pandas`, `numpy`, `matplotlib`, `scipy`, `opencv-python>=4.7`, `joblib`, `typeguard`, `requests`, `tqdm`.
- Development extras: `pytest`.
- Documentation extras: `sphinx`, `numpydoc`, `pydata-sphinx-theme`, `nbsphinx`, `ruff`, `isort`.

## Code Style

- **Formatter**: [Ruff](https://docs.astral.sh/ruff/) — run `ruff format .`
- **Import order**: [isort](https://pycqa.github.io/isort/) with `--profile black` — run `isort --profile black .`
- Both are enforced automatically in CI (`.github/workflows/main.yml`) on every push to `main` or `dev`.
- Line length follows Ruff defaults (88 characters).
- Prefer `pathlib.Path` over raw strings for filesystem paths.
- Use `warnings.warn(..., UserWarning)` for recoverable issues; raise typed exceptions for programming errors.

## Docstrings

- Use **NumPy docstring format** throughout (enforced via `numpydoc` in Sphinx).
- Reuse common parameter/return documentation via the `fill_doc` decorator:

  ```python
  from .utils.doc_decorators import fill_doc

  @fill_doc
  def my_func(max_gap_ms: int = 500):
      """
      Short one-line summary.

      Parameters
      ----------
      %(max_gap_ms_param)s
      """
  ```

  Add new reusable snippets to `pyneon/utils/doc_decorators.py` in the `DOC` dict.

## Data Conventions

- All timestamps are **UNIX timestamps in nanoseconds** stored as `int64`.
- `Stream.data` is a `pandas.DataFrame` indexed by `"timestamp [ns]"`.
- `Events.data` is a `pandas.DataFrame` indexed by the event-type's ID column (e.g. `"blink id"`, `"fixation id"`, `"event id"`).
- Column names and dtypes are governed by `pyneon/utils/variables.py`:
  - `data_types`: maps known column names to their expected pandas/NumPy dtype.
  - `nominal_sampling_rates`: maps stream types to their nominal Hz.
  - `native_to_cloud_column_map`: maps native-format column names to Cloud-format names.
- When adding a new data column, add its dtype to `data_types` in `variables.py`.

## Testing

- Test runner: **pytest**.
- Tests live in `tests/`; shared fixtures are in `tests/conftest.py`.
- Run the full suite: `pytest tests -p no:cacheprovider -p no:faulthandler -p no:unraisableexception`
- Fixtures that require sample data call `get_sample_data()` to download from OSF; these are `scope="package"` fixtures.
- Synthetic (offline) fixtures (e.g. `sim_gaze`, `sim_imu`) are constructed with `numpy`/`pandas` and do not need network access.
- When adding a new feature, add matching tests using the existing synthetic or sample-data fixtures where possible.

## Common Patterns

### Inplace vs. returning a new instance

Many `Stream` and `Events` methods accept an `inplace` parameter:

```python
def crop(self, ..., inplace: bool = False):
    # compute new_data ...
    if inplace:
        self.data = new_data
        return None
    return Stream(new_data)
```

### Adding a new stream type to `Recording`

1. Add the expected column(s) to `data_types` in `variables.py`.
2. Add the nominal sampling rate (if applicable) to `nominal_sampling_rates`.
3. Add a `@cached_property` to `Recording` that reads and returns a `Stream`.
4. Expose it in `Recording.__repr__` / `info` as appropriate.

### Export to BIDS

BIDS exporters live in `pyneon/export/`. Follow the existing pattern in `export_bids.py` and `_bids_parameters.py`.
