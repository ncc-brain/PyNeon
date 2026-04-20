---
applyTo: "pyneon/**/*.py,tests/**/*.py"
description: "Use when editing PyNeon Python source or tests."
---

# Python Source And Test Guidance

- Target Python 3.10+ and use modern type hints (`X | Y`, `list[str]`).
- Keep timestamp conventions intact: nanosecond UNIX timestamps as `int64`.
- Preserve existing DataFrame index conventions:
  - `Stream`: index is `timestamp [ns]`.
  - `Events`: index is the event ID column (for example `event id`, `blink id`).
- When adding data columns, update dtype mappings in `pyneon/utils/variables.py`.
- Prefer `pathlib.Path` for filesystem operations.
- Use `warnings.warn(..., UserWarning)` for recoverable runtime issues and typed exceptions for programming errors.
- Follow existing API patterns around `inplace` behavior (mutate + `None` vs new instance return).

# Testing Expectations

- Add or update `pytest` tests in `tests/` for behavior changes.
- Favor synthetic fixtures when possible; only use sample-data fixtures when required.
- Keep tests deterministic and avoid unnecessary network dependency.
