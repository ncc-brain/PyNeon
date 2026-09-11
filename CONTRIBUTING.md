# Contributing to PyNeon

Thank you for your interest in contributing to PyNeon. We welcome contributions in the form of code, documentation, bug reports, and suggestions for new features or sample datasets. This guide outlines the contribution process and the expectations for contributors.

## Reporting Bugs or Suggesting Features

Contributions are welcome even if you do not plan to submit code. Bugs, documentation issues, feature requests, and suggestions for sample datasets can be reported via GitHub issues. Please search existing issues before opening a new one to avoid duplication.

GitHub issue templates are provided to help ensure that reports include the necessary information. To get started, click `New issue` and select the appropriate template.

## Contributing Code or Documentation

Contributions of code or documentation are submitted via pull requests (PRs). The general workflow is:

1. Set up a development environment.
2. Make and test your changes.
3. Submit a pull request for review.

### Create a Development Environment

We recommend using a virtual environment to manage development dependencies. For example, using `conda`:

```bash
conda create -n pyneon-dev python=3.10
```

Ensure that the environment uses a Python version supported by PyNeon (Python ≥ 3.10).

### Forking and Cloning the Repository

Fork the PyNeon repository on GitHub and clone your fork locally:

```bash
git clone https://github.com/<username>/PyNeon.git
cd PyNeon
```

### Installing PyNeon in Editable Mode

Install PyNeon in editable mode along with development dependencies:

```bash
pip install -e .[dev]
```

### Code Edits

When editing code, please follow the existing style and conventions, and reuse existing utilities (e.g., those in `utils` module) whenever possible to ensure consistency and maintainability. If you are adding a new feature, please also include corresponding tests, using existing synthetic or sample‑data fixtures where feasible.

PyNeon mandates input type checking with `typeguard`. All public functions and methods must include type annotations, and these annotations should be accurate, complete, and meaningful.

### Policy on AI Assistance

PyNeon permits the use of AI‑assisted tools in contributions. Contributors remain responsible for the scientific content, correctness, and integration of any such contributions. Familiarity with the PyNeon codebase, appropriate domain knowledge, and human judgment are required to ensure that contributions meet project standards. AI tools should be used as an aid to development and documentation, not as a substitute for authorial responsibility.

Issues and pull requests involving AI‑assisted work should include a declaration using the [GAIDeT (Generative AI delegation taxonomy)](https://doi.org/10.1080/08989621.2025.2544331). Contributors are encouraged to create their declaration with the [GAIDeT declaration template](https://panbibliotekar.github.io/gaidet-declaration/) and include it in the issue or pull request description, or, where appropriate, in individual commits.

All AI‑assisted contributions must be reviewed by a human contributor, who assumes full responsibility for their content, originality, and compliance with PyNeon’s standards. Contributors are also expected to engage directly with maintainers during the review process.

The maintainers may request clarification regarding the use of AI assistance and reserve the right to close issues or pull requests that do not comply with these guidelines.

### Pull Requests and Testing

Once your changes are ready, open a pull request against the `dev` branch of the main repository (`ncc-brain/PyNeon`). Include a clear description of the changes and their motivation.

Continuous integration checks, including tests and code formatting, are run automatically on all pull requests. Contributors are encouraged to run these checks locally before submission. Typical commands:

```bash
# run tests
pytest tests -q

# fix import order
isort --profile black .

# format code
ruff format .
```

Pull requests are reviewed by the maintainers, who may request revisions or clarification. We appreciate your contributions and your efforts to help improve PyNeon!
