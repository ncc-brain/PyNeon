# Contributing to PyNeon

Thank you for your interest in contributing to PyNeon. We welcome bug reports, feature requests, documentation improvements, and code changes from everyone.

## Use Issues to Report Problems or Suggest Features

Even if you do not plan to submit code, you can still help by reporting bugs or suggesting new features. Please check existing issues before opening a new one to avoid duplicates.

We provide GitHub issue templates for both bug reports and feature requests to help you include the right details. Click "New issue" and select the appropriate template to get started.

## Contributing Code or Documentation

If you would like to contribute code or documentation, you will need to set up a development environment, make your changes, and submit a Pull Request (PR). The general process is as follows:

### Create a Development Environment

We recommend using a virtual environment to manage your development dependencies. For example, with `conda`:

```bash
conda create -n pyneon-dev python=3.10
```

Make sure to create the environment with a Python version supported by PyNeon (>=3.10).

### Fork the Repository and Clone Your Fork Locally

Fork the repository on GitHub and then clone your fork to your local machine:

```bash
git clone https://github.com/<username>/PyNeon.git
cd PyNeon
```

### Install PyNeon in Editable Mode (`-e`) with Development Dependencies

```bash
pip install -e .[dev]
```

### Pull Requests and Testing

Once you have made your changes, open a Pull Request (PR) against the `dev` branch of the original repository (`ncc-brain/PyNeon`). Please include a clear description of your changes and the motivation behind them.

On GitHub, we run tests and check code formatting automatically on every PR. This helps ensure that contributions meet our quality standards before they are merged. If you would like to run the tests and formatter locally before submitting your PR, use the following commands:

```bash
pytest tests -q
isort --profile black .
ruff format .
```

The maintainers will review your PR and may request changes or ask questions. We appreciate your help in improving PyNeon!
