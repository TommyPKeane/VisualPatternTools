# VisualPatternTools Python Package

Custom Python Package `visualpatterntools` (VisualPatternTools) provides some common utilities, classes, structures, and coding patterns for working in Computer Vision and Machine Learning with PyTorch and various other common Python Packages.

Data Preprocessing, Algorithm Development, Algorithm Training, Productionalization, and Post-Processing functionality is all centralized into this package in generic utilities that then are used in some examples in the `scripts/` directory.

## Table of Contents

<!-- MarkdownTOC -->

- [Developer Setup](#developer-setup)
	- [Development Tools Setup \(macOS\)](#development-tools-setup-macos)
	- [Install Dependencies for Developers](#install-dependencies-for-developers)
	- [Developer Loop](#developer-loop)
- [Build and Publish the Package](#build-and-publish-the-package)
- [Accessing the Built Packages](#accessing-the-built-packages)
- [References](#references)

<!-- /MarkdownTOC -->

<a id="developer-setup"></a>
## Developer Setup

These subsections explain how to setup your system to have the proper development tools, and then the final subsection gives you the common steps for a developer setup of the package locally so you can run the unit-tests or the example scripts.

<a id="development-tools-setup-macos"></a>
### Development Tools Setup (macOS)

1. Follow https://github.com/TommyPKeane/example-bash-configuration to install `bash`, `brew`, and `pyenv`
1. `brew update`
1. `brew upgrade`
1. `brew install uv`
1. `uv tool install ruff`

<a id="install-dependencies-for-developers"></a>
### Install Dependencies for Developers

1. `pyenv install` (only needed once)
1. `direnv allow`
1. `uv tool upgrade --all`
1. `uv sync --active` (flag needed to avoid extra `venv` creation already handled by `direnv`)

<a id="developer-loop"></a>
### Developer Loop

As you make changes, you should run the following to keep the code formatted and checking for syntax consistency/best-practices through the `ruff` linter with:

1. `ruff format ./`
1. `ruff check --fix ./`

All of the `ruff` formatting and linting (checks) are configured in the `pyproject.toml` file.

Once you've gotten your updates formattted and checked, you can then run the unit-testing with:

```bash
pytest
```

The PyTest configuration is also in the `pyproject.toml` file, and it will run the Code Coverage summary as well.

<a id="build-and-publish-the-package"></a>
## Build and Publish the Package

> _TBD..._

<a id="accessing-the-built-packages"></a>
## Accessing the Built Packages

> _TBD..._

<a id="references"></a>
## References

- PyTorch: https://docs.pytorch.org/docs/stable/index.html
- PyTorch Lightning: https://lightning.ai/docs/pytorch/stable/
- NumPy: https://numpy.org/doc/stable/reference/index.html#reference
- scikit-learn: https://scikit-learn.org/stable/api/index.html
