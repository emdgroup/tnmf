[![Flake8 Linter](https://github.com/emdgroup/tnmf/actions/workflows/flake8.yml/badge.svg)](https://github.com/emdgroup/tnmf/actions/workflows/flake8.yml)
[![Pylint Linter](https://github.com/emdgroup/tnmf/actions/workflows/pylint.yml/badge.svg)](https://github.com/emdgroup/tnmf/actions/workflows/pylint.yml)
[![Pytest](https://github.com/emdgroup/tnmf/actions/workflows/pytest.yml/badge.svg)](https://github.com/emdgroup/tnmf/actions/workflows/pytest.yml)
[![Build Documentation](https://github.com/emdgroup/tnmf/actions/workflows/sphinx.yml/badge.svg)](https://github.com/emdgroup/tnmf/actions/workflows/sphinx.yml)

[![Logo](doc/logos/tnmf_header.svg)](https://github.com/emdgroup/tnmf)

# Transform-Invariant Non-Negative Matrix Factorization

A comprehensive Python package for **Non-Negative Matrix Factorization (NMF)** with a focus on learning *transform-invariant representations*.

The packages supports multiple optimization backends and can be easily extended to handle application-specific types of transforms.

# General Introduction
A general introduction to Non-Negative Matrix Factorization and the purpose of this package can be found [here](doc/GeneralIntroduction.md) and on the package's [PyPI page](https://pypi.org/project/tnmf/).

# Installation
For using this package, you will at least need Python version 3.6 (or higher).

Installation is easiest using pip:

    pip install tnmf

# Demos and Examples

The package comes with a number of demos and examples that demonstrate the capabilities of the TNMF model and provide a
good starting point for your own experiments.
* To execute a particular streamlit demo, run `streamlit run demos/<demo_name>`.
* A specific example can be executed by calling `python examples/<example_name>`.

# License
Copyright (c) 2021 Merck KGaA, Darmstadt, Germany

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

The full text of the license can be found in the file [LICENSE](LICENSE) in the repository root directory.

# Contributing
Contributions to the package are always welcome and can be submitted via a pull request.
Please note, that you have to agree to the [Contributor License Agreement](CONTRIBUTING.md) to contribute.

## Working with the Code
To checkout the code and set up a working environment with all required Python packages, execute the following commands:

```
git checkout https://github.com/emdgroup/tnmf.git ./tnmf
cd tmnf
python3 -m virtualenv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Now, you should be able to execute the unit tests by calling `pytest` to verify that the code is running as expected.

## Pull Requests
Before creating a pull request, you should always try to ensure that the automated code quality and unit tests do not fail.
To execute them, change into the repository root directory and run the following commands:

```
flake8
pylint tnmf
pytest
```

The output of the individual commands should be pretty instructive, so fixing potential issues is usually rather straightforward.

## Building the Documentation
To build the documentation locally, change into the `doc` subdirectory and run `make html`.
Then, the documentation resides at `doc\_build\html\index.html`.
