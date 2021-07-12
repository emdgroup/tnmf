# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2021-07-12
### Added (since creation of the changelog)
- Initial version of a documentation
- Example: convergence control
- Example: shift-invariant decomposition
- `tnmf` console entry point
- Packaging information
- GitHub workflow for publishing to PyPI and TestPyPI
- Support for `circular` boundary conditions in the Numpy_FFT backend
- Pytorch_FFT backend
- Support for `full` and `circular` convolution in CachingFFT backend
- Demo: 1-D synthetic signals
- Demo: 2-D synthetic signals
- Online Demo via Streamlit Sharing

### Changed (since creation of the changelog)
- Only upload Sphinx artifacts from GitHub actions that run on the main branch
- Fix PyTorch based backends for non-2D signals
- Fix links in README.md and setup.cfg
