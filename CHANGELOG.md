# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.1.0] - 2023-01-23
### Added
- Project examples.

### Changed
- Use CMake instead of Makefile for build system.

## [1.0.0] - 2022-08-20
### Added
- Data log visualizer script
- Sample input sequences
- System tests

### Changed
- Align method returns aligned sequences instead of the final score

## [0.8.0] - 2022-01-30
### Added
- Traceback matrix.

### Fixed
- Arbitrary sequence length alignment issue.

### Removed
- CUDA stream support.

## [0.7.0] - 2022-01-18
### Changed
- Asynchronous device to host memory transfer.

### Changed
- Start using payload threshold for submatrix partitioning.

## [0.6.0] - 2022-01-09
### Changed
- Start using CUDA streams.

### Fixed
- Anti-diagonal copy boundary issue.

## [0.5.0] - 2021-12-23
### Changed
- Start using submatrix fill in each iteration.

### Fixed
- Lower matrix line miscalculating issue.

## [0.4.0] - 2021-12-21
### Changed
- Start using anti-diagonal major ordering for score matrix.

### Fixed
- Vector indexing issue for source sequence bigger than reference.

## [0.3.0] - 2021-12-19
### Changed
- Start using iterative kernel launch in matrix generation.

### Fixed
- CUDA thread count overflow for the kernel launch.
- Amino-acid sequence limit issue.

## [0.2.0] - 2021-12-18
### Added
- Random sequence generator.

### Changed
- Start using CUDA cooperative groups.

## [0.1.0] - 2021-12-13
### Added
- Final score calculator in serial approach.
- Final score calculator with CUDA.
- Score matrix generator in serial approach.
- Score matrix generator with CUDA.
