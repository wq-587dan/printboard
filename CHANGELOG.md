# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-06-11

### Added
- `@tb_log` decorator: captures print output and auto-logs metrics to TensorBoard
- `tb_print()` function: explicit metric logging with dual output
- Multi-strategy regex parser supporting 5+ common print formats
- Custom regex pattern support via named capture groups
- Thread-safe TensorBoard writer with caching by log directory
- StreamProxy for dual-write stdout (terminal + internal buffer)
- Bilingual documentation (English + Chinese)
- GitHub Actions CI with Python 3.8-3.12 matrix
- Example scripts: basic training demo and custom pattern demo
- MIT License

### Technical Details
- 47 unit tests, 92% code coverage
- Python >= 3.8
- Dependencies: torch >= 1.8.0, tensorboard >= 2.8.0
- Zero runtime overhead for lines without metrics (early-exit optimization)
