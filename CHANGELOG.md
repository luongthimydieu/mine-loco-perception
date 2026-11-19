# Changelog

## [1.1.0] — 2026-11-19

### Added
- Repository packaging: `pyproject.toml`, `requirements.txt`, `requirements-dev.txt`, editable install,
  `mine-loco` console entry point.
- CI (`.github/workflows/ci.yml`): Ubuntu + Windows, Python 3.11/3.13, ruff, compileall, pytest, site link check.
- One-click scripts: `scripts/run_all.py`, `scripts/verify_workspace.py`,
  `scripts/download_exdark.py`, `scripts/download_weights.py`, `scripts/make_800_subset.py`.
- Analysis: `scripts/per_class_analysis.py` (per-class AP), `scripts/threshold_sweep.py`
  (deployment confidence/F1), `scripts/check_site_links.py`.
- Docs: model card, data card, technical report, roadmap, FAQ, architecture, deployment.
- Static project page `site/` with a visibility calculator, demo video, and OG metadata.
- Desktop packaging: `packaging/mine_loco.spec`, `build_windows.bat`, release workflow,
  `docs/packaging.md`, `THIRD_PARTY_NOTICES.md`.

### Changed
- All experiment scripts now resolve paths through `mine_loco_system.paths`; data/weights/outputs are
  configurable via `MINE_LOCO_DATA`, `MINE_LOCO_WEIGHTS`, `MINE_LOCO_OUTPUTS`.
- System package imports use repository-relative paths; demo works with full or 800-image ExDark.
- Fixed a dead-code/undefined-name bug in the perception page and a missing `kind` argument in the
  fusion demo; removed unused imports and variables (ruff clean).

### Fixed
- `leak_diagnostic.py` wrote its JSON to an undefined `ROOT`; now writes `outputs/repro/leak_diagnostic.json`.
- `redraw_figs.py` output path now `outputs/figures/corrected/`.
- `run_scaleup.bat` logs to `outputs/logs/`.

## [1.0.0] — 2026-11-19

Initial public code release:
- Corrected three-way evaluation protocol and leakage diagnostic.
- 800-image negative result and 5000-image scale-up + ablation.
- E2 robust fusion, E3 LiDAR boundary, E4 health/RUL.
- PySide6 six-page demo.
- V=30/15 extended evaluation, TTA, model soup, benchmark.
