# Repository Guidelines

## Project Structure & Module Organization

This is a Python 3.12 research/application repository managed with `uv`.
The CLI entry point is `main.py`; monitoring helpers start from `monitor.py`.
Core architecture mutation and variant code lives in `src/gnn_archs/`, while
GNN model training, evaluation, and pipeline logic lives in `src/gnn_model/`.
Shared helpers belong in `src/common/`. Configuration files are under
`config/arch/`, `config/gnn_model/`, and `config/monitor/`. Tests live in
`tests/`; scripts for one-off maintenance live in `scripts/`; longer design
notes and migration records live in `docs/`.

## Build, Test, and Development Commands

- `uv sync --dev`: install runtime and development dependencies.
- `uv run python main.py --config config/arch/resnet_variants.yaml --output_dir output --gpu_node node0`: run one architecture config.
- `PYTHONPATH=src uv run python scripts/migrate_arch_configs.py --help`: inspect the config migration script.
- `uv run ruff check main.py src scripts tests`: lint core paths.
- `uv run ruff format main.py src scripts tests`: format core paths.
- `uv run ty check src main.py tests`: run static type checks.
- `uv run python -m pytest -q`: run the full test suite.

## Coding Style & Naming Conventions

Use future annotations in Python modules: `from __future__ import annotations`.
Keep imports ordered as stdlib, third-party, then local package; let Ruff handle
sorting and formatting. Use `snake_case` for functions, variables, and modules,
`PascalCase` for classes and Pydantic models, and `UPPER_SNAKE_CASE` for
constants. Prefer explicit public function signatures and precise container
types such as `dict[str, float]`. Keep shared utilities in `src/common/`; keep
module-specific helpers local to their package.

## Testing Guidelines

Tests use `pytest` and should be behavior-oriented with names like
`test_main_processes_single_config_file`. Prefer deterministic fixtures and
`tmp_path` for filesystem effects. When changing config schema, migration, or
variant expansion, update tests around `test_config_migration.py`,
`test_variant_expander.py`, and `test_arch_configs.py`. When changing runtime or
mutation behavior, update `test_variant_runner.py` or `test_main.py`.

## Commit & Pull Request Guidelines

Recent history follows Conventional Commit-style subjects, for example
`feat: add recommender model support`, `fix(config): update result JSON paths`,
and `chore: add paper repo as submodule`. Keep commits small and scoped. Pull
requests should explain the behavior change, list validation commands run, link
related issues when available, and include screenshots or sample output only
when user-facing output changes.

## Security & Configuration Tips

Do not commit generated outputs, local credentials, or large experimental
artifacts. Keep reusable configuration in `config/`; treat `tmp/` as
non-core/experimental unless a task explicitly targets it.
