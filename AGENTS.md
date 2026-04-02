# AGENTS Guide for `gnn_predict`

This guide is for coding agents operating in this repository.
It captures runnable commands and repository-specific engineering norms.

## Scope
- Apply these rules to `main.py`, `src/`, `tests/`, and `scripts/`.
- Treat `tmp/` as non-core/experimental unless explicitly asked to modify it.
- Prefer small, reviewable changes over broad refactors.

## 1. Environment and setup
- Python: `3.12` (see `.python-version`).
- Dependency manager/runtime: `uv`.
- Install/sync deps (including dev): `uv sync --dev`.
- If `uv` is unavailable, install it first: `pip install uv` (or platform installer).
- Root working directory is expected for all commands below.

## 2. Repository map
- Entry point CLI: `main.py`.
- Core package: `src/gnn_archs/`.
- Config schema and validation: `src/gnn_archs/config.py`.
- Variant expansion: `src/gnn_archs/util/variant_expander.py`.
- Legacy config migration: `src/gnn_archs/util/config_migration.py`.
- Runtime execution/train/infer/export: `src/gnn_archs/variant_runner.py`.
- Mutation implementations: `src/gnn_archs/mutations.py`.
- Output/result models: `src/gnn_archs/result.py`.
- Tests: `tests/`.
- Arch configs: `config/arch/*.yaml`.

## 3. Build / run commands
Notes:
- This is an application/research repo, not a wheel-focused package build.
- "Build" is primarily environment sync plus runtime/test validation.

- Setup/build environment:
  - `uv sync --dev`
- Run CLI on one config:
  - `uv run python main.py --config config/arch/resnet_variants.yaml --output_dir output --gpu_node node0`
- Run CLI on multiple configs:
  - `uv run python main.py --config config/arch/resnet_variants.yaml config/arch/vit_variants.yaml --output_dir output --gpu_node node0`
- Run migration script (requires `PYTHONPATH`):
  - `PYTHONPATH=src uv run python scripts/migrate_arch_configs.py --help`

## 4. Lint / format commands
- Lint core code:
  - `uv run ruff check main.py src scripts tests`
- Auto-fix lint where safe:
  - `uv run ruff check --fix main.py src scripts tests`
- Format:
  - `uv run ruff format main.py src scripts tests`
- Check formatting only:
  - `uv run ruff format --check main.py src scripts tests`

Lint scope guidance:
- Prefer linting touched files or core paths above.
- `tmp/` currently contains non-core files that are not lint-clean.

## 5. Type-check commands
- Type checker available: `ty`.
- Run type checks:
  - `uv run ty check src main.py tests`
- Current baseline includes existing diagnostics in `src/gnn_archs/mutations.py`.
- Do not introduce new typing regressions in files you touch.

## 6. Test commands
- Full test suite:
  - `uv run python -m pytest -q`
- Verbose with stop-on-first-failure:
  - `uv run python -m pytest -x -vv`
- Run a single file:
  - `uv run python -m pytest -q tests/test_variant_runner.py`
- Run a single test (node id):
  - `uv run python -m pytest -q tests/test_main.py::test_main_processes_single_config_file`
- Run a single parametrized case:
  - `uv run python -m pytest -q "tests/test_arch_configs.py::test_arch_configs_validate_and_expand[resnet_variants.yaml]"`
- Run by keyword expression:
  - `uv run python -m pytest -q tests/test_variant_runner.py -k "convnext and dropout"`

Important test notes:
- Tests rely on `tests/conftest.py` adding `src/` to `sys.path`.
- Some tests import `torch` at collection time.
- If the environment has CUDA/NCCL mismatch, full test collection may fail before running tests.

## 7. Import and module conventions
- Use `from __future__ import annotations` at top of Python modules.
- Order imports as: stdlib, third-party, local package.
- Let Ruff's import sorter (`I`) enforce ordering.
- Put type-only imports under `if TYPE_CHECKING:` when possible.
- Avoid runtime imports solely for annotations (Ruff `TC00x` rules are enabled).

## 8. Formatting and general style
- Target line length: 88 characters (Ruff default `E501` behavior).
- Prefer clear, short functions with explicit names.
- Prefer early validation and early returns over deep nesting.
- Keep side effects localized; favor pure helpers for transformations.
- Use UTF-8 file IO with explicit `encoding="utf-8"` when reading/writing text.

## 9. Typing guidelines
- Annotate public function signatures (including return types).
- Use precise container types (`list[int]`, `dict[str, float]`, etc.).
- Prefer `X | None` over `Optional[X]` for consistency with project style.
- Avoid `Any` unless unavoidable around third-party model outputs.
- When needed, isolate dynamic typing boundaries in small helper functions.

## 10. Naming conventions
- Functions/variables/modules: `snake_case`.
- Classes/dataclasses/pydantic models: `PascalCase`.
- Constants: `UPPER_SNAKE_CASE`.
- Test names: `test_<behavior>`.
- Mutation type names are string identifiers in `CamelCase` and should match canonical sets in `IMAGE_MUTATION_TYPES` and `TEXT_MUTATION_TYPES`.

## 11. Config, models, and data-shape rules
- Pydantic models are strict (`extra="forbid"` via `StrictModel`).
- Validate external/loaded YAML with `ArchConfig.model_validate(...)`.
- Keep config migration logic in `util/config_migration.py` rather than ad-hoc transforms.
- Keep variant expansion logic in `util/variant_expander.py`.
- Serialize result payloads through pydantic models and `model_dump(mode="json")`.

## 12. Error handling and logging
- Raise `ValueError` for invalid values/constraints.
- Raise `TypeError` for invalid runtime object/module types.
- Raise `NotImplementedError` for unsupported mutations/paths not yet migrated.
- Do not silently swallow exceptions; avoid bare `except:`.
- In runtime paths, prefer structured `logging` over `print`.
- Keep exception messages specific and actionable (include parameter/layer names).

## 13. Testing style expectations
- Use `pytest` with focused, behavior-oriented tests.
- Prefer `tmp_path` for filesystem effects.
- Keep tests deterministic (fixed seeds and deterministic fake inputs where applicable).
- When adding features, include at least one positive-path and one failure-path test.
- For mutation behavior, assert both structural change and successful forward-pass shape.

## 14. External agent-rule files status
The following rule files were checked and are currently absent:
- `.cursor/rules/`
- `.cursorrules`
- `.github/copilot-instructions.md`

If any of these files are added later, treat them as higher-priority supplements to this guide.

## 15. Change checklist for agents
- Run relevant Ruff checks on touched files.
- Run the smallest meaningful pytest scope first, then broaden.
- If touching config schema/expansion/migration logic, add or update tests in `tests/test_config_migration.py`.
- If touching config schema/expansion/migration logic, add or update tests in `tests/test_variant_expander.py`.
- If touching config schema/expansion/migration logic, add or update tests in `tests/test_arch_configs.py`.
- If touching runtime execution/mutations, update `tests/test_variant_runner.py` and/or `tests/test_main.py`.
- Keep outputs backward-compatible unless the task explicitly changes schema/CLI contracts.
- Document any environment-dependent failures (for example CUDA/NCCL) in final notes.
