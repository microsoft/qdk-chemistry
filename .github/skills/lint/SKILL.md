---
name: lint
description: "Use when: running linters, fixing lint errors, formatting code, running pre-commit hooks, diagnosing mypy/ruff/clang-format failures. Handles all pre-commit hooks for the qdk-chemistry project."
---

# Lint & Format

## Logging

All commands **must** redirect their full output to a log file so the user can `tail -f` it. Use `> <logfile> 2>&1` to keep output out of agent context. Always tell the user which log file you're writing to. These log files are your primary tool for diagnosing errors — read the log file rather than re-running commands or keeping full output in context.

## Procedure

### 1. Run all hooks

```bash
pre-commit run --all-files > pre-commit.log 2>&1
```

### 2. Run a specific hook

```bash
pre-commit run ruff --all-files > pre-commit.log 2>&1
pre-commit run ruff-format --all-files > pre-commit.log 2>&1
pre-commit run clang-format --all-files > pre-commit.log 2>&1
pre-commit run mypy --all-files > pre-commit.log 2>&1
pre-commit run interrogate --all-files > pre-commit.log 2>&1
pre-commit run check-license-headers --all-files > pre-commit.log 2>&1
pre-commit run check-version-alignment --all-files > pre-commit.log 2>&1
pre-commit run check-pyi-stubs --all-files > pre-commit.log 2>&1
```

### 3. Run on staged files only

```bash
pre-commit run > pre-commit.log 2>&1
```

## Hooks Reference

### Standard hooks (pre-commit/pre-commit-hooks)

- `check-added-large-files` — max 500 MB
- `check-merge-conflict`
- `check-yaml`
- `check-json`
- `check-ast`
- `mixed-line-ending` — auto-fixes to LF
- `trailing-whitespace`
- `end-of-file-fixer`
- `debug-statements`
- `check-case-conflict`
- `check-docstring-first`
- `detect-private-key`

### Formatting & Linting

| Hook | What it does |
|------|-------------|
| `sphinx-lint` | RST validation |
| `gitleaks` | Secret detection |
| `interrogate` | Docstring coverage (80%+, ignores `__init__`) |
| `clang-format` | C/C++ formatting (v20.1.7, `.cpp/.h/.hpp/.cxx/.cc` files) |
| `pretty-format-yaml` | Auto-fix YAML formatting |
| `ruff-format` | Python formatting (double quotes, 120 char lines) |
| `ruff` | Python lint with `--fix` (output-format=concise) |
| `markdownlint` | Markdown lint with `--fix` (line-length disabled, MD024/MD029/MD033 disabled) |
| `mypy` | Type checking (`--ignore-missing-imports --explicit-package-bases --check-untyped-defs`) |

### Local hooks

| Hook | What it does |
|------|-------------|
| `check-license-headers` | Verifies Microsoft license header on `.cpp/.h/.hpp/.cxx/.cc/.c/.py` files (excludes `external/` and `build/`) |
| `check-version-alignment` | Verifies `VERSION` file matches `cpp/CMakeLists.txt`, `python/CMakeLists.txt`, and `docs/source/conf.py` |
| `check-pyi-stubs` | Verifies required `.pyi` type stub files exist for pybind11 modules |

## Key Paths

| Path | Purpose |
|------|---------|
| `.pre-commit-config.yaml` | Hook configuration |
| `python/pyproject.toml` | Ruff and mypy settings (line-length=120, select rules, etc.) |
| `.github/scripts/check_license_headers.py` | License header checker |
| `.github/scripts/check_version_alignment.py` | Version alignment checker |
| `.github/scripts/check_pyi_stubs.py` | `.pyi` stub checker |

## Error Diagnosis / Common Fixes

When something fails, **read `pre-commit.log` first** rather than re-running commands. Pre-commit shows which hook failed and on which file. Many hooks auto-fix — just re-stage the modified files and re-run.

### ruff errors

- Ruff runs with `--fix` so most issues auto-fix. Re-run to check remaining issues.
- Config in `python/pyproject.toml`: line-length=120, select includes `D` (pydocstyle), `SIM` (simplify), and many others.
- `SIM115` (use context manager for open): wrap `open()` calls in `with` statements.
- `D` rules (docstrings): use Google-style docstrings with Parameters, Returns, Raises, Examples.

### ruff-format

- Double quotes, spaces (not tabs), 120 char lines.
- Auto-fixes on run — just re-stage the files.

### clang-format

- Uses `.clang-format` file or Google fallback style.
- Auto-fixes on run — re-stage files after running.

### mypy errors

- Runs with `--ignore-missing-imports --explicit-package-bases --check-untyped-defs`.
- Common fix: add type annotations to function parameters and return types.
- For C++ extension types: ensure `.pyi` stubs exist and are up to date.

### interrogate (docstring coverage)

- Must maintain 80%+ coverage, ignores `__init__` methods.
- Fix: add Google-style docstrings to undocumented public functions/classes.
- Keep each `Args:` parameter description on a **single line** (no continuation/wrap to the next line).

### check-license-headers

- Every `.py` and `.cpp/.h` file needs the Microsoft license header.
- The script at `.github/scripts/check_license_headers.py` has the expected header format.
- Excludes `external/` and `build/` directories.

### check-version-alignment

- Ensures the `VERSION` file at repo root matches versions in `cpp/CMakeLists.txt`, `python/CMakeLists.txt`, and `docs/source/conf.py`.
- Fix: update the `VERSION` file, or update the downstream files to match.

### check-pyi-stubs

- Ensures `.pyi` type stub files exist for pybind11 modules.
- Fix: regenerate stubs or create missing ones.
