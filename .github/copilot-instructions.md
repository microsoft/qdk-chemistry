# Copilot Instructions for qdk-chemistry

QDK/Chemistry is a C++20 and Python package in the Azure Quantum Development Kit.
It provides tools for quantum chemistry: molecular setup, Hamiltonian generation,
quantum algorithm execution, and results analysis. C++ core (`cpp/`), Python API
(`python/`) via pybind11. Package: `qdk-chemistry` (pip) / `qdk_chemistry` (import).

## Design Principles

- **Strict separation of algorithms and data.** Data classes are immutable.
  Algorithms are stateless — all configuration lives in `Settings` objects, never
  as ad-hoc instance variables.
- **Fixed, explicit run signatures.** Each algorithm type has one `_run_impl()`
  signature (see the `domain-reference` skill for the full table). No overload
  families in C++. No runtime `**kwargs` in Python — configuration belongs in
  `Settings`, not in `run()` arguments. The `create(..., **kwargs)` kwargs go to
  `Settings.update()`, which is the configuration mechanism.
- **Algorithm internal state is always Settings.** If an algorithm needs
  configuration, declare it via `settings().set_default()` in `__init__`.

## Working Style

- **Never commit on behalf of the developer.** Stage and describe changes, but
  do not run `git commit`.
- **Consult the developer** on public API changes, new dependencies, algorithm
  semantics, data model changes, and architecture decisions. The developer is a
  domain expert in quantum chemistry, quantum algorithms, and general programming.
- **Tests are long — be surgical.** Run only the specific test files or functions
  relevant to the change, not the full suite. CI/CD handles full verification.
  Use the `cpp-build` and `python-build` skills for test commands and guidance.
- **Use skills.** Skills exist for building, testing, linting, docs, adding
  algorithms, adding bindings, and domain reference. Invoke them rather than
  guessing at commands. If an important workflow doesn't have a skill, write one
  and ask the developer for feedback.
- **Use the `domain-reference` skill** when you need context about quantum
  chemistry conventions, data structure semantics, plugin conversion details,
  algorithm registry mechanics, or design rationale.

## Documentation Principles

- **Be generic, not specific** in conceptual and user-facing docs. Don't
  restrict a component's description to its current consumers — describe its
  general purpose, then mention specific uses as examples. Exempt: API reference
  docs, behavioral constraints, and compatibility notes should be precise.
- **Don't advertise unadvertised features.** Some implementations exist in
  source but are not ready for public-facing documentation. Currently omitted:
  qDRIFT (`"qdrift"`) and partially randomized (`"partially_randomized"`) time
  evolution builders.
- **Link liberally.** Use `:class:`, `:doc:`, `:ref:`, and `:term:`
  cross-references whenever mentioning an algorithm class, data class, or
  glossary term.

## Key Conventions

### C++

- Namespace: `qdk::chemistry`
- Private members: `_prefix` (e.g., `_data`)
- Headers: `cpp/include/qdk/chemistry/`, implementations: `cpp/src/qdk/chemistry/`

### Python

- Google-style docstrings (Parameters, Returns, Raises, Examples)
- Type hints required (PEP 484)
- Line length: 120 characters, double quotes (Ruff)
- **Single-line docstring params.** Keep each `Args:` / `Returns:` / `Raises:`
  parameter description on one line — no continuation/wrap. The Sphinx extension
  order causes `(ERROR/3) Unexpected indentation` on multi-line params.

## Available Skills

| Skill | When to use |
|-------|-------------|
| `cpp-build` | Building C++ code, running C++ tests, diagnosing CMake errors |
| `python-build` | Installing Python package, running pytest, diagnosing pip errors |
| `lint` | Running pre-commit hooks, fixing lint/format/type errors |
| `docs-build` | Building Sphinx documentation, diagnosing doc warnings |
| `add-algorithm` | Adding a new algorithm variant to the registry |
| `add-binding` | Adding pybind11 bindings for a C++ class |
| `domain-reference` | Domain context: conventions, data structures, plugins, design rationale |
| `acquire-codebase-knowledge` | Full codebase scan and structural documentation |
