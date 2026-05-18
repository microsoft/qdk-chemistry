# Codebase Concerns

## Core Sections (Required)

### 1) TODO/FIXME/HACK Count

**Production code only** (excluding tests and tooling): **3 TODOs total**

| File | Count | Details |
|------|-------|---------|
| `python/src/qdk_chemistry/plugins/pyscf/conversion.py` | 2 | "Handle Cartesian basis sets" (lines 160, 271) |
| `cpp/src/qdk/chemistry/algorithms/microsoft/utils.cpp` | 1 | "Support Cartesian Atomic Orbitals" (line 189) |

Evidence: `grep -r 'TODO\|FIXME\|HACK'` across `python/src/` and `cpp/src/` + `cpp/include/`

### 2) Top Risks (Prioritized)

| Severity | Concern | Evidence | Impact | Suggested action |
|----------|---------|----------|--------|------------------|
| Medium | Incomplete Cartesian basis set support | `python/src/qdk_chemistry/plugins/pyscf/conversion.py:160,271`, `cpp/src/qdk/chemistry/algorithms/microsoft/utils.cpp:189` | Cartesian AOs not handled — will produce incorrect results for affected basis sets | Implement Cartesian-to-spherical conversion or native Cartesian support |
| Medium | Large C++ files with mixed responsibilities | `element_data.cpp` (3763 lines), `basis_set.cpp` (2393), `orbitals.cpp` (2051), `wavefunction.cpp` (2047) | Hard to navigate, review, and test in isolation | Decompose into smaller compilation units |
| Low | Hardcoded telemetry endpoint and key | `python/src/qdk_chemistry/utils/telemetry.py:54-79` | Not a security risk (public ingestion key), but complicates deployment to different environments | Consider making endpoint configurable |

### 3) Technical Debt

| Debt item | Why it exists | Where | Risk if ignored | Suggested fix |
|-----------|---------------|-------|-----------------|---------------|
| Cartesian AO TODOs | Spherical-only implementation was sufficient for initial scope | `pyscf/conversion.py`, `utils.cpp` | Silent incorrect results for Cartesian basis sets | Implement conversion or raise explicit error |
| Large monolithic C++ files | Organic growth of data class implementations | `cpp/src/qdk/chemistry/data/` | Slower compilation, harder code review | Split by sub-functionality (e.g., serialization vs. computation) |
| Settings.cpp at 1965 lines | Settings is a complex key-value store with type coercion | `cpp/src/qdk/chemistry/data/settings.cpp` | Maintenance burden | Consider splitting validation/serialization logic |

### 4) Security Concerns

| Risk | OWASP category | Evidence | Current mitigation | Gap |
|------|----------------|----------|--------------------|-----|
| Untrusted deserialization | A08 (Software and Data Integrity) | `SECURITY.md:16-29` | Documented warning against pickle; recommends JSON/HDF5 | No runtime enforcement — users could still load pickle files |
| Telemetry data transmission | N/A | `telemetry.py:236-263` | Opt-out via env var; anonymous data only | Endpoint/key are hardcoded |

### 5) Performance and Scaling Concerns

| Concern | Evidence | Current symptom | Scaling risk | Suggested improvement |
|---------|----------|-----------------|-------------|-----------------------|
| High-churn Hamiltonian code | `cpp/src/qdk/chemistry/algorithms/microsoft/cholesky_hamiltonian.cpp` (6 changes in 90 days) | Frequent modifications suggest active development / instability | Regression risk | Increase test coverage for Cholesky path |
| Large binary test data | `external/macis/tests/ref_data/` (27MB+ binary files) | Slow clone / CI checkout | Repository bloat | Consider Git LFS or external data hosting |

### 6) Fragile/High-Churn Areas

| Area | Why fragile | Churn signal | Safe change strategy |
|------|-------------|-------------|----------------------|
| `python/pyproject.toml` | Central build/dependency config | 8 changes in 90 days | Test install from clean venv after changes |
| `docs/source/conf.py` | Sphinx config with multiple extensions | 8 changes in 90 days | Run `make clean all` in docs/ after changes |
| `python/tests/test_phase_estimation_iterative.py` | Complex quantum algorithm test | 6 changes in 90 days | Run targeted test; check for flakiness |
| `cpp/src/qdk/chemistry/algorithms/microsoft/cholesky_hamiltonian.cpp` | Active feature development | 6 changes in 90 days | Run related C++ tests after changes |
| `cpp/include/qdk/chemistry/data/wavefunction.hpp` + `.cpp` | Core data class under active evolution | 4 changes each in 90 days | Run full wavefunction test suite |

### 7) `[ASK USER]` Questions

1. [ASK USER] Is there a target minimum coverage threshold for Python or C++ tests?
2. [ASK USER] Are there plans to support Cartesian basis sets, or should the code raise an explicit error when they are encountered?
3. [ASK USER] Should the large C++ data files (element_data.cpp, basis_set.cpp, etc.) be refactored into smaller units, or is the current structure intentional?
4. [ASK USER] Are the binary test data files in `external/macis/tests/ref_data/` intended to stay in-repo, or should they be moved to Git LFS?

### 8) Evidence

- Scan output: TODO/FIXME section, high-churn files, code metrics
- `python/src/qdk_chemistry/plugins/pyscf/conversion.py:160,271`
- `cpp/src/qdk/chemistry/algorithms/microsoft/utils.cpp:189`
- `python/src/qdk_chemistry/utils/telemetry.py`
- `SECURITY.md`
- `.github/workflows/build-and-test.yaml`
