# Python Test Conventions

## Folder Structure

```
tests/
├── data/              # Data structures (Structure, Orbitals, Hamiltonian, etc.)
├── algorithms/        # Algorithm tests (SCF, MC, energy estimator, etc.)
├── qubit_mapping/     # Qubit mapper, Pauli commutation, circuit mapping
├── time_evolution/    # Trotter, QDrift, controlled evolution
├── utils/             # Settings, logger, constants, utility functions
└── integrations/      # PySCF, Qiskit, OpenFermion, workflow tests
    └── workflows/     # End-to-end sample workflows
```

Mirrors `src/qdk_chemistry/`. New tests go in the matching folder.

## Rules

### No tautological tests
Never write `obj.x = 5; assert obj.x == 5`. Every test must validate **computation**, **logic**, or **error handling** — not that Python assignment works.

### Max ~500 lines per file
If a test file exceeds 500 lines, split by concern. See `integrations/test_pyscf_*.py` for an example.

### Use `@pytest.mark.parametrize` instead of copy-paste
When testing the same logic with different inputs, use parametrize:
```python
@pytest.mark.parametrize("n_alpha,n_beta,expected_mult", [(2,2,1), (2,1,2), (3,1,3)])
def test_spin_multiplicity(n_alpha, n_beta, expected_mult):
    assert Symmetries(n_alpha=n_alpha, n_beta=n_beta).spin_multiplicity == expected_mult
```

### One serialization roundtrip per format per class
Test JSON and HDF5 roundtrips with one representative object. Don't write separate tests per field.

### Shared infrastructure
- `conftest.py` (root): shared fixtures available to all subfolders
- `test_helpers.py`: factory functions (`create_test_orbitals`, `create_test_hamiltonian`, etc.)
- `reference_tolerances.py`: all numerical tolerances in one place

### Conditional dependency logic
Put `pytest.mark.skipif` for optional deps (Qiskit, PySCF) in the test module or `integrations/conftest.py`. Don't repeat skip decorators per test method.

### Imports from parent
Files in subfolders use `from ..reference_tolerances import ...` and `from ..test_helpers import ...`.
Files in `integrations/workflows/` use `from ...reference_tolerances import ...` (three dots).
