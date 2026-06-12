# Plugin Architecture — How Third-Party Integrations Work

Plugins extend the algorithm registry with implementations backed by external libraries. Each follows the same pattern.

## Registration Pattern

```python
# In plugins/<name>/__init__.py
_loaded = False
def load():
    global _loaded
    if _loaded: return
    _loaded = True
    if importlib.util.find_spec("external_lib") is None: return  # graceful skip
    from qdk_chemistry.algorithms import register
    register(lambda: MyPluginAlgorithm())  # deferred instantiation
```

Plugins are loaded automatically at package import time. If the dependency isn't installed, the plugin silently skips.

## PySCF Plugin (`plugins/pyscf/`)

**What it provides**: Alternative SCF solver, coupled cluster, localization, AVAS active space, MCSCF, stability analysis

**Key conversions**:

- `Structure` → `pyscf.gto.Mole` (atom labels + Bohr coords)
- PySCF MO coefficients/energies → `Orbitals` + `StateVectorContainer` (single determinant)
- Unrestricted references: separate alpha/beta coefficient matrices and energies

**Algorithms registered**: `PyscfScfSolver`, `PyscfCoupledClusterCalculator`, `PyscfLocalizer`, `PyscfAVAS`, `PyscfMcscfCalculator`, `PyscfStabilityChecker`

## Qiskit Plugin (`plugins/qiskit/`)

**What it provides**: Qubit mapping via qiskit-nature, circuit execution via Aer, energy estimation, QPE, state preparation, transpilation, noise models

**Key conversions**:

- `Hamiltonian` → `qiskit_nature.ElectronicEnergy` (reshape 1e to 2D, 2e to 4D tensor)
- Qiskit `SparsePauliOp` → `QubitHamiltonian` (Pauli strings + coefficients)
- `Circuit` ↔ Qiskit `QuantumCircuit` (via QASM3 or QIR)

**Algorithms registered**: `QiskitQubitMapper`, `QiskitStandardPhaseEstimation`, `RegularIsometryStatePreparation`, `QiskitEnergyEstimator`, `QiskitAerSimulator`

## OpenFermion Plugin (`plugins/openfermion/`)

**What it provides**: Qubit mapping + bidirectional `QubitOperator` conversion

**Key conversions** (critical details):

- **Spin-orbital ordering**: OpenFermion uses *interleaved* `[α₀,β₀,α₁,β₁,…]`, QDK uses *blocked* `[α₀,α₁,…,β₀,β₁,…]` — requires index permutation at conversion boundaries
- **Chemist↔Physicist notation**: 2e integrals transposed via `(0,3,2,1)` pattern
- **Pauli string encoding**: OpenFermion sparse `((0,'X'),(1,'Z'))` → QDK dense `"XZI..."` with little-endian qubit ordering (qubit 0 is rightmost in string representation)
- **Core energy**: OpenFermion folds into identity Pauli term; QDK stores separately — must extract/inject at conversion boundaries

**Algorithms registered**: `OpenFermionQubitMapper`
