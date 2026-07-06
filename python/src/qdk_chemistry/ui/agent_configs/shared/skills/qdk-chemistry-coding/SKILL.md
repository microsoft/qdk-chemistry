---
name: qdk-chemistry-coding
version: '{{QDK_CHEMISTRY_VERSION}}'
description: 'Write Python code using the QDK Chemistry SDK. Use when: writing scripts, notebooks, parameter sweeps, custom workflows, or any task that needs programmatic control over quantum chemistry calculations. Covers the algorithm factory API, data types, settings, multi-step pipelines, content hashing, and best practices.'
---

# QDK Chemistry Python SDK Coding

## When to Use

- Writing Python scripts or Jupyter notebooks for quantum chemistry
- Building parameter sweeps, bond dissociation curves, or automated workflows
- Integrating QDK Chemistry with other Python libraries (NumPy, SciPy)
- Implementing custom algorithms or extending the framework
- Content-addressed caching of computation results

## Core API: Algorithm Factory

All algorithms are created through the factory pattern:

```python
from qdk_chemistry.algorithms import create, available

# List all implementations of a type
available("scf_solver")  # e.g., ["qdk", "pyscf"]

# Create with defaults
scf = create("scf_solver")

# Create with specific implementation
scf = create("scf_solver", "pyscf")

# Create with settings overrides
scf = create("scf_solver", "pyscf", convergence_threshold=1e-8)
```

### Algorithm Types

| Type | Purpose | Common Implementations |
|------|---------|----------------------|
| `scf_solver` | Hartree-Fock / DFT | `qdk`, `pyscf` |
| `active_space_selector` | Active orbital selection | `qdk_valence`, `qdk_autocas_eos`, `qdk_occupation` |
| `orbital_localizer` | Orbital localization | `qdk_pipek_mezey`, `qdk_mp2_natural_orbitals`, `qdk_vvhv` |
| `multi_configuration_calculator` | CASCI / Selected CI | `macis_cas`, `macis_asci` |
| `multi_configuration_scf` | CASSCF | (default) |
| `hamiltonian_constructor` | Fermionic Hamiltonian | `qdk` |
| `qubit_mapper` | Fermion-to-qubit encoding | `qdk`, `qiskit` |
| `state_prep` | State preparation circuit | `sparse_isometry_gf2x` |
| `dynamical_correlation_calculator` | MP2 / CCSD / CCSD(T) | `pyscf_mp2`, `pyscf_ccsd` |
| `stability_checker` | SCF solution stability | `qdk` |
| `phase_estimation` | QPE | `iterative` |
| `time_evolution_builder` | U = exp(-iHt) | (default) |
| `controlled_evolution_circuit_mapper` | Controlled-U circuit | (default) |
| `circuit_executor` | Circuit simulation | (default) |
| `energy_estimator` | Shot-based energy | (default) |
| `qubit_hamiltonian_solver` | Exact diagonalization | (default) |

### Settings

```python
scf = create("scf_solver")

# Read settings
scf.settings()                    # Returns Settings object
scf.settings().get("max_iterations")

# Modify before running
scf.settings().set("convergence_threshold", 1e-10)
scf.settings().set("max_iterations", 200)

# Or pass at creation
scf = create("scf_solver", "pyscf", convergence_threshold=1e-10)
```

## Data Types

```python
from qdk_chemistry.data import (
    Structure,          # Molecular geometry
    Wavefunction,       # Electronic wavefunction
    Orbitals,           # Orbital information
    Hamiltonian,        # Fermionic Hamiltonian
    QubitHamiltonian,   # Qubit-mapped Hamiltonian
    Circuit,            # Quantum circuit
    BasisSet,           # Basis set definition
    QpeResult,          # Phase estimation result
    StabilityResult,    # SCF stability result
)
```

### Creating Structures

```python
import numpy as np
from qdk_chemistry.data import Structure

# From arrays (coordinates in Bohr)
structure = Structure(
    symbols=["H", "H"],
    coordinates=np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]])
)

# From XYZ file
from pathlib import Path
structure = Structure.from_xyz_file(Path("molecules/h2o.xyz"))
```

### Coordinate Units

**Coordinates MUST be in Bohr.** Convert from Ångströms:

```python
from qdk_chemistry.constants import ANGSTROM_TO_BOHR

coords_angstrom = np.array([[0, 0, 0], [0, 0, 0.74]])
coords_bohr = coords_angstrom * ANGSTROM_TO_BOHR
```

## Standard Workflow — Complete Example

```python
from qdk_chemistry.algorithms import create
from qdk_chemistry.data import Structure
from qdk_chemistry.utils import compute_valence_space_parameters
import numpy as np

# 1. Define structure (Bohr)
structure = Structure(
    symbols=["N", "N"],
    coordinates=np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 2.1]])
)

# 2. SCF
scf = create("scf_solver")
e_hf, wfn_hf = scf.run(structure, charge=0, spin_multiplicity=1, basis_or_guess="cc-pvdz")
print(f"HF energy: {e_hf:.6f} Ha")

# 3. Stability check
stability = create("stability_checker")
stability_result = stability.run(wfn_hf)

# 4. Active space selection (valence)
num_e, num_o = compute_valence_space_parameters(wfn_hf, charge=0)
selector = create("active_space_selector", "qdk_valence",
                  num_active_electrons=num_e, num_active_orbitals=num_o)
wfn_as = selector.run(wfn_hf)

# 5. Orbital localization (optional, improves convergence)
localizer = create("orbital_localizer", "qdk_mp2_natural_orbitals")
indices = wfn_as.get_orbitals().get_active_space_indices()
wfn_loc = localizer.run(wfn_as, *indices)

# 6. Hamiltonian construction
ham_constructor = create("hamiltonian_constructor")
orbitals = wfn_loc.get_orbitals()
hamiltonian = ham_constructor.run(orbitals)

# 7. Multi-configuration calculation (CASCI)
alpha_e, beta_e = wfn_loc.get_active_num_electrons()
casci = create("multi_configuration_calculator", "macis_cas")
# Enable RDMs and mutual information for downstream AutoCAS
casci.settings().set("calculate_one_rdm", True)
casci.settings().set("calculate_two_rdm", True)
casci.settings().set("calculate_mutual_information", True)
e_cas, wfn_cas = casci.run(hamiltonian, alpha_e, beta_e)
print(f"CASCI energy: {e_cas:.6f} Ha")

# 8. AutoCAS refinement (optional)
autocas = create("active_space_selector", "qdk_autocas_eos")
wfn_refined = autocas.run(wfn_cas)

# 9. Qubit mapping
qubit_mapper = create("qubit_mapper")
qubit_ham = qubit_mapper.run(hamiltonian)

# 10. State preparation
state_prep = create("state_prep")
circuit = state_prep.run(wfn_cas)

# 11. Phase estimation (QPE)
qpe = create("phase_estimation", "iterative",
             num_bits=10, evolution_time=0.5, shots_per_bit=3)
# ... configure time evolution and run
```

## Content Hashing for Caching

Every algorithm and data object supports deterministic content hashing:

```python
from qdk_chemistry.algorithms import create

scf = create("scf_solver", "pyscf")

# Hash the run without executing
run_hash = scf.hash(structure, charge=0, spin_multiplicity=1, basis_or_guess="sto-3g")

# Use for caching
cache = {}
if run_hash in cache:
    energy, wfn = cache[run_hash]
else:
    energy, wfn = scf.run(structure, 0, 1, "sto-3g")
    cache[run_hash] = (energy, wfn)

# Data objects also have content hashes
structure.content_hash()  # 16-char hex string
```

## Remote Execution

Any algorithm can be transparently offloaded to a remote backend:

```python
scf = create("scf_solver")

# Run on SSH server
scf_ssh = scf.on_remote("ssh", host="compute-server.example.com")
energy, wfn = scf_ssh.run(structure, charge=0, spin_multiplicity=1, basis_or_guess="cc-pvdz")
```

See the `remote-execution` skill for full details.

## Parameter Sweep Example

```python
import numpy as np
from qdk_chemistry.algorithms import create
from qdk_chemistry.data import Structure

scf = create("scf_solver", "pyscf")
distances = np.linspace(0.5, 5.0, 20)
energies = []

for d in distances:
    structure = Structure(
        symbols=["H", "H"],
        coordinates=np.array([[0, 0, 0], [0, 0, d]])  # Bohr
    )
    energy, wfn = scf.run(structure, charge=0, spin_multiplicity=1, basis_or_guess="cc-pvdz")
    energies.append(energy)

print("Bond distances (Bohr) and energies (Ha):")
for d, e in zip(distances, energies):
    print(f"  {d:.2f}  {e:.8f}")
```

## Reference Documents

These files contain detailed API reference and worked examples. Load them when writing code:

- [Python SDK Reference](./references/python-sdk-reference.md) — all `create()` calls, utility functions, file I/O patterns, Hamiltonian filtering
- [Example: Benzene State Prep](./references/example-benzene-state-prep.md) — worked example of state preparation with sparse isometry and energy measurement
- [Example: Stretched N₂ QPE](./references/example-n2-stretched.md) — worked example of iterative QPE with multi-trial voting

## Critical Rules

1. **Coordinates in Bohr** — always check units before creating Structure objects
2. **Always run stability checker** after SCF — unstable solutions break downstream
3. **RDMs for AutoCAS** — set `calculate_one_rdm=True`, `calculate_two_rdm=True`, and `calculate_mutual_information=True` on SCI/CASCI before calling AutoCAS
4. **Don't hardcode algorithm names** — use `available()` to check what's installed
5. **Settings are locked during execution** — configure before calling `.run()`
6. **Use the factory** — always use `create()`, never instantiate algorithm classes directly
7. **Tuple returns** — most algorithms return `(energy, wavefunction)` tuples
