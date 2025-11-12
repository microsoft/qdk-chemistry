# PyMACIS - Python Bindings for MACIS

PyMACIS provides Python bindings to the MACIS (Many-Body Adaptive Configuration Interaction Suite) C++ library, allowing
Python users to leverage the performance of C++ for quantum chemistry calculations.

## Features

- Read and write FCIDUMP files
- Perform active space transformations
- Run CASCI calculations
- Find minimal SCI (Selected CI) spaces
- Read and write wavefunctions

## Installation

### From Source

To build PyMACIS from source:

```bash
# Clone the MACIS repository
git clone <macis-repo-url>
cd macis

# Create a build directory
mkdir build
cd build

# Configure with CMake (enable Python bindings)
cmake .. -DMACIS_ENABLE_PYTHON=ON

# Build the library and Python module
make -j

# The Python module will be available in build/python/
# Add to Python path or install
export PYTHONPATH=$PYTHONPATH:/path/to/macis/build/python
```

Note: PyMACIS requires:

- C++17 compatible compiler
- Python 3.8 or later
- NumPy
- pybind11 (automatically downloaded during build)

## Examples

### Reading a FCIDUMP file

```python
import pymacis
import numpy as np

# Read complete Hamiltonian from FCIDUMP file (recommended)
H = pymacis.read_fcidump("h2o_sto3g.fcidump")
print(f"Number of orbitals: {H.norb}")
print(f"Core energy: {H.core_energy:.12f}")
print(f"One-electron integrals shape: {H.T.shape}")
print(f"Two-electron integrals shape: {H.V.shape}")

# Alternatively, read components separately
norb = pymacis.read_fcidump_norb("h2o_sto3g.fcidump")
ecore = pymacis.read_fcidump_core("h2o_sto3g.fcidump")
t_mat = pymacis.read_fcidump_1body("h2o_sto3g.fcidump")
v_tensor = pymacis.read_fcidump_2body("h2o_sto3g.fcidump")
```

### Running CASCI calculations

```python
import pymacis

# Read molecular integrals
H = pymacis.read_fcidump("h2o_sto3g.fcidump")

# Compute active space Hamiltonian (optional - if using subset of orbitals)
H_active = pymacis.compute_active_hamiltonian(nactive=6, ninactive=2, H=H)

# Run CASCI calculation
result = pymacis.casci(nalpha=5, nbeta=5, H=H_active,
                       settings={"ci_res_tol": 1e-8, "return_determinants": True})

print(f"CASCI energy: {result['energy']:.8f} Hartree")
print(f"Number of determinants: {len(result['determinants'])}")
```

### Running ASCI calculations

```python
import pymacis

# Read molecular integrals
H = pymacis.read_fcidump("system.fcidump")

# Generate canonical HF determinant as initial guess
initial_guess = [pymacis.canonical_hf_determinant(nalpha=5, nbeta=5, norb=10)]
C0 = [1.0]

# Calculate initial energy estimate using HF determinant
E0 = pymacis.compute_wfn_energy(initial_guess, C0, H)

# ASCI settings
settings = {
    "ci_max_subspace": 200,
    "asci_grow_factor": 2,
    "asci_ntdets_max": 1000,
}

# Run ASCI calculation
result = pymacis.asci(initial_guess, C0, E0, H, settings)
print(f"ASCI energy: {result['energy']:.8f} Hartree")
print(f"Energy improvement: {(result['energy'] - E0):.8f} Hartree")
```

### Wavefunction analysis and I/O

```python
import pymacis

# Calculate energy of a wavefunction
H = pymacis.read_fcidump("system.fcidump")
wavefunction = ["22u000", "2u2000"]  # Example determinants
coefficients = [0.8, 0.6]           # Corresponding coefficients
energy = pymacis.compute_wfn_energy(wavefunction, coefficients, H)

# Save wavefunction to file
pymacis.write_wavefunction("result.wfn", norb=6,
                          determinants=wavefunction,
                          coefficients=coefficients)
```

### Determinant format and canonical HF determinant

PyMACIS uses strings to represent Slater determinants, where each character represents the occupation of a molecular
orbital:

- '0' = unoccupied orbital
- 'u' = singly occupied (alpha electron only)
- 'd' = singly occupied (beta electron only)
- '2' = doubly occupied (both alpha and beta electrons)

```python
import pymacis

# Generate canonical HF determinant
# For a system with 4 electrons (2 alpha, 2 beta) in 6 orbitals
hf_det = pymacis.canonical_hf_determinant(nalpha=2, nbeta=2, norb=6)
print(f"HF determinant: {hf_det}")  # Output: "220000"

# This represents:
# Orbital 0: doubly occupied (alpha + beta)
# Orbital 1: doubly occupied (alpha + beta)
# Orbitals 2-5: unoccupied

# For open-shell system: 3 alpha, 2 beta electrons in 6 orbitals
uhf_det = pymacis.canonical_hf_determinant(nalpha=3, nbeta=2, norb=6)
print(f"UHF determinant: {uhf_det}")  # Output: "22u000"

# For a radical system: 5 alpha, 4 beta electrons in 6 orbitals
radical_det = pymacis.canonical_hf_determinant(nalpha=5, nbeta=4, norb=6)
print(f"Radical determinant: {radical_det}")  # Output: "222u00"
```

## API Reference

### Hamiltonian Class

**`Hamiltonian()`** Container for molecular integrals and system information. This class contains all the molecular
orbital integrals needed for CI calculations, including one-electron (T) and two-electron (V) integrals, along with
system parameters like number of orbitals and core energy.

**Properties:**

- `norb` (int): Number of active molecular orbitals
- `nbasis` (int): Total number of basis functions
- `core_energy` (float): Nuclear repulsion energy plus inactive orbital contributions
- `T` (numpy.ndarray): One-electron integrals matrix (kinetic + nuclear attraction), shape (norb, norb)
- `V` (numpy.ndarray): Two-electron integrals tensor in physicist's notation, shape (norb, norb, norb, norb)
- `F_inactive` (numpy.ndarray): Inactive Fock matrix for active space calculations

### FCIDUMP Operations

**`read_fcidump(filename)`** *(recommended)* Read complete molecular integral data from a FCIDUMP file. This is the
primary function for loading molecular integrals. It reads all necessary data and returns a complete Hamiltonian object
ready for CI calculations.

- **Args:** `filename` (str) - Path to the FCIDUMP file
- **Returns:** `Hamiltonian` object containing norb, core_energy, T, and V
- **Raises:** `FileNotFoundError`, `RuntimeError`

**`read_fcidump_norb(filename)`** Read the number of orbitals from a FCIDUMP file header.

**`read_fcidump_core(filename)`** Read the core energy (nuclear repulsion + inactive contributions) from a FCIDUMP file.

**`read_fcidump_1body(filename)`** Read one-electron integrals (kinetic energy + nuclear attraction) as a 2D NumPy
array.

**`read_fcidump_2body(filename)`** Read two-electron repulsion integrals as a 4D NumPy array in physicist's notation
V[i,j,k,l] = ⟨ij|kl⟩.

### Active Space Operations

**`compute_active_hamiltonian(nactive, ninactive, H)`** Transform a full molecular Hamiltonian to an active space
representation by incorporating the mean-field effects of inactive (doubly occupied) orbitals into the one-electron part
of the active space Hamiltonian.

- **Args:**
  - `nactive` (int): Number of active orbitals to include in CI calculation
  - `ninactive` (int): Number of inactive (doubly occupied) orbitals
  - `H` (Hamiltonian): Full system Hamiltonian containing all molecular orbitals
- **Returns:** `Hamiltonian` with norb=nactive and modified integrals
- **Note:** Active orbitals are assumed to be the first nactive orbitals in the orbital ordering

### Determinant Operations

**`canonical_hf_determinant(nalpha, nbeta, norb)`** Generate the canonical Hartree-Fock determinant for a given number
of alpha and beta electrons. Creates the unique reference determinant for the Hartree-Fock wavefunction.

- **Args:**
  - `nalpha` (int): Number of alpha (spin-up) electrons
  - `nbeta` (int): Number of beta (spin-down) electrons
  - `norb` (int): Total number of orbitals in the system
- **Returns:** String representing the canonical HF determinant
- **Raises:** `ValueError` if number of electrons or orbitals is invalid

### CI Calculations

**`casci(nalpha, nbeta, H, settings={})`** Run a Complete Active Space Configuration Interaction (CASCI) calculation.
Performs full CI within the active space by generating all possible determinants and diagonalizing the CI Hamiltonian
matrix.

- **Args:**
  - `nalpha` (int): Number of alpha electrons in active space
  - `nbeta` (int): Number of beta electrons in active space
  - `H` (Hamiltonian): Active space Hamiltonian containing molecular integrals
  - `settings` (dict, optional): Calculation settings
- **Settings:**
  - `'ci_max_subspace'`: Maximum Davidson iterations (default: 200)
  - `'ci_res_tol'`: Convergence tolerance for energy (default: 1e-8)
  - `'ci_matel_tol'`: Matrix element threshold (default: machine epsilon)
  - `'return_determinants'`: Include determinants in output (default: False)
- **Returns:** Dict with `'energy'`, `'coefficients'`, and optionally `'determinants'`

**`asci(initial_guess, C0, E0, H, settings={})`** Run an Adaptive Sampling Configuration Interaction (ASCI) calculation.
Performs selected CI by iteratively growing the determinant space using perturbative selection criteria.

- **Args:**
  - `initial_guess` (list): List of determinant strings for initial CI space
  - `C0` (list): Initial CI coefficients corresponding to initial_guess
  - `E0` (float): Initial energy estimate
  - `H` (Hamiltonian): Active space Hamiltonian
  - `settings` (dict, optional): Calculation settings
- **Key ASCI Settings:**
  - `'asci_ntdets_max'`: Maximum determinants in final CI space (default: 100000)
  - `'asci_grow_factor'`: Growth factor for determinant selection (default: 8)
  - `'asci_max_refine_iter'`: Maximum refinement iterations (default: 6)
  - `'asci_ham_el_tol'`: Hamiltonian matrix element threshold (default: 1e-8)
- **Returns:** Dict with `'energy'`, `'coefficients'`, and `'determinants'`

### Wavefunction Operations

**`compute_wfn_energy(wfn, C, H)`** Calculate the energy of a CI wavefunction given its determinants and coefficients
using the provided Hamiltonian.

- **Args:**
  - `wfn` (list): List of determinant strings
  - `C` (numpy.ndarray): Array of CI coefficients
  - `H` (Hamiltonian): Hamiltonian object containing molecular integrals
- **Returns:** Total energy of the wavefunction in Hartree units
- **Raises:** `ValueError` if wfn and C have different lengths

**`write_wavefunction(filename, norb, determinants, coefficients)`** Save CI wavefunction (determinants and
coefficients) to a binary file for later analysis or as input to other calculations.

- **Args:**
  - `filename` (str): Output filename for wavefunction data
  - `norb` (int): Number of molecular orbitals
  - `determinants` (list): List of determinant strings
  - `coefficients` (numpy.ndarray): CI coefficients
- **Raises:** `ValueError`, `IOError`

**`read_wavefunction(filename)`** Load a previously saved CI wavefunction from file.

- **Args:** `filename` (str) - Input filename containing wavefunction data
- **Returns:** Dict with `'norbitals'`, `'determinants'`, and `'coefficients'`
- **Raises:** `FileNotFoundError`, `IOError`
