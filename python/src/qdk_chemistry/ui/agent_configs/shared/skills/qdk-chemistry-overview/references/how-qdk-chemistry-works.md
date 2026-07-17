# How QDK Chemistry Works

## The `create()` Factory

Everything in QDK Chemistry flows through one function:

```python
from qdk_chemistry.algorithms import create

scf = create("scf_solver")                                    # default algorithm
scf = create("scf_solver", "pyscf", max_iterations=100)       # specific algorithm + kwargs
mapper = create("qubit_mapper", algorithm_name="qiskit")       # alternative syntax
```

There are 15 algorithm types. Each has a default algorithm and optional settings. Use `get_algorithm_default_type(type_name)` to discover the default, and `get_algorithm_default_settings(type_name)` or `get_algorithm_default_settings(type_name, algorithm_name)` to see what settings are available.

## MCP Tool Pattern

Every MCP tool follows the same structure:

**Input:** `project_name` + input filenames + output filename + optional `algorithm_name` + optional `settings` dict

**Output:** Always JSON with one of three statuses:

```json
{"status": "ok",     "result": "<value or [value1, value2]>"}
{"status": "error",  "message": "ERROR: ...", "error_type": "ValueError"}
{"status": "exists", "message": "EXISTS: Output file already exists..."}
```

**Auto-behaviors built into the tools:**

- Projects are auto-created if they don't exist
- Filenames are auto-corrected (e.g., `water.json` → `water.structure.json`)
- Full paths are auto-stripped to basenames
- Output files are never silently overwritten — you get `"status": "exists"` instead

## File Naming Convention

Every data file uses the pattern `{name}.{type_marker}.{ext}`:

| Data Type | Marker | Example |
|---|---|---|
| Structure | `.structure.` | `mol.structure.json` |
| Wavefunction | `.wavefunction.` | `mol.wavefunction.json` |
| Hamiltonian | `.hamiltonian.` | `mol.hamiltonian.json` |
| Orbitals | `.orbitals.` | `mol.orbitals.json` |
| QubitHamiltonian | `.qubit_hamiltonian.` | `mol.qubit_hamiltonian.json` |
| Circuit | `.circuit.` | `prep.circuit.json` |
| Ansatz | `.ansatz.` | `mol.ansatz.json` |
| QpeResult | `.qpe_result.` | `result.qpe_result.json` |
| EnergyResult | `.energy_result.` | `energy.energy_result.json` |

If you forget the marker, the tool will insert it. But it's better to get it right.

## What Each Tool Returns

Most tools return a filename. Some return a tuple:

- `run_scf` → `[total_energy, wavefunction_filename]`
- `run_multi_configuration_calculation` → `[energy, wavefunction_filename]`
- `run_energy_estimator` → `[energy_result_file, measurement_data_file]`
- `create_structure` → `structure_filename`
- Everything else → single filename

## Coordinate Units

`create_structure` takes coordinates in **Bohr**. It does NOT auto-detect or auto-convert. If the user gives Angstrom (common in papers, PDB files, and molecular editors), convert first: multiply by 1.8897259886. The `convert_coordinates` utility can do this.

Symptom of wrong units: SCF energy is wildly wrong or convergence fails immediately.

## Auto-Computed Values

Several things you don't need to guess — the package computes them:

- **Active space size:** `compute_valence_space_parameters(wfn, charge)` returns `(num_electrons, num_orbitals)` automatically from the SCF wavefunction
- **Evolution time:** `compute_evolution_time(qubit_hamiltonian, num_bits)` computes the correct time parameter for QPE based on the Hamiltonian's spectral norm
- **Valence active space:** When using `qdk_valence` with `charge`, the MCP tool auto-computes and sets `num_active_electrons` and `num_active_orbitals` — you don't set them manually
