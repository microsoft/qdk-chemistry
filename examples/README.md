# QDK/Chemistry examples

This directory contains example scripts demonstrating how to use QDK/Chemistry for various quantum computing chemistry tasks.

## Standalone examples and data

- `data`: Data directory for examples
- `factory_list.ipynb`: Jupyter notebook that lists available factory methods in QDK/Chemistry along with their descriptions and settings
- `language/cpp`: C++ example programs using the QDK/Chemistry C++ API
- `language/sample_sci_workflow.py`: Python script demonstrating a sample classical workflow for selected CI quantum chemistry calculations.
- `qpe_stretched_n2.ipynb`: Jupyter notebook demonstrating multi-reference quantum chemistry state preparation and iterative quantum phase estimation
- `state_prep_energy.ipynb`: Jupyter notebook demonstrating quantum state preparation and energy calculation using quantum simulators.

## Examples of interoperability with other quantum computing frameworks

### PennyLane

The [`interoperability/pennylane`](interoperability/pennylane/) directory contains example programs demonstrating interoperability between QDK/Chemistry and [PennyLane](https://pennylane.ai/), including:

- [`qpe_no_trotter.py`](interoperability/pennylane/qpe_no_trotter.py): Example of Quantum Phase Estimation (QPE) without Trotterization using PennyLane and QDK/Chemistry.

### Qiskit

The [`interoperability/qiskit`](interoperability/qiskit) directory contains example programs demonstrating interoperability between QDK/Chemistry and [Qiskit](https://qiskit.org/), including:

- [`iqpe_model_hamiltonian.py`](interoperability/qiskit/iqpe_model_hamiltonian.py): Example of Iterative Quantum Phase Estimation (IQPE) using a model Hamiltonian with Qiskit and QDK/Chemistry.
- [`iqpe_no_trotter.py`](interoperability/qiskit/iqpe_no_trotter.py): Example of Iterative Quantum Phase Estimation (IQPE) without Trotterization using Qiskit and QDK/Chemistry.
- [`iqpe_trotter.py`](interoperability/qiskit/iqpe_trotter.py): Example of Iterative Quantum Phase Estimation (IQPE) with Trotterization using Qiskit and QDK/Chemistry.

### OpenFermion

The [`interoperability/openFermion`](interoperability/openFermion/) directory contains example programs demonstrating interoperability between QDK/Chemistry and [OpenFermion](https://quantumai.google/openfermion), including:

- [`molecular_hamiltonian_jordan_wigner.py`](interoperability/openFermion/molecular_hamiltonian_jordan_wigner.py): Example of Jordan-Wigner transformation using OpenFermion and QDK/Chemistry.

### RDKit

The [`interoperability/rdkit`](interoperability/rdkit/) directory contains example programs demonstrating interoperability between QDK/Chemistry and [RDKit](https://www.rdkit.org/), including:

- [`sample_rdkit_geometry.py`](interoperability/rdkit/sample_rdkit_geometry.py): Example of obtaining geometry from RDKit and calculate a simple energy with QDK/Chemistry.
