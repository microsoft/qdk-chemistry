# QDK/Chemistry examples

This directory contains example scripts demonstrating how to use QDK/Chemistry for various quantum computing chemistry tasks.

## Standalone examples and data

- `cpp`: C++ example programs using the QDK/Chemistry C++ API
- `data`: Data directory for examples
- `factory_list.ipynb`: Jupyter notebook that lists available factory methods in QDK/Chemistry along with their descriptions and settings
- `sample_sci_workflow.py`: Python script demonstrating a sample classical workflow for selected CI quantum chemistry calculations.
- `state_prep_energy.ipynb`: Jupyter notebook demonstrating quantum state preparation and energy calculation using quantum simulators.

## Examples of interoperability with other quantum computing frameworks

### PennyLane

The [`pennylane`](pennylane) directory contains example programs demonstrating interoperability between QDK/Chemistry and [PennyLane](https://pennylane.ai/), including:

- [`qpe_no_trotter.py`](pennylane/qpe_no_trotter.py): Example of Quantum Phase Estimation (QPE) without Trotterization using PennyLane and QDK/Chemistry.

### Qiskit

The [`qiskit`](qiskit) directory contains example programs demonstrating interoperability between QDK/Chemistry and [Qiskit](https://qiskit.org/), including:

- [`iqpe_model_hamiltonian.py`](qiskit/iqpe_model_hamiltonian.py): Example of Iterative Quantum Phase Estimation (IQPE) using a model Hamiltonian with Qiskit and QDK/Chemistry.
- [`iqpe_no_trotter.py`](qiskit/iqpe_no_trotter.py): Example of Iterative Quantum Phase Estimation (IQPE) without Trotterization using Qiskit and QDK/Chemistry.
- [`iqpe_trotter.py`](qiskit/iqpe_trotter.py): Example of Iterative Quantum Phase Estimation (IQPE) with Trotterization using Qiskit and QDK/Chemistry.

### Q\#

The [`qsharp`](qsharp) directory contains example programs demonstrating interoperability between QDK/Chemistry and [Q#](https://github.com/microsoft/qdk), including:

- [`iqpe_no_trotter.qs`](qsharp/iqpe_no_trotter.qs): Example of Iterative Quantum Phase Estimation (IQPE) without Trotterization using Q#.
