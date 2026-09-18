# Ground-state molecular energies with QPE: lab notebook

## Setup and provenance

- QDK/Chemistry version:
- Built-in implementation verification result:

## Calculation goal and reference plan

- Molecular system:
- Target quantity:
- Teaching target:
- Basis-set comparison (which related calculations will be compared, and what does their difference measure?):
- Active-space comparison (which molecular models will be compared, and what changes between them?):
- Quantum-algorithm reference (which classical energy for the same selected-space Hamiltonian will validate the final result?):
- Error sources outside the final algorithmic comparison:

## Molecular input and mean-field calculation

- Geometry source:
- Nitrogen-nitrogen distance and units:
- Molecular charge:
- Spin multiplicity:
- First basis set:
- First Hartree-Fock energy and units:
- Second basis set:
- Second Hartree-Fock energy and units:
- Basis-set sensitivity and units:
- Interpretation of the basis-set sensitivity:

## Active-space model

- Orbital representation used for selection:
- Initial active electrons and spatial orbitals:
- Initial correlated method:
- Orbital entropies and entropy-gap evidence:
- Selected active electrons and spatial orbitals:
- Selected active-space energy and units:
- Comparison with the initial active space:
- Interpretation of the energy-change and size tradeoff:
- Algorithmic reference energy and units:

## Qubit representation

- Fermion-to-qubit encoding:
- Number of active spatial orbitals:
- Number of active spin orbitals:
- Number of compute-register qubits:
- Number of Pauli terms:
- Fixed-electron-number subspace:
- Core energy and units:
- Mapped active-space ground-state energy and units:
- Mapped selected-space total energy and units:
- Difference from the algorithmic reference and units:
- Interpretation of the energy comparison:
- Quantities excluded from the compute-register qubit count:

## Trial state

- Source wavefunction:
- Leading reference determinants, amplitudes, weights, and cumulative weights:
- Determinant-selection rule before PMC re-optimization:
- Determinant counts compared:
- Fidelity for each trial state:
- State-preparation method:
- Compute qubits for each trial state:
- Preparation logical gate count and logical gate-family counts:
- Interpretation of fidelity and circuit cost:
- Trial state selected for IQPE and rationale:

## Phase-estimation calculation

- Evolution time and units:
- Hamiltonian-simulation method and settings:
- Number of phase bits:
- Shots per bit:
- Number of complete IQPE runs:
- Readout ancillas:
- Simulator seed range:
- Complete-run bitstring counts:
- Most frequent bitstring:
- Measured active-space energy and units:
- Core energy added after phase estimation:
- Total molecular energy estimate and units:
- Difference from the algorithmic reference and units:
- Result relative to the $1\,\mathrm{m}E_{\mathrm{h}}$ teaching target:
- Observed runtime:

## Conclusion

- Main result:
- Evidence supporting the result:
- Basis-set limitations:
- Active-space limitations:
- Quantum-algorithm limitations:
- One change that would strengthen the calculation:
