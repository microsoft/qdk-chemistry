# Ground-state molecular energies with QPE: completed lab notebook

This completed example records the reference results for the documented QDK/Chemistry 2.2.0 workflow.

## Setup and provenance

- QDK/Chemistry version: 2.2.0
- Built-in implementation verification result: Passed for all required implementations.

## Calculation goal and reference plan

- Molecular system: Neutral singlet N2 at a fixed 1.85 angstrom bond length.
- Target quantity: Fixed-geometry selected-space total ground-state energy.
- Teaching target: Absolute IQPE error no greater than 1 milliHartree.
- Basis-set comparison: Hartree-Fock cc-pVDZ versus cc-pVTZ energy sensitivity.
- Active-space comparison: Initial valence CASCI versus entropy-selected CASCI.
- Quantum-algorithm reference: Exact CASCI energy of the selected active-space Hamiltonian.
- Error sources outside the final algorithmic comparison: Geometry, Born-Oppenheimer approximation, finite basis, and active-space truncation.

## Molecular input and mean-field calculation

- Geometry source: Two nitrogen atoms on the z axis.
- Nitrogen-nitrogen distance: 1.85 angstrom.
- Molecular charge: 0.
- Spin multiplicity: 1 (singlet).
- First basis set: cc-pVDZ.
- First Hartree-Fock energy: -108.418633697214 Hartree.
- Second basis set: cc-pVTZ.
- Second Hartree-Fock energy: -108.445215657498 Hartree.
- Basis-set sensitivity: 0.026581960284 Hartree = 26.581960 milliHartree = 69.790927 kJ/mol.
- Interpretation: This is the observed change between two finite-basis Hartree-Fock calculations, not the exact basis-set error.

## Active-space model

- Orbital representation used for selection: Natural orbitals from the initial valence-space CASCI one-particle RDM.
- Initial active space: CAS(10e,8o), alpha/beta counts (5,5), orbitals 2-9.
- Initial correlated method: Exact CASCI within the initial valence space.
- Initial determinant count: 3,136.
- Initial CASCI energy: -108.778369520882 Hartree.
- Natural-orbital CASCI energy: -108.778369520882 Hartree; change indistinguishable from zero at the displayed precision.
- Single-orbital entropies for orbitals 2-9: 0.021695559, 0.029962840, 0.547855655, 0.963884534, 0.963884534, 0.966011520, 0.966011520, 0.554009346.
- Entropy-gap evidence: Orbitals 4-9 form the high-entropy group; orbitals 2-3 are near 0.03 or below.
- Selected active space: CAS(6e,6o), alpha/beta counts (3,3), orbitals 4-9.
- Selected partition: Four inactive, six active, and 18 virtual spatial orbitals.
- Selected determinant count: 400.
- Selected active-space energy: -108.771051792909 Hartree.
- Energy increase from refinement: 0.007317727973 Hartree.
- Interpretation: The compact space retains the strongest entropy-based correlation evidence while documenting excluded correlation as model error.
- Algorithmic reference energy: -108.771051792909 Hartree.

## Qubit representation

- Fermion-to-qubit encoding: Jordan-Wigner with blocked alpha-then-beta mode ordering.
- Active spatial orbitals: 6.
- Active spin orbitals: 12.
- Compute-register qubits: 12.
- Pauli terms: 247 using a 1e-10 Hartree Pauli-coefficient threshold.
- Fixed-electron-number subspace: 3 alpha and 3 beta electrons; 400 basis states.
- Core energy: -99.117775726922 Hartree.
- Mapped active-space ground-state energy: -9.653276065987 Hartree.
- Mapped selected-space total energy: -108.771051792909 Hartree.
- Difference from algorithmic reference: indistinguishable from zero at the displayed precision.
- Interpretation: The mapping and fixed-sector construction reproduce the selected-space Hamiltonian within numerical precision.
- Excluded from compute-qubit count: Readout/workspace ancillas, error-correction overhead, and physical qubits.

## Trial state

- Source wavefunction: Selected-space CASCI ground state.
- Leading determinants (occupation, amplitude, weight, cumulative weight):
  - 222000, +0.694657453275, 0.482548977390, 0.482548977390
  - 202200, -0.333212130081, 0.111030323633, 0.593579301023
  - 220020, -0.333212130081, 0.111030323633, 0.704609624656
  - 200220, +0.200148175782, 0.040059292269, 0.744668916925
  - 2dudu0, +0.186829733892, 0.034905349466, 0.779574266391
  - 2udud0, +0.186829733892, 0.034905349466, 0.814479615857
  - 022002, -0.146520835129, 0.021468355127, 0.835947970984
  - 2duud0, +0.127624440015, 0.016287997689, 0.852235968673
- Selection rule: Retain the largest one, two, or four CASCI coefficients, then re-optimize amplitudes with PMC.
- One-determinant trial: Fidelity 0.482548977390; 12 compute qubits; 6 preparation logical gates; {'X': 6}.
- Two-determinant trial: Fidelity 0.586414650360; 12 compute qubits; 14 preparation logical gates; {'CNOT': 6, 'H': 2, 'Rz': 2, 'S': 2, 'X': 2}.
- Four-determinant trial: Fidelity 0.732385025483; 12 compute qubits; 30 preparation logical gates; {'CNOT': 16, 'H': 4, 'Rz': 4, 'S': 4, 'X': 2}.
- State-preparation method: QDK/Chemistry sparse-isometry implementation.
- Interpretation: More determinants improve fidelity but change preparation cost; register size is fixed by the selected spin-orbital space.

## Phase-estimation calculation

- Evolution time: 0.162738437655 inverse Hartree.
- Hamiltonian simulation: First-order Trotter product formula, one division, repeated approximate base unitary for controlled powers.
- Phase bits: 6.
- Shots per bit: 3.
- Complete IQPE runs: 20.
- Readout ancillas: 1 per iteration circuit.
- Simulator seeds: 42-61.
- Complete-run bitstring counts: {'010000': 19, '001111': 1}.
- Most frequent bitstring: 010000.
- Measured active-space energy: -9.652276065987 Hartree.
- Core energy added after phase estimation: -99.117775726922 Hartree.
- Total molecular energy estimate: -108.770051792909 Hartree.
- Algorithmic reference: -108.771051792909 Hartree.
- Signed difference: +0.001000000000 Hartree.
- Teaching-target result: Meets the 1 milliHartree target at its boundary.
- Observed repeated-run phase: approximately six minutes in the measured development run; runtime varies by computer.

## Conclusion

- Main result: The configured IQPE workflow reproduces the selected-space CASCI reference within 1 milliHartree.
- Supporting evidence: Exact mapped/CASCI agreement, 19 of 20 complete runs at bitstring 010000, and the reconstructed total-energy comparison.
- Basis-set limitation: cc-pVDZ is finite; the cc-pVDZ/cc-pVTZ difference is sensitivity evidence, not exact error.
- Active-space limitation: Refinement excludes correlation and raises the CASCI energy by about 7.32 milliHartree relative to the initial valence space.
- Quantum-algorithm limitations: Reference-guided evolution-time tuning is circular; first-order Trotter evolution, six phase bits, and finite sampling remain approximations.
- One strengthening change: Select evolution time without the known target energy, then increase phase precision and Trotter accuracy while measuring the resulting cost.
