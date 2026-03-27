=========
Changelog
=========

Version 1.1.0
=============

- **Streamlined Q# circuit integration**: Moved to Q# as the internal circuit
  representation, enabling a drastically optimized end-to-end QPE code path.
  Qiskit is now an optional dependency.
- **Model Hamiltonians**: Added native support for constructing fermionic and
  spin model Hamiltonians: Hückel, Pariser-Parr-Pople (PPP), Fermi-Hubbard,
  Ising, and Heisenberg.
- **More robust quantum primitives and improved Trotter support**: Introduced
  arbitrary-order Trotter-Suzuki product formulas, accuracy-aware
  parameterization, and error-bound analysis utilities.
- **Native ROHF in QDK SCF**: Added DIIS-accelerated restricted open-shelsl
  Hartree-Fock (ROHF) to improve open-shell system support.
- **Cholesky-based AO-to-MO transformation**: Added Cholesky decomposition
  for two-electron integral transformation, enabling treatment of larger
  molecular systems.
- Improved MACIS active-orbital limits, orbital localization, and ASCI
  refinement robustness
- Added OpenFermion plugin, Pauli math operators, and QDK-native qubit mapper
- Extensive documentation improvements and restructuring
- Added macOS wheels and broadened CI/CD coverage

Version 1.0.2
=============

- Make qiskit-aer and qiskit-nature optional dependencies
- Loosen matplotlib version requirement to >=3.10.0
- Fixed installation instructions for Ubuntu compatibility
- Improved iQPE demo notebook

Version 1.0.1
=============

- Added support for Python 3.10
- Enhanced INSTALL.md with clearer installation steps

Version 1.0.0
=============

- Initial release of QDK/Chemistry
