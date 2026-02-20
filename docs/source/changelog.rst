=========
Changelog
=========

Unreleased
==========

- Added accuracy-aware Trotter parameterization with automatic step-count
  determination via ``target_accuracy`` parameter
- Added unified Pauli commutation utility module
  (``qdk_chemistry.utils.pauli_commutation``)
- Added Trotter error estimation functions (``trotter_steps_naive``,
  ``trotter_steps_commutator``) with commutator-based bounds from
  Childs *et al.* (2021)
- **Breaking:** Renamed Trotter parameter ``num_trotter_steps`` to
  ``num_divisions``
- **Breaking:** Renamed Trotter parameter ``tolerance`` to
  ``weight_threshold``
- New Trotter keyword-only parameters: ``target_accuracy``,
  ``error_bound``

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
