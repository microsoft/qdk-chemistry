"""QDK/Chemistry data module for quantum chemistry data structures and settings.

This module provides access to core quantum chemistry data types including molecular
structures, basis sets, wavefunctions, and computational settings. It serves as the
primary interface for managing quantum chemical data within the QDK/Chemistry framework.

Exposed classes are:

- ``Ansatz``: Quantum chemical ansatz combining a Hamiltonian and wavefunction for energy calculations.
- ``BasisSet``: Gaussian basis set definitions for quantum calculations.
- ``AOType``: Enumeration of basis set types (STO-3G, 6-31G, etc.).
- ``CasWavefunctionContainer``: Complete Active Space (CAS) wavefunction with CI coefficients and determinants.
- ``Configuration``: Electronic configuration state information.
- ``CoupledClusterAmplitudes``: Amplitudes for coupled cluster calculations.
- ``DataClass``: Base data class.
- ``ElectronicStructureSettings``: Specialized settings for electronic structure calculations.
- ``Element``: Represents a chemical element with its properties.
- ``EnergyExpectationResult``: Result for Hamiltonian energy expectation value and variance.
- ``Hamiltonian``: Quantum mechanical Hamiltonian operator representation.
- ``MeasurementData``: Measurement bitstring data and metadata for ``QubitHamiltonian`` objects.
- ``ModelOrbitals``: Simple orbital representation for model systems without full basis set information.
- ``Orbitals``: Molecular orbital information and properties.
- ``OrbitalType``: Enumeration of orbital angular momentum types (s, p, d, f, etc.).
- ``QpeResult``: Result of quantum phase estimation workflows, including phase, energy, and metadata.
- ``QubitHamiltonian``: Molecular electronic Hamiltonians mapped to qubits.
- ``SciWavefunctionContainer``: Selected Configuration Interaction (SCI) wavefunction with CI coefficients.
- ``Settings``: Configuration settings for quantum chemistry calculations.
- ``SettingValue``: Type-safe variant for storing different setting value types.
- ``Shell``: Individual shell within a basis set.
- ``SlaterDeterminantContainer``: Single Slater determinant wavefunction representation.
- ``StabilityResult``: Result of stability analysis for electronic structure calculations.
- ``Structure``: Molecular structure and geometry information.
- ``Wavefunction``: Electronic wavefunction data and coefficients.
- ``WavefunctionContainer``: Abstract base class for different wavefunction representations.
- ``WavefunctionType``: Enumeration of wavefunction types (SelfDual, NotSelfDual).

Exposed exceptions are

- ``SettingNotFound`` / ``SettingNotFoundError``: Raised when a requested setting is not found.
- ``SettingTypeMismatch`` / ``SettingTypeMismatchError``: Raised when a setting value has an incorrect type.

"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from contextlib import suppress

from qdk_chemistry._core.data import (
    Ansatz,
    AOType,
    BasisSet,
    CasWavefunctionContainer,
    Configuration,
    CoupledClusterAmplitudes,
    ElectronicStructureSettings,
    Element,
    Hamiltonian,
    HamiltonianType,
    ModelOrbitals,
    Orbitals,
    OrbitalType,
    SciWavefunctionContainer,
    SettingNotFound,
    Settings,
    SettingsAreLocked,
    SettingTypeMismatch,
    SettingValue,
    Shell,
    SlaterDeterminantContainer,
    SpinChannel,
    StabilityResult,
    Structure,
    Wavefunction,
    WavefunctionContainer,
    WavefunctionType,
)
from qdk_chemistry.data.base import DataClass
from qdk_chemistry.data.estimator_data import EnergyExpectationResult, MeasurementData
from qdk_chemistry.data.qpe_result import QpeResult
from qdk_chemistry.data.qubit_hamiltonian import QubitHamiltonian

# Give Users the option to use "Error" suffix for exceptions if they prefer
SettingNotFoundError = SettingNotFound
SettingTypeMismatchError = SettingTypeMismatch
SettingsAreLockedError = SettingsAreLocked


__all__ = [
    "AOType",
    "Ansatz",
    "BasisSet",
    "CasWavefunctionContainer",
    "Configuration",
    "CoupledClusterAmplitudes",
    "DataClass",
    "ElectronicStructureSettings",
    "Element",
    "EnergyExpectationResult",
    "Hamiltonian",
    "HamiltonianType",
    "MeasurementData",
    "ModelOrbitals",
    "OrbitalType",
    "Orbitals",
    "QpeResult",
    "QubitHamiltonian",
    "SciWavefunctionContainer",
    "SettingNotFound",
    "SettingNotFoundError",
    "SettingTypeMismatch",
    "SettingTypeMismatchError",
    "SettingValue",
    "Settings",
    "SettingsAreLocked",
    "SettingsAreLockedError",
    "Shell",
    "SlaterDeterminantContainer",
    "SpinChannel",
    "StabilityResult",
    "Structure",
    "Wavefunction",
    "WavefunctionContainer",
    "WavefunctionType",
]
