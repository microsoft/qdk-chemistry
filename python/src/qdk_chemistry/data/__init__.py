"""QDK/Chemistry data module for quantum chemistry data structures and settings.

This module provides access to core quantum chemistry data types including molecular
structures, basis sets, wavefunctions, and computational settings. It serves as the
primary interface for managing quantum chemical data within the QDK/Chemistry framework.

Exposed classes are:

- :class:`Ansatz`: Quantum chemical ansatz combining a Hamiltonian and wavefunction for energy calculations.
- :class:`AOType`: Enumeration of basis set types (STO-3G, 6-31G, etc.).
- :class:`BasisSet`: Gaussian basis set definitions for quantum calculations.
- :class:`CanonicalFourCenterHamiltonianContainer`: Container for four-center two-electron integrals in canonical form.
- :class:`CasWavefunctionContainer`: Complete Active Space (CAS) wavefunction with CI coefficients and determinants.
- :class:`CholeskyHamiltonianContainer`: Container for Hamiltonians represented using Cholesky-decomposed integrals.
- :class:`Circuit`: Quantum circuit information.
- :class:`Configuration`: Electronic configuration state information.
- :class:`ConfigurationSet`: Collection of electronic configurations with associated orbital information.
- :class:`ControlledTimeEvolutionUnitary`: Controlled time evolution unitary.
- :class:`CoupledClusterContainer`: Container for coupled cluster wavefunction amplitudes and determinants.
- :class:`DataClass`: Base data class.
- :class:`ElectronicStructureSettings`: Specialized settings for electronic structure calculations.
- :class:`Element`: Represents a chemical element with its properties.
- :class:`EnergyExpectationResult`: Result for Hamiltonian energy expectation value and variance.
- :class:`Hamiltonian`: Quantum mechanical Hamiltonian operator representation.
- :class:`HamiltonianContainer`: Abstract base class for different Hamiltonian storage formats.
- :class:`HamiltonianType`: Enumeration of Hamiltonian types (Hermitian, NonHermitian).
- :class:`LatticeGraph`: Lattice graph defining the connectivity and geometry of a model Hamiltonian.
- :class:`MeasurementData`: Measurement bitstring data and metadata for QubitHamiltonian objects.
- :class:`SparseHamiltonianContainer`: Container for lattice model Hamiltonians with sparse internal storage.
- :class:`ModelOrbitals`: Simple orbital representation for model systems without full basis set information.
- :class:`MP2Container`: Container for MP2 wavefunction with Hamiltonian reference and optional amplitudes.
- :class:`Orbitals`: Molecular orbital information and properties.
- :class:`OrbitalType`: Enumeration of orbital angular momentum types (s, p, d, f, etc.).
- :class:`PauliOperator`: Pauli operator (I, X, Y, Z) for quantum operator expressions with arithmetic support.
- :class:`PauliProductFormulaContainer`: Container for Pauli product formula representation of time evolution unitary.
- :class:`QpeResult`: Result of quantum phase estimation workflows, including phase, energy, and metadata.
- :class:`QuantumErrorProfile`: Information about quantum gates and error properties.
- :class:`QubitHamiltonian`: Molecular electronic Hamiltonians mapped to qubits.
- :class:`SciWavefunctionContainer`: Selected Configuration Interaction (SCI) wavefunction with CI coefficients.
- :class:`Settings`: Configuration settings for quantum chemistry calculations.
- :class:`SettingValue`: Type-safe variant for storing different setting value types.
- :class:`Shell`: Individual shell within a basis set.
- :class:`SlaterDeterminantContainer`: Single Slater determinant wavefunction representation.
- :class:`StabilityResult`: Result of stability analysis for electronic structure calculations.
- :class:`Structure`: Molecular structure and geometry information.
- :class:`Symmetries`: Physical symmetries of an electronic state for symmetry-exploiting algorithms.
- :class:`TermPartition`: Index-based partition of Hamiltonian terms.
  See :class:`FlatPartition` and :class:`LayeredPartition`.
- :class:`TimeEvolutionUnitary`: Time evolution unitary.
- :class:`TimeEvolutionUnitaryContainer`: Abstract base class for different time evolution unitary representation.
- :class:`Wavefunction`: Electronic wavefunction data and coefficients.
- :class:`WavefunctionContainer`: Abstract base class for different wavefunction representations.
- :class:`WavefunctionType`: Enumeration of wavefunction types (SelfDual, NotSelfDual).

Exposed exceptions are:

- :exc:`SettingNotFound` / :exc:`SettingNotFoundError`: Raised when a requested setting is not found.
- :exc:`SettingTypeMismatch` / :exc:`SettingTypeMismatchError`: Raised when a setting value has an incorrect type.

"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from contextlib import suppress

from qdk_chemistry._core.data import (
    AlgorithmRef,
    Ansatz,
    AOType,
    BasisSet,
    CanonicalFourCenterHamiltonianContainer,
    CasWavefunctionContainer,
    CholeskyHamiltonianContainer,
    Configuration,
    ConfigurationSet,
    CoupledClusterContainer,
    ElectronicStructureSettings,
    Element,
    Hamiltonian,
    HamiltonianContainer,
    HamiltonianType,
    LatticeGraph,
    ModelOrbitals,
    MP2Container,
    Orbitals,
    OrbitalType,
    PauliOperator,
    PauliTermAccumulator,
    SciWavefunctionContainer,
    SettingNotFound,
    Settings,
    SettingsAreLocked,
    SettingTypeMismatch,
    SettingValue,
    Shell,
    SlaterDeterminantContainer,
    SparseHamiltonianContainer,
    SpinChannel,
    StabilityResult,
    Structure,
    Wavefunction,
    WavefunctionContainer,
    WavefunctionType,
    get_current_ciaaw_version,
)
from qdk_chemistry.data.base import DataClass
from qdk_chemistry.data.circuit import Circuit
from qdk_chemistry.data.circuit_executor_data import CircuitExecutorData
from qdk_chemistry.data.encoding_validation import EncodingMismatchError, validate_encoding_compatibility
from qdk_chemistry.data.enums.fermion_mode_order import FermionModeOrder
from qdk_chemistry.data.estimator_data import EnergyExpectationResult, MeasurementData
from qdk_chemistry.data.noise_models import QuantumErrorProfile
from qdk_chemistry.data.qpe_result import QpeResult
from qdk_chemistry.data.qubit_hamiltonian import QubitHamiltonian
from qdk_chemistry.data.symmetries import Symmetries
from qdk_chemistry.data.term_partition import FlatPartition, LayeredPartition, TermPartition
from qdk_chemistry.data.time_evolution.base import TimeEvolutionUnitary
from qdk_chemistry.data.time_evolution.containers.base import TimeEvolutionUnitaryContainer
from qdk_chemistry.data.time_evolution.containers.pauli_product_formula import PauliProductFormulaContainer
from qdk_chemistry.data.time_evolution.controlled_time_evolution import ControlledTimeEvolutionUnitary

# Give Users the option to use "Error" suffix for exceptions if they prefer
SettingNotFoundError = SettingNotFound
SettingTypeMismatchError = SettingTypeMismatch
SettingsAreLockedError = SettingsAreLocked


__all__ = [
    "AOType",
    "AlgorithmRef",
    "Ansatz",
    "BasisSet",
    "CanonicalFourCenterHamiltonianContainer",
    "CasWavefunctionContainer",
    "CholeskyHamiltonianContainer",
    "Circuit",
    "CircuitExecutorData",
    "Configuration",
    "ConfigurationSet",
    "ControlledTimeEvolutionUnitary",
    "CoupledClusterContainer",
    "DataClass",
    "ElectronicStructureSettings",
    "Element",
    "EncodingMismatchError",
    "EnergyExpectationResult",
    "FermionModeOrder",
    "FlatPartition",
    "Hamiltonian",
    "HamiltonianContainer",
    "HamiltonianType",
    "LatticeGraph",
    "LayeredPartition",
    "MP2Container",
    "MeasurementData",
    "ModelOrbitals",
    "OrbitalType",
    "Orbitals",
    "PauliOperator",
    "PauliProductFormulaContainer",
    "PauliTermAccumulator",
    "QpeResult",
    "QuantumErrorProfile",
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
    "SparseHamiltonianContainer",
    "SpinChannel",
    "StabilityResult",
    "Structure",
    "Symmetries",
    "TermPartition",
    "TimeEvolutionUnitary",
    "TimeEvolutionUnitaryContainer",
    "Wavefunction",
    "WavefunctionContainer",
    "WavefunctionType",
    "get_current_ciaaw_version",
    "validate_encoding_compatibility",
]


# ---------------------------------------------------------------------------
# LatticeGraph.edge_coloring overlay
# ---------------------------------------------------------------------------
#
# LatticeGraph is bound from C++ and computes its edge coloring in C++ via
# :meth:`LatticeGraph._edge_coloring_raw`, returning a ``{(i, j): color}`` dict.
# The Python overlay below wraps that dict in a
# :class:`~qdk_chemistry.geometry.HypergraphEdgeColoring`, the richer Python
# representation used by downstream consumers (e.g. the spin-model Hamiltonian
# builders).  Cached colorings on the C++ side mean repeat calls are cheap.


def _lattice_edge_coloring(self, *, seed: int | None = 0, trials: int = 1):
    """Compute an edge coloring of this lattice.

    Delegates to the C++ implementation on :class:`LatticeGraph` (which uses
    a deterministic optimal coloring for ``CHAIN`` and ``SQUARE`` kinds and
    a cached randomised greedy coloring otherwise) and wraps the result in a
    :class:`~qdk_chemistry.geometry.HypergraphEdgeColoring`.

    Args:
        self: The :class:`LatticeGraph` instance whose edges are to be colored.
        seed: Random seed for the greedy fallback (ignored for deterministic
            kinds; ``None`` is treated as 0).
        trials: Number of randomised trials for the greedy fallback; the
            coloring with the fewest colors is returned.

    Returns:
        :class:`~qdk_chemistry.geometry.HypergraphEdgeColoring`: A coloring
        whose ``hypergraph`` carries the same undirected edges as this
        :class:`LatticeGraph`.

    """
    # Imported lazily to avoid a circular import at module load time.
    from qdk_chemistry.geometry.hypergraph import Hyperedge, Hypergraph, HypergraphEdgeColoring  # noqa: PLC0415

    raw = self._edge_coloring_raw(seed=0 if seed is None else int(seed), trials=int(trials))
    edges = [Hyperedge(list(e)) for e in raw]
    hypergraph = Hypergraph(edges)
    coloring = HypergraphEdgeColoring(hypergraph)
    for edge in edges:
        coloring.add_edge(edge, raw[tuple(edge.vertices)])
    return coloring


LatticeGraph.edge_coloring = _lattice_edge_coloring  # type: ignore[attr-defined]
del _lattice_edge_coloring
