"""Code snippets for v1.1.0 release notes.

Each cell is a runnable snippet included in the Sphinx release notes via
``literalinclude`` with ``start-after`` / ``end-before`` markers.  The file
is executed end-to-end by the ``test_docs_examples.py`` test harness.
"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import importlib.util
from pathlib import Path

import numpy as np
from qdk_chemistry.algorithms import create
from qdk_chemistry.data import Structure

# ---------------------------------------------------------------------------
# Shared pipeline: H2 molecule → SCF → active-space → Hamiltonian → QubitHam
# (used by cells below that need a fermionic or qubit Hamiltonian)
# ---------------------------------------------------------------------------
_h2_structure = Structure.from_xyz_file(
    Path(__file__).parent / "../data/h2.structure.xyz"
)
_scf = create("scf_solver")
_E_scf, _wfn_scf = _scf.run(
    _h2_structure, charge=0, spin_multiplicity=1, basis_or_guess="sto-3g"
)
_orbitals = _wfn_scf.get_orbitals()

_ham_constructor = create("hamiltonian_constructor")
_hamiltonian = _ham_constructor.run(_orbitals)

_qdk_mapper = create("qubit_mapper", "qdk")
_qubit_hamiltonian = _qdk_mapper.run(_hamiltonian)

# ===========================================================================
# Model Hamiltonians — fermionic
# ===========================================================================

################################################################################
# start-cell-model-fermionic
from qdk_chemistry.data import LatticeGraph  # noqa: E402
from qdk_chemistry.utils.model_hamiltonians import (  # noqa: E402
    create_huckel_hamiltonian,
    create_hubbard_hamiltonian,
    create_ppp_hamiltonian,
    ohno_potential,
)

# 8-site open chain with tight-binding (Hückel) hopping
chain = LatticeGraph.chain(8)
ham = create_huckel_hamiltonian(chain, epsilon=-0.5, t=1.0)

# Add on-site Coulomb repulsion (Hubbard)
ham = create_hubbard_hamiltonian(chain, epsilon=-0.5, t=1.0, U=4.0)

# 4-site periodic ring with intersite Coulomb via Ohno potential (PPP)
ring = LatticeGraph.chain(4, periodic=True)
V = ohno_potential(ring, U=11.26, R=1.4, nearest_neighbor_only=True)
ham = create_ppp_hamiltonian(ring, epsilon=0.0, t=2.4, U=11.26, V=V, z=1.0)
# end-cell-model-fermionic
################################################################################

# ===========================================================================
# Model Hamiltonians — spin
# ===========================================================================

################################################################################
# start-cell-model-spin
from qdk_chemistry.utils.model_hamiltonians import (  # noqa: E402
    create_ising_hamiltonian,
    create_heisenberg_hamiltonian,
)

# Transverse-field Ising model (ZZ coupling + transverse X field)
ising = create_ising_hamiltonian(chain, j=1.0, h=0.5)

# Isotropic Heisenberg (XXX) model
heisenberg = create_heisenberg_hamiltonian(chain, jx=1.0, jy=1.0, jz=1.0)
# end-cell-model-spin
################################################################################

# ===========================================================================
# Lattice geometries
# ===========================================================================

################################################################################
# start-cell-lattice
LatticeGraph.chain(8, periodic=True)  # ring
LatticeGraph.square(4, 4)  # 2D grid
LatticeGraph.triangular(3, 3)
LatticeGraph.honeycomb(3, 3)
LatticeGraph.kagome(3, 3)

# Or from an adjacency matrix / edge dict
LatticeGraph({(0, 1): 1.0, (1, 2): 0.5}, num_sites=3)
# end-cell-lattice
################################################################################

# ===========================================================================
# Trotter-Suzuki — build a time-evolution unitary
# ===========================================================================

qubit_hamiltonian = _qubit_hamiltonian  # alias for display

################################################################################
# start-cell-trotter
from qdk_chemistry.algorithms.time_evolution.builder.trotter import Trotter  # noqa: E402

builder = Trotter(order=2, target_accuracy=1e-3, error_bound="commutator")
unitary = builder.run(qubit_hamiltonian, time=1.0)
# end-cell-trotter
################################################################################

# ===========================================================================
# Trotter error-bound utilities
# ===========================================================================

################################################################################
# start-cell-trotter-error
from qdk_chemistry.algorithms.time_evolution.builder.trotter_error import (  # noqa: E402
    trotter_steps_commutator,
)

n_steps = trotter_steps_commutator(
    qubit_hamiltonian, time=1.0, target_accuracy=1e-3, order=2
)
# end-cell-trotter-error
################################################################################

# ===========================================================================
# Native ROHF — restricted open-shell Hartree-Fock
# ===========================================================================

# Use lithium atom (doublet) for an open-shell example
structure = Structure(np.array([[0.0, 0.0, 0.0]]), symbols=["Li"])

################################################################################
# start-cell-rohf
scf_solver = create("scf_solver")
scf_solver.settings().set("method", "hf")
scf_solver.settings().set("scf_type", "restricted")
scf_solver.settings().set("enable_gdm", False)
energy, wavefunction = scf_solver.run(
    structure, charge=0, spin_multiplicity=2, basis_or_guess="sto-3g"
)
# end-cell-rohf
################################################################################

print(f"ROHF energy: {energy:.10f} Hartree")

# ===========================================================================
# Cholesky-based AO→MO transformation
# ===========================================================================

orbitals = _orbitals  # alias for display

################################################################################
# start-cell-cholesky
constructor = create("hamiltonian_constructor", "qdk_cholesky")
constructor.settings().set("cholesky_tolerance", 1e-8)
constructor.settings().set("eri_threshold", 1e-12)
constructor.settings().set("store_cholesky_vectors", True)
cholesky_hamiltonian = constructor.run(orbitals)
# end-cell-cholesky
################################################################################

print(
    f"Cholesky Hamiltonian core energy: {cholesky_hamiltonian.get_core_energy():.10f}"
)

# ===========================================================================
# MACIS — single-orbital entropies and mutual information
# ===========================================================================

hamiltonian = _hamiltonian  # alias for display

################################################################################
# start-cell-macis-entropies
mc = create("multi_configuration_calculator", "macis_asci")
mc.settings().set("calculate_single_orbital_entropies", True)
mc.settings().set("calculate_mutual_information", True)
E_asci, wfn_asci = mc.run(hamiltonian, 1, 1)
# end-cell-macis-entropies
################################################################################

print(f"ASCI energy: {E_asci:.10f} Hartree")

# ===========================================================================
# Energy estimator
# ===========================================================================

# Build a trivial state-preparation circuit for H2; reuse existing objects
_cas = create("multi_configuration_calculator", "macis_cas")
_E_cas, _wfn_cas = _cas.run(_hamiltonian, 1, 1)

_state_prep = create("state_prep", "sparse_isometry_gf2x")
_circuit = _state_prep.run(_wfn_cas)

circuit_executor = create("circuit_executor", "qdk_sparse_state_simulator")
circuit = _circuit  # alias for display

################################################################################
# start-cell-energy-estimator
estimator = create("energy_estimator", "qdk")
energy_result, measurement_data = estimator.run(
    circuit,
    qubit_hamiltonian,
    circuit_executor,
    total_shots=10_000,
)
# end-cell-energy-estimator
################################################################################

print(f"Estimated energy: {energy_result.energy_expectation_value:.6f}")

# ===========================================================================
# Symmetries data class
# ===========================================================================

################################################################################
# start-cell-symmetries
from qdk_chemistry.data import Symmetries  # noqa: E402

sym = Symmetries(n_alpha=3, n_beta=2)
sym.n_particles  # 5
sym.sz  # 0.5
sym.spin_multiplicity  # 2

# Or construct from an existing wavefunction
sym = Symmetries.from_wavefunction(wavefunction)
# end-cell-symmetries
################################################################################

print(f"Symmetries: n_particles={sym.n_particles}, sz={sym.sz}")

# ===========================================================================
# OpenFermion plugin (optional — skipped if openfermion is not installed)
# ===========================================================================
_HAS_OPENFERMION = importlib.util.find_spec("openfermion") is not None

if _HAS_OPENFERMION:
    ############################################################################
    # start-cell-openfermion
    from qdk_chemistry.data import Symmetries  # noqa: E402, F811

    mapper = create("qubit_mapper", "openfermion", encoding="jordan-wigner")
    qh = mapper.run(hamiltonian)

    # Symmetry-conserving Bravyi-Kitaev (reduces qubit count by 2)
    sym = Symmetries(n_alpha=1, n_beta=1)
    mapper = create(
        "qubit_mapper",
        "openfermion",
        encoding="symmetry-conserving-bravyi-kitaev",
    )
    qh = mapper.run(hamiltonian, sym)
    # end-cell-openfermion
    ############################################################################

    print(f"OpenFermion SCBK qubit count: {qh.num_qubits}")
