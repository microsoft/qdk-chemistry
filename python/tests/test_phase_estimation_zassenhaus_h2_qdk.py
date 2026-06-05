"""End-to-end H2/STO-3G IQPE test using Zassenhaus evolution on QDK backends."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import os

import numpy as np
import pytest

# The SCF stack imports matplotlib transitively in some environments.  Point the
# cache at a writable location so this slow integration test does not warn or
# fail when the user's home directory is read-only under pytest.
os.environ.setdefault("MPLCONFIGDIR", "/tmp")

from qdk_chemistry.algorithms import create
from qdk_chemistry.data import AlgorithmRef, MajoranaMapping, PauliProductFormulaContainer, Structure
from qdk_chemistry.utils import Logger
from qdk_chemistry.utils.phase import resolve_energy_aliases

from .reference_tolerances import float_comparison_relative_tolerance

try:
    import pyscf  # noqa: F401

    PYSCF_AVAILABLE = True
except ImportError:
    PYSCF_AVAILABLE = False

_RUN_SLOW_TESTS = os.getenv("QDK_CHEMISTRY_RUN_SLOW_TESTS", "").lower() in {"1", "true", "yes"}

_LOG_SLOW_TEST_DETAILS = os.getenv("QDK_CHEMISTRY_LOG_SLOW_TEST_DETAILS", "").lower() in {"1", "true", "yes"}


def _log_if_requested(message: str) -> None:
    """Print opt-in diagnostics for this slow workflow test."""
    if _LOG_SLOW_TEST_DETAILS:
        print(message)


def _qir_size(circuit) -> tuple[int, int]:
    """Return line and character counts for generated QIR as a size proxy."""
    qir = str(circuit.get_qir())
    return len(qir.splitlines()), len(qir)


# This is intentionally a slow, opt-in regression test.  It exercises the same
# high-level plumbing as the public IQPE examples, but keeps the backend path
# entirely inside QDK: QDK electronic-structure setup, QDK qubit mapping, QDK
# state preparation, QDK circuit mapping, and QDK full-state simulation.
@pytest.mark.slow
@pytest.mark.skipif(not PYSCF_AVAILABLE, reason="PySCF not available")
@pytest.mark.skipif(
    not _RUN_SLOW_TESTS,
    reason="Skipping slow test. Set QDK_CHEMISTRY_RUN_SLOW_TESTS=1 to enable.",
)
def test_qdk_iqpe_zassenhaus_h2_sto3g_ground_state_energy() -> None:
    """Estimate the H2/STO-3G ground-state energy with QDK IQPE and Zassenhaus evolution."""
    Logger.set_global_level("error")

    # Build the molecular input for H2/STO-3G.  The coordinates are in bohr,
    # matching the existing IQPE example scripts.
    structure = Structure(
        np.array([[0.0, 0.0, -0.72], [0.0, 0.0, 0.72]], dtype=float),
        ["H", "H"],
    )

    scf_solver = create("scf_solver")
    scf_total_energy, scf_wavefunction = scf_solver.run(
        structure,
        charge=0,
        spin_multiplicity=1,
        basis_or_guess="sto-3g",
    )

    # Select the minimal two-electron/two-orbital active space and compute the
    # CASCI reference.  IQPE estimates the electronic active-space energy, so the
    # core energy is removed before alias resolution and added back afterward.
    selector = create(
        "active_space_selector",
        "qdk_valence",
        num_active_electrons=2,
        num_active_orbitals=2,
    )
    active_orbitals = selector.run(scf_wavefunction).get_orbitals()

    constructor = create("hamiltonian_constructor")
    active_hamiltonian = constructor.run(active_orbitals)

    mc_calculator = create("multi_configuration_calculator")
    casci_total_energy, casci_wavefunction = mc_calculator.run(active_hamiltonian, 1, 1)
    reference_electronic_energy = casci_total_energy - active_hamiltonian.get_core_energy()
    _log_if_requested(
        "\n".join(
            [
                "H2/STO-3G reference energies:",
                f"  SCF total energy:   {scf_total_energy:.12f} Hartree",
                f"  CASCI total energy: {casci_total_energy:.12f} Hartree",
                f"  Core energy:        {active_hamiltonian.get_core_energy():.12f} Hartree",
                f"  CASCI electronic:   {reference_electronic_energy:.12f} Hartree",
            ]
        )
    )

    # Use the QDK qubit mapper so this test validates the
    # Zassenhaus builder as a drop-in source for the QDK phase-estimation stack.
    # Note: For this tiny H2 Hamiltonian, coefficient pruning does not seem to
    # reduce the term count; all 15 Jordan-Wigner terms are non-negligible.
    n_spin_orbitals = 2 * active_hamiltonian.get_orbitals().get_num_molecular_orbitals()
    qubit_hamiltonian = create("qubit_mapper", "qdk").run(
        active_hamiltonian,
        MajoranaMapping.jordan_wigner(n_spin_orbitals),
    )
    state_preparation = create("state_prep", "sparse_isometry_gf2x").run(casci_wavefunction)
    _log_if_requested(
        "\n".join(
            [
                "Qubit Hamiltonian:",
                f"  Qubits: {qubit_hamiltonian.num_qubits}",
                f"  Pauli terms: {len(qubit_hamiltonian.pauli_strings)}",
            ]
        )
    )

    # Use second-order Zassenhaus with enough divisions to keep the finite-step
    # error below chemical accuracy (simulating a method deployable on near-term
    # hardware).
    evolution_time = 1.0
    zassenhaus_order = 2
    zassenhaus_num_divisions = 2
    unitary_builder_ref = AlgorithmRef(
        "hamiltonian_unitary_builder",
        "zassenhaus",
        time=evolution_time,
        order=zassenhaus_order,
        num_divisions=zassenhaus_num_divisions,
    )
    if _LOG_SLOW_TEST_DETAILS:
        unitary_builder = create(
            "hamiltonian_unitary_builder",
            "zassenhaus",
            time=evolution_time,
            order=zassenhaus_order,
            num_divisions=zassenhaus_num_divisions,
        )
        unitary = unitary_builder.run(qubit_hamiltonian)
        container = unitary.get_container()
        if isinstance(container, PauliProductFormulaContainer):
            _log_if_requested(
                "\n".join(
                    [
                        "Zassenhaus product-formula size:",
                        f"  Order: {zassenhaus_order}",
                        f"  Requested divisions: {zassenhaus_num_divisions}",
                        f"  Step terms: {len(container.step_terms)}",
                        f"  Step repetitions: {container.step_reps}",
                        f"  Total exponentials: {len(container.step_terms) * container.step_reps}",
                    ]
                )
            )

    iqpe = create("phase_estimation", "iterative", num_bits=12, shots_per_bit=1)
    iqpe.settings().set(
        "circuit_executor",
        AlgorithmRef("circuit_executor", "qdk_full_state_simulator", seed=42),
    )
    iqpe.settings().set(
        "unitary_builder",
        unitary_builder_ref,
    )
    iqpe.settings().set(
        "circuit_mapper",
        AlgorithmRef("controlled_circuit_mapper", "pauli_sequence"),
    )

    result = iqpe.run(
        state_preparation=state_preparation,
        qubit_hamiltonian=qubit_hamiltonian,
    )
    if _LOG_SLOW_TEST_DETAILS:
        iteration_circuits = iqpe.get_circuits()
        circuit_sizes = [_qir_size(circuit) for circuit in iteration_circuits]
        largest_lines, largest_chars = max(circuit_sizes)
        first_lines, first_chars = circuit_sizes[0]
        last_lines, last_chars = circuit_sizes[-1]
        _log_if_requested(
            "\n".join(
                [
                    "Generated IQPE circuit sizes (QIR text proxy):",
                    f"  Iteration circuits: {len(iteration_circuits)}",
                    f"  First circuit: {first_lines} QIR lines, {first_chars} chars",
                    f"  Last circuit: {last_lines} QIR lines, {last_chars} chars",
                    f"  Largest circuit: {largest_lines} QIR lines, {largest_chars} chars",
                ]
            )
        )

    estimated_electronic_energy = resolve_energy_aliases(
        result.raw_energy,
        evolution_time=evolution_time,
        reference_energy=reference_electronic_energy,
        shift_range=range(-3, 4),
    )
    estimated_total_energy = estimated_electronic_energy + active_hamiltonian.get_core_energy()
    _log_if_requested(
        "\n".join(
            [
                "QPE energy estimate:",
                f"  Measured bits: {result.bits_msb_first}",
                f"  Phase fraction: {result.phase_fraction:.12f}",
                f"  Raw electronic energy: {result.raw_energy:.12f} Hartree",
                f"  Resolved electronic energy: {estimated_electronic_energy:.12f} Hartree",
                f"  Estimated total energy: {estimated_total_energy:.12f} Hartree",
                f"  CASCI total energy: {casci_total_energy:.12f} Hartree",
                f"  Total-energy error: {estimated_total_energy - casci_total_energy:+.12e} Hartree",
            ]
        )
    )

    # The 1.6e-3 Hartree tolerance is the usual chemical-accuracy margin of error:
    assert np.isclose(
        estimated_total_energy,
        casci_total_energy,
        rtol=float_comparison_relative_tolerance,
        atol=1.6e-3,
    )
