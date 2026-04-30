"""Sample sparse-CI finder workflow combining QDK/Chemistry primitives with PMC.

This script performs a complete SCF → CASCI → sparse-CI finder sequence for a
user provided geometry and reports the determinant subset that reproduces the
CASCI energy to within a specific accuracy (default 1 mHartree).
The CLI exposes knobs for the initial valence active space selection (number of electrons and orbitals),
active-space solver (including the MACIS ASCI solver), sparse-CI
tolerance, and maximum determinant budget so users can explore different
heuristics without editing the code.
"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import argparse
from collections.abc import Sequence
from pathlib import Path

from qdk_chemistry.algorithms import create
from qdk_chemistry.data import Structure, BasisSet, Ansatz
from qdk_chemistry.utils import Logger, compute_valence_space_parameters

DEFAULT_ENERGY_TOLERANCE = 1.0e-3  # Hartree
DEFAULT_MAX_DETERMINANTS = 2000


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI options in the same order as the workflow steps."""
    parser = argparse.ArgumentParser(description="End-to-end sparse-CI finder demo")
    parser.add_argument(
        "--xyz",
        type=Path,
        help="Path to an XYZ geometry file. Defaults to examples/data/water.structure.xyz.",
    )
    parser.add_argument(
        "--basis",
        default="cc-pvdz",
        help="Basis set applied to the SCF solver (default: cc-pvdz).",
    )
    parser.add_argument(
        "--aux_basis",
        default="cc-pvdz-rifit",
        help="Auxiliary basis set applied to the SCF solver for density fitting (default: cc-pvdz-rifit).",
    )
    parser.add_argument(
        "--charge", type=int, default=0, help="Total molecular charge (default: 0)."
    )
    parser.add_argument(
        "--spin",
        type=int,
        default=1,
        help="Spin multiplicity (2S+1). Default assumes a singlet (1).",
    )
    parser.add_argument(
        "--num-active-electrons",
        type=int,
        default=None,
        help="Override the heuristic valence electron count (optional).",
    )
    parser.add_argument(
        "--num-active-orbitals",
        type=int,
        default=None,
        help="Override the heuristic valence orbital count (optional).",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Drive the simplified SCF → CASCI → sparse-CI workflow."""
    args = parse_args(argv)

    ########################################################################################
    # 1. Load the target structure (fallback to the water example bundled with these demos).
    ########################################################################################
    default_structure = (
        Path(__file__).resolve().parent.parent / "data" / "water.structure.xyz"
    )
    structure_path = args.xyz or default_structure
    if not structure_path.is_file():
        raise FileNotFoundError(f"XYZ file {structure_path} not found.")
    structure = Structure.from_xyz_file(structure_path)
    Logger.info(structure.get_summary())

    ########################################################################################
    # 2. Run the SCF stage to obtain the reference wavefunction.
    ########################################################################################
    # Note that the SCF energy computed here uses the full four-center integrals, whereas
    # the MP2 calculation later uses the density-fitted Hamiltonian, so the reference energy
    # reported by the MP2 calculator will differ from this SCF energy.
    scf_solver = create("scf_solver")
    scf_solver.settings().set("method", "hf")
    scf_solver.settings().set("integral_type", "four_center")
    basis = BasisSet.from_basis_name(args.basis, args.aux_basis, structure)
    e_scf, scf_wavefunction = scf_solver.run(structure, args.charge, args.spin, basis)
    Logger.info(f"SCF energy = {e_scf:.8f} Hartree")

    ########################################################################################
    # 3. Select the valence active space (heuristic or user overrides).
    ########################################################################################
    inferred_e, inferred_orb = compute_valence_space_parameters(
        scf_wavefunction, args.charge
    )
    electrons = (
        args.num_active_electrons
        if args.num_active_electrons is not None
        else inferred_e
    )
    orbitals = (
        args.num_active_orbitals
        if args.num_active_orbitals is not None
        else inferred_orb
    )

    selector = create("active_space_selector", "qdk_valence")
    settings = selector.settings()
    settings.set("num_active_electrons", electrons)
    settings.set("num_active_orbitals", orbitals)

    active_orbital_wavefunction = selector.run(scf_wavefunction)
    active_orbitals = active_orbital_wavefunction.get_orbitals()
    Logger.info(active_orbitals.get_summary())

    ########################################################################################
    # 4. Build the active-space density fitted Hamiltonian.
    ########################################################################################
    hamiltonian_constructor = create(
        "hamiltonian_constructor", "qdk_density_fitted_hamiltonian"
    )
    active_hamiltonian = hamiltonian_constructor.run(active_orbitals)
    Logger.info(active_hamiltonian.get_summary())

    ########################################################################################
    # 5. Run the MP2 calculation.
    ########################################################################################
    # MP2 energy via factory
    ansatz = Ansatz(active_hamiltonian, active_orbital_wavefunction)
    mp2_calculator = create("dynamical_correlation_calculator", "qdk_mp2_calculator")

    # Note that the MP2 calculator recomputes reference energy based on the Density Fitted
    # Hamiltonian, so the reference energy differs from the four-center SCF energy computed earlier.
    mp2_total_energy, _, _ = mp2_calculator.run(ansatz)
    Logger.info(f"MP2 total energy = {mp2_total_energy:.8f} Hartree")


if __name__ == "__main__":
    Logger.set_global_level("info")
    main()
