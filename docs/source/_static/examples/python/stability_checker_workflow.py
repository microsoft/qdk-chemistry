"""Complete stability checker workflow example with orbital rotation."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

################################################################################
# start-cell-create
import numpy as np
from qdk_chemistry.algorithms import create
from qdk_chemistry.data import Structure
from qdk_chemistry.utils import rotate_orbitals
from qdk_chemistry.constants import ANGSTROM_TO_BOHR

# Create the default StabilityChecker instance
stability_checker = create("stability_checker", "qdk")
# end-cell-create
################################################################################

################################################################################
# start-cell-configure
# Configure stability checker settings
stability_checker.settings().set("internal", True)
# Will be adjusted based on calculation type
stability_checker.settings().set("external", True)
stability_checker.settings().set("stability_tolerance", -1e-4)
stability_checker.settings().set("davidson_tolerance", 1e-4)
# Maximum subspace size for Davidson solver
stability_checker.settings().set("max_subspace", 30)
stability_checker.settings().set("method", "hf")

# end-cell-configure
################################################################################

################################################################################
# start-cell-run
# Create N2 molecule at stretched geometry (1.4 Angstrom)
symbols = ["N", "N"]
coords = np.array([[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]])
coords *= ANGSTROM_TO_BOHR
n2 = Structure(symbols, coords)

# Create and configure SCF solver with auto scf_type
scf_solver = create("scf_solver", "qdk")
scf_solver.settings().set("scf_type", "auto")
scf_solver.settings().set("method", "hf")

# Run initial SCF calculation
spin_multiplicity = 1
energy, wavefunction = scf_solver.run(
    n2, charge=0, spin_multiplicity=1, basis_or_guess="def2-svp"
)
print(f"Initial SCF Energy: {energy:.10f} Hartree")

# Determine if calculation is restricted and configure stability checker accordingly
is_restricted = wavefunction.get_orbitals().is_restricted() and spin_multiplicity == 1
if is_restricted:
    stability_checker.settings().set("external", True)
else:
    stability_checker.settings().set("external", False)

# Iterative workflow: Check stability and rotate orbitals until convergence
max_iterations = 5
iteration = 0

print("\n=== Starting Iterative Stability Workflow ===\n")

while iteration < max_iterations:
    iteration += 1
    print(f"--- Iteration {iteration} ---")
    print(f"Current Energy: {energy:.10f} Hartree")

    # Check stability - handle potential Davidson convergence failure
    try:
        is_stable, result = stability_checker.run(wavefunction)
    except RuntimeError as e:
        if "Davidson Did Not Converge!" in str(e):
            print("Try increasing max_subspace or adjusting davidson_tolerance")
            raise RuntimeError(f"Davidson solver did not converge: {str(e)}")
        else:
            raise RuntimeError(f"Stability check failed: {str(e)}")

    if is_stable:
        print("\nConverged to stable wavefunction!")
        break

    # Determine rotation type based on which instability is present
    do_external = False
    if not result.is_internal_stable():
        smallest_eigenvalue, rotation_vector = (
            result.get_smallest_internal_eigenvalue_and_vector()
        )
        print(
            f"Internal instability detected. Smallest eigenvalue: {smallest_eigenvalue:.6f}"
        )
    elif not result.is_external_stable() and result.has_external_result():
        smallest_eigenvalue, rotation_vector = (
            result.get_smallest_external_eigenvalue_and_vector()
        )
        print(
            f"External instability detected. Smallest eigenvalue: {smallest_eigenvalue:.6f}"
        )
        do_external = True
    else:
        print("Unexpected state: neither internal nor external instability detected")
        break

    # Rotate orbitals along the instability direction
    num_alpha, num_beta = wavefunction.get_total_num_electrons()
    orbitals = wavefunction.get_orbitals()
    rotated_orbitals = rotate_orbitals(
        orbitals, rotation_vector, num_alpha, num_beta, do_external
    )

    # If external instability detected, switch to unrestricted calculation
    if do_external:
        print("Switching to unrestricted calculation due to external instability")
        # Create new solver instances with updated settings
        scf_solver_name = scf_solver.name()
        stability_checker_name = stability_checker.name()

        # Copy settings and update for unrestricted calculation
        scf_settings_map = scf_solver.settings().to_dict()
        scf_settings_map["scf_type"] = "unrestricted"
        new_scf_solver = create("scf_solver", scf_solver_name)
        new_scf_solver.settings().from_dict(scf_settings_map)

        stability_settings_map = stability_checker.settings().to_dict()
        stability_settings_map["external"] = False
        new_stability_checker = create("stability_checker", stability_checker_name)
        new_stability_checker.settings().from_dict(stability_settings_map)

        scf_solver = new_scf_solver
        stability_checker = new_stability_checker

    # Re-run SCF with rotated orbitals as initial guess
    energy, wavefunction = scf_solver.run(
        n2, charge=0, spin_multiplicity=1, basis_or_guess=rotated_orbitals
    )
    print(f"New Energy after rotation: {energy:.10f} Hartree")
    print()

print(f"\nFinal Energy: {energy:.10f} Hartree")
print(f"Final stability status: {is_stable}")
# end-cell-run
################################################################################

################################################################################
# start-cell-list-implementations
from qdk_chemistry.algorithms import registry  # noqa: E402

print("available backend choices for scf_solver and stability_checker:")
print(registry.available("stability_checker"))
# end-cell-list-implementations
################################################################################
