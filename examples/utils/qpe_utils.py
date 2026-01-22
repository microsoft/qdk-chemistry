"""Utility functions for QPE."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import numpy as np
from qdk_chemistry.data import SciWavefunctionContainer, Wavefunction


def prepare_2_dets_trial_state(
    wf: Wavefunction, rotation_angle: float = np.pi / 12
) -> tuple[Wavefunction, float]:
    """Scan rotation angles for 2-determinant wavefunction.

        psi(theta) = cos(theta)*|D1> + sin(theta)*|D2|

    Args:
        wf: Original wavefunction (used to extract determinants)
        rotation_angle: Rotation angle (in radians)

    Returns:
        wavefunction: Wavefunction object for the given rotation angle
        fidelity: Fidelity with respect to the exact wavefunction

    """
    dets = wf.get_top_determinants(max_determinants=2)
    orbitals = wf.get_orbitals()

    c1_new = np.cos(round(rotation_angle, 4))
    c2_new = np.sin(round(rotation_angle, 4))

    # Only include terms with non-zero coefficients
    coeffs_new = []
    dets_new = []

    for coeff, det in zip([c1_new, c2_new], dets):
        if not np.isclose(coeff, 0.0):
            coeffs_new.append(coeff)
            dets_new.append(det)

    # Convert to numpy arrays and normalize
    coeffs_new = np.array(coeffs_new, dtype=float)
    coeffs_new /= np.linalg.norm(coeffs_new)

    # Construct trial wavefunction
    rotated_wf = Wavefunction(SciWavefunctionContainer(coeffs_new, dets_new, orbitals))

    # Fidelity with original reference wf
    coeffs_wf = np.array(list(dets.values()))
    fidelity = np.abs(np.vdot(coeffs_new, coeffs_wf)) ** 2

    return rotated_wf, fidelity
