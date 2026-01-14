"""PySCF orbital localization module for qdk_chemistry.

This module provides orbital localization capabilities using the PySCF library
and integrates PySCF localization algorithms into the QDK/Chemistry framework.

The implementation supports both restricted (closed-shell) and
unrestricted (open-shell) orbitals: occupied, singly-occupied and
virtual spaces are detected and localized separately to preserve
orthogonality and physical character.

Supported localization methods (selected via the settings `method`):

* "pipek-mezey" :cite:`Boughton1993` (Pipek-Mezey, supports a configurable population analysis),
* "foster-boys" (Foster-Boys),
* "edmiston-ruedenberg" (Edmiston-Ruedenberg),
* "cholesky" (Cholesky-based orthogonalization/localization).

This module registers a ``pyscf`` localizer with the QDK/Chemistry localizer registry at
import time, making the functionality available via
``qdk_chemistry.algorithms.create("orbital_localizer", "pyscf")``.

Requires: PySCF (the code uses the ``pyscf.lo`` localization routines).
"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from pyscf import lo

from qdk_chemistry.algorithms import OrbitalLocalizer
from qdk_chemistry.data import Orbitals, SciWavefunctionContainer, Settings, SlaterDeterminantContainer, Wavefunction
from qdk_chemistry.plugins.pyscf.conversion import basis_to_pyscf_mol
from qdk_chemistry.utils import Logger

__all__ = ["PyscfLocalizer", "PyscfLocalizerSettings"]


class PyscfLocalizerSettings(Settings):
    """Configuration settings for PySCF orbital localization.

    This class manages settings for orbital localization procedures using
    PySCF. It inherits from the Settings base class and exposes the
    configurable options used by :class:`PyscfLocalizer`.

    Attributes:
        method (str): The localization algorithm to use (default = "pipek-mezey").
            Supported values (case-insensitive) include "pipek-mezey", "foster-boys", "edmiston-ruedenberg", and
            "cholesky".

        population_method (str): The population analysis used for the Pipek-Mezey localization (default = "mulliken").
            Passed through to PySCF's PM implementation (for example, "mulliken").

        occupation_threshold (float): Tolerance threshold used to classify orbitals as occupied,
            singly-occupied (for ROHF/UHF), or virtual (default = 1e-10). Orbitals with occupations below this
            threshold are considered unoccupied.

    Examples:
        >>> settings = PyscfLocalizerSettings()
        >>> settings.get("occupation_threshold")
        1e-10
        >>> settings.set("occupation_threshold", 1e-8)
        >>> settings.set("population_method", "mulliken")

    """

    def __init__(self):
        """Initialize the localization object with default parameters."""
        Logger.trace_entering()
        super().__init__()
        self._set_default("method", "string", "pipek-mezey")
        self._set_default("occupation_threshold", "double", 1e-10)
        self._set_default("population_method", "string", "mulliken")


class PyscfLocalizer(OrbitalLocalizer):
    """PySCF-based orbital localizer for quantum chemistry calculations.

    This class implements orbital localization using routines from PySCF.
    It supports multiple localization algorithms and works with both
    restricted (closed-shell) and unrestricted (open-shell) orbital inputs.

    Key behavior:

        - Supported algorithms: Pipek-Mezey (PM), Foster-Boys (FB), Edmiston-Ruedenberg (ER), and a Cholesky-based
          localizer.
        - It is the user's responsibility to provide appropriate orbital indices (e.g., only occupied orbitals or only
          virtual orbitals).

    Examples:
        >>> localizer = PyscfLocalizer()
        >>> # Localize occupied orbitals (user provides appropriate indices)
        >>> occ_indices = [0, 1, 2, 3, 4]  # indices of occupied orbitals
        >>> localized_orbitals = localizer.run(canonical_orbitals, occ_indices, occ_indices)

    Notes:
        Uses Mulliken population analysis for the Pipek-Mezey method by default.
        The population method can be configured via the `population_method` setting.

    """

    def __init__(self):
        """Initialize the PySCF localizer with default settings."""
        Logger.trace_entering()
        super().__init__()
        self._settings = PyscfLocalizerSettings()

    def _run_impl(self, wavefunction: Wavefunction, loc_indices_a: list[int], loc_indices_b: list[int]):
        """Perform orbital localization using PySCF localization method.

        This method operates on an input Wavefunction object and returns a new Wavefunction
        instance containing localized molecular orbitals. Only orbitals specified in
        the index lists are localized, with all specified orbitals localized together
        as a single group.

        The method supports both restricted (closed-shell) and unrestricted (open-shell)
        orbital inputs. For restricted orbitals, loc_indices_a and loc_indices_b must
        be identical.

        Args:
            wavefunction: The wavefunction with molecular orbitals to be localized.
                Can be restricted or unrestricted (spin-polarized).
            loc_indices_a: Indices of alpha orbitals to localize (must be sorted, empty for no localization).
            loc_indices_b: Indices of beta orbitals to localize (must be sorted, empty for no localization).
                For restricted orbitals, this must be identical to loc_indices_a.

        Returns:
            A new Orbitals object containing the localized molecular orbitals.

        Raises:
            ValueError: If an unknown localization method is requested, if
                loc_indices_a and loc_indices_b differ for restricted orbitals,
                or if the indices are not sorted.

        """
        Logger.trace_entering()
        # Validate that indices are sorted
        if loc_indices_a != sorted(loc_indices_a):
            raise ValueError("loc_indices_a must be sorted")
        if loc_indices_b != sorted(loc_indices_b):
            raise ValueError("loc_indices_b must be sorted")

        # If both index vectors are empty, return original orbitals unchanged
        if len(loc_indices_a) == 0 and len(loc_indices_b) == 0:
            return wavefunction

        pop_method = self._settings.get("population_method")
        loc_method = self._settings.get("method").lower()

        orbitals = wavefunction.get_orbitals()

        # Convert basis set to PySCF Mole object
        mol = basis_to_pyscf_mol(orbitals.get_basis_set())

        def _do_loc(inp, indices):
            """Perform localization on the specified orbitals."""
            if len(indices) in [0, 1]:
                return inp
            if loc_method == "pipek-mezey":
                return lo.PM(mol, inp[:, indices], mf=None, pop_method=pop_method).kernel()
            if loc_method == "foster-boys":
                return lo.Boys(mol, inp[:, indices]).kernel()
            if loc_method == "edmiston-ruedenberg":
                return lo.ER(mol, inp[:, indices]).kernel()
            if loc_method == "cholesky":
                return lo.cholesky_mos(inp[:, indices])
            raise ValueError(f"Unknown localization method: {loc_method}")

        # Preserve active/inactive space indices from input orbitals if they exist
        # For restricted: indices = (active, inactive)
        # For unrestricted: indices = (active_alpha, active_beta, inactive_alpha, inactive_beta)
        if orbitals.has_active_space():
            active_alpha, active_beta = orbitals.get_active_space_indices()
            inactive_alpha, inactive_beta = orbitals.get_inactive_space_indices()
        else:
            active_alpha = active_beta = inactive_alpha = inactive_beta = None

        # Perform localization and populate localized orbitals instance
        if orbitals.is_restricted():
            if loc_indices_a != loc_indices_b:
                raise ValueError("For restricted orbitals, loc_indices_a and loc_indices_b must be identical")

            # Start with original coefficients
            mo_coeffs = orbitals.get_coefficients()[0]  # For restricted, get alpha coefficients
            mo_loc = mo_coeffs.copy()

            localized_mos = _do_loc(mo_coeffs, loc_indices_a)
            for i, idx in enumerate(loc_indices_a):
                mo_loc[:, idx] = localized_mos[:, i]

            loc_orbitals = Orbitals(
                coefficients=mo_loc,
                energies=orbitals.get_energies()[0] if orbitals.has_energies() else None,
                ao_overlap=orbitals.get_overlap_matrix() if orbitals.has_overlap_matrix() else None,
                basis_set=orbitals.get_basis_set(),
                indices=(list(active_alpha), list(inactive_alpha)) if active_alpha is not None else None,
            )
        else:
            # Unrestricted case - handle alpha and beta separately
            mo_coeffs_alpha, mo_coeffs_beta = orbitals.get_coefficients()
            mo_a = mo_coeffs_alpha.copy()
            mo_b = mo_coeffs_beta.copy()

            localized_a = _do_loc(mo_coeffs_alpha, loc_indices_a)
            for i, idx in enumerate(loc_indices_a):
                mo_a[:, idx] = localized_a[:, i]

            localized_b = _do_loc(mo_coeffs_beta, loc_indices_b)
            for i, idx in enumerate(loc_indices_b):
                mo_b[:, idx] = localized_b[:, i]

            energies_alpha, energies_beta = orbitals.get_energies() if orbitals.has_energies() else (None, None)
            loc_orbitals = Orbitals(
                coefficients_alpha=mo_a,
                coefficients_beta=mo_b,
                energies_alpha=energies_alpha,
                energies_beta=energies_beta,
                ao_overlap=orbitals.get_overlap_matrix() if orbitals.has_overlap_matrix() else None,
                basis_set=orbitals.get_basis_set(),
                indices=(
                    list(active_alpha),
                    list(active_beta),
                    list(inactive_alpha),
                    list(inactive_beta),
                )
                if active_alpha is not None
                else None,
            )
        if len(wavefunction.get_active_determinants()) == 1:
            # Single determinant case - return new wavefunction with localized orbitals
            return Wavefunction(SlaterDeterminantContainer(wavefunction.get_active_determinants()[0], loc_orbitals))
        return Wavefunction(SciWavefunctionContainer(wavefunction.get_active_determinants(), loc_orbitals))

    def name(self) -> str:
        """Return the settings for the localizer."""
        Logger.trace_entering()
        return "pyscf_multi"
