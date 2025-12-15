"""PySCF AVAS active space selector implementation for qdk_chemistry.

This module provides an interface to the PySCF Automated Valence Active Space (AVAS)
method for selecting active spaces in quantum chemistry calculations. The AVAS method
automatically constructs molecular active spaces from atomic valence orbitals.

The module contains:

* :class:`PyscfAVASSettings`: Configuration class for AVAS parameters
* :class:`PyscfAVAS`: Main active space selector implementing the AVAS algorithm
* Registration functions to integrate with the QDK/Chemistry framework

References
----------
>>> from qdk_chemistry.plugins.pyscf.active_space import PyscfAVAS
>>> avas_selector = PyscfAVAS()
>>> avas_selector.settings().set("ao_labels", ["Fe 3d", "Fe 4d"])
>>> active_orbitals = avas_selector.run(molecular_orbitals)

"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from __future__ import annotations

import re

import numpy as np
from pyscf.mcscf import avas

from qdk_chemistry.algorithms import ActiveSpaceSelector
from qdk_chemistry.data import Configuration, Orbitals, Settings, SlaterDeterminantContainer, Wavefunction
from qdk_chemistry.plugins.pyscf.conversion import orbitals_to_scf
from qdk_chemistry.utils import Logger

__all__ = ["PyscfAVAS", "PyscfAVASSettings"]


class PyscfAVASSettings(Settings):
    """Settings for the PySCF AVAS Active Space Selector.

    This class manages the configuration parameters for the PySCF AVAS, providing a convenient
    interface for setting and getting options.

    Attributes:
        ao_labels (list[str]): The atomic orbital labels to be included in the active space (default = None).
        canonicalize (bool): Whether to canonicalize the active space orbitals after selection (default = False).
        openshell_option (int): How to handle singly-occupied orbitals in the active space (default = 2).
            The singly-occupied orbitals are projected as part of alpha orbitals if ``openshell_option=2``, or
            completely kept in active space if ``openshell_option=3``.

    Examples:
        >>> settings = PyscfAVASSettings()
        >>> settings.get("ao_labels")
        []
        >>> settings.set("ao_labels", ["1s", "2s", "2p"])
        >>> settings.get("ao_labels")
        ['1s', '2s', '2p']

    """

    def __init__(self):
        """Initialize the settings with default values."""
        Logger.trace_entering()
        super().__init__()
        self._set_default("ao_labels", "vector<string>", [])
        self._set_default("canonicalize", "bool", False)
        self._set_default("openshell_option", "int", 2)


class PyscfAVAS(ActiveSpaceSelector):
    """PySCF-based Active Space Selector for quantum chemistry calculations.

    This class exposes AVAS active space selection using the PySCF library.

    The details of the AVAS method can be found in the following publication:

        Sayfutyarova et al. (2017) doi:10.1021/acs.jctc.7b00128. :cite:`Sayfutyarova2017`

    Example:
        >>> avas = PyscfAVAS()
        >>> avas.settings().set("ao_labels", ["Fe 3d", "Fe 4d"])
        >>> active_orbitals = avas.run(wavefunction)

    Notes:
        The selection criteria can be customized through the settings object.

    """

    def __init__(self):
        """Initialize the PySCF AVAS with default settings."""
        Logger.trace_entering()
        super().__init__()
        self._settings = PyscfAVASSettings()

    def _run_impl(self, wavefunction) -> Orbitals:
        """Select the active space from the provided wavefunction.

        Args:
            wavefunction: The wavefunction object containing orbital information.

        Returns:
            Orbitals with the active space identified and populated. AVAS may
            rotate/canonicalize molecular orbitals and recompute occupations,
            so the returned coefficients/occupations can differ from the
            input. The input orbitals are not modified.

        """
        Logger.trace_entering()
        # Convert QDK/Chemistry -> PySCF SCF object
        orbitals = wavefunction.get_orbitals()
        alpha_occs, beta_occs = wavefunction.get_total_orbital_occupations()
        open_shell = np.any(alpha_occs != beta_occs)
        if orbitals.is_restricted():
            mf = orbitals_to_scf(orbitals, alpha_occs, beta_occs)
            mol = mf.mol
        else:
            raise ValueError("PySCF-QDK/Chemistry AVAS Plugin only supports restricted orbitals.")

        ao_labels = self._settings.get("ao_labels")
        canonicalize = self._settings.get("canonicalize")
        openshell_option = self._settings.get("openshell_option")

        if len(ao_labels) == 0:
            raise ValueError("No atomic orbital labels provided for AVAS selection.")

        # Sanitize the AO labels
        _atom_symbols = [s.split()[1].strip() for s in mol.ao_labels()]
        atom_symbols = set(_atom_symbols)
        atom_symbols_no_idx = [str(re.sub(r"\d+", "", s)) for s in atom_symbols]
        ao_labels_clean = []
        for label in ao_labels:
            atom_symb, orb_type = label.split()

            if atom_symb in atom_symbols:
                # If there is an exact match, don't override
                ao_labels_clean.append(label)
            elif "*" in atom_symb:
                # If the atom symbol is a wildcard, keep it
                ao_labels_clean.append(label)
            elif atom_symb in atom_symbols_no_idx:
                # If the atom symbol refers to an atom that is indexed but not unique, append a wildcard
                count = atom_symbols_no_idx.count(atom_symb)
                if count > 1:
                    ao_labels_clean.append(atom_symb + "* " + orb_type)
                else:
                    ao_labels_clean.append(label)
            else:
                raise ValueError(
                    f"Atom symbol '{atom_symb}' in ao_label '{label}' not found in molecule. "
                    f"Available atom symbols: {atom_symbols_no_idx} "
                    f"Or with indices: {atom_symbols}"
                )

        avas_obj = avas.AVAS(mf, ao_labels_clean, canonicalize=canonicalize, openshell_option=openshell_option)
        norb_act, _, mo_coeff = avas_obj.kernel()

        # Extract active indices
        inactive_range = int(mol.nelectron / 2) - norb_act
        active_indices = [inactive_range + i for i in range(norb_act)]
        inactive_indices = range(inactive_range)

        if open_shell:
            # Create active orbitals
            active_orbitals = Orbitals(
                mo_coeff,
                mo_coeff,  # AVAS returns same coefficients for alpha/beta
                None,
                None,
                orbitals.get_overlap_matrix() if orbitals.has_overlap_matrix() else None,
                orbitals.get_basis_set(),
                [active_indices, active_indices, inactive_indices, inactive_indices],
            )
        else:
            # Create active orbitals
            active_orbitals = Orbitals(
                mo_coeff,
                None,
                orbitals.get_overlap_matrix() if orbitals.has_overlap_matrix() else None,
                orbitals.get_basis_set(),
                [active_indices, inactive_indices],
            )
        if len(wavefunction.get_active_determinants()) == 1:
            # Single determinant case - return new wavefunction with localized orbitals
            old_config = wavefunction.get_active_determinants()[0]

            # Get old and new active space indices
            old_orbitals = wavefunction.get_orbitals()
            old_active_indices = old_orbitals.get_active_space_indices()[0]
            new_active_indices = active_orbitals.get_active_space_indices()[0]

            # Map from old active space to new active space
            # The old determinant is already shortened to the old active space
            old_config_str = old_config.to_string()
            new_config_chars = []

            for new_idx in new_active_indices:
                # Find position of this orbital in the old active space
                try:
                    old_pos = old_active_indices.index(new_idx)
                    # Get the occupation from the old determinant
                    new_config_chars.append(old_config_str[old_pos])
                except ValueError:
                    # This orbital wasn't in the old active space, so it's unoccupied
                    new_config_chars.append("0")

            active_config = Configuration("".join(new_config_chars))
            return Wavefunction(SlaterDeterminantContainer(active_config, active_orbitals))
        raise NotImplementedError(
            "PySCF AVAS active space selector currently only supports single-determinant wavefunctions."
        )

    def name(self) -> str:
        """Return the name of the active space selector."""
        Logger.trace_entering()
        return "pyscf_avas"
