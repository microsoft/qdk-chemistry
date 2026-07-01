"""Utility functions for orbital visualization."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from __future__ import annotations

from qdk_chemistry.data import Orbitals, Wavefunction
from qdk_chemistry.utils.cubegen import generate_cubefiles_from_orbitals


def orbital_label(idx: int, n_occupied: int) -> str:
    """Return a human-readable label (e.g., HOMO-1, LUMO+2) for a molecular orbital index.

    Args:
        idx: Zero-based molecular orbital index.
        n_occupied: Number of occupied orbitals (HOMO index is ``n_occupied - 1``).

    Returns:
        Label string such as "HOMO", "HOMO-2", "LUMO", or "LUMO+1".

    """
    offset = idx - (n_occupied - 1)
    if offset == 0:
        return "HOMO"
    elif offset < 0:
        return f"HOMO{offset}"
    elif offset == 1:
        return "LUMO"
    else:
        return f"LUMO+{offset - 1}"


def generate_cube_data_with_info(
    orbitals: Orbitals,
    n_occupied: int,
    indices: list[int] | None = None,
    grid_size: tuple[int, int, int] = (40, 40, 40),
    margin: float = 10.0,
) -> dict[str, dict]:
    """Generate cube data with info overlays for the MoleculeViewer widget.

    Each entry in the returned dictionary is keyed by a human-readable orbital label
    (e.g., "HOMO-1") and contains ``"data"`` (cube file string) and ``"info"``
    (metadata dict with energy and occupation).

    Args:
        orbitals: Orbitals object containing MO coefficients and energies.
        n_occupied: Number of occupied orbitals.
        indices: Orbital indices to include. If None, all orbitals are included.
        grid_size: Grid dimensions for cube file generation.
        margin: Margin in Bohr around the molecule for the cube grid.

    Returns:
        Dictionary suitable for passing as ``cube_data`` to ``MoleculeViewer``.

    """
    if orbitals.is_unrestricted():
        raise ValueError(
            "generate_cube_data_with_info only supports restricted orbitals. "
            "Unrestricted orbitals have separate alpha/beta channels that require different handling."
        )

    energies = orbitals.get_energies_alpha()

    cube_data_raw = generate_cubefiles_from_orbitals(
        orbitals=orbitals,
        grid_size=grid_size,
        margin=margin,
        indices=indices,
    )

    cube_data_with_info: dict[str, dict] = {}
    for raw_label, cube_str in cube_data_raw.items():
        idx = int(raw_label.split("_")[1]) - 1
        cube_data_with_info[orbital_label(idx, n_occupied)] = {
            "data": cube_str,
            "info": {
                "Energy (Ha)": f"{energies[idx]:.4f}",
                "Occupation": "occupied" if idx < n_occupied else "virtual",
            },
        }

    return cube_data_with_info


def generate_cube_data_with_correlation_info(
    wavefunction: Wavefunction,
    indices: list[int] | None = None,
    grid_size: tuple[int, int, int] = (40, 40, 40),
    margin: float = 10.0,
) -> dict[str, dict]:
    """Generate cube data with natural-occupation and entropy overlays for the MoleculeViewer widget.

    Builds the ``cube_data`` dictionary for the orbitals of a ``Wavefunction``,
    attaching each orbital's natural occupation number and single-orbital entropy as metadata.

    The wavefunction must carry one- and two-particle RDMs
    so that occupations and entropies are available.

    Args:
        wavefunction: wavefunction with 1- and 2-RDMs and an active space.
        indices: Orbital indices to include. If None, all active-space orbitals are included.
        grid_size: Grid dimensions for cube file generation.
        margin: Margin in Bohr around the molecule for the cube grid.

    Returns:
        Dictionary suitable for passing as ``cube_data`` to ``MoleculeViewer``.

    """
    orbitals = wavefunction.get_orbitals()
    if orbitals.is_unrestricted():
        raise ValueError(
            "generate_cube_data_with_correlation_info only supports restricted orbitals. "
            "Unrestricted orbitals have separate alpha/beta channels that require different handling."
        )

    try:
        entropies = wavefunction.get_single_orbital_entropies()
    except RuntimeError as exc:
        raise ValueError(
            "Wavefunction does not have single-orbital entropies. "
            "Ensure the calculator was run with calculate_one_rdm=True and calculate_two_rdm=True."
        ) from exc

    if not orbitals.has_active_space():
        raise ValueError(
            "No active space is defined on the wavefunction's orbitals. "
            "Set an active space on the Orbitals before requesting occupation/entropy overlays."
        )

    if indices is None:
        indices, _ = orbitals.get_active_space_indices()
        indices = list(indices)

    # Occupations and entropies are indexed by active-space position.
    # Build a lookup from MO index to position for direct access.
    active_indices, _ = orbitals.get_active_space_indices()
    invalid = [i for i in indices if i not in active_indices]
    if invalid:
        raise ValueError(
            f"indices {invalid} are not in the active space {list(active_indices)}. "
            "Occupations and entropies are only available for active-space orbitals."
        )
    try:
        occ_alpha, occ_beta = wavefunction.get_active_orbital_occupations()
    except RuntimeError as exc:
        raise ValueError(
            "Wavefunction does not have active-orbital occupations available for overlay."
        ) from exc
    mo_to_pos = {mo_idx: pos for pos, mo_idx in enumerate(active_indices)}

    cube_data_raw = generate_cubefiles_from_orbitals(
        orbitals=orbitals,
        grid_size=grid_size,
        margin=margin,
        indices=indices,
    )

    cube_data_with_info: dict[str, dict] = {}
    for raw_label, cube_str in cube_data_raw.items():
        mo_idx = int(raw_label.split("_")[1]) - 1
        pos = mo_to_pos[mo_idx]
        cube_data_with_info[f"MO {mo_idx + 1}"] = {
            "data": cube_str,
            "info": {
                "Occupation": f"{float(occ_alpha[pos]) + float(occ_beta[pos]):.3f}",
                "Entropy": f"{float(entropies[pos]):.3f}",
            },
        }

    return cube_data_with_info
