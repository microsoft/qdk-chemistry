# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

"""Export qdk-chemistry SCF data to ExaChem's serial HDF5 format.

Writes MO coefficients and density matrices as HDF5 files compatible with
ExaChem's ``USE_SERIAL_IO`` restart path, enabling DUCC calculations that
skip ExaChem's internal SCF.

The HDF5 format matches ExaChem's ``write_scf_mat`` / ``read_scf_mat`` in
``scf_outputs.cpp``: a flat 1-D dataset of doubles (row-major) named after
the file extension, plus an ``rdims`` dataset storing ``[nrows, ncols]``.

AO basis function reordering
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

qdk-chemistry and ExaChem/Libint2 differ in AO ordering in two ways:

1. **Inter-shell ordering** (Pople bases with SP shells): qdk groups all AOs
   by angular momentum within each atom (all s first, then all p, then d, ...),
   while ExaChem/Libint2 keeps shells in ``.g94`` file order where SP-shell
   splits produce adjacent (s, p) groups.

2. **Within-p-shell ordering**: qdk uses m = (-1, 0, +1) ordering while
   ExaChem uses (m=0, m=+1, m=-1) ordering for p-shell components.

The :func:`reorder_ao_qdk_to_exachem` function generates the full permutation
combining both corrections.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

try:
    import h5py

    _HAS_H5PY = True
except ImportError:
    _HAS_H5PY = False


def _require_h5py() -> None:
    if not _HAS_H5PY:
        raise ImportError("h5py is required for ExaChem SCF export. Install with: pip install h5py")


def _parse_g94_atom_shells(g94_path: str | Path, element_symbol: str) -> list[str]:
    """Parse a .g94 file and return the shell types for a given element.

    Returns a list like ``["S", "SP", "SP"]`` for Li in 6-31G, reflecting
    the order shells appear in the file (which is ExaChem/Libint2's ordering).
    """
    g94_path = Path(g94_path)
    element_symbol = element_symbol.strip().title()
    shells: list[str] = []
    in_element = False

    with open(g94_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("!"):
                continue
            if line == "****":
                if in_element:
                    break
                continue
            parts = line.split()
            # Element header line: "Li     0"
            if len(parts) == 2 and parts[1] == "0" and parts[0].isalpha():
                if parts[0] == element_symbol:
                    in_element = True
                elif in_element:
                    break
                continue
            if in_element:
                # Shell line: "S    6   1.00" or "SP   3   1.00"
                if parts[0] in ("S", "P", "SP", "D", "F", "G", "H", "I"):
                    shells.append(parts[0])
    return shells


def _find_libint2_basis_file(basis_name: str, exachem_binary: str | Path | None = None) -> Path | None:
    """Locate the .g94 basis file from Libint2's share directory.

    Searches relative to the ExaChem binary: ``<prefix>/share/libint/*/basis/<name>.g94``.
    """
    if exachem_binary is None:
        return None
    binary = Path(exachem_binary).resolve()
    # ExaChem binary is typically at <prefix>/bin/ExaChem
    prefix = binary.parent.parent
    share_libint = prefix / "share" / "libint"
    if not share_libint.exists():
        # Try the install prefix used in this environment
        for candidate in [Path("/workspaces/exachem_install/share/libint"),
                          Path("/workspaces/exachem_install_serial_io/../exachem_install/share/libint")]:
            if candidate.exists():
                share_libint = candidate
                break
    if not share_libint.exists():
        return None
    # Find the version directory
    for version_dir in share_libint.iterdir():
        basis_file = version_dir / "basis" / f"{basis_name}.g94"
        if basis_file.exists():
            return basis_file
    return None


def _num_ao_for_shell_type(shell_type: str) -> int:
    """Number of AOs produced by a shell type (spherical harmonics)."""
    return {"S": 1, "P": 3, "SP": 4, "D": 5, "F": 7, "G": 9, "H": 11, "I": 13}[shell_type]


def reorder_ao_qdk_to_exachem(basis_set, basis_name: str | None = None,
                              elements: list | None = None,
                              exachem_binary: str | Path | None = None) -> np.ndarray:
    """Compute the AO index permutation from qdk-chemistry to ExaChem ordering.

    Handles two ordering differences:

    1. **Inter-shell** (Pople SP-shell bases): qdk groups AOs by angular
       momentum within each atom ``[all s, all p, all d, ...]``, while
       ExaChem/Libint2 keeps SP-split shells adjacent in .g94 file order.

    2. **Within-p-shell**: qdk uses ``(m=-1, m=0, m=+1)`` component ordering
       while ExaChem uses ``(m=0, m=+1, m=-1)`` for p-shells.

    The returned permutation ``perm`` satisfies ``C_exachem = C_qdk[perm, :]``.

    Args:
        basis_set: A :class:`~qdk_chemistry.data.BasisSet` instance.
        basis_name: Basis set name (e.g. ``"6-31g"``). Required for inter-shell reordering.
        elements: List of element symbols in molecule order (e.g. ``["Li", "H"]``).
        exachem_binary: Path to ExaChem binary for locating .g94 files.

    Returns:
        Permutation array of shape ``(nao,)`` mapping qdk AO indices to
        ExaChem AO indices.

    """
    nao = basis_set.get_num_atomic_orbitals()
    perm = np.arange(nao)

    # ── Step 1: Inter-shell reordering (if we can find the .g94 file) ──
    g94_file = None
    if basis_name and exachem_binary:
        g94_file = _find_libint2_basis_file(basis_name, exachem_binary)

    if g94_file and elements:
        perm = _compute_full_permutation(basis_set, g94_file, elements)
    else:
        # Fallback: only within-p-shell reorder (original behavior)
        perm = _within_p_shell_reorder(basis_set)

    return perm


def _within_p_shell_reorder(basis_set) -> np.ndarray:
    """Original within-p-shell reorder only (no inter-shell fix)."""
    nao = basis_set.get_num_atomic_orbitals()
    perm = np.arange(nao)
    i = 0
    for s in range(basis_set.get_num_shells()):
        ao_indices = []
        while i < nao:
            info = basis_set.get_atomic_orbital_info(i)
            if info[0] == s:
                ao_indices.append(i)
                i += 1
            else:
                break
        if len(ao_indices) == 3:
            a = ao_indices[0]
            perm[a] = a + 1
            perm[a + 1] = a + 2
            perm[a + 2] = a
    return perm


def _compute_full_permutation(basis_set, g94_file: Path, elements: list) -> np.ndarray:
    """Compute full permutation including inter-shell and within-p-shell reorder.

    Algorithm:
    - For each atom, read ExaChem's .g94 shell order (e.g. [S, SP, SP])
    - ExaChem splits SP → adjacent (S, P): [S, S, P, S, P]
    - qdk sorts by angular momentum: [S, S, S, P, P]
    - Build mapping from qdk's position → ExaChem's position
    - Apply within-p-shell component reorder on top
    """
    nao = basis_set.get_num_atomic_orbitals()

    # Identify atom boundaries and shell structure from qdk's BasisSet
    qdk_shells = []  # list of (shell_idx, [ao_indices], l)
    i = 0
    for s in range(basis_set.get_num_shells()):
        ao_indices = []
        while i < nao:
            info = basis_set.get_atomic_orbital_info(i)
            if info[0] == s:
                ao_indices.append(i)
                i += 1
            else:
                break
        l = {1: 0, 3: 1, 5: 2, 7: 3, 9: 4, 11: 5, 13: 6}.get(len(ao_indices), -1)
        qdk_shells.append((s, ao_indices, l))

    # Determine atom boundaries: shells for atom k are contiguous in qdk,
    # and elements tells us how many atoms and their types.
    # Use the .g94 data to figure out how many shells each atom contributes.
    atom_shell_ranges = _assign_shells_to_atoms(qdk_shells, elements, g94_file)

    # Build the full permutation
    perm = np.arange(nao)

    for atom_idx, (shell_start, shell_end) in enumerate(atom_shell_ranges):
        elem = elements[atom_idx]
        g94_shells = _parse_g94_atom_shells(g94_file, elem)

        # Build ExaChem's AO layout for this atom (relative positions)
        ec_layout = []  # list of (l, count) in ExaChem order
        for sh_type in g94_shells:
            if sh_type == "S":
                ec_layout.append((0, 1))
            elif sh_type == "SP":
                ec_layout.append((0, 1))
                ec_layout.append((1, 3))
            elif sh_type == "P":
                ec_layout.append((1, 3))
            elif sh_type == "D":
                ec_layout.append((2, 5))
            elif sh_type == "F":
                ec_layout.append((3, 7))
            else:
                ang = {"G": 4, "H": 5, "I": 6}.get(sh_type, -1)
                ec_layout.append((ang, 2 * ang + 1))

        # qdk's AO layout for this atom: sorted by l
        qdk_atom_shells = qdk_shells[shell_start:shell_end]
        qdk_layout = []  # list of (l, count, global_ao_start)
        for _, ao_indices, l in qdk_atom_shells:
            qdk_layout.append((l, len(ao_indices), ao_indices[0]))

        # Both layouts should have the same total AO count
        ec_total = sum(c for _, c in ec_layout)
        qdk_total = sum(c for _, c, _ in qdk_layout)
        if ec_total != qdk_total:
            logger.warning(
                "AO count mismatch for %s: ec=%d, qdk=%d. Skipping inter-shell reorder.",
                elem, ec_total, qdk_total,
            )
            continue

        # Map: for each shell in ExaChem order, find the corresponding shell in qdk order
        # ExaChem processes shells by l in file order; qdk has them sorted by l.
        # Within the same l, both maintain the same relative order.
        ec_s_idx = 0  # counter for s-shells consumed in EC order
        ec_p_idx = 0  # counter for p-shells consumed
        ec_d_idx = 0  # etc.
        l_counters = {}

        # qdk groups: all shells of l=0, then l=1, then l=2...
        qdk_by_l: dict[int, list[int]] = {}  # l → list of qdk_layout indices
        for idx, (l, count, ao_start) in enumerate(qdk_layout):
            qdk_by_l.setdefault(l, []).append(idx)

        # Track which qdk shell (within each l-group) we've consumed
        l_consumed = {l: 0 for l in qdk_by_l}

        # Build the AO-level mapping
        ec_ao_offset = qdk_layout[0][2]  # global start of this atom's AOs in qdk
        # Actually we need the atom's global AO start:
        atom_ao_start = qdk_atom_shells[0][1][0]

        ec_pos = 0  # position within ExaChem's atom block
        mapping = []  # (ec_relative_pos, qdk_global_ao_idx, count)

        for l_ec, count_ec in ec_layout:
            # This EC shell corresponds to the next unconsumed qdk shell of the same l
            qdk_idx_in_group = l_consumed[l_ec]
            qdk_shell_list_idx = qdk_by_l[l_ec][qdk_idx_in_group]
            _, _, qdk_global_start = qdk_layout[qdk_shell_list_idx]
            l_consumed[l_ec] += 1

            for k in range(count_ec):
                perm[atom_ao_start + ec_pos + k] = qdk_global_start + k
            ec_pos += count_ec

    # ── Step 2: Within-p-shell component reorder ──
    # After inter-shell reorder, p-shells need component reordering.
    # perm currently maps: EC_position → qdk_index (for inter-shell)
    # We need to also apply the within-p-shell component swap.
    # qdk p-components: (m=-1, m=0, m=+1) at consecutive positions
    # ExaChem p-components: (m=0, m=+1, m=-1)
    # So within each p-shell block of 3 in the perm, we swap components:
    #   EC[a] ← qdk[a+1] (m=0), EC[a+1] ← qdk[a+2] (m=+1), EC[a+2] ← qdk[a] (m=-1)
    for atom_idx, (shell_start, shell_end) in enumerate(atom_shell_ranges):
        elem = elements[atom_idx]
        g94_shells = _parse_g94_atom_shells(g94_file, elem)
        atom_ao_start = qdk_shells[shell_start][1][0]

        ec_pos = 0
        for sh_type in g94_shells:
            if sh_type == "S":
                ec_pos += 1
            elif sh_type == "SP":
                ec_pos += 1  # s-part
                # p-part: 3 components need reorder
                a = atom_ao_start + ec_pos
                p0, p1, p2 = perm[a], perm[a + 1], perm[a + 2]
                perm[a] = p0 + 1      # EC gets qdk m=0 (offset +1 from base)
                perm[a + 1] = p0 + 2  # EC gets qdk m=+1 (offset +2)
                perm[a + 2] = p0      # EC gets qdk m=-1 (offset +0)
                ec_pos += 3
            elif sh_type == "P":
                a = atom_ao_start + ec_pos
                p0 = perm[a]
                perm[a] = p0 + 1
                perm[a + 1] = p0 + 2
                perm[a + 2] = p0
                ec_pos += 3
            elif sh_type == "D":
                ec_pos += 5
            elif sh_type == "F":
                ec_pos += 7
            else:
                ang = {"G": 4, "H": 5, "I": 6}.get(sh_type, 0)
                ec_pos += 2 * ang + 1

    return perm


def _assign_shells_to_atoms(qdk_shells, elements, g94_file):
    """Determine which qdk shells belong to each atom.

    Uses the .g94 file to count how many shells each atom should have
    (after splitting SP → S + P), then assigns qdk shells sequentially.
    """
    atom_shell_ranges = []
    shell_cursor = 0
    for elem in elements:
        g94_types = _parse_g94_atom_shells(g94_file, elem)
        # Count shells after SP splitting
        n_shells = 0
        for sh_type in g94_types:
            if sh_type == "SP":
                n_shells += 2  # splits into S + P
            else:
                n_shells += 1
        atom_shell_ranges.append((shell_cursor, shell_cursor + n_shells))
        shell_cursor += n_shells
    return atom_shell_ranges


def write_exachem_matrix(filepath: str | Path, matrix: np.ndarray) -> None:
    """Write a matrix in ExaChem's serial-IO HDF5 format.

    Args:
        filepath: Output path. The dataset name is derived from the file extension (e.g. ``"movecs"`` for ``.alpha.movecs``).
        matrix: 2-D numpy array to write. Stored as a flat row-major 1-D dataset.

    Raises:
        ImportError: If h5py is not installed.

    """
    _require_h5py()
    filepath = Path(filepath)
    # Dataset name = last extension without the dot (e.g. ".alpha.movecs" → "movecs")
    dataset_name = filepath.suffix.lstrip(".")
    if not dataset_name:
        dataset_name = "data"

    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError(f"Expected 2-D array, got shape {matrix.shape}")

    # ExaChem uses row-major (Eigen::RowMajor). Ensure C-contiguous.
    matrix_c = np.ascontiguousarray(matrix)

    with h5py.File(str(filepath), "w") as f:
        f.create_dataset(dataset_name, data=matrix_c.ravel())
        f.create_dataset("rdims", data=np.array(matrix_c.shape, dtype=np.int64))

    logger.debug("Wrote %s: shape=%s, dataset=%r", filepath, matrix.shape, dataset_name)


def export_scf_files(
    files_prefix: str | Path,
    mo_coeff_alpha: np.ndarray,
    density_alpha: np.ndarray,
    ao_tilesize: int = 30,
    runcontext_prefix: str | Path | None = None,
    basis_set=None,
    basis_name: str | None = None,
    elements: list | None = None,
    exachem_binary: str | Path | None = None,
) -> list[Path]:
    """Write SCF restart files for ExaChem's ``noscf`` mode.

    Creates the minimum files required by ExaChem's SCF restart path:
    MO coefficients and density matrices in serial-IO HDF5 format, plus
    a run context JSON with the AO tile size.

    When ``basis_set`` is provided, the AO rows of the MO coefficient and
    density matrices are reordered from qdk-chemistry's convention to
    ExaChem's convention (see :func:`reorder_ao_qdk_to_exachem`).

    Args:
        files_prefix: Path prefix for output files (e.g. ``"workdir/scf/h2o.cc-pvdz"``). Files will be named ``<prefix>.alpha.movecs``, etc.
        mo_coeff_alpha: Alpha MO coefficient matrix, shape ``(nbf, nmo)``.
        density_alpha: Alpha density matrix in AO basis, shape ``(nbf, nbf)``.
        ao_tilesize: AO tile size for ExaChem (default: 30).
        runcontext_prefix: Path prefix for the run context JSON. If None, uses ``files_prefix``.
        basis_set: A :class:`~qdk_chemistry.data.BasisSet` instance for AO reordering. If None, no reordering is applied.
        basis_name: Basis set name for inter-shell reordering (e.g. ``"6-31g"``).
        elements: List of element symbols for inter-shell reordering.
        exachem_binary: Path to ExaChem binary for locating .g94 files.

    Returns:
        List of paths to the created files.

    Raises:
        ImportError: If h5py is not installed.

    """
    _require_h5py()
    files_prefix = Path(files_prefix)
    files_prefix.parent.mkdir(parents=True, exist_ok=True)

    # Apply AO reordering if basis_set provided
    if basis_set is not None:
        perm = reorder_ao_qdk_to_exachem(
            basis_set, basis_name=basis_name, elements=elements, exachem_binary=exachem_binary
        )
        mo_coeff_alpha = mo_coeff_alpha[perm, :]
        density_alpha = density_alpha[np.ix_(perm, perm)]

    created: list[Path] = []

    # Alpha movecs and density (always required)
    movecs_path = Path(str(files_prefix) + ".alpha.movecs")
    write_exachem_matrix(movecs_path, mo_coeff_alpha)
    created.append(movecs_path)

    density_path = Path(str(files_prefix) + ".alpha.density")
    write_exachem_matrix(density_path, density_alpha)
    created.append(density_path)

    # Run context JSON (required by ExaChem's scf_restart to read ao_tilesize)
    rc_prefix = Path(runcontext_prefix) if runcontext_prefix is not None else files_prefix
    rc_prefix.parent.mkdir(parents=True, exist_ok=True)
    runcontext_path = Path(str(rc_prefix) + ".runcontext.json")
    import json as _json

    run_context = {"ao_tilesize": ao_tilesize}
    runcontext_path.write_text(_json.dumps(run_context, indent=2))
    created.append(runcontext_path)

    logger.info("Exported %d SCF restart files with prefix %s", len(created), files_prefix)
    return created
