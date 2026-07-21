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

qdk-chemistry and ExaChem/Libint2 can differ in AO ordering in two ways: the
*inter-shell* order (how the shells of an atom are laid out, which for general
contractions such as cc-pVTZ differs between qdk-chemistry's grouped order and
Libint2's ``.g94`` file order) and the *within-shell* order (how the ``2l+1``
spherical components of a shell are ordered).

The preferred, numerically-consistent approach (see :func:`write_qdk_basis_g94`)
is to feed ExaChem *qdk-chemistry's own* basis: qdk-chemistry's shells are
written to a Gaussian-94 file in qdk-chemistry's native shell order and ExaChem
is pointed at it via the ``LIBINT_DATA_PATH`` environment variable.  ExaChem then
builds its AO basis with the identical inter-shell order and identical basis
parameters, so the *only* remaining difference is the within-shell m-component
ordering.

That within-shell ordering is **not** encoded in the basis file -- it is a
property of each code's Libint build (qdk-chemistry links Libint 2.9.0, ExaChem
links Libint 2.11.2), so feeding the same basis cannot fix it.  Empirically the
two Libint versions agree on ascending-m ordering (``-l, ..., 0, ..., +l``) for
every angular momentum **except p (l=1)**: qdk-chemistry orders p as
``(m=-1, 0, +1)`` while ExaChem consumes it as ``(m=0, +1, -1)``.
:func:`_within_shell_m_reorder` applies exactly that p-only swap (d/f/g were
verified to match; permuting them corrupts the imported reference).
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


_L_SYMBOL = {0: "S", 1: "P", 2: "D", 3: "F", 4: "G", 5: "H", 6: "I"}


def _within_shell_m_reorder(basis_set) -> np.ndarray:
    """AO permutation correcting the within-shell m-component ordering.

    This assumes ExaChem reads qdk-chemistry's own basis (identical inter-shell
    order, see :func:`write_qdk_basis_g94`), so the only remaining difference is
    the ordering of the spherical components *inside* a shell.

    Empirically, qdk-chemistry (Libint 2.9.0) and ExaChem (Libint 2.11.2) agree
    on the ascending-m ordering (``-l, ..., 0, ..., +l``) for every angular
    momentum **except p (l=1)**: qdk-chemistry orders p as ``(m=-1, 0, +1)``
    while ExaChem consumes it as ``(m=0, +1, -1)``.  (d/f/g were verified to
    match; applying a permutation to them corrupts the imported reference.)
    Only the p components are therefore reordered.  The permutation is built
    from qdk-chemistry's AO metadata (``get_atomic_orbital_info`` returns
    ``(shell_index, m)``) and satisfies ``C_exachem = C_qdk[perm, :]``.
    """
    nao = basis_set.get_num_atomic_orbitals()
    perm = np.arange(nao)
    # Group AO indices by shell, recording the m quantum number (qdk order).
    shell_map: dict[int, list[tuple[int, int]]] = {}
    for i in range(nao):
        shell, m = basis_set.get_atomic_orbital_info(i)
        shell_map.setdefault(int(shell), []).append((int(m), i))
    for aos in shell_map.values():
        if len(aos) != 3:
            # Only p shells (l=1, three components) need reordering; d/f/g match.
            continue
        # The p-orbital swap is needed ONLY because qdk-chemistry and ExaChem link
        # DIFFERENT Libint versions (qdk 2.9.0 vs ExaChem 2.11.2) whose real
        # solid-harmonic component ordering differs for l=1: qdk emits
        # (m=-1, 0, +1) while ExaChem consumes (m=0, +1, -1).  This ordering is a
        # property of the Libint build, not of the .g94 basis file, so feeding the
        # same basis cannot fix it.
        # TODO: build qdk-chemistry and ExaChem against the SAME Libint version so
        # the p ordering matches and this swap can be removed entirely.
        idxs = [i for _m, i in sorted(aos)]  # ascending m: (m=-1, 0, +1)
        start = idxs[0]
        # ExaChem p order (m=0, +1, -1) -> qdk indices (1, 2, 0)
        perm[start + 0] = idxs[1]
        perm[start + 1] = idxs[2]
        perm[start + 2] = idxs[0]
    return perm


def write_qdk_basis_g94(
    basis_set, elements: list, basis_data_dir: str | Path, basis_name: str
) -> Path:
    """Write qdk-chemistry's basis as a Gaussian-94 file in qdk's shell order.

    ExaChem/Libint2 reads a basis from ``<LIBINT_DATA_PATH>/basis/<name>.g94``.
    :meth:`BasisSet.get_shell` returns each shell's exponents and contraction
    coefficients in the raw g94 convention, so writing qdk-chemistry's shells in
    qdk-chemistry's native order makes ExaChem build its AO basis with exactly
    the same inter-shell order *and* numerically identical parameters.  Only the
    within-shell m-component order then differs (handled by
    :func:`_within_shell_m_reorder`).

    Args:
        basis_set: A :class:`~qdk_chemistry.data.BasisSet` instance.
        elements: Element symbols in molecule (atom) order.
        basis_data_dir: Directory to write into; the file is placed at
            ``<basis_data_dir>/basis/<basis_name>.g94``.  Point
            ``LIBINT_DATA_PATH`` at ``basis_data_dir`` when running ExaChem.
        basis_name: Basis set name (used for the file name).

    Returns:
        The ``basis_data_dir`` path, to be used as ``LIBINT_DATA_PATH``.
    """
    # Group shell indices by atom (shells are already in qdk's native order).
    atom_shells: dict[int, list[int]] = {}
    for s in range(basis_set.get_num_shells()):
        atom_shells.setdefault(int(basis_set.get_shell(s).atom_index), []).append(s)

    # One block per unique element (all atoms of an element share the basis).
    blocks: list[str] = []
    seen: set[str] = set()
    for atom_idx in sorted(atom_shells):
        elem = str(elements[atom_idx]).strip().title()
        if elem in seen:
            continue
        seen.add(elem)
        lines = [f"{elem}     0"]
        for s in atom_shells[atom_idx]:
            sh = basis_set.get_shell(s)
            l = sh.get_angular_momentum()
            exps = list(sh.exponents)
            coeffs = list(sh.coefficients)
            lines.append(f"{_L_SYMBOL[l]}   {len(exps)}   1.00")
            for e, c in zip(exps, coeffs):
                lines.append(f"{e: .14E}   {c: .14E}")
        blocks.append("\n".join(lines))

    basis_dir = Path(basis_data_dir) / "basis"
    basis_dir.mkdir(parents=True, exist_ok=True)
    g94_path = basis_dir / f"{basis_name}.g94"
    content = (
        f"! qdk-chemistry basis '{basis_name}' (qdk-chemistry shell order)\n"
        + "****\n"
        + "\n****\n".join(blocks)
        + "\n****\n"
    )
    g94_path.write_text(content)
    logger.info("Wrote qdk-chemistry basis for ExaChem: %s", g94_path)
    return Path(basis_data_dir)


def export_scf_files(
    files_prefix: str | Path,
    mo_coeff_alpha: np.ndarray,
    density_alpha: np.ndarray,
    basis_set,
    basis_name: str,
    elements: list,
    basis_data_dir: str | Path,
    ao_tilesize: int = 30,
    runcontext_prefix: str | Path | None = None,
) -> list[Path]:
    """Write SCF restart files for ExaChem's ``noscf`` mode.

    Creates the minimum files required by ExaChem's SCF restart path:
    MO coefficients and density matrices in serial-IO HDF5 format, plus
    a run context JSON with the AO tile size.

    qdk-chemistry's own basis is written to ``basis_data_dir`` (see
    :func:`write_qdk_basis_g94`) so ExaChem reads it via ``LIBINT_DATA_PATH``,
    giving both codes an identical inter-shell order and identical basis
    parameters.  The AO rows/columns of the MO coefficient and density matrices
    are then corrected for the within-shell p-component ordering only
    (:func:`_within_shell_m_reorder`).

    Args:
        files_prefix: Path prefix for output files (e.g. ``"workdir/scf/h2o.cc-pvdz"``). Files will be named ``<prefix>.alpha.movecs``, etc.
        mo_coeff_alpha: Alpha MO coefficient matrix, shape ``(nbf, nmo)``.
        density_alpha: Alpha density matrix in AO basis, shape ``(nbf, nbf)``.
        ao_tilesize: AO tile size for ExaChem (default: 30).
        runcontext_prefix: Path prefix for the run context JSON. If None, uses ``files_prefix``.
        basis_set: A :class:`~qdk_chemistry.data.BasisSet` instance (required for the basis export and AO reorder).
        basis_name: Basis set name (e.g. ``"cc-pvdz"``); used for the written ``.g94`` file name.
        elements: Element symbols in molecule (atom) order.
        basis_data_dir: Directory to write qdk-chemistry's basis into as a
            Gaussian-94 file (at ``<basis_data_dir>/basis/<basis_name>.g94``) for
            ExaChem to read via ``LIBINT_DATA_PATH``.

    Returns:
        List of paths to the created files.

    Raises:
        ImportError: If h5py is not installed.

    """
    _require_h5py()
    files_prefix = Path(files_prefix)
    files_prefix.parent.mkdir(parents=True, exist_ok=True)

    # Feed qdk-chemistry's own basis to ExaChem (read via LIBINT_DATA_PATH) so the
    # inter-shell order and basis parameters match exactly; only the within-shell
    # p-component order then needs correcting.
    write_qdk_basis_g94(basis_set, elements, basis_data_dir, basis_name)
    perm = _within_shell_m_reorder(basis_set)
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
