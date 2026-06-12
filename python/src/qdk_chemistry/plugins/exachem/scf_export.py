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

qdk-chemistry and ExaChem differ in how p-orbitals (l=1) are ordered.
qdk-chemistry sets ``pure=false`` for l<2 in Libint2, giving **Cartesian**
ordering ``(px, py, pz)`` for p-shells. ExaChem sets ``pure=true`` for all
shells, giving **spherical harmonic** ordering ``(py, pz, px)`` for p-shells.

For l >= 2 (d, f, g, ...), both codes use spherical harmonic ordering
``m = -l, ..., 0, ..., +l`` with identical conventions. No reordering is
needed for those shells.

The :func:`reorder_ao_qdk_to_exachem` function generates the permutation
that maps p-shell AO rows from qdk's Cartesian order to ExaChem's spherical
order.
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


def reorder_ao_qdk_to_exachem(basis_set) -> np.ndarray:
    """Compute the AO index permutation from qdk-chemistry to ExaChem ordering.

    Only p-shells (l=1) differ: qdk uses Cartesian ``(px, py, pz)`` while
    ExaChem uses spherical ``(py, pz, px)``. For all other angular momenta
    the ordering is identical.

    The returned permutation ``perm`` satisfies ``C_exachem = C_qdk[perm, :]``.

    Args:
        basis_set: A :class:`~qdk_chemistry.data.BasisSet` instance.

    Returns:
        Permutation array of shape ``(nao,)`` mapping qdk AO indices to
        ExaChem AO indices.

    """
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

        # Only reorder p-shells (3 functions)
        if len(ao_indices) == 3:
            # qdk Cartesian: [px, py, pz] at positions [a, a+1, a+2]
            # ExaChem spherical: [py, pz, px]
            # EC[a]   ← qdk[a+1] (py)
            # EC[a+1] ← qdk[a+2] (pz)
            # EC[a+2] ← qdk[a]   (px)
            a = ao_indices[0]
            perm[a] = a + 1  # EC pos a   gets qdk py
            perm[a + 1] = a + 2  # EC pos a+1 gets qdk pz
            perm[a + 2] = a  # EC pos a+2 gets qdk px

    return perm


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
        perm = reorder_ao_qdk_to_exachem(basis_set)
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
