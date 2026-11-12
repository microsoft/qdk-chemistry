"""Export qdk-chemistry SCF data to ExaChem's serial HDF5 format.

Writes MO coefficients and density matrices as HDF5 files compatible with
ExaChem's ``USE_SERIAL_IO`` restart path, enabling post-Hartree-Fock
calculations that skip ExaChem's internal SCF.

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

The interface feeds ExaChem *qdk-chemistry's own* basis (see
:func:`write_qdk_basis_g94`): qdk-chemistry's shells are written to a
Gaussian-94 file in qdk-chemistry's native shell order and ExaChem is pointed at
it via the ``LIBINT_DATA_PATH`` environment variable.  ExaChem then builds its AO
basis with the identical inter-shell order and identical basis parameters, so
the *only* remaining difference is the within-shell m-component ordering.
Effective core potentials are written next to the basis the same way (see
:func:`write_qdk_ecp`).

That within-shell ordering is **not** encoded in the basis file -- it reflects a
qdk-chemistry AO-layout convention: qdk-chemistry represents p orbitals in
**Cartesian** form (``px, py, pz``) while ExaChem/Libint uses **spherical** p in
ascending-m order (``m=-1, 0, +1`` = ``py, pz, px``).  Every other angular
momentum already matches (s is a single function; d/f/g are spherical in both),
so ``_within_shell_m_reorder`` applies exactly that p-only swap (permuting
d/f/g corrupts the imported reference).
"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import h5py
import numpy as np

from qdk_chemistry.utils import Logger

__all__ = [
    "basis_name_to_exachem",
    "ecp_name_to_exachem",
    "export_scf_files",
    "write_exachem_matrix",
    "write_qdk_basis_g94",
    "write_qdk_ecp",
]

if TYPE_CHECKING:
    from qdk_chemistry.data import BasisSet

_L_SYMBOL = {0: "S", 1: "P", 2: "D", 3: "F", 4: "G", 5: "H", 6: "I"}


def basis_name_to_exachem(name: str) -> str:
    """Translate a qdk-chemistry basis set name into ExaChem's spelling.

    ExaChem lowercases basis set names, replaces spaces with underscores, and prefixes names starting with
    ``aug`` with ``ec-``. The returned name is unchanged by that rewrite, so it names both ExaChem's library
    entry and the file written by :func:`write_qdk_basis_g94`.

    Args:
        name: Basis set name, as returned by :meth:`~qdk_chemistry.data.BasisSet.get_name`.

    Returns:
        The basis set name as ExaChem spells it.

    Examples:
        >>> basis_name_to_exachem("aug-cc-pVDZ")
        'ec-aug-cc-pvdz'

    """
    Logger.trace_entering()
    exachem_name = name.lower().replace(" ", "_")
    return f"ec-{exachem_name}" if exachem_name.startswith("aug") else exachem_name


def ecp_name_to_exachem(name: str) -> str:
    """Translate a qdk-chemistry ECP name into the name of ExaChem's ECP library file.

    ExaChem stores an ECP shared by several basis sets once: every def2 basis set uses ``def2-ecp``, and
    LANL2DZ and LANL2DZdp use ``lanl2dz_ecp``. Other names are spelled as in :func:`basis_name_to_exachem`.
    The result names both ExaChem's library file and the file written by :func:`write_qdk_ecp`.

    Args:
        name: ECP name, as returned by :meth:`~qdk_chemistry.data.BasisSet.get_ecp_name`.

    Returns:
        The name of ExaChem's ``.ecp`` file holding the ECP.

    Examples:
        >>> ecp_name_to_exachem("def2-svp")
        'def2-ecp'

    """
    Logger.trace_entering()
    exachem_name = basis_name_to_exachem(name)
    if exachem_name.startswith("def2-"):
        return "def2-ecp"
    if exachem_name in ("lanl2dz", "lanl2dzdp"):
        return "lanl2dz_ecp"
    return exachem_name


def _within_shell_m_reorder(basis_set: BasisSet) -> np.ndarray:
    """AO permutation correcting the within-shell m-component ordering.

    This assumes ExaChem reads qdk-chemistry's own basis (identical inter-shell
    order, see :func:`write_qdk_basis_g94`), so the only remaining difference is
    the ordering of the spherical components *inside* a shell.

    qdk-chemistry represents p orbitals in **Cartesian** form (``px, py, pz``)
    while ExaChem/Libint uses **spherical** p in ascending-m order
    (``m=-1, 0, +1`` = ``py, pz, px``); s is a single function and d/f/g are
    spherical in both, so they already match.  Only the p components are
    therefore reordered.  The permutation is built from qdk-chemistry's AO
    metadata (``get_atomic_orbital_info`` returns ``(shell_index, m)``) and
    satisfies ``C_exachem = C_qdk[perm, :]``.
    """
    Logger.trace_entering()
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
        # The p-orbital swap is needed because qdk-chemistry treats p orbitals as
        # CARTESIAN (px, py, pz), whereas ExaChem/Libint uses SPHERICAL p in
        # ascending-m order (m=-1, 0, +1) = (py, pz, px).  qdk-chemistry keeps s
        # and p Cartesian and makes only d/f/g spherical (see the l >= 2 predicate
        # in convert_to_libint_shell), which is why d/f/g already match and only p
        # needs reordering.  For l=1 the Cartesian and spherical p functions span
        # the same space, so a pure permutation suffices.  (This is a qdk-chemistry
        # AO-layout convention, not a Libint version difference.)
        idxs = [i for _m, i in sorted(aos)]  # qdk Cartesian p: (px, py, pz)
        start = idxs[0]
        # ExaChem spherical p (py, pz, px) -> qdk Cartesian indices (1, 2, 0)
        perm[start + 0] = idxs[1]
        perm[start + 1] = idxs[2]
        perm[start + 2] = idxs[0]
    return perm


def write_qdk_basis_g94(basis_set: BasisSet, elements: list, basis_data_dir: str | Path, basis_name: str) -> Path:
    """Write qdk-chemistry's basis as a Gaussian-94 file in qdk's shell order.

    ExaChem/Libint2 reads a basis from ``<LIBINT_DATA_PATH>/basis/<name>.g94``.
    :meth:`~qdk_chemistry.data.BasisSet.get_shell` returns each shell's exponents and contraction
    coefficients in the raw g94 convention, so writing qdk-chemistry's shells in
    qdk-chemistry's native order makes ExaChem build its AO basis with exactly
    the same inter-shell order *and* numerically identical parameters.  Only the
    within-shell m-component order then differs (handled by
    ``_within_shell_m_reorder``).

    Args:
        basis_set: A :class:`~qdk_chemistry.data.BasisSet` instance.
        elements: Element symbols in molecule (atom) order.
        basis_data_dir: Directory to write into; the file is placed at ``<basis_data_dir>/basis/<basis_name>.g94``.
        basis_name: Basis set name (used for the file name).

    Returns:
        The ``basis_data_dir`` path, to be used as ``LIBINT_DATA_PATH``.

    Raises:
        ValueError: If the basis name is not a safe file name or atoms of one element carry different shells.

    """
    Logger.trace_entering()
    # basis_name becomes a file name, so reject anything that could escape the basis dir.
    if not basis_name or "/" in basis_name or "\\" in basis_name or ".." in basis_name:
        raise ValueError(f"Invalid basis name for a Gaussian-94 file name: {basis_name!r}")

    # Group shell indices by atom (shells are already in qdk's native order).
    atom_shells: dict[int, list[int]] = {}
    for s in range(basis_set.get_num_shells()):
        atom_shells.setdefault(int(basis_set.get_shell(s).atom_index), []).append(s)

    # One block per element, so every atom of an element must carry the same shells.
    blocks: list[str] = []
    seen: dict[str, list] = {}
    for atom_idx in sorted(atom_shells):
        elem = str(elements[atom_idx]).strip().title()
        shells = [
            (sh.get_angular_momentum(), list(sh.exponents), list(sh.coefficients))
            for sh in map(basis_set.get_shell, atom_shells[atom_idx])
        ]
        if elem in seen:
            if shells != seen[elem]:
                raise ValueError(f"ExaChem needs one basis per element, but the {elem} atoms carry different shells.")
            continue
        seen[elem] = shells
        lines = [f"{elem}     0"]
        for s in atom_shells[atom_idx]:
            sh = basis_set.get_shell(s)
            angular_momentum = sh.get_angular_momentum()
            exps = list(sh.exponents)
            coeffs = list(sh.coefficients)
            lines.append(f"{_L_SYMBOL[angular_momentum]}   {len(exps)}   1.00")
            for e, c in zip(exps, coeffs, strict=False):
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
    Logger.info(f"Wrote qdk-chemistry basis for ExaChem: {g94_path}")
    return Path(basis_data_dir)


def write_qdk_ecp(basis_set: BasisSet, elements: list, basis_data_dir: str | Path, ecp_name: str) -> Path:
    """Write qdk-chemistry's effective core potentials as an ExaChem ``.ecp`` file.

    The counterpart of :func:`write_qdk_basis_g94` for ECPs. ExaChem builds whose basis lookup honors
    ``LIBINT_DATA_PATH`` read ``<LIBINT_DATA_PATH>/basis/<name>.ecp``; others read their own library file of that
    name, so ``ecp_name`` must be ExaChem's library name (see :func:`ecp_name_to_exachem`). Each element's block
    lists qdk-chemistry's ECP shells, one radial power, exponent, and coefficient per line, with the local term
    (``OrbitalType.UL``) as ExaChem's ``ul`` channel; an ECP without a local term gets a zero one.

    Args:
        basis_set: A :class:`~qdk_chemistry.data.BasisSet` instance carrying ECP shells.
        elements: Element symbols in molecule (atom) order.
        basis_data_dir: Directory to write into; the file is placed at ``<basis_data_dir>/basis/<ecp_name>.ecp``.
        ecp_name: ECP name (used for the file name).

    Returns:
        The ``basis_data_dir`` path, to be used as ``LIBINT_DATA_PATH``.

    Raises:
        ValueError: If the ECP name is not a safe file name or atoms of one element carry different ECPs.

    """
    Logger.trace_entering()
    # ecp_name becomes a file name, so reject anything that could escape the basis dir.
    if not ecp_name or "/" in ecp_name or "\\" in ecp_name or ".." in ecp_name:
        raise ValueError(f"Invalid ECP name for an ExaChem ECP file name: {ecp_name!r}")

    # One block per element, so every atom of an element must carry the same ECP.
    blocks: list[str] = []
    seen: dict[str, tuple] = {}
    for atom_idx, core_electrons in enumerate(basis_set.get_ecp_electrons()):
        if not core_electrons:
            continue
        elem = str(elements[atom_idx]).strip().title()
        channels = [
            (sh.get_angular_momentum(), [int(n) for n in sh.rpowers], list(sh.exponents), list(sh.coefficients))
            for sh in basis_set.get_ecp_shells_for_atom(atom_idx)
        ]
        if elem in seen:
            if (core_electrons, channels) != seen[elem]:
                raise ValueError(f"ExaChem needs one ECP per element, but the {elem} atoms carry different ECPs.")
            continue
        seen[elem] = (core_electrons, channels)
        lines = [f"{elem} nelec {core_electrons}"]
        if all(channel[0] >= 0 for channel in channels):
            # ExaChem treats "ul" as the highest channel, so without one the highest semi-local term would turn local.
            lines += [f"{elem} ul", f"2   {1.0: .14E}   {0.0: .14E}"]
        for angular_momentum, rpowers, exponents, coefficients in sorted(channels, key=lambda channel: channel[0]):
            lines.append(f"{elem} {'ul' if angular_momentum < 0 else _L_SYMBOL[angular_momentum]}")
            lines.extend(
                f"{n}   {e: .14E}   {c: .14E}" for n, e, c in zip(rpowers, exponents, coefficients, strict=True)
            )
        blocks.append("\n".join(lines))

    basis_dir = Path(basis_data_dir) / "basis"
    basis_dir.mkdir(parents=True, exist_ok=True)
    ecp_path = basis_dir / f"{ecp_name}.ecp"
    # ExaChem reads from the first line containing "ECP" to the first containing "END", so no header comment.
    ecp_path.write_text("ECP\n" + "\n".join(blocks) + "\nEND\n")
    Logger.info(f"Wrote qdk-chemistry ECP for ExaChem: {ecp_path}")
    return Path(basis_data_dir)


def write_exachem_matrix(filepath: str | Path, matrix: np.ndarray) -> None:
    """Write a matrix in ExaChem's serial-IO HDF5 format.

    Args:
        filepath: Output path whose extension names the dataset (``.alpha.movecs`` -> ``"movecs"``).
        matrix: 2-D numpy array to write. Stored as a flat row-major 1-D dataset.

    Raises:
        ValueError: If ``matrix`` is not two-dimensional.

    """
    Logger.trace_entering()
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

    Logger.debug(f"Wrote {filepath}: shape={matrix.shape}, dataset={dataset_name!r}")


def export_scf_files(
    files_prefix: str | Path,
    mo_coeff_alpha: np.ndarray,
    density_alpha: np.ndarray,
    basis_set: BasisSet,
    basis_name: str,
    elements: list,
    basis_data_dir: str | Path,
    runcontext_prefix: str | Path,
    mo_coeff_beta: np.ndarray | None = None,
    density_beta: np.ndarray | None = None,
    ecp_name: str | None = None,
) -> list[Path]:
    """Write SCF restart files for ExaChem's ``noscf`` mode.

    Creates the minimum files required by ExaChem's SCF restart path:
    MO coefficients and density matrices in serial-IO HDF5 format, plus
    a run context JSON with the AO tile size.

    qdk-chemistry's own basis is written to ``basis_data_dir`` (see
    :func:`write_qdk_basis_g94`) so ExaChem reads it via ``LIBINT_DATA_PATH``,
    giving both codes an identical inter-shell order and identical basis
    parameters, and its ECP is written next to it when ``ecp_name`` is given
    (see :func:`write_qdk_ecp`).  The AO rows/columns of the MO coefficient and
    density matrices are then corrected for the within-shell p-component
    ordering only (``_within_shell_m_reorder``).

    For a restricted (closed-shell) reference, pass only the alpha quantities;
    ExaChem reads the total density from ``.alpha.density`` (so ``density_alpha``
    should be the total alpha+beta density). For an unrestricted reference, also
    pass ``mo_coeff_beta`` and ``density_beta``; ExaChem then reads the per-spin
    densities from ``.alpha.density`` and ``.beta.density`` separately.

    Args:
        files_prefix: Path prefix for output files; files are named ``<prefix>.alpha.movecs`` etc.
        mo_coeff_alpha: Alpha MO coefficient matrix, shape ``(nbf, nmo)``.
        density_alpha: Alpha (or total, for restricted) AO density matrix, shape ``(nbf, nbf)``.
        runcontext_prefix: Path prefix for the run context JSON.
        basis_set: A :class:`~qdk_chemistry.data.BasisSet` instance (required for the basis export and AO reorder).
        mo_coeff_beta: Beta MO coefficients (unrestricted), shape ``(nbf, nmo)``; None writes no beta files.
        density_beta: Beta AO density (unrestricted), shape ``(nbf, nbf)``; None writes no beta density.
        basis_name: Basis set name (e.g. ``"cc-pvdz"``); used for the written ``.g94`` file name.
        elements: Element symbols in molecule (atom) order.
        basis_data_dir: Directory receiving qdk-chemistry's basis as a Gaussian-94 file for ExaChem to read.
        ecp_name: ExaChem ECP name (e.g. ``"def2-ecp"``); if given, qdk-chemistry's ECP is also written for ExaChem.

    Returns:
        List of paths to the created files.

    Raises:
        ValueError: If :func:`write_qdk_basis_g94` or :func:`write_qdk_ecp` rejects the basis set.

    """
    Logger.trace_entering()
    files_prefix = Path(files_prefix)
    files_prefix.parent.mkdir(parents=True, exist_ok=True)

    # Feed qdk-chemistry's own basis and ECP to ExaChem (read via LIBINT_DATA_PATH) so the
    # inter-shell order and basis parameters match exactly; only the within-shell
    # p-component order then needs correcting.
    write_qdk_basis_g94(basis_set, elements, basis_data_dir, basis_name)
    if ecp_name is not None:
        write_qdk_ecp(basis_set, elements, basis_data_dir, ecp_name)
    perm = _within_shell_m_reorder(basis_set)
    mo_coeff_alpha = mo_coeff_alpha[perm, :]
    density_alpha = density_alpha[np.ix_(perm, perm)]
    if mo_coeff_beta is not None:
        mo_coeff_beta = mo_coeff_beta[perm, :]
    if density_beta is not None:
        density_beta = density_beta[np.ix_(perm, perm)]

    created: list[Path] = []

    # Alpha movecs and density (always required)
    movecs_path = Path(str(files_prefix) + ".alpha.movecs")
    write_exachem_matrix(movecs_path, mo_coeff_alpha)
    created.append(movecs_path)

    density_path = Path(str(files_prefix) + ".alpha.density")
    write_exachem_matrix(density_path, density_alpha)
    created.append(density_path)

    # Beta movecs and density (unrestricted references only)
    if mo_coeff_beta is not None:
        beta_movecs_path = Path(str(files_prefix) + ".beta.movecs")
        write_exachem_matrix(beta_movecs_path, mo_coeff_beta)
        created.append(beta_movecs_path)
    if density_beta is not None:
        beta_density_path = Path(str(files_prefix) + ".beta.density")
        write_exachem_matrix(beta_density_path, density_beta)
        created.append(beta_density_path)

    # ExaChem's SCF restart aborts unless this matches its AO tile size (30 by default).
    rc_prefix = Path(runcontext_prefix)
    rc_prefix.parent.mkdir(parents=True, exist_ok=True)
    runcontext_path = Path(str(rc_prefix) + ".runcontext.json")

    run_context = {"ao_tilesize": 30}
    runcontext_path.write_text(json.dumps(run_context, indent=2))
    created.append(runcontext_path)

    Logger.info(f"Exported {len(created)} SCF restart files with prefix {files_prefix}")
    return created
