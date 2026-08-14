"""Generate cube files for molecular orbitals and electron densities."""
# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import os
import tempfile
from collections.abc import Callable
from pathlib import Path

from qdk_chemistry.data import Orbitals
from qdk_chemistry.data._spin_channels import spin_channel_matrix
from qdk_chemistry.data.symmetry import axes
from qdk_chemistry.utils import CubeGenerator, CubeGrid, Logger

__all__ = [
    "generate_cubefiles_from_orbitals",
]


def generate_cubefiles_from_orbitals(
    orbitals: Orbitals,
    output_folder: str | Path | None = None,
    indices: list[int] | None = None,
    grid_size: tuple = (40, 40, 40),
    margin: float = 3.0,
    label_maker: Callable[[int], str] | None = None,
    backend: str = "native",
) -> list[str] | dict[str, str]:
    """Generate volumetric cube data for molecular orbitals.

    This method creates cube files containing the spatial distribution of molecular
    orbitals on a 3D grid. It supports both canonical and localized orbitals.

    Args:
        orbitals:  The orbitals object containing the molecular orbital coefficients and basis set
        output_folder:  The folder where the cube files will be saved.

            If None, files are not saved to temporary storage.

        indices: Specific molecular orbital indices to generate cube files for. If None, all orbitals are processed.
        grid_size: The size of the grid in each dimension (nx, ny, nz). Default is (40, 40, 40).
        margin: The margin (in Bohr radii) to extend around molecule. Default is 3.
        label_maker: A function that takes an orbital index and returns a base label without the ``.cube`` extension.

            If None, files are labeled with the zero-based orbital index; orbital ``0`` generates
            ``orbital_0000.cube``.

        backend: Which evaluator to use, either ``"native"`` (default) or ``"pyscf"``.

            The native backend evaluates the orbitals in process with gauXC and needs no
            third-party quantum chemistry package. The ``"pyscf"`` backend delegates to
            ``pyscf.tools.cubegen`` and is kept for comparison and for falling back if a
            difference is ever suspected. PySCF is not installed on Windows, so the
            ``"pyscf"`` backend is unavailable there and raises ``ImportError``.

            Both backends place grid points identically: the origin is the nuclear bounding
            box corner minus ``margin``, and the step is the padded extent divided by
            ``n - 1`` along each axis. They also share the same atomic orbital ordering, so
            the same coefficient vector means the same thing to both.

    Returns:
        list[str] | dict[str, str]: Paths or contents of the generated cube files.

    Raises:
        ValueError: If ``backend`` is not ``"native"`` or ``"pyscf"``.

    """
    Logger.trace_entering()
    if output_folder is not None:
        output_folder = Path(output_folder)
        output_folder.mkdir(parents=True, exist_ok=True)

    basis_set = orbitals.get_basis_set()
    nmo = orbitals.get_num_molecular_orbitals()
    mo_range = range(nmo)
    nx, ny, nz = grid_size

    if backend == "native":
        grid = CubeGrid.from_basis_set(basis_set, nx, ny, nz, margin)
        generator = CubeGenerator(basis_set)

        def _write_cube(coeff, outfile_name) -> None:
            generator.orbital(coeff, grid, outfile=str(outfile_name))

    elif backend == "pyscf":
        # Imported lazily so that the default path, and therefore this module,
        # does not require PySCF to be installed.
        from pyscf.tools import cubegen  # noqa: PLC0415

        from qdk_chemistry.plugins.pyscf.conversion import basis_to_pyscf_mol  # noqa: PLC0415

        try:
            mol = basis_to_pyscf_mol(basis_set, charge=0, multiplicity=1)
        except RuntimeError:
            mol = basis_to_pyscf_mol(basis_set, charge=0, multiplicity=2)

        def _write_cube(coeff, outfile_name) -> None:
            cubegen.orbital(mol, outfile=str(outfile_name), coeff=coeff, nx=nx, ny=ny, nz=nz, margin=margin)

    else:
        raise ValueError(f"Unknown cube backend '{backend}'. Expected 'native' or 'pyscf'.")

    mo_a = spin_channel_matrix(orbitals.coefficients(), axes.alpha())
    mo_b = spin_channel_matrix(orbitals.coefficients(), axes.beta())

    cubefile_paths: list[str] | dict[str, str] = [] if output_folder is not None else {}

    def _generate_cube(coeff, label):
        fd = None
        if output_folder is None:
            fd, outfile_name = tempfile.mkstemp()
            os.close(fd)
        else:
            outfile_name = output_folder / label
        _write_cube(coeff, outfile_name)

        if output_folder is None:
            with open(outfile_name, encoding="utf-8") as f:
                assert isinstance(cubefile_paths, dict)
                cubefile_paths[label.replace(".cube", "")] = f.read()
            os.remove(outfile_name)
        else:
            assert isinstance(cubefile_paths, list)
            cubefile_paths.append(outfile_name)

    if label_maker is None:
        label_maker = lambda p: f"orbital_{p:04d}"  # noqa: E731

    # Loop over all the MOs
    for _, p in enumerate(mo_range):
        if indices is not None and p not in indices:
            continue
        if orbitals.is_restricted():
            coeff = mo_a[:, p]
            label = f"{label_maker(p)}.cube"
            _generate_cube(coeff, label)
        else:
            coeff_a = mo_a[:, p]
            label_a = f"{label_maker(p)}_a.cube"
            _generate_cube(coeff_a, label_a)

            coeff_b = mo_b[:, p]
            label_b = f"{label_maker(p)}_b.cube"
            _generate_cube(coeff_b, label_b)

    if output_folder is not None:
        assert isinstance(cubefile_paths, list)
        return [str(i) for i in cubefile_paths]

    assert isinstance(cubefile_paths, dict)
    return cubefile_paths
