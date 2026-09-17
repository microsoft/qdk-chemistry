"""Tests for the native cube generator bindings."""
# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import numpy as np
import pytest

try:
    import pyscf  # noqa: F401
    from pyscf import gto
    from pyscf.tools import cubegen

    PYSCF_AVAILABLE = True
except ImportError:
    PYSCF_AVAILABLE = False

from qdk_chemistry.data import AOType, BasisSet, Orbitals, OrbitalType, Shell, Structure
from qdk_chemistry.utils import (
    CubeGenerator,
    CubeGrid,
    generate_orbital_cubes,
)
from qdk_chemistry.utils.cubegen import generate_cubefiles_from_orbitals

if PYSCF_AVAILABLE:
    from qdk_chemistry.plugins.pyscf.conversion import pyscf_mol_to_qdk_basis


O2_COORDS = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 2.282]])


def _o2(basis: str):
    """Build a matching PySCF Mole and QDK basis set for O2."""
    structure = Structure(["O", "O"], O2_COORDS)
    mol = gto.M(
        atom=[("O", O2_COORDS[0]), ("O", O2_COORDS[1])],
        basis=basis,
        unit="Bohr",
        spin=2,
    )
    return mol, pyscf_mol_to_qdk_basis(mol, structure, basis)


def _pyscf_grid(mol, n: int, margin: float = 3.0):
    """Return the exact grid PySCF's cubegen would use, as a CubeGrid.

    PySCF stores fractional axis samples and a box matrix, so the step in Bohr
    is the box diagonal divided by ``n - 1``.
    """
    cube = cubegen.Cube(mol, n, n, n, margin=margin)
    origin = np.asarray(cube.boxorig, dtype=float)
    spacing = np.diag(np.asarray(cube.box, dtype=float)) / (n - 1)
    return cube, CubeGrid(origin, spacing, n, n, n)


class TestCubeGrid:
    """Tests for the grid description."""

    def test_num_points_is_product_of_extents(self):
        """The point count is the product of the three extents."""
        grid = CubeGrid([0.0, 0.0, 0.0], [0.2, 0.2, 0.2], 4, 5, 6)
        assert grid.num_points() == 120

    def test_defaults_are_eighty_cubed(self):
        """Omitted extents default to 80 points per axis."""
        grid = CubeGrid([0.0, 0.0, 0.0], [0.2, 0.2, 0.2])
        assert (grid.nx, grid.ny, grid.nz) == (80, 80, 80)

    @pytest.mark.skipif(not PYSCF_AVAILABLE, reason="PySCF not available")
    def test_from_basis_set_encloses_molecule_with_margin(self):
        """The derived grid covers every nucleus plus the requested margin."""
        _, basis_set = _o2("sto-3g")
        grid = CubeGrid.from_basis_set(basis_set, 16, 16, 16, 3.0)
        far_corner = np.asarray(grid.origin) + np.asarray(grid.spacing) * np.array(
            [grid.nx - 1, grid.ny - 1, grid.nz - 1]
        )
        assert np.all(np.asarray(grid.origin) <= O2_COORDS.min(axis=0) - 3.0 + 1e-9)
        assert np.all(far_corner >= O2_COORDS.max(axis=0) + 3.0 - 1e-9)


def test_pyscf_backend_rejects_cartesian_basis_before_import(tmp_path):
    """PySCF cannot consume the Cartesian AO coefficient layout."""
    structure = Structure(["O"], np.zeros((1, 3)))
    shell = Shell(0, OrbitalType.D, np.array([1.0]), np.array([1.0]))
    basis_set = BasisSet("cartesian-d", [shell], structure, AOType.Cartesian)
    nbf = basis_set.get_num_atomic_orbitals()
    orbitals = Orbitals(np.eye(nbf), None, None, basis_set)

    with pytest.raises(ValueError, match="does not support Cartesian"):
        generate_cubefiles_from_orbitals(
            orbitals,
            output_folder=tmp_path,
            indices=[0],
            grid_size=(2, 2, 2),
            backend="pyscf",
        )
    assert list(tmp_path.iterdir()) == []


@pytest.mark.skipif(not PYSCF_AVAILABLE, reason="PySCF not available")
class TestCubeGenerator:
    """Tests for orbital and density evaluation."""

    def test_orbital_returns_grid_shaped_array(self):
        """The returned field is shaped like the grid, not flat."""
        mol, basis_set = _o2("sto-3g")
        grid = CubeGrid([-3.0, -3.0, -3.0], [0.5, 0.5, 0.5], 6, 7, 8)
        field = CubeGenerator(basis_set).orbital(np.zeros(mol.nao_nr()), grid)
        assert field.shape == (6, 7, 8)
        assert np.all(field == 0.0)

    def test_orbital_writes_cube_file_only_when_requested(self, tmp_path):
        """A file appears only when an output path is supplied."""
        mol, basis_set = _o2("sto-3g")
        generator = CubeGenerator(basis_set)
        grid = CubeGrid([-3.0, -3.0, -3.0], [1.0, 1.0, 1.0], 4, 4, 4)
        coeff = np.zeros(mol.nao_nr())
        coeff[0] = 1.0

        generator.orbital(coeff, grid)
        assert list(tmp_path.iterdir()) == []

        outfile = tmp_path / "written.cube"
        generator.orbital(coeff, grid, outfile=str(outfile), comment="hello")
        assert outfile.exists()
        assert "hello" in outfile.read_text().splitlines()[0]

    @pytest.mark.parametrize("basis", ["sto-3g", "cc-pvdz"])
    def test_matches_pyscf_atomic_orbital_evaluation(self, basis):
        """The native evaluator must reproduce PySCF on an identical grid.

        This pins the atomic orbital ordering, the Cartesian p convention, the
        spherical d ordering, and the normalisation all at once: any
        disagreement in those would show up as an order-one error here.
        """
        mol, basis_set = _o2(basis)
        n = 12
        cube, grid = _pyscf_grid(mol, n)
        ao = mol.eval_gto("GTOval", cube.get_coords())
        generator = CubeGenerator(basis_set)

        rng = np.random.default_rng(0)
        coeff = rng.normal(size=mol.nao_nr())
        native = generator.orbital(coeff, grid).ravel()
        reference = ao @ coeff
        assert np.allclose(native, reference, rtol=1e-10, atol=1e-10 * np.max(np.abs(reference)))

        # Per atomic orbital, so a permutation cannot hide inside a contraction.
        for index in range(mol.nao_nr()):
            unit = np.zeros(mol.nao_nr())
            unit[index] = 1.0
            got = generator.orbital(unit, grid).ravel()
            want = ao[:, index]
            assert np.allclose(got, want, rtol=1e-10, atol=1e-10 * np.max(np.abs(want))), (
                f"atomic orbital {index} ({mol.ao_labels()[index]}) disagrees with PySCF"
            )

    def test_density_of_rank_one_matrix_is_squared_orbital(self):
        """density(c c^T) must equal orbital(c) squared, point by point."""
        mol, basis_set = _o2("sto-3g")
        generator = CubeGenerator(basis_set)
        grid = CubeGrid([-3.0, -3.0, -3.0], [0.75, 0.75, 0.75], 8, 8, 8)

        rng = np.random.default_rng(1)
        coeff = rng.normal(size=mol.nao_nr())
        orbital = generator.orbital(coeff, grid)
        density = generator.density(np.outer(coeff, coeff), grid)
        assert np.allclose(density, orbital**2, rtol=1e-12, atol=1e-12)

    def test_density_is_used_verbatim(self):
        """Scaling the density matrix scales the field, with no hidden factor."""
        mol, basis_set = _o2("sto-3g")
        generator = CubeGenerator(basis_set)
        grid = CubeGrid([-3.0, -3.0, -3.0], [1.0, 1.0, 1.0], 6, 6, 6)

        rng = np.random.default_rng(2)
        matrix = rng.normal(size=(mol.nao_nr(), mol.nao_nr()))
        matrix = matrix + matrix.T
        single = generator.density(matrix, grid)
        doubled = generator.density(2.0 * matrix, grid)
        assert np.allclose(doubled, 2.0 * single, rtol=1e-12, atol=1e-12)


@pytest.mark.skipif(not PYSCF_AVAILABLE, reason="PySCF not available")
class TestGenerateOrbitalCubes:
    """Tests for the batch cube writer."""

    @staticmethod
    def _h2_orbitals():
        """Build minimal restricted H2 orbitals and their basis set."""
        coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]])
        structure = Structure(["H", "H"], coords)
        mol = gto.M(atom=[("H", coords[0]), ("H", coords[1])], basis="sto-3g", unit="Bohr")
        basis_set = pyscf_mol_to_qdk_basis(mol, structure, "sto-3g")
        nao = mol.nao_nr()
        return Orbitals(np.eye(nao), None, None, basis_set), basis_set

    def test_writes_zero_based_names_for_restricted_orbitals(self, tmp_path):
        """Restricted orbitals get one zero-based cube per orbital."""
        orbitals, basis_set = self._h2_orbitals()
        grid = CubeGrid.from_basis_set(basis_set, 8, 8, 8, 3.0)
        paths = generate_orbital_cubes(orbitals, [0, 1], str(tmp_path), grid)
        assert [p.rsplit("/", 1)[-1] for p in paths] == [
            "orbital_0000.cube",
            "orbital_0001.cube",
        ]
        assert all((tmp_path / name).exists() for name in ["orbital_0000.cube", "orbital_0001.cube"])

    def test_respects_label_prefix(self, tmp_path):
        """A custom prefix replaces the default orbital label."""
        orbitals, basis_set = self._h2_orbitals()
        grid = CubeGrid.from_basis_set(basis_set, 8, 8, 8, 3.0)
        paths = generate_orbital_cubes(orbitals, [0], str(tmp_path), grid, "mo_")
        assert paths[0].endswith("mo_0000.cube")
