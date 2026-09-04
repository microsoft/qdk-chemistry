"""Tests for cube file generation utilities."""
# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import numpy as np
import pytest

try:
    import pyscf  # noqa: F401
    from pyscf import gto

    PYSCF_AVAILABLE = True
except ImportError:
    PYSCF_AVAILABLE = False

pytestmark = pytest.mark.skipif(not PYSCF_AVAILABLE, reason="PySCF not available")

if PYSCF_AVAILABLE:
    from qdk_chemistry.data import Orbitals, Structure
    from qdk_chemistry.plugins.pyscf.conversion import basis_to_pyscf_mol, pyscf_mol_to_qdk_basis
    from qdk_chemistry.utils.cubegen import generate_cubefiles_from_orbitals


def _diatomic_orbitals(symbols: tuple[str, str], bond_length: float, multiplicity: int):
    """Build identity orbitals for a diatomic in the STO-3G basis."""
    coordinates = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, bond_length]])
    structure = Structure(list(symbols), coordinates)
    mol = gto.M(
        atom=list(zip(symbols, coordinates, strict=True)),
        basis="sto-3g",
        unit="Bohr",
        spin=multiplicity - 1,
    )
    basis_set = pyscf_mol_to_qdk_basis(mol, structure, "sto-3g")
    return Orbitals(np.eye(mol.nao_nr()), None, None, basis_set)


def _o2_orbitals():
    """Return identity orbitals for triplet O2."""
    return _diatomic_orbitals(("O", "O"), bond_length=2.282, multiplicity=3)


def _no_orbitals():
    """Return identity orbitals for doublet NO."""
    return _diatomic_orbitals(("N", "O"), bond_length=2.175, multiplicity=2)


def _parse_cube_text(text: str):
    """Parse cube text into origin, axis vectors, shape, and the scalar field."""
    lines = text.splitlines()
    natoms = int(lines[2].split()[0])
    origin = np.array([float(v) for v in lines[2].split()[1:4]])
    shape, vectors = [], []
    for i in range(3):
        parts = lines[3 + i].split()
        shape.append(int(parts[0]))
        vectors.append([float(v) for v in parts[1:4]])
    field = np.array([float(v) for v in " ".join(lines[6 + abs(natoms) :]).split()])
    return origin, np.array(vectors), tuple(shape), field


@pytest.mark.skipif(not PYSCF_AVAILABLE, reason="PySCF not available")
class TestCubegen:
    """Tests for cube file generation utilities."""

    def test_generate_cubefiles_singlet(self):
        """An empty index list produces no cubes for a singlet."""
        orbitals = _o2_orbitals()
        assert generate_cubefiles_from_orbitals(orbitals, indices=[]) == {}

    def test_generate_cubefiles_doublet(self):
        """An empty index list produces no cubes for a doublet."""
        orbitals = _no_orbitals()
        assert generate_cubefiles_from_orbitals(orbitals, indices=[]) == {}

    def test_generate_cubefiles_uses_zero_based_default_labels(self, tmp_path):
        """Default labels embed the zero-based orbital index."""
        orbitals = _o2_orbitals()

        cubes = generate_cubefiles_from_orbitals(orbitals, indices=[0], grid_size=(4, 4, 4))
        paths = generate_cubefiles_from_orbitals(
            orbitals,
            output_folder=tmp_path,
            indices=[0],
            grid_size=(4, 4, 4),
        )

        assert list(cubes) == ["orbital_0000"]
        assert paths == [str(tmp_path / "orbital_0000.cube")]

    def test_generate_cubefiles_are_identical_for_singlet_and_triplet(self, monkeypatch):
        """The PySCF backend must not depend on the multiplicity it guesses for the Mole."""
        from qdk_chemistry.plugins.pyscf import conversion  # noqa: PLC0415

        orbitals = _o2_orbitals()
        basis_set = orbitals.get_basis_set()
        molecules = iter(
            [
                basis_to_pyscf_mol(basis_set, charge=0, multiplicity=1),
                basis_to_pyscf_mol(basis_set, charge=0, multiplicity=3),
            ]
        )
        monkeypatch.setattr(conversion, "basis_to_pyscf_mol", lambda *_args, **_kwargs: next(molecules))

        singlet_cubes = generate_cubefiles_from_orbitals(orbitals, indices=[0], grid_size=(4, 4, 4), backend="pyscf")
        triplet_cubes = generate_cubefiles_from_orbitals(orbitals, indices=[0], grid_size=(4, 4, 4), backend="pyscf")

        # get first item in map and remove header (which includes current time) to compare only the data
        singlet_cubes = next(iter(singlet_cubes.values())).splitlines()[2:]
        triplet_cubes = next(iter(triplet_cubes.values())).splitlines()[2:]

        assert singlet_cubes == triplet_cubes

    def test_native_backend_matches_pyscf_backend(self):
        """Both backends must place the grid identically and agree on every field value.

        This is the guard on making the native backend the default: switching
        it must not move a single grid point or change a single value beyond
        the precision the cube format itself can represent.
        """
        orbitals = _o2_orbitals()
        kwargs = {"indices": [0, 3], "grid_size": (8, 8, 8)}
        native = generate_cubefiles_from_orbitals(orbitals, backend="native", **kwargs)
        reference = generate_cubefiles_from_orbitals(orbitals, backend="pyscf", **kwargs)

        assert set(native) == set(reference)
        for label in native:
            got = _parse_cube_text(native[label])
            want = _parse_cube_text(reference[label])
            assert got[2] == want[2], f"{label}: grid shape differs"
            np.testing.assert_allclose(got[0], want[0], atol=1e-6, err_msg=f"{label}: origin")
            np.testing.assert_allclose(got[1], want[1], atol=1e-6, err_msg=f"{label}: axes")
            # Cube files store values as %13.5E, so agreement is bounded by the
            # format, not by the evaluators.
            scale = max(float(np.max(np.abs(want[3]))), 1.0)
            np.testing.assert_allclose(got[3], want[3], atol=2e-5 * scale, err_msg=f"{label}: field values")

    def test_rejects_unknown_backend(self):
        """An unrecognised backend name fails loudly rather than silently picking one."""
        orbitals = _o2_orbitals()
        with pytest.raises(ValueError, match="Unknown cube backend"):
            generate_cubefiles_from_orbitals(orbitals, indices=[0], backend="nope")
