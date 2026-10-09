"""Basic Python API tests for MPS storage."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import numpy as np
import pytest

from qdk_chemistry.data import ModelOrbitals, MPSContainer, MPSSite, Wavefunction
from qdk_chemistry.data import symmetry as sym


def make_site(values):
    """Wrap an array as one trivial-symmetry block."""
    left, physical, right = values.shape
    product, label = sym.SymmetryProduct([]), sym.SymmetryLabel([])
    tensor_type = sym.SymmetryBlockedTensorRank3Complex if np.iscomplexobj(values) else sym.SymmetryBlockedTensorRank3
    tensor = tensor_type(
        [product] * 3,
        [{label: left}, {label: physical}, {label: right}],
        [((label,) * 3, values.reshape(left * physical, right))],
    )
    return MPSSite(tensor, [label], [label], [label])


@pytest.fixture
def site_data():
    """Use reproducible random input rather than a particular physical state."""
    rng = np.random.default_rng(42)
    return [rng.random((1, 4, 3)), rng.random((3, 4, 1))]


def test_construction_and_accessors(site_data):
    """The container retains supplied sites and optional metadata."""
    sites = [make_site(values) for values in site_data]
    orbitals = ModelOrbitals(2)
    counts = sym.SymmetryBlockedScalarCount([sym.SymmetryProduct([])], [((sym.SymmetryLabel([]),), 2)])
    wavefunction = Wavefunction(
        MPSContainer(
            sites,
            orbitals,
            total_num_particles=counts,
            orthogonality_center=1,
            site_to_orbital_order=[1, 0],
        )
    )
    assert "num_sites=2" in repr(wavefunction)
    container = wavefunction.get_container()
    del wavefunction
    assert isinstance(container, MPSContainer)
    assert container.num_sites == 2
    assert container.max_bond_dimension == 3
    assert not container.is_complex
    assert container.orthogonality_center == 1
    assert container.site_to_orbital_order == [1, 0]
    assert container.total_num_particles.value(sym.SymmetryLabel([])) == 2
    assert not container.has_active_num_particles()
    with pytest.raises(RuntimeError, match="Active particle-count is not set"):
        _ = container.active_num_particles
    for site, values in zip(container.sites, site_data, strict=True):
        assert site.shape == values.shape
        np.testing.assert_array_equal(site.to_dense().reshape(site.shape), values)


@pytest.mark.parametrize("complex_values", [False, True])
def test_serialization(site_data, complex_values, tmp_path):
    """Python JSON and file factories preserve real and complex tensor data."""
    if complex_values:
        rng = np.random.default_rng(43)
        site_data = [values + 1j * rng.random(values.shape) for values in site_data]
    original = Wavefunction(MPSContainer([make_site(values) for values in site_data], ModelOrbitals(2)))
    restored = [Wavefunction.from_json(original.to_json())]
    for extension, format_name in [("json", "json"), ("h5", "hdf5")]:
        path = tmp_path / f"mps.wavefunction.{extension}"
        original.to_file(path, format_name)
        restored.append(Wavefunction.from_file(path, format_name))
    for wavefunction in restored:
        assert wavefunction.content_hash() == original.content_hash()
        container = wavefunction.get_container()
        assert container.is_complex is complex_values
        for site, values in zip(container.sites, site_data, strict=True):
            assert site.to_dense().dtype == values.dtype
            np.testing.assert_array_equal(site.to_dense().reshape(site.shape), values)


@pytest.mark.parametrize("dimension", [2, 4])
def test_numpy_input_and_output_copies(dimension):
    """Strided input and exported arrays do not alias stored data."""
    values = np.random.default_rng(42).random((2, dimension, 3))[:, :, ::-1]
    expected = values.copy()
    site = make_site(values)
    values[:] = 0
    exported = site.to_dense()
    assert exported.shape == (2 * dimension, 3)
    np.testing.assert_array_equal(exported.reshape(site.shape), expected)
    exported[:] = 1
    block = site.tensor.block((sym.SymmetryLabel([]),) * 3)
    block[:] = 2
    np.testing.assert_array_equal(site.to_dense().reshape(site.shape), expected)


@pytest.mark.parametrize("complex_values", [False, True])
def test_supplied_rdms(site_data, complex_values):
    """Each RDM keyword reaches its corresponding inherited getter."""
    rng = np.random.default_rng(42)
    one = [rng.random((2, 2)) for _ in range(3)]
    two = [rng.random(16) for _ in range(4)]
    if complex_values:
        one = [values + 1j * rng.random(values.shape) for values in one]
        two = [values + 1j * rng.random(values.shape) for values in two]
    wavefunction = Wavefunction(
        MPSContainer(
            [make_site(values) for values in site_data],
            ModelOrbitals(2),
            one_rdm_spin_traced=one[0],
            one_rdm_aa=one[1],
            one_rdm_bb=one[2],
            two_rdm_spin_traced=two[0],
            two_rdm_aaaa=two[1],
            two_rdm_aabb=two[2],
            two_rdm_bbbb=two[3],
        )
    )
    assert wavefunction.has_one_rdm_spin_dependent()
    assert wavefunction.has_two_rdm_spin_dependent()
    actual_one = [wavefunction.get_active_one_rdm_spin_traced(), *wavefunction.get_active_one_rdm_spin_dependent()]
    actual_two = [wavefunction.get_active_two_rdm_spin_traced(), *wavefunction.get_active_two_rdm_spin_dependent()]
    for actual, expected in zip(actual_one + actual_two, one + two, strict=True):
        np.testing.assert_array_equal(actual, expected)


def test_invalid_constructor_arguments():
    """Native argument failures reach Python as ValueError."""
    label = sym.SymmetryLabel([])
    with pytest.raises(ValueError, match="MPS site tensor must not be null"):
        MPSSite(None, [label], [label], [label])
    with pytest.raises(ValueError, match="MPS requires nonempty sites"):
        MPSContainer([], ModelOrbitals(1))
