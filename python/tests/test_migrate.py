"""Tests for the v1 -> v2 schema migration utilities (``qdk_chemistry.migrate``).

Old-schema fixtures are written by hand to mirror exactly what qdk-chemistry
<= 1.1.0 produced (matrix = list-of-rows in JSON; Eigen column-major matrices in
the ``[rows, cols]`` HDF5 datasets written by ``save_matrix_to_group``).
"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from __future__ import annotations

import json

import h5py
import numpy as np
import pytest

from qdk_chemistry import migrate
from qdk_chemistry.data import Configuration, Hamiltonian, Orbitals, Wavefunction

RNG = np.random.default_rng(20240101)


def _sym(n):
    m = RNG.standard_normal((n, n))
    return (m + m.T) / 2


# --------------------------------------------------------------------------- #
# Old-format JSON builders
# --------------------------------------------------------------------------- #
def _old_orbitals_json(nao, nmo, restricted, coeff, energies=None, active=None, ao_overlap=None):
    doc = {
        "version": "0.1.0",
        "type": "Orbitals",
        "num_atomic_orbitals": nao,
        "num_molecular_orbitals": nmo,
        "is_restricted": restricted,
        "has_overlap_matrix": ao_overlap is not None,
        "coefficients": {"alpha": coeff[0].tolist(), "beta": coeff[1].tolist()},
    }
    if energies is not None:
        doc["energies"] = {"alpha": energies[0].tolist(), "beta": energies[1].tolist()}
    if active is not None:
        doc["active_space_indices"] = {"alpha": active, "beta": active}
    if ao_overlap is not None:
        doc["ao_overlap"] = ao_overlap.tolist()
    return doc


def _old_four_center_json(container_type, restricted, h1, h2, fock, orbitals, cholesky=None):
    doc = {
        "version": "0.1.0",
        "container_type": container_type,
        "core_energy": 1.23,
        "type": "Hermitian",
        "is_restricted": restricted,
        "has_one_body_integrals": True,
        "one_body_integrals_alpha": h1[0].tolist(),
        "has_two_body_integrals": True,
        "two_body_integrals": {"aaaa": h2[0].tolist(), "aabb": h2[1].tolist(), "bbbb": h2[2].tolist()},
        "has_inactive_fock_matrix": True,
        "inactive_fock_matrix_alpha": fock[0].tolist(),
        "orbitals": orbitals,
    }
    if not restricted:
        doc["one_body_integrals_beta"] = h1[1].tolist()
        doc["inactive_fock_matrix_beta"] = fock[1].tolist()
    if cholesky is not None:
        doc["has_ao_cholesky_vectors"] = True
        doc["ao_cholesky_vectors"] = cholesky.tolist()
    return doc


# --------------------------------------------------------------------------- #
# Old-format HDF5 builders (replicate the v1.1.0 C++ layout)
# --------------------------------------------------------------------------- #
def _write_matrix(group, name, matrix):
    # save_matrix_to_group writes Eigen column-major data into a [rows, cols]
    # dataset, i.e. the row-major bytes equal the column-major flattening.
    matrix = np.asarray(matrix, dtype=np.float64)
    group.create_dataset(name, data=matrix.flatten("F").reshape(matrix.shape))


def _write_vector(group, name, vector):
    group.create_dataset(name, data=np.asarray(vector, dtype=np.float64).ravel())


def _write_old_orbitals_h5(group, nao, nmo, restricted, coeff, energies=None, active=None, ao_overlap=None):
    group.attrs["version"] = "0.1.0"
    metadata = group.create_group("metadata")
    metadata.attrs["type"] = "Orbitals"
    metadata.create_dataset("num_atomic_orbitals", data=np.uint32(nao))
    metadata.create_dataset("num_molecular_orbitals", data=np.uint32(nmo))
    metadata.create_dataset("is_restricted", data=bool(restricted))
    _write_matrix(group, "coefficients_alpha", coeff[0])
    _write_matrix(group, "coefficients_beta", coeff[1])
    if energies is not None:
        _write_vector(group, "energies_alpha", energies[0])
        _write_vector(group, "energies_beta", energies[1])
    if active is not None:
        _write_vector(group, "active_space_indices_alpha", np.array(active))
        _write_vector(group, "active_space_indices_beta", np.array(active))
    if ao_overlap is not None:
        _write_matrix(group, "ao_overlap", ao_overlap)


def _write_old_four_center_h5(group, container_type, restricted, h1, h2, fock, orb_writer, cholesky=None):
    group.attrs["version"] = "0.1.0"
    group.attrs["container_type"] = container_type
    metadata = group.create_group("metadata")
    metadata.attrs["core_energy"] = 1.23
    metadata.attrs["type"] = "Hermitian"
    metadata.attrs["is_restricted"] = bool(restricted)
    _write_matrix(group, "one_body_integrals_alpha", h1[0])
    _write_vector(group, "two_body_integrals_aaaa", h2[0])
    _write_matrix(group, "inactive_fock_matrix_alpha", fock[0])
    if not restricted:
        _write_matrix(group, "one_body_integrals_beta", h1[1])
        _write_vector(group, "two_body_integrals_aabb", h2[1])
        _write_vector(group, "two_body_integrals_bbbb", h2[2])
        _write_matrix(group, "inactive_fock_matrix_beta", fock[1])
    if cholesky is not None:
        _write_matrix(group, "ao_cholesky_vectors", cholesky)
    orb_writer(group.create_group("orbitals"))


# --------------------------------------------------------------------------- #
# Orbitals
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("fmt", ["json", "hdf5"])
def test_orbitals_restricted(tmp_path, fmt):
    nao, nmo = 4, 3
    coeff = RNG.standard_normal((nao, nmo))
    energies = RNG.standard_normal(nmo)
    overlap = _sym(nao)
    src = tmp_path / f"a.orbitals.{'json' if fmt == 'json' else 'h5'}"
    dst = tmp_path / f"a_new.orbitals.{'json' if fmt == 'json' else 'h5'}"
    if fmt == "json":
        src.write_text(
            json.dumps(_old_orbitals_json(nao, nmo, True, (coeff, coeff), (energies, energies), [0, 1, 2], overlap))
        )
    else:
        with h5py.File(src, "w") as handle:
            _write_old_orbitals_h5(handle, nao, nmo, True, (coeff, coeff), (energies, energies), [0, 1, 2], overlap)

    migrate.convert_file(src, dst)
    orb = Orbitals.from_file(str(dst), fmt)
    alpha_coeff, _ = orb.get_coefficients()
    alpha_energies, _ = orb.get_energies()
    assert np.allclose(alpha_coeff, coeff)
    assert np.allclose(alpha_energies, energies)
    assert orb.is_restricted()
    assert orb.get_num_atomic_orbitals() == nao


@pytest.mark.parametrize("fmt", ["json", "hdf5"])
def test_orbitals_unrestricted(tmp_path, fmt):
    nao, nmo = 4, 3
    ca, cb = RNG.standard_normal((nao, nmo)), RNG.standard_normal((nao, nmo))
    src = tmp_path / f"b.orbitals.{'json' if fmt == 'json' else 'h5'}"
    dst = tmp_path / f"b_new.orbitals.{'json' if fmt == 'json' else 'h5'}"
    if fmt == "json":
        src.write_text(json.dumps(_old_orbitals_json(nao, nmo, False, (ca, cb))))
    else:
        with h5py.File(src, "w") as handle:
            _write_old_orbitals_h5(handle, nao, nmo, False, (ca, cb))
    migrate.convert_file(src, dst)
    orb = Orbitals.from_file(str(dst), fmt)
    x, y = orb.get_coefficients()
    assert not orb.is_restricted()
    assert np.allclose(x, ca)
    assert np.allclose(y, cb)


# --------------------------------------------------------------------------- #
# Hamiltonian: four-center and cholesky -> four-center
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("fmt", ["json", "hdf5"])
@pytest.mark.parametrize("container_type", ["canonical_four_center", "cholesky"])
def test_hamiltonian_restricted(tmp_path, fmt, container_type):
    norb = 3
    ocoeff = RNG.standard_normal((norb, norb))
    h1, fock = _sym(norb), _sym(norb)
    h2 = RNG.standard_normal(norb**4)
    cholesky = RNG.standard_normal((norb * norb, 5)) if container_type == "cholesky" else None
    src = tmp_path / f"c.hamiltonian.{'json' if fmt == 'json' else 'h5'}"
    dst = tmp_path / f"c_new.hamiltonian.{'json' if fmt == 'json' else 'h5'}"

    if fmt == "json":
        orb = _old_orbitals_json(norb, norb, True, (ocoeff, ocoeff))
        container = _old_four_center_json(container_type, True, (h1, h1), (h2, h2, h2), (fock, fock), orb, cholesky)
        src.write_text(json.dumps({"version": "0.1.0", "container": container}))
    else:
        with h5py.File(src, "w") as handle:
            handle.attrs["version"] = "0.1.0"
            group = handle.create_group("container")
            _write_old_four_center_h5(
                group,
                container_type,
                True,
                (h1, h1),
                (h2, h2, h2),
                (fock, fock),
                lambda g: _write_old_orbitals_h5(g, norb, norb, True, (ocoeff, ocoeff)),
                cholesky,
            )

    migrate.convert_file(src, dst)
    ham = Hamiltonian.from_file(str(dst), fmt)
    # The v1 cholesky container is migrated to a four-center container.
    assert ham.get_container_type() == "canonical_four_center"
    alpha_h1, _ = ham.get_one_body_integrals()
    aaaa, aabb, bbbb = ham.get_two_body_integrals()
    assert np.allclose(alpha_h1, h1)
    assert np.allclose(aaaa, h2)
    assert np.allclose(aabb, h2)
    assert np.allclose(bbbb, h2)
    assert abs(ham.get_core_energy() - 1.23) < 1e-12


@pytest.mark.parametrize("fmt", ["json", "hdf5"])
def test_hamiltonian_unrestricted(tmp_path, fmt):
    norb = 3
    ca, cb = RNG.standard_normal((norb, norb)), RNG.standard_normal((norb, norb))
    h1a, h1b = _sym(norb), _sym(norb)
    fa, fb = _sym(norb), _sym(norb)
    aaaa, aabb, bbbb = (RNG.standard_normal(norb**4) for _ in range(3))
    src = tmp_path / f"d.hamiltonian.{'json' if fmt == 'json' else 'h5'}"
    dst = tmp_path / f"d_new.hamiltonian.{'json' if fmt == 'json' else 'h5'}"

    if fmt == "json":
        orb = _old_orbitals_json(norb, norb, False, (ca, cb))
        container = _old_four_center_json("canonical_four_center", False, (h1a, h1b), (aaaa, aabb, bbbb), (fa, fb), orb)
        src.write_text(json.dumps({"version": "0.1.0", "container": container}))
    else:
        with h5py.File(src, "w") as handle:
            handle.attrs["version"] = "0.1.0"
            group = handle.create_group("container")
            _write_old_four_center_h5(
                group,
                "canonical_four_center",
                False,
                (h1a, h1b),
                (aaaa, aabb, bbbb),
                (fa, fb),
                lambda g: _write_old_orbitals_h5(g, norb, norb, False, (ca, cb)),
            )

    migrate.convert_file(src, dst)
    ham = Hamiltonian.from_file(str(dst), fmt)
    oa, ob = ham.get_one_body_integrals()
    g_aaaa, g_aabb, g_bbbb = ham.get_two_body_integrals()
    assert np.allclose(oa, h1a)
    assert np.allclose(ob, h1b)
    assert np.allclose(g_aaaa, aaaa)
    assert np.allclose(g_aabb, aabb)
    assert np.allclose(g_bbbb, bbbb)


# --------------------------------------------------------------------------- #
# Sparse Hamiltonian
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("fmt", ["json", "hdf5"])
def test_sparse(tmp_path, fmt):
    norb = 3
    one_body = [[0, 1, 1.0], [1, 0, 1.0], [1, 2, 0.5], [2, 1, 0.5]]
    two_body = [[0, 0, 1, 1, 0.7], [0, 1, 1, 0, 0.3]]
    src = tmp_path / f"e.hamiltonian.{'json' if fmt == 'json' else 'h5'}"
    dst = tmp_path / f"e_new.hamiltonian.{'json' if fmt == 'json' else 'h5'}"

    if fmt == "json":
        container = {
            "version": "0.1.0",
            "container_type": "sparse",
            "core_energy": 2.5,
            "type": "Hermitian",
            "is_restricted": True,
            "num_orbitals": norb,
            "has_one_body_integrals": True,
            "one_body_integrals_alpha_sparse": one_body,
            "has_two_body_integrals": True,
            "two_body_integrals_sparse": two_body,
        }
        src.write_text(json.dumps({"version": "0.1.0", "container": container}))
    else:
        with h5py.File(src, "w") as handle:
            handle.attrs["version"] = "0.1.0"
            group = handle.create_group("container")
            group.attrs["version"] = "0.1.0"
            group.attrs["container_type"] = "sparse"
            metadata = group.create_group("metadata")
            metadata.attrs["core_energy"] = 2.5
            metadata.attrs["type"] = "Hermitian"
            metadata.attrs["is_restricted"] = True
            ob = group.create_dataset("one_body_integrals_alpha_sparse", data=np.array(one_body, dtype=np.float64))
            ob.attrs["num_orbitals"] = np.int32(norb)
            group.create_dataset("two_body_integrals_sparse", data=np.array(two_body, dtype=np.float64))

    migrate.convert_file(src, dst)
    ham = Hamiltonian.from_file(str(dst), fmt)
    assert ham.get_container_type() == "sparse"
    assert abs(ham.get_core_energy() - 2.5) < 1e-12
    aaaa, _, _ = ham.get_two_body_integrals()
    flat = np.asarray(aaaa).reshape(norb, norb, norb, norb)
    assert abs(flat[0, 0, 1, 1] - 0.7) < 1e-12
    assert abs(flat[0, 1, 1, 0] - 0.3) < 1e-12


# --------------------------------------------------------------------------- #
# Wavefunction containers: state_vector and amplitude variants
# --------------------------------------------------------------------------- #
def _config(occ):
    return json.loads(Configuration.from_spin_half_string(occ).to_json())


def test_wavefunction_sd(tmp_path):
    norb = 2
    coeff = np.array([[1.0, 0.2], [0.0, 0.9]])
    orb = _old_orbitals_json(norb, norb, True, (coeff, coeff), active=[0, 1])
    src = tmp_path / "a.wavefunction.json"
    dst = tmp_path / "a_new.wavefunction.json"
    doc = {
        "version": "0.1.0",
        "container_type": "sd",
        "container": {
            "version": "0.1.0",
            "container_type": "sd",
            "wavefunction_type": "self_dual",
            "orbitals": orb,
            "determinant": _config("ud"),
        },
    }
    src.write_text(json.dumps(doc))
    migrate.convert_file(src, dst)
    wf = Wavefunction.from_json_file(str(dst))
    assert wf.get_container_type() == "state_vector"


def test_wavefunction_cas_with_rdms(tmp_path):
    norb = 2
    coeff = np.array([[1.0, 0.2], [0.0, 0.9]])
    orb = _old_orbitals_json(norb, norb, True, (coeff, coeff), active=[0, 1])
    one_aa = np.array([[1.0, 0.1], [0.1, 0.5]])
    one_bb = np.array([[0.9, 0.0], [0.0, 0.4]])
    t_aaaa = np.arange(16, dtype=float)
    t_aabb = np.arange(16, dtype=float) + 100
    t_bbbb = np.arange(16, dtype=float) + 200
    container = {
        "version": "0.1.0",
        "container_type": "cas",
        "wavefunction_type": "self_dual",
        "is_complex": False,
        "coefficients": [0.9, 0.1],
        "configuration_set": {"orbitals": orb, "configurations": [_config("ud"), _config("du")]},
        "rdms": {
            "is_one_rdm_aa_complex": False,
            "one_rdm_aa": one_aa.tolist(),
            "is_one_rdm_bb_complex": False,
            "one_rdm_bb": one_bb.tolist(),
            "is_two_rdm_aaaa_complex": False,
            "two_rdm_aaaa": t_aaaa.tolist(),
            "is_two_rdm_aabb_complex": False,
            "two_rdm_aabb": t_aabb.tolist(),
            "is_two_rdm_bbbb_complex": False,
            "two_rdm_bbbb": t_bbbb.tolist(),
        },
    }
    src = tmp_path / "b.wavefunction.json"
    dst = tmp_path / "b_new.wavefunction.json"
    src.write_text(json.dumps({"version": "0.1.0", "container_type": "cas", "container": container}))
    migrate.convert_file(src, dst)
    wf = Wavefunction.from_json_file(str(dst))
    assert wf.get_container_type() == "state_vector"
    aaaa, aabb, bbbb = wf.get_active_two_rdm_spin_dependent()
    assert np.allclose(np.asarray(aaaa).ravel(), t_aaaa)
    assert np.allclose(np.asarray(aabb).ravel(), t_aabb)
    assert np.allclose(np.asarray(bbbb).ravel(), t_bbbb)
    oaa, obb = wf.get_active_one_rdm_spin_dependent()
    assert np.allclose(np.asarray(oaa), one_aa)
    assert np.allclose(np.asarray(obb), one_bb)


def _nested_sd(orb, occ):
    return {
        "version": "0.1.0",
        "container_type": "sd",
        "container": {
            "version": "0.1.0",
            "container_type": "sd",
            "wavefunction_type": "self_dual",
            "orbitals": orb,
            "determinant": _config(occ),
        },
    }


def test_wavefunction_mp2(tmp_path):
    norb = 2
    coeff = np.eye(norb)
    orb = _old_orbitals_json(norb, norb, True, (coeff, coeff), active=[0, 1])
    container = {"version": "0.1.0", "type": "mp2", "orbitals": orb, "wavefunction": _nested_sd(orb, "ud")}
    src = tmp_path / "c.wavefunction.json"
    dst = tmp_path / "c_new.wavefunction.json"
    src.write_text(json.dumps({"version": "0.1.0", "container_type": "mp2", "container": container}))
    migrate.convert_file(src, dst)
    wf = Wavefunction.from_json_file(str(dst))
    assert wf.get_container_type() == "amplitude"


def test_wavefunction_cc(tmp_path):
    norb = 4
    coeff = np.eye(norb)
    orb = _old_orbitals_json(norb, norb, True, (coeff, coeff), active=[0, 1, 2, 3])
    t1 = (np.arange(3, dtype=float) * 0.1).tolist()
    t2 = (np.arange(9, dtype=float) * 0.01).tolist()
    container = {
        "version": "0.1.0",
        "container_type": "coupled_cluster",
        "is_complex": False,
        "orbitals": orb,
        "wavefunction": _nested_sd(orb, "ud00"),
        "t1_amplitudes_aa": t1,
        "t1_amplitudes_bb": t1,
        "t2_amplitudes_abab": t2,
        "t2_amplitudes_aaaa": t2,
        "t2_amplitudes_bbbb": t2,
    }
    src = tmp_path / "d.wavefunction.json"
    dst = tmp_path / "d_new.wavefunction.json"
    src.write_text(json.dumps({"version": "0.1.0", "container_type": "coupled_cluster", "container": container}))
    migrate.convert_file(src, dst)
    wf = Wavefunction.from_json_file(str(dst))
    assert wf.get_container_type() == "amplitude"


# --------------------------------------------------------------------------- #
# Deserializers reject old files and point at the converter
# --------------------------------------------------------------------------- #
def test_old_file_rejected_with_guidance(tmp_path):
    old = tmp_path / "x.orbitals.json"
    old.write_text(json.dumps(_old_orbitals_json(1, 1, True, (np.ones((1, 1)), np.ones((1, 1))))))
    with pytest.raises(RuntimeError, match=r"qdk_chemistry\.migrate") as excinfo:
        Orbitals.from_json_file(str(old))
    assert "qdk_chemistry.migrate" in str(excinfo.value)


def test_unknown_type_raises(tmp_path):
    bad = tmp_path / "x.unknown.json"
    bad.write_text("{}")
    with pytest.raises(migrate.MigrationError):
        migrate.convert_file(bad, tmp_path / "y.unknown.json")
