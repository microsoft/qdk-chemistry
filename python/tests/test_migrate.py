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
import pathlib

import h5py
import numpy as np
import pytest

from qdk_chemistry import migrate
from qdk_chemistry.data import Ansatz, Configuration, Hamiltonian, Orbitals, Wavefunction
from qdk_chemistry.migrate import _wavefunction

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
        # The v1 HDF5 sparse format packs index pairs into the bytes of a double
        # (uint32[2] -> double), so one-body rows are [pack(row, col), value] and
        # two-body rows are [pack(p, q), pack(r, s), value].
        def pack(a, b):
            return np.array([a, b], dtype=np.uint32).view(np.float64)[0]

        ob_packed = np.array([[pack(r, c), v] for r, c, v in one_body], dtype=np.float64)
        tb_packed = np.array([[pack(p, q), pack(r, s), v] for p, q, r, s, v in two_body], dtype=np.float64)
        with h5py.File(src, "w") as handle:
            handle.attrs["version"] = "0.1.0"
            group = handle.create_group("container")
            group.attrs["version"] = "0.1.0"
            group.attrs["container_type"] = "sparse"
            metadata = group.create_group("metadata")
            metadata.attrs["core_energy"] = 2.5
            metadata.attrs["type"] = "Hermitian"
            metadata.attrs["is_restricted"] = True
            ob = group.create_dataset("one_body_integrals_alpha_sparse", data=ob_packed)
            ob.attrs["num_orbitals"] = np.int32(norb)
            group.create_dataset("two_body_integrals_sparse", data=tb_packed)

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


# --------------------------------------------------------------------------- #
# Ground-truth regression tests against real H2/STO-3G files produced by
# qdk-chemistry 1.0.0 (see test_data/migrate/README.md for provenance).
# --------------------------------------------------------------------------- #
_REAL_DATA = pathlib.Path(__file__).parent / "test_data" / "migrate"


def _spot_ref():
    return json.loads((_REAL_DATA / "reference.json").read_text())


@pytest.mark.parametrize("out_fmt", ["json", "hdf5"])
def test_real_orbitals_with_basis_set(tmp_path, out_fmt):
    ref = _spot_ref()["orbitals"]
    dst = tmp_path / f"h2.orbitals.{'json' if out_fmt == 'json' else 'h5'}"
    migrate.convert_file(_REAL_DATA / "h2_v1_0_0.orbitals.h5", dst)
    orb = Orbitals.from_file(str(dst), out_fmt)
    assert orb.has_basis_set()
    assert orb.get_num_molecular_orbitals() == ref["nmo"]
    coeff = np.asarray(orb.get_coefficients()[0])
    assert abs(coeff[0, 0] - ref["coeff_00"]) < 1e-9
    assert abs(coeff[0, 1] - ref["coeff_01"]) < 1e-9


@pytest.mark.parametrize("out_fmt", ["json", "hdf5"])
def test_real_hamiltonian(tmp_path, out_fmt):
    ref = _spot_ref()["hamiltonian"]
    dst = tmp_path / f"h2.hamiltonian.{'json' if out_fmt == 'json' else 'h5'}"
    migrate.convert_file(_REAL_DATA / "h2_v1_0_0.hamiltonian.h5", dst)
    ham = Hamiltonian.from_file(str(dst), out_fmt)
    assert ham.get_container_type() == "canonical_four_center"
    assert abs(ham.get_core_energy() - ref["core"]) < 1e-9
    one_body = np.asarray(ham.get_one_body_integrals()[0])
    assert abs(one_body[0, 0] - ref["one_body_00"]) < 1e-9
    aaaa = np.asarray(ham.get_two_body_integrals()[0]).ravel()
    assert np.allclose(aaaa[:4], ref["two_body_aaaa_head"])


@pytest.mark.parametrize("out_fmt", ["json", "hdf5"])
def test_real_cholesky_to_four_center(tmp_path, out_fmt):
    # The v1 cholesky container stored the full four-center two-body tensor (not
    # MO three-center vectors), so it migrates to a four-center container.
    ref = _spot_ref()["cholesky"]
    dst = tmp_path / f"chol.hamiltonian.{'json' if out_fmt == 'json' else 'h5'}"
    migrate.convert_file(_REAL_DATA / "h2_v1_1_0_chol.hamiltonian.h5", dst)
    ham = Hamiltonian.from_file(str(dst), out_fmt)
    assert ham.get_container_type() == "canonical_four_center"
    assert abs(ham.get_core_energy() - ref["core"]) < 1e-9
    assert abs(np.asarray(ham.get_one_body_integrals()[0])[0, 0] - ref["one_body_00"]) < 1e-9
    aaaa = np.asarray(ham.get_two_body_integrals()[0]).ravel()
    assert np.allclose(aaaa[:4], ref["two_body_aaaa_head"])


@pytest.mark.parametrize("out_fmt", ["json", "hdf5"])
def test_real_sparse(tmp_path, out_fmt):
    ref = _spot_ref()["sparse"]
    dst = tmp_path / f"sparse.hamiltonian.{'json' if out_fmt == 'json' else 'h5'}"
    migrate.convert_file(_REAL_DATA / "h2_v1_1_0_sparse.hamiltonian.h5", dst)
    ham = Hamiltonian.from_file(str(dst), out_fmt)
    assert ham.get_container_type() == "sparse"
    assert abs(ham.get_core_energy() - ref["core"]) < 1e-9
    norb = 4
    two_body = np.asarray(ham.get_two_body_integrals()[0]).reshape(norb, norb, norb, norb)
    assert abs(two_body[0, 0, 0, 0] - ref["two_body_0000"]) < 1e-9
    assert abs(two_body[1, 1, 1, 1] - ref["two_body_1111"]) < 1e-9
    one_body = np.asarray(ham.get_one_body_integrals()[0]).reshape(norb, norb)
    assert abs(one_body[0, 1] - ref["one_body_01"]) < 1e-9


@pytest.mark.parametrize("out_fmt", ["json", "hdf5"])
def test_real_cas_wavefunction_rdms(tmp_path, out_fmt):
    ref = _spot_ref()["cas_rdm"]
    dst = tmp_path / f"h2_cas.wavefunction.{'json' if out_fmt == 'json' else 'h5'}"
    migrate.convert_file(_REAL_DATA / "h2_v1_0_0_cas_rdm.wavefunction.h5", dst)
    wf = Wavefunction.from_file(str(dst), out_fmt)
    assert wf.get_container_type() == "state_vector"
    aaaa, aabb, bbbb = wf.get_active_two_rdm_spin_dependent()
    aaaa, aabb, bbbb = np.asarray(aaaa).ravel(), np.asarray(aabb).ravel(), np.asarray(bbbb).ravel()
    assert np.allclose(aaaa[:4], ref["two_aaaa_head"])
    assert np.allclose(aabb[:4], ref["two_aabb_head"])
    # Closed-shell file omits the beta channels; they alias the alpha channels.
    assert np.allclose(bbbb, aaaa)
    oaa, obb = wf.get_active_one_rdm_spin_dependent()
    assert np.allclose(np.asarray(oaa)[0], ref["one_aa_row0"])
    assert np.allclose(np.asarray(obb), np.asarray(oaa))


@pytest.mark.parametrize("out_fmt", ["json", "hdf5"])
def test_real_ansatz(tmp_path, out_fmt):
    # Ansatz embeds a Hamiltonian and a Wavefunction; both must be migrated.
    ref = _spot_ref()["ansatz"]
    dst = tmp_path / f"h2.ansatz.{'json' if out_fmt == 'json' else 'h5'}"
    migrate.convert_file(_REAL_DATA / "h2_v1_0_0.ansatz.h5", dst)
    ansatz = Ansatz.from_file(str(dst), out_fmt)
    assert ansatz.get_wavefunction().get_container_type() == ref["wfn_container"]
    one_body = np.asarray(ansatz.get_hamiltonian().get_one_body_integrals()[0])
    assert abs(one_body[0, 0] - ref["ham_one_body_00"]) < 1e-9


def test_cas_restricted_rdm_only_alpha(tmp_path):
    # A closed-shell CAS file stores only the alpha/opposite-spin RDM channels.
    norb = 2
    coeff = np.array([[1.0, 0.2], [0.0, 0.9]])
    orb = _old_orbitals_json(norb, norb, True, (coeff, coeff), active=[0, 1])
    one_aa = np.array([[1.0, 0.1], [0.1, 0.5]])
    t_aaaa = np.arange(16, dtype=float)
    t_aabb = np.arange(16, dtype=float) + 50
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
            "is_two_rdm_aaaa_complex": False,
            "two_rdm_aaaa": t_aaaa.tolist(),
            "is_two_rdm_aabb_complex": False,
            "two_rdm_aabb": t_aabb.tolist(),
        },
    }
    src = tmp_path / "r.wavefunction.json"
    dst = tmp_path / "r_new.wavefunction.json"
    src.write_text(json.dumps({"version": "0.1.0", "container_type": "cas", "container": container}))
    migrate.convert_file(src, dst)
    wf = Wavefunction.from_json_file(str(dst))
    aaaa, aabb, bbbb = wf.get_active_two_rdm_spin_dependent()
    assert np.allclose(np.asarray(aaaa).ravel(), t_aaaa)
    assert np.allclose(np.asarray(aabb).ravel(), t_aabb)
    assert np.allclose(np.asarray(bbbb).ravel(), t_aaaa)  # beta aliases alpha


# --------------------------------------------------------------------------- #
# Regression tests for PR review fixes: reject already-migrated inputs and read
# the determinant orbital capacity / statistics from the stored HDF5 attributes.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("fixture", "type_token"),
    [
        ("h2_v1_0_0.orbitals.h5", "orbitals"),
        ("h2_v1_0_0.hamiltonian.h5", "hamiltonian"),
        ("h2_v1_0_0_cas_rdm.wavefunction.h5", "wavefunction"),
        ("h2_v1_0_0.ansatz.h5", "ansatz"),
    ],
)
def test_already_v2_input_raises(tmp_path, fixture, type_token):
    v2_file = tmp_path / f"v2.{type_token}.json"
    migrate.convert_file(_REAL_DATA / fixture, v2_file)
    with pytest.raises(migrate.MigrationError):
        migrate.convert_file(v2_file, tmp_path / f"again.{type_token}.json")


def test_already_v2_wavefunction_does_not_drop_rdms(tmp_path):
    # The v2 schema renamed the active-RDM fields; re-running the converter on a
    # v2 file would silently drop them, so it must fail fast instead.
    v2_file = tmp_path / "v2.wavefunction.json"
    migrate.convert_file(_REAL_DATA / "h2_v1_0_0_cas_rdm.wavefunction.h5", v2_file)
    assert "rdms" in json.loads(v2_file.read_text())["container"]
    with pytest.raises(migrate.MigrationError, match="current schema"):
        migrate.convert_file(v2_file, tmp_path / "again.wavefunction.json")


def test_sd_determinant_capacity_read_from_attribute(tmp_path):
    # "ud0000" (6 orbitals) packs into 2 bytes; inferring capacity as bytes * 4
    # would over-decode to 8 modes. The orbital_capacity attribute is authoritative.
    path = tmp_path / "det.h5"
    with h5py.File(path, "w") as handle:
        dataset = handle.create_dataset("configuration", data=np.array([0x09, 0x00], dtype=np.uint8))
        dataset.attrs["orbital_capacity"] = 6
    with h5py.File(path, "r") as handle:
        decoded = _wavefunction._decode_configuration_dataset(handle["configuration"])
    assert decoded["configuration"] == "ud0000"


def test_sd_determinant_spinless_bits_per_mode(tmp_path):
    # A spinless determinant stores bits_per_mode=1; the decoder must honor it.
    path = tmp_path / "det.h5"
    with h5py.File(path, "w") as handle:
        dataset = handle.create_dataset("configuration", data=np.array([0b00000101], dtype=np.uint8))
        dataset.attrs["orbital_capacity"] = 3
        dataset.attrs["bits_per_mode"] = 1
    with h5py.File(path, "r") as handle:
        decoded = _wavefunction._decode_configuration_dataset(handle["configuration"])
    assert decoded["bits_per_mode"] == 1
    assert decoded["configuration"] == "101"
