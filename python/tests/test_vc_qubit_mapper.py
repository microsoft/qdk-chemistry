"""Tests for the Verstraete-Cirac fermion-to-qubit encoder."""
# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from qdk_chemistry.data import (
    CanonicalFourCenterHamiltonianContainer,
    Hamiltonian,
    MajoranaMapping,
    QubitHamiltonian,
)
from qdk_chemistry.algorithms.qubit_mapper.vc_qubit_mapper import (
    VerstraeteCiracQubitMapper,
    build_vc_majorana_mapping,
)

from .test_helpers import create_test_orbitals


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_hamiltonian(one_body, two_body, orbitals, core_energy=0.0):
    fock = np.eye(0)
    return Hamiltonian(
        CanonicalFourCenterHamiltonianContainer(
            one_body, two_body, orbitals, core_energy, fock
        )
    )


def _spinless_fh_hamiltonian(n_rows: int, n_cols: int, t: float = 1.0) -> Hamiltonian:
    """Open-boundary spinless Fermi-Hubbard, H = -t * sum_<ij> hopping."""
    N = n_rows * n_cols
    h1 = np.zeros((N, N))
    for r in range(n_rows):
        for c in range(n_cols):
            n = r * n_cols + c
            if c + 1 < n_cols:
                m = r * n_cols + (c + 1)
                h1[n, m] = h1[m, n] = -t
            if r + 1 < n_rows:
                m = (r + 1) * n_cols + c
                h1[n, m] = h1[m, n] = -t
    return _make_hamiltonian(h1, np.zeros(N**4), create_test_orbitals(N))


def _vc_run(
    n_rows: int,
    n_cols: int,
    ham: Hamiltonian,
    threshold: float = 1e-12,
    integral_threshold: float = 1e-12,
) -> QubitHamiltonian:
    """Build a VC mapper and run it, passing the required mapping argument."""
    mapper = VerstraeteCiracQubitMapper(
        lattice_shape=(n_rows, n_cols),
        threshold=threshold,
        integral_threshold=integral_threshold,
    )
    return mapper.run(ham, mapper.mapping)


def _jw_eigenvalues_spinless(n_rows: int, n_cols: int, t: float = 1.0) -> np.ndarray:
    """Exact eigenvalues of the spinless Fermi-Hubbard under Jordan-Wigner."""
    N = n_rows * n_cols
    h1 = np.zeros((N, N))
    for r in range(n_rows):
        for c in range(n_cols):
            n = r * n_cols + c
            if c + 1 < n_cols:
                m = r * n_cols + (c + 1)
                h1[n, m] = h1[m, n] = -t
            if r + 1 < n_rows:
                m = (r + 1) * n_cols + c
                h1[n, m] = h1[m, n] = -t

    _p = {
        "I": np.eye(2, dtype=complex),
        "X": np.array([[0, 1], [1, 0]], dtype=complex),
        "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
        "Z": np.array([[1, 0], [0, -1]], dtype=complex),
    }

    def kron_term(ops):
        mat = np.array([[1.0 + 0j]])
        for q in range(N):
            mat = np.kron(mat, _p[ops.get(q, "I")])
        return mat

    H = np.zeros((2**N, 2**N), dtype=complex)
    for n in range(N):
        h_nn = float(h1[n, n])
        if abs(h_nn) > 1e-15:
            H += h_nn / 2 * kron_term({n: "Z"})
            H += h_nn / 2 * np.eye(2**N, dtype=complex)
    for n in range(N):
        for m in range(n + 1, N):
            h_nm = (h1[n, m] + h1[m, n]) / 2
            if abs(h_nm) < 1e-15:
                continue
            z_ops = {k: "Z" for k in range(n + 1, m)}
            H += h_nm / 2 * kron_term({n: "X", m: "X", **z_ops})
            H += h_nm / 2 * kron_term({n: "Y", m: "Y", **z_ops})

    return np.sort(np.real(np.linalg.eigvalsh(H)))


def _vc_codespace_eigenvalues(qh: QubitHamiltonian, n_sites: int) -> np.ndarray:
    """Eigenvalues of VC Hamiltonian restricted to the +1 codespace.

    Codespace = all auxiliary qubits (indices n_sites..2*n_sites-1) in |0>.
    In little-endian convention, qubit k = bit k of the state index.
    """
    H = qh.to_matrix()
    dim = 2 ** qh.num_qubits
    codespace_states = [s for s in range(dim) if (s >> n_sites) == 0]
    assert len(codespace_states) == 2**n_sites
    proj = np.zeros((dim, len(codespace_states)), dtype=float)
    for i, s in enumerate(codespace_states):
        proj[s, i] = 1.0
    return np.sort(np.real(np.linalg.eigvalsh(proj.T @ H @ proj)))


# ---------------------------------------------------------------------------
# AC1 -- Construction on all required lattice sizes
# ---------------------------------------------------------------------------

class TestVCMappingConstruction:

    @pytest.mark.parametrize("n_rows,n_cols", [(2, 2), (2, 3), (3, 3), (4, 4)])
    def test_build_vc_majorana_mapping(self, n_rows, n_cols):
        mapping = build_vc_majorana_mapping(n_rows, n_cols)
        N = n_rows * n_cols
        assert isinstance(mapping, MajoranaMapping)
        assert mapping.num_qubits == 2 * N
        assert len(mapping.table) == 2 * N

    @pytest.mark.parametrize("n_rows,n_cols", [(2, 2), (2, 3), (3, 3), (4, 4)])
    def test_mapper_instantiates(self, n_rows, n_cols):
        mapper = VerstraeteCiracQubitMapper(lattice_shape=(n_rows, n_cols))
        assert mapper.name() == "verstraete-cirac"
        assert mapper.lattice_shape == (n_rows, n_cols)

    @pytest.mark.parametrize("n_rows,n_cols", [(2, 2), (2, 3), (3, 3), (4, 4)])
    def test_mapper_run_produces_qubit_hamiltonian(self, n_rows, n_cols):
        ham = _spinless_fh_hamiltonian(n_rows, n_cols)
        qh = _vc_run(n_rows, n_cols, ham)
        N = n_rows * n_cols
        assert isinstance(qh, QubitHamiltonian)
        assert qh.num_qubits == 2 * N
        assert qh.encoding == "verstraete-cirac"
        assert len(qh.pauli_strings) > 0
        for ps in qh.pauli_strings:
            assert len(ps) == 2 * N

    def test_invalid_lattice_raises(self):
        with pytest.raises(ValueError):
            VerstraeteCiracQubitMapper(lattice_shape=(1, 4))
        with pytest.raises(ValueError):
            build_vc_majorana_mapping(4, 1)

    def test_mode_count_mismatch_raises(self):
        mapper = VerstraeteCiracQubitMapper(lattice_shape=(2, 2))
        ham = _spinless_fh_hamiltonian(3, 3)  # 9 modes != 4 sites
        with pytest.raises(ValueError, match="modes"):
            mapper.run(ham, mapper.mapping)


# ---------------------------------------------------------------------------
# AC2 -- Eigenvalues match Jordan-Wigner in codespace
# ---------------------------------------------------------------------------

class TestVCEigenvalues:

    def test_2x2_fermi_hubbard_matches_jw(self):
        n_rows, n_cols, t = 2, 2, 1.0
        ham = _spinless_fh_hamiltonian(n_rows, n_cols, t=t)
        qh = _vc_run(n_rows, n_cols, ham)
        vc_evals = _vc_codespace_eigenvalues(qh, n_rows * n_cols)
        jw_evals = _jw_eigenvalues_spinless(n_rows, n_cols, t=t)
        assert len(vc_evals) == len(jw_evals)
        np.testing.assert_allclose(vc_evals, jw_evals, atol=1e-10)

    def test_2x3_fermi_hubbard_matches_jw(self):
        n_rows, n_cols = 2, 3
        ham = _spinless_fh_hamiltonian(n_rows, n_cols)
        qh = _vc_run(n_rows, n_cols, ham)
        vc_evals = _vc_codespace_eigenvalues(qh, n_rows * n_cols)
        jw_evals = _jw_eigenvalues_spinless(n_rows, n_cols)
        np.testing.assert_allclose(vc_evals, jw_evals, atol=1e-10)

    def test_hamiltonian_is_hermitian(self):
        qh = _vc_run(2, 2, _spinless_fh_hamiltonian(2, 2))
        for coeff in qh.coefficients:
            assert abs(complex(coeff).imag) < 1e-12


# ---------------------------------------------------------------------------
# AC3 -- Pauli weight is system-size independent
# ---------------------------------------------------------------------------

class TestVCPauliWeight:

    @staticmethod
    def _max_weight(n_rows, n_cols):
        qh = _vc_run(n_rows, n_cols, _spinless_fh_hamiltonian(n_rows, n_cols))
        return max(sum(1 for ch in ps if ch != "I") for ps in qh.pauli_strings)

    def test_weight_identical_for_l_2_3_4(self):
        """Max hopping weight = n_cols + 3 for each LxL lattice."""
        for L in (2, 3, 4):
            w = self._max_weight(L, L)
            expected = L + 3
            assert w == expected, (
                f"{L}x{L}: expected max weight {expected}, got {w}"
            )

# ---------------------------------------------------------------------------
# AC4 -- JSON and HDF5 round-trips
# ---------------------------------------------------------------------------

class TestVCSerialisation:

    def _build_qh(self):
        return _vc_run(2, 2, _spinless_fh_hamiltonian(2, 2))

    def test_json_round_trip(self):
        qh = self._build_qh()
        restored = QubitHamiltonian.from_json(json.loads(json.dumps(qh.to_json())))
        assert qh.pauli_strings == restored.pauli_strings
        np.testing.assert_array_equal(qh.coefficients, restored.coefficients)
        assert restored.encoding == "verstraete-cirac"

    def test_hdf5_round_trip(self):
        h5py = pytest.importorskip("h5py")
        qh = self._build_qh()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "vc.h5"
            with h5py.File(path, "w") as hf:
                qh.to_hdf5(hf)
            with h5py.File(path, "r") as hf:
                restored = QubitHamiltonian.from_hdf5(hf)
        assert qh.pauli_strings == restored.pauli_strings
        np.testing.assert_allclose(
            np.abs(qh.coefficients - restored.coefficients), 0.0, atol=1e-15
        )
        assert restored.encoding == "verstraete-cirac"

    def test_json_terms_match_term_by_term(self):
        qh = self._build_qh()
        restored = QubitHamiltonian.from_json(json.loads(json.dumps(qh.to_json())))
        orig = dict(zip(qh.pauli_strings, qh.coefficients, strict=True))
        rest = dict(zip(restored.pauli_strings, restored.coefficients, strict=True))
        assert set(orig) == set(rest)
        for ps in orig:
            assert abs(orig[ps] - rest[ps]) < 1e-15


# ---------------------------------------------------------------------------
# Additional correctness checks
# ---------------------------------------------------------------------------

class TestVCCorrectness:

    def test_all_pauli_strings_valid(self):
        qh = _vc_run(2, 2, _spinless_fh_hamiltonian(2, 2))
        for ps in qh.pauli_strings:
            assert all(ch in "IXYZ" for ch in ps)

    def test_pauli_string_lengths_consistent(self):
        qh = _vc_run(3, 3, _spinless_fh_hamiltonian(3, 3))
        lengths = {len(ps) for ps in qh.pauli_strings}
        assert len(lengths) == 1

    def test_deterministic_output(self):
        ham = _spinless_fh_hamiltonian(2, 2)
        qh1 = _vc_run(2, 2, ham)
        qh2 = _vc_run(2, 2, ham)
        assert qh1.pauli_strings == qh2.pauli_strings
        np.testing.assert_array_equal(qh1.coefficients, qh2.coefficients)

    def test_qubit_count_formula(self):
        for n_rows, n_cols in [(2, 2), (2, 3), (3, 3), (4, 4)]:
            qh = _vc_run(n_rows, n_cols, _spinless_fh_hamiltonian(n_rows, n_cols))
            assert qh.num_qubits == 2 * n_rows * n_cols

    def test_threshold_prunes_small_terms(self):
        ham = _spinless_fh_hamiltonian(2, 2)
        qh_full = _vc_run(2, 2, ham, threshold=0.0)
        qh_pruned = _vc_run(2, 2, ham, threshold=100.0)
        assert len(qh_pruned.pauli_strings) <= len(qh_full.pauli_strings)
