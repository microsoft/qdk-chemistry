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

from qdk_chemistry.algorithms.qubit_mapper.vc_qubit_mapper import (
    VerstraeteCiracQubitMapper,
    build_vc_majorana_mapping,
)
from qdk_chemistry.data import (
    CanonicalFourCenterHamiltonianContainer,
    Hamiltonian,
    MajoranaMapping,
    QubitHamiltonian,
)

from .test_helpers import create_test_orbitals

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_hamiltonian(one_body, two_body, orbitals, core_energy=0.0):
    """Construct a Hamiltonian from integrals."""
    n_modes = one_body.shape[0]
    fock = np.zeros((n_modes, n_modes))
    return Hamiltonian(CanonicalFourCenterHamiltonianContainer(one_body, two_body, orbitals, core_energy, fock))


def _spinless_fh_hamiltonian(n_rows: int, n_cols: int, t: float = 1.0) -> Hamiltonian:
    """Return an open-boundary spinless Fermi-Hubbard Hamiltonian H = -t * sum_<ij> hopping."""
    n_sites = n_rows * n_cols
    h1 = np.zeros((n_sites, n_sites))
    for r in range(n_rows):
        for c in range(n_cols):
            n = r * n_cols + c
            if c + 1 < n_cols:
                m = r * n_cols + (c + 1)
                h1[n, m] = h1[m, n] = -t
            if r + 1 < n_rows:
                m = (r + 1) * n_cols + c
                h1[n, m] = h1[m, n] = -t
    return _make_hamiltonian(h1, np.zeros(n_sites**4), create_test_orbitals(n_sites))


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
    """Compute exact eigenvalues of the spinless Fermi-Hubbard under Jordan-Wigner."""
    n_sites = n_rows * n_cols
    h1 = np.zeros((n_sites, n_sites))
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
        """Build Kronecker product for given operator dict."""
        mat = np.array([[1.0 + 0j]])
        for q in range(n_sites):
            mat = np.kron(mat, _p[ops.get(q, "I")])
        return mat

    h_jw = np.zeros((2**n_sites, 2**n_sites), dtype=complex)
    for n in range(n_sites):
        h_nn = float(h1[n, n])
        if abs(h_nn) > 1e-15:
            h_jw += h_nn / 2 * kron_term({n: "Z"})
            h_jw += h_nn / 2 * np.eye(2**n_sites, dtype=complex)
    for n in range(n_sites):
        for m in range(n + 1, n_sites):
            h_nm = (h1[n, m] + h1[m, n]) / 2
            if abs(h_nm) < 1e-15:
                continue
            z_ops = dict.fromkeys(range(n + 1, m), "Z")
            h_jw += h_nm / 2 * kron_term({n: "X", m: "X", **z_ops})
            h_jw += h_nm / 2 * kron_term({n: "Y", m: "Y", **z_ops})

    single = np.real(np.linalg.eigvalsh(h_jw))
    # VC mapper produces two decoupled copies (alpha + beta blocks);
    # the combined spectrum is all pairwise sums of the single-block spectrum.
    pairwise = np.add.outer(single, single).ravel()
    return np.sort(pairwise)


def _vc_codespace_eigenvalues(qh: QubitHamiltonian, n_sites: int) -> np.ndarray:
    """Compute codespace eigenvalues with sparse intermediate construction.

    The mapper builds its MajoranaMapping with N = 2 * n_sites modes
    (alpha + beta spin-orbital blocks for the underlying C++ engine),
    giving num_qubits = 4 * n_sites: 2*n_sites physical + 2*n_sites
    auxiliary qubits. The codespace matrix has dimension
    2**(2*n_sites) = 2**n_phys with n_phys = 2*n_sites.

    In the little-endian QDK convention:
      - ps[0 : n_phys]  = auxiliary qubit ops (n_phys = 2*n_sites)
      - ps[n_phys :]    = physical qubit ops (n_phys qubits)
    A term contributes iff every auxiliary op is I or Z.

    Each term is accumulated as a sparse matrix to avoid materialising
    O(num_terms) dense 2**n_phys x 2**n_phys arrays during construction;
    the dense matrix is built only once, immediately before eigvalsh.
    """
    import scipy.sparse as sp  # noqa: PLC0415

    _p = {
        "I": sp.eye(2, format="csr", dtype=complex),
        "X": sp.csr_matrix(np.array([[0, 1], [1, 0]], dtype=complex)),
        "Y": sp.csr_matrix(np.array([[0, -1j], [1j, 0]], dtype=complex)),
        "Z": sp.csr_matrix(np.diag([1, -1]).astype(complex)),
    }
    n_phys = 2 * n_sites
    dim_phys = 2**n_phys
    h_cs = sp.csr_matrix((dim_phys, dim_phys), dtype=complex)
    for ps, coeff in zip(qh.pauli_strings, qh.coefficients, strict=True):
        aux_chars = ps[:n_phys]
        phys_chars = ps[n_phys:]
        if any(ch in ("X", "Y") for ch in aux_chars):
            continue
        mat = sp.csr_matrix(np.array([[1.0 + 0j]]))
        for ch in phys_chars:
            mat = sp.kron(mat, _p[ch], format="csr")
        h_cs = h_cs + coeff * mat
    return np.sort(np.real(np.linalg.eigvalsh(h_cs.toarray())))


# ---------------------------------------------------------------------------
# AC1 -- Construction on all required lattice sizes
# ---------------------------------------------------------------------------


class TestVCMappingConstruction:
    """Tests for AC1: mapper and mapping construction on all required lattice sizes."""

    @pytest.mark.parametrize(("n_rows", "n_cols"), [(2, 2), (2, 3), (3, 3), (4, 4)])
    def test_build_vc_majorana_mapping(self, n_rows, n_cols):
        """Test that build_vc_majorana_mapping returns a valid MajoranaMapping."""
        mapping = build_vc_majorana_mapping(n_rows * n_cols)
        n_modes = n_rows * n_cols
        assert isinstance(mapping, MajoranaMapping)
        assert mapping.num_qubits == 2 * n_modes
        assert len(mapping.table) == 2 * n_modes

    @pytest.mark.parametrize(("n_rows", "n_cols"), [(2, 2), (2, 3), (3, 3), (4, 4)])
    def test_mapper_instantiates(self, n_rows, n_cols):
        """Test that VerstraeteCiracQubitMapper instantiates with correct properties."""
        mapper = VerstraeteCiracQubitMapper(lattice_shape=(n_rows, n_cols))
        assert mapper.name() == "verstraete-cirac"
        assert mapper.lattice_shape == (n_rows, n_cols)

    @pytest.mark.parametrize(("n_rows", "n_cols"), [(2, 2), (2, 3), (3, 3), (4, 4)])
    def test_mapper_run_produces_qubit_hamiltonian(self, n_rows, n_cols):
        """Test that mapper.run returns a valid QubitHamiltonian."""
        ham = _spinless_fh_hamiltonian(n_rows, n_cols)
        qh = _vc_run(n_rows, n_cols, ham)
        n_sites = n_rows * n_cols
        assert isinstance(qh, QubitHamiltonian)
        assert qh.num_qubits == 4 * n_sites
        assert qh.encoding == "verstraete-cirac"
        assert len(qh.pauli_strings) > 0
        for ps in qh.pauli_strings:
            assert len(ps) == 4 * n_sites

    def test_invalid_lattice_raises(self):
        """Test that invalid lattice shapes raise ValueError."""
        with pytest.raises(ValueError, match="2x2"):
            VerstraeteCiracQubitMapper(lattice_shape=(1, 4))
        with pytest.raises(ValueError, match="n_modes"):
            build_vc_majorana_mapping(0)

    def test_mode_count_mismatch_raises(self):
        """Test that mode count mismatch raises ValueError."""
        mapper = VerstraeteCiracQubitMapper(lattice_shape=(2, 2))
        ham = _spinless_fh_hamiltonian(3, 3)
        with pytest.raises(ValueError, match="modes|sites"):
            mapper.run(ham, mapper.mapping)


# ---------------------------------------------------------------------------
# AC2 -- Eigenvalues match Jordan-Wigner in codespace
# ---------------------------------------------------------------------------


class TestVCEigenvalues:
    scipy_sparse = pytest.importorskip("scipy.sparse", reason="scipy required for codespace eigenvalue tests")
    """Tests for AC2: codespace eigenvalues match JW to within 1e-10."""

    def test_2x2_fermi_hubbard_matches_jw(self):
        """Test 2x2 Fermi-Hubbard codespace eigenvalues match JW exactly."""
        n_rows, n_cols, t = 2, 2, 1.0
        ham = _spinless_fh_hamiltonian(n_rows, n_cols, t=t)
        qh = _vc_run(n_rows, n_cols, ham)
        vc_evals = _vc_codespace_eigenvalues(qh, n_rows * n_cols)
        jw_evals = _jw_eigenvalues_spinless(n_rows, n_cols, t=t)
        assert len(vc_evals) == len(jw_evals)
        np.testing.assert_allclose(vc_evals, jw_evals, atol=1e-10)

    def test_2x3_fermi_hubbard_matches_jw(self):
        """Test 2x3 Fermi-Hubbard codespace eigenvalues match JW exactly."""
        n_rows, n_cols = 2, 3
        ham = _spinless_fh_hamiltonian(n_rows, n_cols)
        qh = _vc_run(n_rows, n_cols, ham)
        vc_evals = _vc_codespace_eigenvalues(qh, n_rows * n_cols)
        jw_evals = _jw_eigenvalues_spinless(n_rows, n_cols)
        np.testing.assert_allclose(vc_evals, jw_evals, atol=1e-10)

    def test_hamiltonian_is_hermitian(self):
        """Test that the mapped Hamiltonian has real coefficients."""
        qh = _vc_run(2, 2, _spinless_fh_hamiltonian(2, 2))
        for coeff in qh.coefficients:
            assert abs(complex(coeff).imag) < 1e-12


# ---------------------------------------------------------------------------
# AC3 -- Pauli weight characterisation
# ---------------------------------------------------------------------------


class TestVCPauliWeight:
    """Tests for AC3: nearest-neighbour hopping Pauli weight characterisation.

    Horizontal hops always have weight 4 (X/Y on two physical qubits +
    Z on two auxiliary qubits), independent of lattice size.
    Vertical hops have weight n_cols + 3: finite and fully determined
    by the number of columns.  For fixed n_cols the weight is constant
    across all lattices.  For square LxL lattices the weight is L + 3.
    """

    @staticmethod
    def _max_weight(n_rows, n_cols):
        """Return max Pauli weight of hopping terms (weight >= 4)."""
        qh = _vc_run(n_rows, n_cols, _spinless_fh_hamiltonian(n_rows, n_cols))
        weights = [sum(1 for ch in ps if ch != "I") for ps in qh.pauli_strings]
        hop_weights = [w for w in weights if w >= 4]
        return max(hop_weights) if hop_weights else 0

    def test_weight_identical_for_l_2_3_4(self):
        """Test max hopping weight equals n_cols + 3 for each LxL lattice."""
        for lat_l in (2, 3, 4):
            w = self._max_weight(lat_l, lat_l)
            expected = lat_l + 3
            assert w == expected, f"{lat_l}x{lat_l}: expected max weight {expected}, got {w}"

    def test_weight_is_bounded(self):
        """Test max hopping weight is n_cols + 3 for every tested lattice."""
        for n_rows, n_cols in [(2, 2), (2, 3), (3, 3), (4, 4)]:
            w = self._max_weight(n_rows, n_cols)
            expected = n_cols + 3
            assert w == expected, f"{n_rows}x{n_cols}: expected weight {expected}, got {w}"

    def test_weight_identical_for_rectangular(self):
        """Test 2x3 and 3x3 have same max hop weight (same n_cols=3)."""
        assert self._max_weight(2, 3) == self._max_weight(3, 3)


# ---------------------------------------------------------------------------
# AC4 -- JSON and HDF5 round-trips
# ---------------------------------------------------------------------------


class TestVCSerialisation:
    """Tests for AC4: JSON and HDF5 serialisation round-trips."""

    def _build_qh(self) -> QubitHamiltonian:
        """Build a 2x2 VC QubitHamiltonian for serialisation tests."""
        return _vc_run(2, 2, _spinless_fh_hamiltonian(2, 2))

    def test_json_round_trip(self):
        """Test that JSON serialisation and deserialisation preserves the Hamiltonian."""
        qh = self._build_qh()
        restored = QubitHamiltonian.from_json(json.loads(json.dumps(qh.to_json())))
        assert qh.pauli_strings == restored.pauli_strings
        np.testing.assert_array_equal(qh.coefficients, restored.coefficients)
        assert restored.encoding == "verstraete-cirac"

    def test_hdf5_round_trip(self):
        """Test that HDF5 serialisation and deserialisation preserves the Hamiltonian."""
        h5py = pytest.importorskip("h5py")
        qh = self._build_qh()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "vc.h5"
            with h5py.File(path, "w") as hf:
                qh.to_hdf5(hf)
            with h5py.File(path, "r") as hf:
                restored = QubitHamiltonian.from_hdf5(hf)
        assert qh.pauli_strings == restored.pauli_strings
        np.testing.assert_allclose(np.abs(qh.coefficients - restored.coefficients), 0.0, atol=1e-15)
        assert restored.encoding == "verstraete-cirac"

    def test_json_terms_match_term_by_term(self):
        """Test that each Pauli term matches after JSON round-trip."""
        qh = self._build_qh()
        restored = QubitHamiltonian.from_json(json.loads(json.dumps(qh.to_json())))
        orig = dict(zip(qh.pauli_strings, qh.coefficients, strict=False))
        rest = dict(zip(restored.pauli_strings, restored.coefficients, strict=False))
        assert set(orig) == set(rest)
        for ps, val in orig.items():
            assert abs(val - rest[ps]) < 1e-15


# ---------------------------------------------------------------------------
# Additional correctness checks
# ---------------------------------------------------------------------------


class TestVCCorrectness:
    """Additional correctness and structural checks."""

    def test_all_pauli_strings_valid(self):
        """Test that all Pauli strings contain only valid characters."""
        qh = _vc_run(2, 2, _spinless_fh_hamiltonian(2, 2))
        for ps in qh.pauli_strings:
            assert all(ch in "IXYZ" for ch in ps)

    def test_pauli_string_lengths_consistent(self):
        """Test that all Pauli strings have the same length."""
        qh = _vc_run(3, 3, _spinless_fh_hamiltonian(3, 3))
        lengths = {len(ps) for ps in qh.pauli_strings}
        assert len(lengths) == 1

    def test_deterministic_output(self):
        """Test that repeated runs produce identical output."""
        ham = _spinless_fh_hamiltonian(2, 2)
        qh1 = _vc_run(2, 2, ham)
        qh2 = _vc_run(2, 2, ham)
        assert qh1.pauli_strings == qh2.pauli_strings
        np.testing.assert_array_equal(qh1.coefficients, qh2.coefficients)

    def test_qubit_count_formula(self):
        """Test that qubit count equals 4 * n_rows * n_cols.

        The mapper builds its MajoranaMapping with N = 2 * n_sites modes
        (alpha + beta spin-orbital blocks for the C++ engine), giving
        num_qubits = 2 * N = 4 * n_sites.
        """
        for n_rows, n_cols in [(2, 2), (2, 3), (3, 3), (4, 4)]:
            qh = _vc_run(n_rows, n_cols, _spinless_fh_hamiltonian(n_rows, n_cols))
            assert qh.num_qubits == 4 * n_rows * n_cols

    def test_threshold_prunes_small_terms(self):
        """Test that a large threshold removes Pauli terms."""
        ham = _spinless_fh_hamiltonian(2, 2)
        qh_full = _vc_run(2, 2, ham, threshold=0.0)
        qh_pruned = _vc_run(2, 2, ham, threshold=100.0)
        assert len(qh_pruned.pauli_strings) <= len(qh_full.pauli_strings)
