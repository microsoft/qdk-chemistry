"""Integration tests for model Hamiltonian Python bindings."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import numpy as np
import pytest

from qdk_chemistry.data import Hamiltonian, LatticeGraph, QubitOperator
from qdk_chemistry.utils.model_hamiltonians import (
    create_heisenberg_hamiltonian,
    create_hubbard_hamiltonian,
    create_huckel_hamiltonian,
    create_ising_hamiltonian,
    create_ppp_hamiltonian,
    mataga_nishimoto_potential,
    ohno_potential,
)

from .reference_tolerances import float_comparison_absolute_tolerance


def _get_terms_dict(qh: QubitOperator) -> dict[str, float]:
    """Return a {pauli_string: coefficient} dict for easy assertions."""
    return dict(qh.get_real_coefficients())


class TestModelHamiltonians:
    """Integration test for model Hamiltonians via Python bindings."""

    def test_huckel_chain(self):
        n = 4
        lattice = LatticeGraph.chain(n)
        hamiltonian = create_huckel_hamiltonian(lattice, epsilon=-0.5, t=1.0)

        assert isinstance(hamiltonian, Hamiltonian)
        assert hamiltonian.has_one_body_integrals()
        assert not hamiltonian.has_two_body_integrals()

        h1_alpha, _ = hamiltonian.get_one_body_integrals()

        assert h1_alpha.shape == (n, n)

        h1_expected = np.zeros((n, n))
        np.fill_diagonal(h1_expected, -0.5)
        h1_expected[0, 1] = h1_expected[1, 0] = -1.0
        h1_expected[1, 2] = h1_expected[2, 1] = -1.0
        h1_expected[2, 3] = h1_expected[3, 2] = -1.0

        assert np.allclose(h1_alpha, h1_expected, atol=float_comparison_absolute_tolerance)

    def test_hubbard_chain(self):
        n = 4
        lattice = LatticeGraph.chain(n)
        hamiltonian = create_hubbard_hamiltonian(lattice, epsilon=-0.5, t=1.0, U=0.3)

        assert isinstance(hamiltonian, Hamiltonian)
        assert hamiltonian.has_one_body_integrals()
        assert hamiltonian.has_two_body_integrals()

        h1_alpha, _ = hamiltonian.get_one_body_integrals()
        h1_expected = np.zeros((n, n))
        np.fill_diagonal(h1_expected, -0.5)
        h1_expected[0, 1] = h1_expected[1, 0] = -1.0
        h1_expected[1, 2] = h1_expected[2, 1] = -1.0
        h1_expected[2, 3] = h1_expected[3, 2] = -1.0

        assert np.allclose(h1_alpha, h1_expected, atol=float_comparison_absolute_tolerance)
        for i in range(n):
            assert hamiltonian.get_two_body_element(i, i, i, i) == pytest.approx(
                0.3, abs=float_comparison_absolute_tolerance
            )
        assert hamiltonian.get_two_body_element(0, 0, 1, 1) == pytest.approx(
            0.0, abs=float_comparison_absolute_tolerance
        )
        assert hamiltonian.get_core_energy() == pytest.approx(0.0, abs=float_comparison_absolute_tolerance)

    def test_ppp_with_ohno_potential(self):
        n = 4
        epsilon_r = 0.9
        u_val = 0.4
        r_val = 1.2
        lattice = LatticeGraph.chain(n)
        v = ohno_potential(lattice, U=u_val, R=r_val, epsilon_r=epsilon_r)

        assert v.shape == (n, n)

        epsilon_vec = np.zeros(n)
        t_mat = np.ones((n, n))
        u_vec = np.full(n, u_val)
        z_vec = np.ones(n)
        hamiltonian = create_ppp_hamiltonian(lattice, epsilon=epsilon_vec, t=t_mat, U=u_vec, V=v, z=z_vec)

        assert isinstance(hamiltonian, Hamiltonian)
        assert hamiltonian.has_one_body_integrals()
        assert hamiltonian.has_two_body_integrals()

    def test_ppp_with_mataga_nishimoto_potential(self):
        n = 4
        epsilon_r = 0.9
        u_val = 0.4
        r_val = 1.2
        lattice = LatticeGraph.chain(n)
        v = mataga_nishimoto_potential(lattice, U=u_val, R=r_val, epsilon_r=epsilon_r)

        assert v.shape == (n, n)

        epsilon_vec = np.zeros(n)
        t_mat = np.ones((n, n))
        u_vec = np.full(n, u_val)
        z_vec = np.ones(n)
        hamiltonian = create_ppp_hamiltonian(lattice, epsilon=epsilon_vec, t=t_mat, U=u_vec, V=v, z=z_vec)

        assert isinstance(hamiltonian, Hamiltonian)
        assert hamiltonian.has_one_body_integrals()
        assert hamiltonian.has_two_body_integrals()

    def test_ising_chain(self):
        n = 4
        lattice = LatticeGraph.chain(n)
        j = 1.0
        h = 0.5

        expected = {}
        # edges
        for edge in [(0, 1), (1, 2), (2, 3)]:
            pauli = ["I"] * n
            pauli[n - 1 - edge[0]] = "Z"
            pauli[n - 1 - edge[1]] = "Z"
            expected["".join(pauli)] = j
        # sites
        for i in range(n):
            pauli = ["I"] * n
            pauli[n - 1 - i] = "X"
            expected["".join(pauli)] = h

        # scalar
        qh = create_ising_hamiltonian(lattice, j=j, h=h)
        assert isinstance(qh, QubitOperator)
        assert qh.num_qubits == n
        assert qh.is_hermitian()
        terms = _get_terms_dict(qh)
        assert len(terms) == len(expected)
        for pauli_str, coeff in expected.items():
            assert terms[pauli_str] == pytest.approx(coeff, abs=float_comparison_absolute_tolerance)

        # vector/matrix
        j_mat = np.ones((n, n)) * j
        h_vec = np.full(n, h)
        qh_explicit = create_ising_hamiltonian(lattice, j=j_mat, h=h_vec)
        terms_explicit = _get_terms_dict(qh_explicit)
        assert set(terms_explicit.keys()) == set(terms.keys())
        for k in terms:
            assert terms_explicit[k] == pytest.approx(terms[k], abs=float_comparison_absolute_tolerance)

        # modify j and h
        j_mat[0, 1] = 2.5
        h_vec[2] = 0.9
        qh_modified = create_ising_hamiltonian(lattice, j=j_mat, h=h_vec)
        terms_mod = _get_terms_dict(qh_modified)
        assert terms_mod["IIZZ"] == pytest.approx(2.5, abs=float_comparison_absolute_tolerance)
        assert terms_mod["IZZI"] == pytest.approx(1.0, abs=float_comparison_absolute_tolerance)
        assert terms_mod["ZZII"] == pytest.approx(1.0, abs=float_comparison_absolute_tolerance)
        assert terms_mod["IIIX"] == pytest.approx(0.5, abs=float_comparison_absolute_tolerance)
        assert terms_mod["IXII"] == pytest.approx(0.9, abs=float_comparison_absolute_tolerance)

        # weighted edges
        lattice_w = LatticeGraph.chain(3, t=0.5)
        qh_w = create_ising_hamiltonian(lattice_w, j=2.0, h=1.0)
        terms_w = _get_terms_dict(qh_w)
        assert terms_w["IZZ"] == pytest.approx(1.0, abs=float_comparison_absolute_tolerance)
        assert terms_w["ZZI"] == pytest.approx(1.0, abs=float_comparison_absolute_tolerance)
        assert terms_w["IIX"] == pytest.approx(1.0, abs=float_comparison_absolute_tolerance)
        assert terms_w["IXI"] == pytest.approx(1.0, abs=float_comparison_absolute_tolerance)
        assert terms_w["XII"] == pytest.approx(1.0, abs=float_comparison_absolute_tolerance)

    def test_heisenberg_chain(self):
        n = 4
        lattice = LatticeGraph.chain(n)
        edges = [(0, 1), (1, 2), (2, 3)]
        jx = 1.0
        jy = 2.0
        jz = 3.0
        hx = -1.0
        hy = -2.0
        hz = -3.0

        expected = {}
        # edges
        for edge in edges:
            for pauli_char, j_val in [("X", jx), ("Y", jy), ("Z", jz)]:
                ps = ["I"] * n
                ps[n - 1 - edge[0]] = pauli_char
                ps[n - 1 - edge[1]] = pauli_char
                expected["".join(ps)] = j_val
        # sites
        for i in range(n):
            for pauli_char, h_val in [("X", hx), ("Y", hy), ("Z", hz)]:
                ps = ["I"] * n
                ps[n - 1 - i] = pauli_char
                expected["".join(ps)] = h_val

        # scalar
        qh = create_heisenberg_hamiltonian(lattice, jx=jx, jy=jy, jz=jz, hx=hx, hy=hy, hz=hz)
        assert isinstance(qh, QubitOperator)
        assert qh.num_qubits == n
        assert qh.is_hermitian()
        terms = _get_terms_dict(qh)
        assert len(terms) == len(expected)
        for pauli_str, coeff in expected.items():
            assert terms[pauli_str] == pytest.approx(coeff, abs=float_comparison_absolute_tolerance)

        # vector/matrix
        jx_mat = np.ones((n, n)) * jx
        jy_mat = np.ones((n, n)) * jy
        jz_mat = np.ones((n, n)) * jz
        hx_vec = np.full(n, hx)
        hy_vec = np.full(n, hy)
        hz_vec = np.full(n, hz)
        qh_explicit = create_heisenberg_hamiltonian(
            lattice, jx=jx_mat, jy=jy_mat, jz=jz_mat, hx=hx_vec, hy=hy_vec, hz=hz_vec
        )
        terms_explicit = _get_terms_dict(qh_explicit)
        assert set(terms_explicit.keys()) == set(terms.keys())
        for k in terms:
            assert terms_explicit[k] == pytest.approx(terms[k], abs=float_comparison_absolute_tolerance)

        # modify jx on edge (0,1) and jz on edge (1,2)
        jx_mat[0, 1] = 2.5
        jz_mat[1, 2] = 0.3
        hx_vec[2] = 0.9
        qh_modified = create_heisenberg_hamiltonian(
            lattice, jx=jx_mat, jy=jy_mat, jz=jz_mat, hx=hx_vec, hy=hy_vec, hz=hz_vec
        )
        terms_mod = _get_terms_dict(qh_modified)
        assert terms_mod["IIXX"] == pytest.approx(2.5, abs=float_comparison_absolute_tolerance)
        assert terms_mod["IIYY"] == pytest.approx(jy, abs=float_comparison_absolute_tolerance)
        assert terms_mod["IIZZ"] == pytest.approx(jz, abs=float_comparison_absolute_tolerance)
        assert terms_mod["IXXI"] == pytest.approx(jx, abs=float_comparison_absolute_tolerance)
        assert terms_mod["IYYI"] == pytest.approx(jy, abs=float_comparison_absolute_tolerance)
        assert terms_mod["IZZI"] == pytest.approx(0.3, abs=float_comparison_absolute_tolerance)
        assert terms_mod["IIIX"] == pytest.approx(hx, abs=float_comparison_absolute_tolerance)
        assert terms_mod["IXII"] == pytest.approx(0.9, abs=float_comparison_absolute_tolerance)

        # weighted edges
        lattice_w = LatticeGraph.chain(3, t=2.0)
        qh_w = create_heisenberg_hamiltonian(lattice_w, jx=1.0, jy=1.0, jz=1.0)
        terms_w = _get_terms_dict(qh_w)
        for pauli_char in ["X", "Y", "Z"]:
            for edge in [(0, 1), (1, 2)]:
                ps = ["I"] * 3
                ps[3 - 1 - edge[0]] = pauli_char
                ps[3 - 1 - edge[1]] = pauli_char
                assert terms_w["".join(ps)] == pytest.approx(2.0, abs=float_comparison_absolute_tolerance)

    def test_heisenberg_geometric_neighbor_shells(self):
        n = 3
        j1 = 1.25
        j2 = -0.4
        custom = LatticeGraph.from_dense_matrix(np.zeros((3, 3)))
        with pytest.raises(RuntimeError, match="require lattice positions"):
            custom.mth_nearest_neighbors(1)

        graph = LatticeGraph.square(n, n, t=7.0)
        assert graph.mth_nearest_neighbors(99) == []
        with pytest.raises(ValueError, match="m must be > 0"):
            graph.mth_nearest_neighbors(0)
        with pytest.raises(TypeError):
            graph.mth_nearest_neighbors(-1)
        with pytest.raises(TypeError):
            graph.mth_nearest_neighbors(1.5)
        with pytest.raises(ValueError, match="tolerance must be positive"):
            graph.mth_nearest_neighbors(1, tolerance=np.inf)

        shells = graph.nearest_neighbor_shells([2, 1, 99])
        first_neighbors = set(shells[1])
        second_neighbors = set(shells[2])
        assert shells[99] == []
        assert (0, 1) in first_neighbors
        assert (0, 4) in second_neighbors
        assert (0, 2) not in second_neighbors

        couplings = {1: j1, 2: j2}
        hamiltonian = create_heisenberg_hamiltonian(
            graph,
            jx=couplings,
            jy=couplings,
            jz=couplings,
        )
        assert hamiltonian.term_partition is None
        terms = _get_terms_dict(hamiltonian)
        assert len(terms) == 3 * (len(first_neighbors) + len(second_neighbors))

        for neighbors, expected_coupling in [(first_neighbors, j1), (second_neighbors, j2)]:
            for i, j in neighbors:
                for pauli_char in ["X", "Y", "Z"]:
                    pauli = ["I"] * (n * n)
                    pauli[n * n - 1 - i] = pauli_char
                    pauli[n * n - 1 - j] = pauli_char
                    assert terms["".join(pauli)] == pytest.approx(
                        expected_coupling, abs=float_comparison_absolute_tolerance
                    )

        for pauli_char in ["X", "Y", "Z"]:
            axial_distance_two = ["I"] * (n * n)
            axial_distance_two[n * n - 1] = pauli_char
            axial_distance_two[n * n - 1 - 2] = pauli_char
            assert "".join(axial_distance_two) not in terms

    def test_heisenberg_ungrouped_legacy_term_order(self):
        hamiltonian = create_heisenberg_hamiltonian(
            LatticeGraph.chain(3),
            jx=1.0,
            jy=2.0,
            jz=3.0,
            hx=4.0,
            hy=5.0,
            hz=6.0,
            include_term_groups=False,
        )

        assert hamiltonian.pauli_strings == [
            "IXX",
            "IYY",
            "IZZ",
            "XXI",
            "YYI",
            "ZZI",
            "IIX",
            "IIY",
            "IIZ",
            "IXI",
            "IYI",
            "IZI",
            "XII",
            "YII",
            "ZII",
        ]
        np.testing.assert_array_equal(
            hamiltonian.coefficients,
            np.array([1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 4.0, 5.0, 6.0, 4.0, 5.0, 6.0]),
        )
        assert hamiltonian.content_hash() == "7283ee88ae88a265"

    def test_shell_mapping_order_is_deterministic(self):
        graph = LatticeGraph.square(3, 3)
        forward = {1: 1.25, 2: -0.4}
        reversed_order = {2: -0.4, 1: 1.25}

        forward_hamiltonian = create_heisenberg_hamiltonian(graph, forward, forward, forward)
        reversed_hamiltonian = create_heisenberg_hamiltonian(graph, reversed_order, reversed_order, reversed_order)

        assert reversed_hamiltonian.pauli_strings == forward_hamiltonian.pauli_strings
        np.testing.assert_array_equal(reversed_hamiltonian.coefficients, forward_hamiltonian.coefficients)
        assert reversed_hamiltonian.content_hash() == forward_hamiltonian.content_hash()
