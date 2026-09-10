"""Integration tests for model Hamiltonian Python bindings."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from unittest.mock import patch

import numpy as np
import pytest
import scipy.sparse

from qdk_chemistry.data import Hamiltonian, LatticeGraph, LayeredPartition, QubitOperator
from qdk_chemistry.utils import model_hamiltonians
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

    @pytest.mark.parametrize("include_term_groups", [False, True])
    def test_ising_chain(self, include_term_groups: bool) -> None:
        """Check packed Ising coefficients with and without term grouping."""
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
        qh = create_ising_hamiltonian(lattice, j=j, h=h, include_term_groups=include_term_groups)
        assert isinstance(qh, QubitOperator)
        assert qh.has_sparse_terms
        assert (qh.term_partition is not None) == include_term_groups
        assert qh.num_qubits == n
        assert qh.is_hermitian()
        terms = _get_terms_dict(qh)
        assert len(terms) == len(expected)
        for pauli_str, coeff in expected.items():
            assert terms[pauli_str] == pytest.approx(coeff, abs=float_comparison_absolute_tolerance)

        # vector/matrix
        j_mat = np.ones((n, n)) * j
        h_vec = np.full(n, h)
        qh_explicit = create_ising_hamiltonian(lattice, j=j_mat, h=h_vec, include_term_groups=include_term_groups)
        assert qh_explicit.has_sparse_terms
        terms_explicit = _get_terms_dict(qh_explicit)
        assert set(terms_explicit.keys()) == set(terms.keys())
        for k in terms:
            assert terms_explicit[k] == pytest.approx(terms[k], abs=float_comparison_absolute_tolerance)

        # modify j and h
        j_mat[0, 1] = 2.5
        h_vec[2] = 0.9
        qh_modified = create_ising_hamiltonian(lattice, j=j_mat, h=h_vec, include_term_groups=include_term_groups)
        terms_mod = _get_terms_dict(qh_modified)
        assert terms_mod["IIZZ"] == pytest.approx(2.5, abs=float_comparison_absolute_tolerance)
        assert terms_mod["IZZI"] == pytest.approx(1.0, abs=float_comparison_absolute_tolerance)
        assert terms_mod["ZZII"] == pytest.approx(1.0, abs=float_comparison_absolute_tolerance)
        assert terms_mod["IIIX"] == pytest.approx(0.5, abs=float_comparison_absolute_tolerance)
        assert terms_mod["IXII"] == pytest.approx(0.9, abs=float_comparison_absolute_tolerance)

        # weighted edges
        lattice_w = LatticeGraph.chain(3, t=0.5)
        qh_w = create_ising_hamiltonian(lattice_w, j=2.0, h=1.0, include_term_groups=include_term_groups)
        terms_w = _get_terms_dict(qh_w)
        assert terms_w["IZZ"] == pytest.approx(1.0, abs=float_comparison_absolute_tolerance)
        assert terms_w["ZZI"] == pytest.approx(1.0, abs=float_comparison_absolute_tolerance)
        assert terms_w["IIX"] == pytest.approx(1.0, abs=float_comparison_absolute_tolerance)
        assert terms_w["IXI"] == pytest.approx(1.0, abs=float_comparison_absolute_tolerance)
        assert terms_w["XII"] == pytest.approx(1.0, abs=float_comparison_absolute_tolerance)

    @pytest.mark.parametrize("include_term_groups", [False, True])
    def test_heisenberg_chain(self, include_term_groups: bool) -> None:
        """Check packed Heisenberg coefficients with and without term grouping."""
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
        qh = create_heisenberg_hamiltonian(
            lattice, jx=jx, jy=jy, jz=jz, hx=hx, hy=hy, hz=hz, include_term_groups=include_term_groups
        )
        assert isinstance(qh, QubitOperator)
        assert qh.has_sparse_terms
        assert (qh.term_partition is not None) == include_term_groups
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
            lattice,
            jx=jx_mat,
            jy=jy_mat,
            jz=jz_mat,
            hx=hx_vec,
            hy=hy_vec,
            hz=hz_vec,
            include_term_groups=include_term_groups,
        )
        assert qh_explicit.has_sparse_terms
        terms_explicit = _get_terms_dict(qh_explicit)
        assert set(terms_explicit.keys()) == set(terms.keys())
        for k in terms:
            assert terms_explicit[k] == pytest.approx(terms[k], abs=float_comparison_absolute_tolerance)

        # modify jx on edge (0,1) and jz on edge (1,2)
        jx_mat[0, 1] = 2.5
        jz_mat[1, 2] = 0.3
        hx_vec[2] = 0.9
        qh_modified = create_heisenberg_hamiltonian(
            lattice,
            jx=jx_mat,
            jy=jy_mat,
            jz=jz_mat,
            hx=hx_vec,
            hy=hy_vec,
            hz=hz_vec,
            include_term_groups=include_term_groups,
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
        qh_w = create_heisenberg_hamiltonian(lattice_w, jx=1.0, jy=1.0, jz=1.0, include_term_groups=include_term_groups)
        terms_w = _get_terms_dict(qh_w)
        for pauli_char in ["X", "Y", "Z"]:
            for edge in [(0, 1), (1, 2)]:
                ps = ["I"] * 3
                ps[3 - 1 - edge[0]] = pauli_char
                ps[3 - 1 - edge[1]] = pauli_char
                assert terms_w["".join(ps)] == pytest.approx(2.0, abs=float_comparison_absolute_tolerance)


class TestSparseSpinHamiltonians:
    """Sparse assembly preserves model and grouping contracts without dense allocations."""

    def test_weighted_custom_graph(self) -> None:
        """Use upper-triangle couplings on signed edges, ignoring self-loops and non-edges."""
        graph = LatticeGraph.from_dense_matrix(np.array([[7.0, 1.5, -0.5], [1.5, -2.0, 0.0], [-0.5, 0.0, 3.0]]))
        coupling = np.array([[9.0, 99.0, 99.0], [2.0, 9.0, 99.0], [4.0, 99.0, 9.0]]).T
        actual = create_ising_hamiltonian(graph, coupling, np.array([0.25, 0.0, -0.5]))
        assert actual.has_sparse_terms
        assert actual.term_partition is None
        assert _get_terms_dict(actual) == pytest.approx({"IZZ": 3.0, "ZIZ": -2.0, "IIX": 0.25, "XII": -0.5})

    @pytest.mark.parametrize("field", [0.0, -0.5, np.float32(0.1), np.int64(2)])
    def test_zero_weight_edges_and_onsite_weights(self, field: float) -> None:
        """Zero-weight edges and self-loops leave an exact zero or field-only operator."""
        graph = LatticeGraph({(0, 0): 7.0, (0, 1): 0.0, (1, 0): 0.0, (1, 1): -2.0}, num_sites=3)
        actual = create_heisenberg_hamiltonian(graph, 1.0, 2.0, 3.0, hx=field)
        assert actual.has_sparse_terms
        assert actual.term_partition is None
        assert actual.pauli_strings == (["IIX", "IXI", "XII"] if field else ["III"])
        np.testing.assert_array_equal(actual.coefficients, [field] * (3 if field else 1))

    def test_filtered_supports_share_native_coloring(self) -> None:
        """Color each distinct support once and partition terms into disjoint, equal-axis layers."""
        graph = LatticeGraph.square(3, 2, t=0.5)
        jx = np.zeros((6, 6))
        jz = np.zeros((6, 6))
        jx[0, 1] = jx[1, 2] = jx[3, 4] = 1.0
        jz[0, 3] = jz[1, 4] = jz[2, 5] = -0.5
        jy = -2.0 * jx
        jy[0, 5] = 9.0  # A non-edge does not alter the active support.

        with patch.object(
            model_hamiltonians, "greedy_edge_coloring", wraps=model_hamiltonians.greedy_edge_coloring
        ) as colorer:
            actual = create_heisenberg_hamiltonian(graph, jx, jy, jz)
        assert colorer.call_count == 2
        assert actual.num_terms == 9
        partition = actual.term_partition
        assert isinstance(partition, LayeredPartition)
        assert partition.num_groups == 3
        assert sorted(partition.all_indices()) == list(range(actual.num_terms))
        offsets, qubits, paulis = actual.sparse_term_arrays()
        for group in partition.groups:
            axes: set[int] = set()
            for layer in group:
                occupied: set[int] = set()
                for term in layer:
                    start, end = int(offsets[term]), int(offsets[term + 1])
                    sites = qubits[start:end].tolist()
                    assert occupied.isdisjoint(sites)
                    occupied.update(sites)
                    axes.update(paulis[start:end].tolist())
            assert len(axes) == 1

    @pytest.mark.parametrize("shape", [(1, 2), (2, 1)])
    def test_native_coloring_rejects_rectangular_adjacency(self, shape: tuple[int, int]) -> None:
        """Reject nonsquare adjacency before entering the native colorer."""
        adjacency = scipy.sparse.csr_matrix(np.ones(shape))
        with pytest.raises(ValueError, match="must be square"):
            model_hamiltonians.greedy_edge_coloring(adjacency)

    @pytest.mark.parametrize("trials", [0, -1])
    def test_native_coloring_rejects_nonpositive_trials(self, trials: int) -> None:
        """Reject a trial count that cannot produce a coloring."""
        with pytest.raises(ValueError, match="must be positive"):
            model_hamiltonians.greedy_edge_coloring(scipy.sparse.eye(2, format="csr"), trials=trials)

    @pytest.mark.parametrize("include_term_groups", [False, True])
    def test_broadcast_couplings_do_not_expand(self, include_term_groups: bool) -> None:
        """Multiply a float32 broadcast view in double precision without materializing it."""
        graph = LatticeGraph.chain(40_000, t=1.1)
        coupling = np.broadcast_to(np.float32(0.1), (graph.num_sites, graph.num_sites))
        with patch.object(model_hamiltonians, "to_pair_param", side_effect=AssertionError("Dense conversion")):
            actual = create_ising_hamiltonian(graph, coupling, include_term_groups=include_term_groups)
        assert actual.has_sparse_terms
        assert actual.num_terms == graph.num_sites - 1
        np.testing.assert_array_equal(actual.coefficients, np.full(actual.num_terms, float(coupling[0, 1]) * 1.1))

    @pytest.mark.parametrize("include_term_groups", [False, True])
    def test_large_square_never_materializes_dense_data(self, include_term_groups: bool) -> None:
        """Build a 200-by-200 model with at most linear-size fields and no dense pair matrices or labels."""
        side = 200
        graph = LatticeGraph.square(side, side, t=-0.5)
        num_edges = 2 * side * (side - 1)
        get_attribute = QubitOperator.__getattribute__

        def reject_label_access(operator: QubitOperator, name: str) -> object:
            """Fail if construction requests the dense-label compatibility view."""
            if name == "pauli_strings":
                raise AssertionError("Dense labels requested")
            return get_attribute(operator, name)

        with (
            patch.object(LatticeGraph, "adjacency_matrix", side_effect=AssertionError("Dense adjacency requested")),
            patch.object(model_hamiltonians, "to_pair_param", side_effect=AssertionError("Scalar matrix expanded")),
            patch.object(QubitOperator, "__getattribute__", new=reject_label_access),
        ):
            actual = create_heisenberg_hamiltonian(
                graph, 1.0, -2.0, 3.0, hx=1.0, include_term_groups=include_term_groups
            )
            assert actual.has_sparse_terms
            assert actual.num_qubits == side * side
            assert actual.num_terms == 3 * num_edges + side * side
