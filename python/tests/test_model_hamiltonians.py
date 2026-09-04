"""Integration tests for model Hamiltonian Python bindings."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import numpy as np
import pytest

from qdk_chemistry.data import (
    BondFlavorDefinition,
    Hamiltonian,
    LatticeGraph,
    QubitOperator,
)
from qdk_chemistry.utils.model_hamiltonians import (
    KitaevBondFlavor,
    create_heisenberg_hamiltonian,
    create_hubbard_hamiltonian,
    create_huckel_hamiltonian,
    create_ising_hamiltonian,
    create_kitaev_hamiltonian,
    create_ppp_hamiltonian,
    kitaev_honeycomb_bond_flavors,
    mataga_nishimoto_potential,
    ohno_potential,
)

from .reference_tolerances import float_comparison_absolute_tolerance


def _get_terms_dict(qh: QubitOperator) -> dict[str, float]:
    """Return a {pauli_string: coefficient} dict for easy assertions."""
    return dict(qh.get_real_coefficients())


def _with_standard_kitaev_flavors(graph: LatticeGraph) -> LatticeGraph:
    """Attach the standard honeycomb Kitaev flavor IDs for inspection."""
    return graph.with_bond_flavors(kitaev_honeycomb_bond_flavors())


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

    def test_honeycomb_positions_define_one_cell_shells(self):
        unit_cell = LatticeGraph.honeycomb(1, 1)
        assert unit_cell.num_sites == 2
        assert unit_cell.num_edges == 1

        graph = LatticeGraph.honeycomb_plaquettes(1, 1)
        positions = graph.positions
        assert positions.shape == (6, 2)

        shells = graph.nearest_neighbor_shells([1, 2, 3])
        for shell, expected_distance in ((1, 1.0), (2, np.sqrt(3.0)), (3, 2.0)):
            distances = [np.linalg.norm(positions[site_j] - positions[site_i]) for site_i, site_j in shells[shell]]
            assert distances == pytest.approx(
                [expected_distance] * len(distances),
                abs=float_comparison_absolute_tolerance,
            )

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

    def test_kitaev_nearest_neighbor_components(self):
        graph = LatticeGraph.honeycomb(2, 2, periodic_x=True, periodic_y=True)
        flavored_graph = _with_standard_kitaev_flavors(graph)
        cases = (
            (
                {"j": 4.0},
                {
                    KitaevBondFlavor.X: {"XX": 1.0, "YY": 1.0, "ZZ": 1.0},
                    KitaevBondFlavor.Y: {"XX": 1.0, "YY": 1.0, "ZZ": 1.0},
                    KitaevBondFlavor.Z: {"XX": 1.0, "YY": 1.0, "ZZ": 1.0},
                },
            ),
            (
                {"kx": 8.0, "ky": 12.0, "kz": 16.0},
                {
                    KitaevBondFlavor.X: {"XX": 2.0},
                    KitaevBondFlavor.Y: {"YY": 3.0},
                    KitaevBondFlavor.Z: {"ZZ": 4.0},
                },
            ),
            (
                {"gamma": 20.0},
                {
                    KitaevBondFlavor.X: {"YZ": 5.0, "ZY": 5.0},
                    KitaevBondFlavor.Y: {"XZ": 5.0, "ZX": 5.0},
                    KitaevBondFlavor.Z: {"XY": 5.0, "YX": 5.0},
                },
            ),
            (
                {"gamma_prime": 24.0},
                {
                    KitaevBondFlavor.X: {"XY": 6.0, "YX": 6.0, "XZ": 6.0, "ZX": 6.0},
                    KitaevBondFlavor.Y: {"XY": 6.0, "YX": 6.0, "YZ": 6.0, "ZY": 6.0},
                    KitaevBondFlavor.Z: {"XZ": 6.0, "ZX": 6.0, "YZ": 6.0, "ZY": 6.0},
                },
            ),
        )

        connections = flavored_graph.neighbor_connections([1])
        for overrides, expected_by_flavor in cases:
            parameters = {"kx": 0.0, "ky": 0.0, "kz": 0.0, "j": 0.0, "gamma": 0.0, "gamma_prime": 0.0}
            parameters.update(overrides)
            hamiltonian = create_kitaev_hamiltonian(graph, **parameters)
            assert hamiltonian.term_partition is not None
            terms = _get_terms_dict(hamiltonian)
            expected: dict[str, float] = {}
            for connection in connections:
                assert connection.flavor is not None
                for pauli_label, coefficient in expected_by_flavor[KitaevBondFlavor(connection.flavor)].items():
                    pauli = ["I"] * graph.num_sites
                    pauli[graph.num_sites - 1 - connection.site_i] = pauli_label[0]
                    pauli[graph.num_sites - 1 - connection.site_j] = pauli_label[1]
                    expected["".join(pauli)] = coefficient
            assert terms == pytest.approx(expected, abs=float_comparison_absolute_tolerance)

    def test_kitaev_flavored_geometric_shells(self):
        graph = LatticeGraph.honeycomb(5, 5)
        flavored_graph = _with_standard_kitaev_flavors(graph)
        shell_couplings = {1: 4.0, 2: 8.0, 3: 12.0}
        zero_couplings = {1: 0.0, 2: 0.0, 3: 0.0}
        hamiltonian = create_kitaev_hamiltonian(
            graph,
            kx=shell_couplings,
            ky=zero_couplings,
            kz=zero_couplings,
        )
        assert hamiltonian.term_partition is None
        terms = _get_terms_dict(hamiltonian)
        expected: dict[str, float] = {}
        for connection in flavored_graph.neighbor_connections([1, 2, 3]):
            if connection.flavor != KitaevBondFlavor.X:
                continue
            pauli = ["I"] * graph.num_sites
            pauli[graph.num_sites - 1 - connection.site_i] = "X"
            pauli[graph.num_sites - 1 - connection.site_j] = "X"
            expected["".join(pauli)] = shell_couplings[connection.bond_class.shell] / 4.0
        assert terms == pytest.approx(expected, abs=float_comparison_absolute_tolerance)

        heisenberg = create_kitaev_hamiltonian(
            graph,
            kx=zero_couplings,
            ky=zero_couplings,
            kz=zero_couplings,
            j=shell_couplings,
        )
        heisenberg_terms = _get_terms_dict(heisenberg)
        expected_heisenberg: dict[str, float] = {}
        for connection in flavored_graph.neighbor_connections([1, 2, 3]):
            for component in "XYZ":
                pauli = ["I"] * graph.num_sites
                pauli[graph.num_sites - 1 - connection.site_i] = component
                pauli[graph.num_sites - 1 - connection.site_j] = component
                expected_heisenberg["".join(pauli)] = shell_couplings[connection.bond_class.shell] / 4.0
        assert heisenberg_terms == pytest.approx(expected_heisenberg, abs=float_comparison_absolute_tolerance)

    def test_geometric_flavors_are_optional_and_configurable(self):
        square = LatticeGraph.square(3, 3)
        assert all(connection.flavor is None for connection in square.neighbor_connections([1]))

        flavored = square.with_bond_flavors(
            [
                BondFlavorDefinition(shell=1, axis=np.array([1.0, 0.0]), flavor=1000),
                BondFlavorDefinition(shell=1, axis=np.array([0.0, 1.0]), flavor=1001),
                BondFlavorDefinition(shell=2, axis=np.array([1.0, 1.0]), flavor=1002),
                BondFlavorDefinition(shell=2, axis=np.array([1.0, -1.0]), flavor=1003),
            ]
        )
        connections = flavored.neighbor_connections([1, 2])
        assert {connection.flavor for connection in connections} == {1000, 1001, 1002, 1003}

    def test_kitaev_flavor_specific_gamma_terms(self):
        graph = LatticeGraph.honeycomb(3, 3)
        flavored_graph = _with_standard_kitaev_flavors(graph)
        gamma = {KitaevBondFlavor.X: 4.0, KitaevBondFlavor.Y: 8.0, KitaevBondFlavor.Z: 12.0}
        gamma_prime = {KitaevBondFlavor.X: 16.0, KitaevBondFlavor.Y: 20.0, KitaevBondFlavor.Z: 24.0}
        hamiltonian = create_kitaev_hamiltonian(
            graph,
            kx=0.0,
            ky=0.0,
            kz=0.0,
            gamma_x=gamma[KitaevBondFlavor.X],
            gamma_y=gamma[KitaevBondFlavor.Y],
            gamma_z=gamma[KitaevBondFlavor.Z],
            gamma_prime_x=gamma_prime[KitaevBondFlavor.X],
            gamma_prime_y=gamma_prime[KitaevBondFlavor.Y],
            gamma_prime_z=gamma_prime[KitaevBondFlavor.Z],
            include_term_groups=False,
        )
        terms = _get_terms_dict(hamiltonian)
        flavor_index = {KitaevBondFlavor.X: 0, KitaevBondFlavor.Y: 1, KitaevBondFlavor.Z: 2}
        components = "XYZ"
        expected: dict[str, float] = {}
        for connection in flavored_graph.neighbor_connections([1]):
            assert connection.flavor is not None
            flavor = KitaevBondFlavor(connection.flavor)
            selected = flavor_index[flavor]
            other = tuple(index for index in range(3) if index != selected)
            matrix = np.zeros((3, 3))
            matrix[other[0], other[1]] = matrix[other[1], other[0]] = gamma[flavor] / 4.0
            for index in other:
                matrix[selected, index] = matrix[index, selected] = gamma_prime[flavor] / 4.0
            for first in range(3):
                for second in range(3):
                    if matrix[first, second] == 0.0:
                        continue
                    pauli = ["I"] * graph.num_sites
                    pauli[graph.num_sites - 1 - connection.site_i] = components[first]
                    pauli[graph.num_sites - 1 - connection.site_j] = components[second]
                    expected["".join(pauli)] = matrix[first, second]
        assert terms == pytest.approx(expected, abs=float_comparison_absolute_tolerance)

    def test_kitaev_crystallographic_magnetic_field(self):
        graph = LatticeGraph.honeycomb(2, 2)
        field_abc = np.array([2.0, -3.0, 5.0])
        g_abc = np.array([1.5, 2.0, 0.5])
        bohr_magneton = 0.25
        transform = np.array(
            [
                [1 / np.sqrt(6), 1 / np.sqrt(6), -2 / np.sqrt(6)],
                [-1 / np.sqrt(2), 1 / np.sqrt(2), 0.0],
                [1 / np.sqrt(3), 1 / np.sqrt(3), 1 / np.sqrt(3)],
            ]
        )

        def expected_field_terms(coefficients: np.ndarray) -> dict[str, float]:
            expected: dict[str, float] = {}
            for site in range(graph.num_sites):
                for component, coefficient in zip("XYZ", coefficients, strict=True):
                    pauli = ["I"] * graph.num_sites
                    pauli[graph.num_sites - 1 - site] = component
                    expected["".join(pauli)] = coefficient
            return expected

        cubic = create_kitaev_hamiltonian(
            graph,
            0.0,
            0.0,
            0.0,
            magnetic_field_abc=field_abc,
            g_factors_abc=g_abc,
            bohr_magneton=bohr_magneton,
            crystallographic_transform=transform,
            include_term_groups=False,
        )
        cubic_coefficients = bohr_magneton * transform.T @ (g_abc * field_abc) / 2.0
        assert _get_terms_dict(cubic) == pytest.approx(
            expected_field_terms(cubic_coefficients), abs=float_comparison_absolute_tolerance
        )

        crystallographic = create_kitaev_hamiltonian(
            graph,
            0.0,
            0.0,
            0.0,
            magnetic_field_abc=field_abc,
            g_factors_abc=g_abc,
            bohr_magneton=bohr_magneton,
            crystallographic_transform=transform,
            spin_basis_transform=transform,
            include_term_groups=False,
        )
        abc_coefficients = bohr_magneton * g_abc * field_abc / 2.0
        assert _get_terms_dict(crystallographic) == pytest.approx(
            expected_field_terms(abc_coefficients), abs=float_comparison_absolute_tolerance
        )

        identity_crystal_frame = create_kitaev_hamiltonian(
            graph,
            0.0,
            0.0,
            0.0,
            magnetic_field_abc=field_abc,
            g_factors_abc=g_abc,
            bohr_magneton=bohr_magneton,
            crystallographic_transform=np.eye(3),
            include_term_groups=False,
        )
        assert _get_terms_dict(identity_crystal_frame) == pytest.approx(
            expected_field_terms(abc_coefficients), abs=float_comparison_absolute_tolerance
        )

        grouped = create_kitaev_hamiltonian(
            graph,
            0.0,
            0.0,
            0.0,
            magnetic_field_abc=field_abc,
            g_factors_abc=g_abc,
            bohr_magneton=bohr_magneton,
            crystallographic_transform=transform,
        )
        assert grouped.term_partition is not None
        assert _get_terms_dict(grouped) == pytest.approx(
            expected_field_terms(cubic_coefficients), abs=float_comparison_absolute_tolerance
        )

    def test_kitaev_field_matches_explicit_crystallographic_spin_operators(self):
        field_abc = np.array([2.0, -3.0, 5.0])
        g_abc = np.array([1.5, 2.0, 0.5])
        bohr_magneton = 0.25
        transform = np.array(
            [
                [1 / np.sqrt(6), 1 / np.sqrt(6), -2 / np.sqrt(6)],
                [-1 / np.sqrt(2), 1 / np.sqrt(2), 0.0],
                [1 / np.sqrt(3), 1 / np.sqrt(3), 1 / np.sqrt(3)],
            ]
        )
        spin_xyz = np.array(
            [
                [[0.0, 0.5], [0.5, 0.0]],
                [[0.0, -0.5j], [0.5j, 0.0]],
                [[0.5, 0.0], [0.0, -0.5]],
            ],
            dtype=complex,
        )
        spin_abc = np.einsum("ai,ijk->ajk", transform, spin_xyz)
        expected = bohr_magneton * np.einsum("a,aij->ij", g_abc * field_abc, spin_abc)

        actual = create_kitaev_hamiltonian(
            LatticeGraph.chain(1),
            0.0,
            0.0,
            0.0,
            magnetic_field_abc=field_abc,
            g_factors_abc=g_abc,
            bohr_magneton=bohr_magneton,
            crystallographic_transform=transform,
            include_term_groups=False,
        )

        np.testing.assert_allclose(actual.to_matrix(), expected, atol=1e-15)

    def test_kitaev_crystallographic_transform(self):
        graph = LatticeGraph.honeycomb(3, 3)
        j = 1.1
        k = -2.3
        gamma = 0.7
        gamma_prime = -0.2
        transform = np.array(
            [
                [1 / np.sqrt(6), 1 / np.sqrt(6), -2 / np.sqrt(6)],
                [-1 / np.sqrt(2), 1 / np.sqrt(2), 0.0],
                [1 / np.sqrt(3), 1 / np.sqrt(3), 1 / np.sqrt(3)],
            ]
        )
        actual = _get_terms_dict(
            create_kitaev_hamiltonian(
                graph,
                kx=k,
                ky=k,
                kz=k,
                j=j,
                gamma=gamma,
                gamma_prime=gamma_prime,
                spin_basis_transform=transform,
                include_term_groups=False,
            )
        )

        coefficient_a = k / 3 + 2 * (gamma - gamma_prime) / 3
        coefficient_b = k / 3 - (gamma - gamma_prime) / 3
        j_ab = j + coefficient_b - gamma_prime
        j_c = j + coefficient_a + 2 * gamma_prime
        phases = {
            KitaevBondFlavor.Z: 0.0,
            KitaevBondFlavor.X: 2 * np.pi / 3,
            KitaevBondFlavor.Y: 4 * np.pi / 3,
        }
        expected: dict[str, float] = {}
        components = ("X", "Y", "Z")
        for connection in _with_standard_kitaev_flavors(graph).neighbor_connections([1]):
            assert connection.flavor is not None
            flavor = KitaevBondFlavor(connection.flavor)
            cosine = np.cos(phases[flavor])
            sine = np.sin(phases[flavor])
            exchange = np.array(
                [
                    [j_ab + coefficient_a * cosine, -coefficient_a * sine, -np.sqrt(2) * coefficient_b * cosine],
                    [-coefficient_a * sine, j_ab - coefficient_a * cosine, -np.sqrt(2) * coefficient_b * sine],
                    [-np.sqrt(2) * coefficient_b * cosine, -np.sqrt(2) * coefficient_b * sine, j_c],
                ]
            )
            exchange[np.abs(exchange) < 100 * np.finfo(float).eps * np.max(np.abs(exchange))] = 0.0
            for first_index, first in enumerate(components):
                for second_index, second in enumerate(components):
                    coefficient = exchange[first_index, second_index] / 4.0
                    if coefficient == 0.0:
                        continue
                    pauli = ["I"] * graph.num_sites
                    pauli[graph.num_sites - 1 - connection.site_i] = first
                    pauli[graph.num_sites - 1 - connection.site_j] = second
                    expected["".join(pauli)] = coefficient
        assert actual == pytest.approx(expected, abs=float_comparison_absolute_tolerance)

    def test_kitaev_crystallographic_transform_preserves_spin_algebra(self):
        transform = np.array(
            [
                [1 / np.sqrt(6), 1 / np.sqrt(6), -2 / np.sqrt(6)],
                [-1 / np.sqrt(2), 1 / np.sqrt(2), 0.0],
                [1 / np.sqrt(3), 1 / np.sqrt(3), 1 / np.sqrt(3)],
            ]
        )
        spin_xyz = np.array(
            [
                [[0.0, 0.5], [0.5, 0.0]],
                [[0.0, -0.5j], [0.5j, 0.0]],
                [[0.5, 0.0], [0.0, -0.5]],
            ],
            dtype=complex,
        )
        spin_abc = np.einsum("ai,ijk->ajk", transform, spin_xyz)

        np.testing.assert_allclose(transform @ transform.T, np.eye(3), atol=1e-15)
        assert np.linalg.det(transform) == pytest.approx(1.0, abs=1e-15)
        for first, second, third in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
            commutator = spin_abc[first] @ spin_abc[second] - spin_abc[second] @ spin_abc[first]
            np.testing.assert_allclose(commutator, 1.0j * spin_abc[third], atol=1e-15)

    def test_kitaev_accumulates_periodic_flavor_collisions(self):
        graph = LatticeGraph.honeycomb(2, 2, periodic_x=True, periodic_y=True)
        flavored_graph = _with_standard_kitaev_flavors(graph)
        couplings = {KitaevBondFlavor.X: 4.0, KitaevBondFlavor.Y: 8.0, KitaevBondFlavor.Z: 12.0}
        hamiltonian = create_kitaev_hamiltonian(
            graph,
            kx={3: couplings[KitaevBondFlavor.X]},
            ky={3: couplings[KitaevBondFlavor.Y]},
            kz={3: couplings[KitaevBondFlavor.Z]},
        )
        terms = _get_terms_dict(hamiltonian)
        expected: dict[str, float] = {}
        pauli_for_flavor = {KitaevBondFlavor.X: "X", KitaevBondFlavor.Y: "Y", KitaevBondFlavor.Z: "Z"}
        for connection in flavored_graph.neighbor_connections([3]):
            assert connection.flavor is not None
            flavor = KitaevBondFlavor(connection.flavor)
            pauli_char = pauli_for_flavor[flavor]
            pauli = ["I"] * graph.num_sites
            pauli[graph.num_sites - 1 - connection.site_i] = pauli_char
            pauli[graph.num_sites - 1 - connection.site_j] = pauli_char
            pauli_string = "".join(pauli)
            expected[pauli_string] = expected.get(pauli_string, 0.0) + couplings[flavor] / 4.0
        assert terms == pytest.approx(expected, abs=float_comparison_absolute_tolerance)

    def test_kitaev_requires_flavored_connections(self):
        with pytest.raises(ValueError, match="requires X, Y, or Z flavor IDs"):
            create_kitaev_hamiltonian(LatticeGraph.square(3, 3), kx=1.0, ky=1.0, kz=1.0)

        unsupported = LatticeGraph.square(3, 3).with_bond_flavors([BondFlavorDefinition(1, np.array([1.0, 0.0]), 1000)])
        with pytest.raises(ValueError, match="requires X, Y, or Z flavor IDs"):
            create_kitaev_hamiltonian(unsupported, kx=1.0, ky=1.0, kz=1.0)

        graph = LatticeGraph.honeycomb(3, 3)
        invalid_transforms = (
            (np.eye(2), "shape"),
            (np.array([[np.nan, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]), "finite"),
            (np.diag([2.0, 1.0, 1.0]), "orthogonal"),
            (np.diag([-1.0, 1.0, 1.0]), "right-handed"),
        )
        for transform, message in invalid_transforms:
            with pytest.raises(ValueError, match=message):
                create_kitaev_hamiltonian(graph, 1.0, 1.0, 1.0, spin_basis_transform=transform)
            with pytest.raises(ValueError, match=message):
                create_kitaev_hamiltonian(
                    graph,
                    0.0,
                    0.0,
                    0.0,
                    magnetic_field_abc=np.ones(3),
                    crystallographic_transform=transform,
                )

        with pytest.raises(ValueError, match="orthogonal"):
            create_kitaev_hamiltonian(
                graph,
                0.0,
                0.0,
                0.0,
                crystallographic_transform=np.diag([2.0, 1.0, 1.0]),
            )

        with pytest.raises(ValueError, match="shape"):
            create_kitaev_hamiltonian(graph, 0.0, 0.0, 0.0, magnetic_field_abc=np.zeros(2))
        with pytest.raises(ValueError, match="crystallographic_transform"):
            create_kitaev_hamiltonian(graph, 0.0, 0.0, 0.0, magnetic_field_abc=np.ones(3))
        with pytest.raises(ValueError, match="finite"):
            create_kitaev_hamiltonian(graph, 0.0, 0.0, 0.0, g_factors_abc=np.array([1.0, np.nan, 1.0]))
        with pytest.raises(ValueError, match="bohr_magneton"):
            create_kitaev_hamiltonian(graph, 0.0, 0.0, 0.0, bohr_magneton=np.inf)

    def test_kitaev_explicit_flavors_override_defaults(self):
        graph = LatticeGraph.honeycomb_plaquettes(1, 1)
        flavor_swap = {
            KitaevBondFlavor.X: KitaevBondFlavor.Z,
            KitaevBondFlavor.Y: KitaevBondFlavor.Y,
            KitaevBondFlavor.Z: KitaevBondFlavor.X,
        }
        swapped = graph.with_bond_flavors(
            [
                BondFlavorDefinition(
                    definition.shell,
                    definition.axis,
                    flavor_swap[KitaevBondFlavor(definition.flavor)],
                )
                for definition in kitaev_honeycomb_bond_flavors()
            ]
        )

        actual = _get_terms_dict(create_kitaev_hamiltonian(swapped, 4.0, 0.0, 0.0, include_term_groups=False))
        expected = {}
        for connection in swapped.neighbor_connections([1]):
            if connection.flavor != KitaevBondFlavor.X:
                continue
            pauli = ["I"] * graph.num_sites
            pauli[graph.num_sites - 1 - connection.site_i] = "X"
            pauli[graph.num_sites - 1 - connection.site_j] = "X"
            expected["".join(pauli)] = 1.0

        assert actual == expected

    def test_kitaev_without_interactions_does_not_require_geometry(self):
        graph = LatticeGraph.from_dense_matrix(np.zeros((2, 2)))
        zero = create_kitaev_hamiltonian(graph, 0.0, 0.0, 0.0)
        np.testing.assert_array_equal(zero.to_matrix(), np.zeros((4, 4)))

        field = create_kitaev_hamiltonian(
            graph,
            0.0,
            0.0,
            0.0,
            magnetic_field_abc=(0.0, 1.0, 0.0),
            crystallographic_transform=np.eye(3),
            include_term_groups=False,
        )
        assert field.num_qubits == 2
        assert field.is_hermitian()

        with pytest.raises(RuntimeError, match="require lattice positions"):
            create_kitaev_hamiltonian(graph, {1: 1.0}, 0.0, 0.0)

    def test_kitaev_shell_mapping_order_is_deterministic(self):
        graph = LatticeGraph.honeycomb(3, 3)
        forward = {1: 1.0, 2: -0.5, 3: 0.25}
        reversed_order = {3: 0.25, 2: -0.5, 1: 1.0}
        forward_hamiltonian = create_kitaev_hamiltonian(graph, forward, forward, forward)
        reversed_hamiltonian = create_kitaev_hamiltonian(graph, reversed_order, reversed_order, reversed_order)
        assert reversed_hamiltonian.pauli_strings == forward_hamiltonian.pauli_strings
        np.testing.assert_array_equal(reversed_hamiltonian.coefficients, forward_hamiltonian.coefficients)
        assert reversed_hamiltonian.content_hash() == forward_hamiltonian.content_hash()

    def test_kitaev_open_hexagon_matches_explicit_matrix(self):
        graph = LatticeGraph.honeycomb_plaquettes(1, 1)
        bonds = {
            (1, KitaevBondFlavor.X): {(0, 1), (4, 5)},
            (1, KitaevBondFlavor.Y): {(0, 3), (2, 5)},
            (1, KitaevBondFlavor.Z): {(1, 2), (3, 4)},
            (2, KitaevBondFlavor.X): {(0, 4), (1, 5)},
            (2, KitaevBondFlavor.Y): {(0, 2), (3, 5)},
            (2, KitaevBondFlavor.Z): {(1, 3), (2, 4)},
            (3, KitaevBondFlavor.X): {(2, 3)},
            (3, KitaevBondFlavor.Y): {(1, 4)},
            (3, KitaevBondFlavor.Z): {(0, 5)},
        }
        actual_bonds: dict[tuple[int, KitaevBondFlavor], set[tuple[int, int]]] = {}
        for connection in _with_standard_kitaev_flavors(graph).neighbor_connections([1, 2, 3]):
            assert connection.flavor is not None
            key = (connection.bond_class.shell, KitaevBondFlavor(connection.flavor))
            actual_bonds.setdefault(key, set()).add((connection.site_i, connection.site_j))
        assert actual_bonds == bonds

        transform = np.array(
            [
                [1 / np.sqrt(6), 1 / np.sqrt(6), -2 / np.sqrt(6)],
                [-1 / np.sqrt(2), 1 / np.sqrt(2), 0.0],
                [1 / np.sqrt(3), 1 / np.sqrt(3), 1 / np.sqrt(3)],
            ]
        )
        kitaev = {1: -13.3, 2: -0.67, 3: 0.1}
        heisenberg = {1: -1.3, 2: 0.0, 3: 1.0}
        gamma = 9.4
        gamma_prime = -2.3
        field_abc = np.array([0.0, 10.0, 0.0])
        g_factors = np.array([2.3, 2.3, 1.3])
        bohr_magneton = 5.988e-2

        actual = create_kitaev_hamiltonian(
            graph,
            kx=kitaev,
            ky=kitaev,
            kz=kitaev,
            j={1: heisenberg[1], 3: heisenberg[3]},
            gamma=gamma,
            gamma_prime=gamma_prime,
            magnetic_field_abc=field_abc,
            g_factors_abc=g_factors,
            bohr_magneton=bohr_magneton,
            crystallographic_transform=transform,
            include_term_groups=False,
        )
        explicit_zero_j2 = create_kitaev_hamiltonian(
            graph,
            kx=kitaev,
            ky=kitaev,
            kz=kitaev,
            j=heisenberg,
            gamma=gamma,
            gamma_prime=gamma_prime,
            magnetic_field_abc=field_abc,
            g_factors_abc=g_factors,
            bohr_magneton=bohr_magneton,
            crystallographic_transform=transform,
            include_term_groups=False,
        )
        np.testing.assert_allclose(actual.to_matrix(), explicit_zero_j2.to_matrix(), atol=0.0, rtol=0.0)

        pauli = {
            "X": np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex),
            "Y": np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex),
            "Z": np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex),
        }
        identity = np.eye(2, dtype=complex)

        def term_matrix(operators: dict[int, str]) -> np.ndarray:
            result = np.array([[1.0 + 0.0j]])
            for site in reversed(range(6)):
                result = np.kron(result, pauli[operators[site]] if site in operators else identity)
            return result

        expected = np.zeros((64, 64), dtype=complex)
        flavor_index = {KitaevBondFlavor.X: 0, KitaevBondFlavor.Y: 1, KitaevBondFlavor.Z: 2}
        components = "XYZ"
        for (shell, flavor), pairs in bonds.items():
            selected = flavor_index[flavor]
            other = tuple(index for index in range(3) if index != selected)
            exchange = np.eye(3) * heisenberg[shell]
            exchange[selected, selected] += kitaev[shell]
            if shell == 1:
                exchange[other[0], other[1]] = exchange[other[1], other[0]] = gamma
                for index in other:
                    exchange[selected, index] = exchange[index, selected] = gamma_prime
            for site_i, site_j in pairs:
                for first in range(3):
                    for second in range(3):
                        expected += (
                            exchange[first, second]
                            / 4.0
                            * term_matrix({site_i: components[first], site_j: components[second]})
                        )
        field_xyz = bohr_magneton * transform.T @ (g_factors * field_abc) / 2.0
        for site in range(6):
            for index, component in enumerate(components):
                expected += field_xyz[index] * term_matrix({site: component})

        np.testing.assert_allclose(actual.to_matrix(), expected, atol=1e-12, rtol=0.0)
        eigenvalues = np.linalg.eigvalsh(expected)
        assert eigenvalues[0] == pytest.approx(-31.590058786230344, abs=1e-12)
        assert eigenvalues[1] - eigenvalues[0] == pytest.approx(4.214007173911412, abs=1e-12)
