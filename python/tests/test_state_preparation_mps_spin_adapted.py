"""Tests for spin-adapted (U(1) x SU(2)) MPS input to the matrix product state preparation.

The reduced tensors come from the ``mps_h2`` fixture folded into SU(2) form, and from random spin-adapted
MPS built here. A reference contraction with the Clebsch-Gordan coefficients of arXiv:2611.19728 unfolds
them into determinant states, against which the prepared states are checked.
"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import numpy as np
import pytest
from scipy import sparse

from qdk_chemistry.algorithms import create
from qdk_chemistry.data import AlgorithmRef, Configuration, MPSContainer, MPSSite, QubitOperator, Wavefunction
from qdk_chemistry.data import symmetry as sym
from qdk_chemistry.utils.qsharp import QSHARP_UTILS

from .helpers import apply_operation, create_test_orbitals
from .test_state_preparation_mps import _FIXTURES, _fixture_site_tensors, _target_state, make_mps, synthesis_ref

_PRECISION = 10
_TOLERANCE = 1e-3
_HOUSEHOLDER = AlgorithmRef("circuit_mapper", "unitary_synthesis_householder")
# (n_P, 2m_P) -> physical index in the ('0', 'u', 'd', '2') basis.
_LOCAL = {(0, 0): 0, (1, 1): 1, (1, -1): 2, (2, 0): 3}
# Left sector offset (Δn, Δ2S) and physical sector of each fusion channel.
_CHANNELS = (((0, 0), (0, 0)), ((-2, 0), (2, 0)), ((-1, -1), (1, 1)), ((-1, 1), (1, 1)))
# name -> (sites, electrons, 2S, largest multiplicity, seed)
_RANDOM = {"singlet": (4, 4, 0, 3, 1), "doublet": (5, 5, 1, 3, 2), "triplet": (4, 4, 2, 3, 3)}


def _label(n: int, two_s: int) -> sym.SymmetryLabel:
    """Return the (particle number, total spin) sector label."""
    return sym.SymmetryLabel([sym.axes.particle_number_value(n), sym.TotalSpinValue(two_s)])


def _random_sites(num_sites: int, num_electrons: int, two_spin: int, max_multiplicity: int, seed: int) -> list:
    """Return random left-isometric reduced blocks ``[(labels, values), ...]`` per site with the target sector."""
    successors = [lambda s, c=channel: (s[0] - c[0][0], s[1] - c[0][1]) for channel in _CHANNELS]
    reachable = [{(0, 0)}]
    for _ in range(num_sites):
        reachable.append({f(s) for s in reachable[-1] for f in successors if f(s)[1] >= 0})
    # Keep only the sectors from which the target stays reachable.
    kept: list[set[tuple[int, int]]] = [set() for _ in range(num_sites)] + [{(num_electrons, two_spin)}]
    for bond in range(num_sites - 1, -1, -1):
        kept[bond] = {s for s in reachable[bond] if any(f(s) in kept[bond + 1] for f in successors)}
    rng = np.random.default_rng(seed)
    multiplicities, sites = [{(0, 0): 1}], []
    for bond in range(1, num_sites + 1):
        blocks, multiplicity = [], {}
        for right in sorted(kept[bond]):
            channels = [
                ((right[0] + dn, right[1] + ds), physical)
                for (dn, ds), physical in _CHANNELS
                if (right[0] + dn, right[1] + ds) in multiplicities[-1]
            ]
            rows = sum(multiplicities[-1][left] for left, _ in channels)
            multiplicity[right] = 1 if bond == num_sites else min(max_multiplicity, rows)
            q_matrix = np.linalg.qr(rng.standard_normal((rows, multiplicity[right])))[0]
            offset = 0
            for left, physical in channels:
                size = multiplicities[-1][left]
                blocks.append(([left, physical, right], q_matrix[offset : offset + size]))
                offset += size
        multiplicities.append(multiplicity)
        sites.append(blocks)
    return sites


def _h2_sites() -> tuple[list, np.ndarray]:
    """Fold the ``mps_h2`` singlet into two spin-adapted sites; also return its chain-order state."""
    first, second = (np.asarray(tensor) for tensor in _fixture_site_tensors("mps_h2", (0, 1)))
    chain = np.einsum("apb,bqc->pq", first, second)
    # <1/2 +1/2; 1/2 -1/2 | 0 0> = 1/sqrt(2) and <1/2 -1/2; 1/2 +1/2 | 0 0> = -1/sqrt(2) (Eq. 10).
    np.testing.assert_allclose(chain[2, 1], -chain[1, 2], atol=1e-12)
    unit = np.ones((1, 1))
    sites = [
        [([(0, 0), sector, sector], unit) for sector in ((0, 0), (1, 1), (2, 0))],
        [
            ([(0, 0), (2, 0), (2, 0)], np.full((1, 1), chain[0, 3])),
            ([(1, 1), (1, 1), (2, 0)], np.full((1, 1), np.sqrt(2) * chain[1, 2])),
            ([(2, 0), (0, 0), (2, 0)], np.full((1, 1), chain[3, 0])),
        ],
    ]
    return sites, chain.reshape(-1)


def _container(sites: list, orthogonality_center: int | None = None, counts=None, singly_occupied="u") -> MPSContainer:
    """Build the spin-adapted MPS container of reduced blocks, optionally with (alpha, beta) particle counts."""
    labels = [label for blocks in sites for block_labels, _ in blocks for label in block_labels]
    product = sym.SymmetryProduct(
        [
            sym.axes.particle_number(max(n for n, _ in labels)),
            sym.SymmetryAxis(
                sym.AxisName.TotalSpin, [sym.TotalSpinValue(s) for s in range(max(s for _, s in labels) + 1)], False
            ),
        ]
    )
    physical_order = [_label(0, 0), _label(1, 1), _label(2, 0)]
    # Each bond holds the sectors that either neighbouring site uses, in sorted (N, 2S) order.
    bonds: list[dict[tuple[int, ...], int]] = [{} for _ in range(len(sites) + 1)]
    for index, blocks in enumerate(sites):
        for (left, _, right), values in blocks:
            bonds[index][tuple(left)], bonds[index + 1][tuple(right)] = np.shape(values)
    orders = [[_label(*key) for key in sorted(bond)] for bond in bonds]
    extents = [{_label(*key): bond[key] for key in sorted(bond)} for bond in bonds]
    mps_sites = []
    for index, blocks in enumerate(sites):
        tensor = sym.SymmetryBlockedTensorRank3(
            [product] * 3,
            [extents[index], dict.fromkeys(physical_order, 1), extents[index + 1]],
            [(tuple(_label(*label) for label in block_labels), np.asarray(values)) for block_labels, values in blocks],
        )
        basis = [Configuration.from_spin_half_string(state) for state in ("0", singly_occupied, "2")]
        mps_sites.append(MPSSite(tensor, orders[index], physical_order, orders[index + 1], basis))
    particle_count = None
    if counts is not None:
        particle_count = sym.SymmetryBlockedScalarCount(
            [sym.SymmetryProduct([sym.axes.spin(1, False)])],
            [
                ((sym.SymmetryLabel([sym.axes.alpha()]),), counts[0]),
                ((sym.SymmetryLabel([sym.axes.beta()]),), counts[1]),
            ],
        )
    num_sites = len(sites)
    return MPSContainer(
        mps_sites,
        create_test_orbitals(num_sites),
        particle_count,
        num_sites - 1 if orthogonality_center is None else orthogonality_center,
    )


def _clebsch_gordan(two_sl: int, two_ml: int, two_sp: int, two_mp: int, two_sr: int, two_mr: int) -> float:
    """Return <S_L m_L; S_P m_P | S_R m_R> for S_P = 0 or 1/2 (arXiv:2611.19728, Eqs. 9-10) from doubled arguments."""
    if two_ml + two_mp != two_mr or abs(two_ml) > two_sl or abs(two_mr) > two_sr:
        return 0.0
    if two_sp == 0:
        return float(two_sl == two_sr)
    s_l, m_r = two_sl / 2, two_mr / 2
    plus, minus = np.sqrt((s_l + m_r + 0.5) / (2 * s_l + 1)), np.sqrt((s_l - m_r + 0.5) / (2 * s_l + 1))
    if two_sl == two_sr - 1:
        return plus if two_mp == 1 else minus
    if two_sl == two_sr + 1:
        return -minus if two_mp == 1 else plus
    return 0.0


def _unfold(sites: list, two_projection: int) -> np.ndarray:
    """Contract reduced blocks with Clebsch-Gordan coefficients into the chain-order determinant state.

    The state is indexed by the ('0', 'u', 'd', '2') index of each site, site zero most significant, with
    the modes created in chain order, alpha before beta.
    """
    states: dict[tuple, np.ndarray] = {((0, 0), 0, 0): np.ones(1)}
    for blocks in sites:
        unfolded: dict[tuple, np.ndarray] = {}
        for ((n_l, two_sl), (n_p, two_sp), (n_r, two_sr)), values in blocks:
            for ((n, two_s), alpha, two_ml), vector in states.items():
                if (n, two_s) != (n_l, two_sl):
                    continue
                for two_mp in range(-two_sp, two_sp + 1, 2):
                    coefficient = _clebsch_gordan(two_sl, two_ml, two_sp, two_mp, two_sr, two_ml + two_mp)
                    if coefficient:
                        local = np.eye(4)[_LOCAL[(n_p, two_mp)]]
                        for alpha_r, value in enumerate(np.asarray(values)[alpha]):
                            key = ((n_r, two_sr), alpha_r, two_ml + two_mp)
                            unfolded[key] = unfolded.get(key, 0.0) + coefficient * value * np.kron(vector, local)
        states = unfolded
    ((target, _, _),) = {key for key in states if key[2] == two_projection}
    return states[(target, 0, two_projection)]


def _right_canonical_tensors(chain: np.ndarray, num_sites: int) -> list[np.ndarray]:
    """Split a chain-order state into right-canonical (left, 4, right) site tensors by SVD."""
    tensors: list[np.ndarray] = []
    remaining = chain.reshape(-1, 1)
    for _ in range(num_sites - 1):
        right = remaining.shape[1]
        u, s, vt = np.linalg.svd(remaining.reshape(-1, 4 * right), full_matrices=False)
        rank = int(np.sum(s > 1e-12))
        tensors.insert(0, vt[:rank].reshape(rank, 4, right))
        remaining = u[:, :rank] * s[:rank]
    return [remaining.reshape(1, 4, -1), *tensors]


def _spin(chain: np.ndarray, num_sites: int) -> tuple[float, float]:
    """Return <S^2> and <S_z> of a chain-order state."""
    dim = 4**num_sites
    annihilators = []
    for mode in range(2 * num_sites):
        bit = 2 * (num_sites - 1 - mode // 2) + mode % 2
        mask = sum(1 << (2 * (num_sites - 1 - earlier // 2) + earlier % 2) for earlier in range(mode))
        occupied = [index for index in range(dim) if index >> bit & 1]
        signs = [(-1) ** bin(index & mask).count("1") for index in occupied]
        annihilators.append(
            sparse.csr_array((signs, ([index ^ (1 << bit) for index in occupied], occupied)), shape=(dim, dim))
        )
    number = [a.T @ a for a in annihilators]
    s_z = 0.5 * sum(number[2 * i] - number[2 * i + 1] for i in range(num_sites))
    s_plus = sum(annihilators[2 * i].T @ annihilators[2 * i + 1] for i in range(num_sites))
    state = chain / np.linalg.norm(chain)
    return float(state @ ((s_plus @ s_plus.T + s_z @ s_z - s_z) @ state)), float(state @ (s_z @ state))


def _prepare(container: MPSContainer, **settings) -> np.ndarray:
    """Return the state the spin-adapted circuit prepares on its qubits."""
    circuit = create(
        "state_prep",
        "matrix_product_state",
        unitary_synthesis=_HOUSEHOLDER,
        rotation_bit_precision=_PRECISION,
        **settings,
    ).run(Wavefunction(container))
    return apply_operation(circuit._qsharp_op, np.eye(1, 1 << circuit.num_qubits).ravel())


@pytest.mark.parametrize(
    ("name", "two_projection"),
    [(name, two_m) for name, args in _RANDOM.items() for two_m in range(-args[2], args[2] + 1, 2)],
)
def test_unfolded_reference_is_spin_eigenstate(name, two_projection):
    """Unfolding random reduced tensors with the paper's Clebsch-Gordan coefficients gives an S, M_S eigenstate."""
    num_sites, _, two_spin, _, _ = _RANDOM[name]
    chain = _unfold(_random_sites(*_RANDOM[name]), two_projection)
    spin_square, s_z = _spin(chain, num_sites)
    assert np.linalg.norm(chain) == pytest.approx(1.0, abs=1e-10)
    assert spin_square == pytest.approx(two_spin / 2 * (two_spin / 2 + 1), abs=1e-10)
    assert s_z == pytest.approx(two_projection / 2, abs=1e-12)


@pytest.mark.parametrize(
    ("name", "counts"), [("singlet", None), ("doublet", (2, 3)), ("triplet", (2, 2)), ("triplet", None)]
)
def test_prepares_unfolded_state(name, counts):
    """The circuit prepares the unfolded determinant state of the requested M_S on the blocked Jordan-Wigner qubits."""
    num_sites, _, two_spin, _, _ = _RANDOM[name]
    sites = _random_sites(*_RANDOM[name])
    two_projection = two_spin if counts is None else counts[0] - counts[1]
    target = _target_state(make_mps(_right_canonical_tensors(_unfold(sites, two_projection), num_sites)))
    state = _prepare(_container(sites, counts=counts))
    assert np.linalg.norm(state) == pytest.approx(1.0, abs=_TOLERANCE)
    assert np.abs(np.vdot(target, state)) ** 2 > 1 - _TOLERANCE


def test_folded_h2_singlet_has_ground_state_energy():
    """The mps_h2 ground state folded into SU(2) form prepares the H2 ground state and its energy."""
    sites, chain = _h2_sites()
    np.testing.assert_allclose(_unfold(sites, 0), chain, atol=1e-12)
    fixture = _FIXTURES["mps_h2"]
    hamiltonian = (
        QubitOperator(pauli_strings=fixture["pauli_strings"], coefficients=fixture["coefficients"])
        .get_container()
        .to_matrix()
    )
    state = _prepare(_container(sites))
    assert (
        np.abs(np.vdot(_target_state(make_mps(_fixture_site_tensors("mps_h2", (0, 1)))), state)) ** 2 > 1 - _TOLERANCE
    )
    # The dump has qubit 0 most significant; the Pauli matrix has it least significant.
    state = state.reshape((2,) * 4).transpose().reshape(-1)
    assert np.real(np.vdot(state, hamiltonian @ state)) == pytest.approx(fixture["reference_eigenvalue"], abs=5e-3)


def test_matches_dense_matrix_product_state():
    """The spin-adapted circuit and the dense circuit on the determinant form prepare the same state."""
    sites = _random_sites(*_RANDOM["doublet"])
    determinant = make_mps(_right_canonical_tensors(_unfold(sites, 1), _RANDOM["doublet"][0]))
    circuit = create(
        "state_prep",
        "matrix_product_state",
        rotation_bit_precision=_PRECISION,
        unitary_synthesis=synthesis_ref("dense"),
    ).run(Wavefunction(determinant))
    dense = apply_operation(circuit._qsharp_op, np.eye(1, 1 << circuit.num_qubits).ravel())
    assert np.abs(np.vdot(dense, _prepare(_container(sites)))) ** 2 > 1 - 2 * _TOLERANCE


def test_shared_phase_gradient_prepares_target():
    """Without its own gradient the circuit acts on [state | gradient], prepares the MPS and restores the gradient."""
    sites = _random_sites(*_RANDOM["singlet"])
    target = _target_state(make_mps(_right_canonical_tensors(_unfold(sites, 0), 4)))
    circuit = create(
        "state_prep",
        "matrix_product_state",
        unitary_synthesis=_HOUSEHOLDER,
        rotation_bit_precision=8,
        allocate_phase_gradient=False,
    ).run(Wavefunction(_container(sites)))
    assert circuit.metadata.num_phase_gradient_ancillas == 8
    op = QSHARP_UTILS.CircuitComposition.MakeSharedAncillaOp(
        circuit._qsharp_op, QSHARP_UTILS.PhaseGradient.PreparePhaseGradientState, 8
    )
    output = apply_operation(op, np.eye(1, 1 << circuit.num_qubits).ravel()).reshape(1 << 8, 1 << 8)
    np.testing.assert_allclose(output[:, 1:], 0.0, atol=1e-6, err_msg="phase gradient was not restored")
    assert np.abs(np.vdot(target, output[:, 0])) ** 2 > 1 - 1e-2


@pytest.mark.parametrize(
    ("settings", "container_options", "error", "match"),
    [
        ({"unitary_synthesis": synthesis_ref("dense")}, {}, TypeError, "unitary_synthesis_householder"),
        ({}, {"orthogonality_center": 0}, ValueError, "left-canonical"),
        ({}, {"counts": (3, 0)}, ValueError, "cannot prepare 2M_S = 3"),
        ({}, {"singly_occupied": "d"}, ValueError, "reduced physical basis"),
    ],
)
def test_input_validation(settings, container_options, error, match):
    """The mapper, canonical form, spin projection and reduced basis of a spin-adapted MPS are validated."""
    container = _container(_random_sites(*_RANDOM["doublet"]), **container_options)
    preparer = create("state_prep", "matrix_product_state", **{"unitary_synthesis": _HOUSEHOLDER, **settings})
    with pytest.raises(error, match=match):
        preparer.run(Wavefunction(container))


def test_resource_estimate_regression():
    """Pin the logical resource counts of the random doublet circuit against a silent regression."""
    circuit = create(
        "state_prep", "matrix_product_state", unitary_synthesis=_HOUSEHOLDER, rotation_bit_precision=6
    ).run(Wavefunction(_container(_random_sites(*_RANDOM["doublet"]))))
    counts = circuit.estimate().logical_counts
    assert (counts["numQubits"], counts["cczCount"], counts["ccixCount"]) == (47, 9768, 0)
