"""Tests for matrix product state preparation with general unitary synthesis.

Tests the classical preprocessing (cosine-sine and Givens decompositions), the public
algorithm, the full Q# circuit (state preparation fidelity via statevector simulation),
and its resource estimates.
"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import itertools
from dataclasses import dataclass

import numpy as np
import pytest
from qdk.qsharp import QSharpError

from qdk_chemistry.algorithms import create
from qdk_chemistry.algorithms.state_preparation.matrix_product_state import (
    GivensLayerData,
    MatrixProductStatePreparationData,
    generate_matrix_product_state_preparation_data,
)
from qdk_chemistry.data import Circuit, Configuration, MPSContainer, MPSSite, Orbitals, Wavefunction
from qdk_chemistry.data import symmetry as sym
from qdk_chemistry.utils.qsharp import QSHARP_UTILS, create_qsharp_context, use_qsharp_context
from qdk_chemistry.utils.unitary_synthesis import decompose_mps, dense_unitary_synthesis

from .mps_test_helpers import (
    JORDAN_WIGNER_CONVENTION_CASES,
    REFERENCE_MPS_EXPECTED_STATE,
    REFERENCE_MPS_TENSORS,
    SPINLESS_JORDAN_WIGNER_CONVENTION_CASES,
    blocked_jordan_wigner_state,
    contract_mps,
    dense_site,
    dense_target,
    make_mps,
    make_site,
    particle_number_blocked_sites,
    random_mps,
    random_orthogonal,
    random_particle_number_tensors,
    reconstruct_givens,
    right_normalized_mps,
    right_normalized_tensors,
    simulate_mps_preparation,
    site_isometry,
)
from .test_helpers import create_test_basis_set, create_test_wavefunction

_OPERATION = "QDKChemistry.Utils.MPSSequential.MPSSequential"
_SITE_STRUCT = "QDKChemistry.Utils.MPSSequential.SequentialSiteDecomposition"

# Non-zero spin MPS from the Qualtran MPSPreparation tests (Apache-2.0): a 4-site system
# whose first site carries a left bond of dimension 3 (singlet embedding).
_NON_ZERO_SPIN_RAW_TENSORS = (
    np.array(
        [
            [
                [-0.00110206, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.00316609, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, -0.57734054, 0.0, 0.0],
            ],
            [
                [0.0, 0.00110206, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, -0.00223876, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, -0.00223876, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.57734054, 0.0],
            ],
            [
                [0.0, 0.0, -0.00110206, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.00316609, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.57734054],
            ],
        ]
    ),
    np.array(
        [
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ],
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [-0.70710678, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, -0.70710678, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, -0.0, -0.0, 0.0, 0.0, 0.0],
            ],
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ],
            [
                [-0.55872176, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.82920795, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.01562571, 0.0],
            ],
            [
                [0.0, -0.55872176, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.82920795, -0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, -0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.01562571],
            ],
            [
                [0.0, 0.0, -0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ],
            [
                [0.0, 0.0, 0.0, 0.0, -0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.70710678, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.70710678],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ],
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, -0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ],
        ]
    ),
    np.array(
        [
            [
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ],
            [
                [0.0, 0.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ],
            [
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ],
            [
                [-0.99960484, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, -0.0, 0.0],
                [0.0, 0.0, 0.0, 0.02810986],
            ],
            [
                [0.0, 0.0, 0.0, 0.0],
                [0.0, -0.70710678, 0.0, 0.0],
                [0.0, 0.0, -0.70710678, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ],
            [
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ],
            [
                [0.0, -0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, -1.0],
                [0.0, 0.0, 0.0, 0.0],
            ],
            [
                [0.0, 0.0, -0.0, 0.0],
                [0.0, 0.0, 0.0, -1.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ],
        ]
    ),
    np.array(
        [
            [[0.0], [0.0], [0.0], [1.0]],
            [[0.0], [0.0], [1.0], [0.0]],
            [[0.0], [1.0], [0.0], [0.0]],
            [[1.0], [0.0], [0.0], [0.0]],
        ]
    ),
)

NON_ZERO_SPIN_EXPECTED_STATE = np.array(
    [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
        0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
        0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
        0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
        0.        ,  0.        ,  0.        , -0.00110206,  0.        ,
        0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
        0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
        0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
        0.        ,  0.        ,  0.        , -0.00110206,  0.        ,
        0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
        0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
        0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
        0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
        0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
        0.        ,  0.00176896,  0.        ,  0.        ,  0.        ,
       -0.00125085,  0.        ,  0.        ,  0.        ,  0.        ,
        0.        ,  0.        ,  0.        , -0.00262431,  0.        ,
        0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
        0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
        0.        ,  0.        ,  0.        ,  0.        ,  0.00185567,
        0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
        0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
        0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
        0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
        0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
        0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
        0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
       -0.00125085,  0.        ,  0.        ,  0.        ,  0.00176896,
        0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
        0.        ,  0.        ,  0.00185567,  0.        ,  0.        ,
        0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
        0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
        0.        ,  0.        ,  0.        , -0.00262431,  0.        ,
        0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
        0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
        0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
        0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
        0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
        0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
        0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
        0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
        0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
        0.        ,  0.        ,  0.57734054,  0.        ,  0.        ,
        0.        , -0.40824141,  0.        ,  0.        ,  0.        ,
        0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
        0.        ,  0.        ,  0.        , -0.40824141,  0.        ,
        0.        ,  0.        ,  0.57734054,  0.        ,  0.        ,
        0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
        0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
        0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
        0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
        0.        ])  # fmt: skip

# Contracting the singlet-embedding left bond with the all-ones boundary vector (as
# ``contract_mps`` does) gives an equivalent open-boundary MPS.
NON_ZERO_SPIN_TENSORS = (_NON_ZERO_SPIN_RAW_TENSORS[0].sum(axis=0, keepdims=True), *_NON_ZERO_SPIN_RAW_TENSORS[1:])

# Qualtran resource estimates (QROM mode) for cross-validation, from QubitCount and
# QECGatesCost (and_bloq + cswap). The non-zero spin estimates use the left bond of dimension 3.
QUALTRAN_COST_DENSE = {"num_qubits": 26, "toffoli": 600}
QUALTRAN_COST_SPARSE = {"num_qubits": 32, "toffoli": 321}
QUALTRAN_COST_NON_ZERO_SPIN_DENSE = {"num_qubits": 29, "toffoli": 734}
QUALTRAN_COST_NON_ZERO_SPIN_SPARSE = {"num_qubits": 29, "toffoli": 258}


@dataclass(frozen=True)
class QualtranCase:
    """A reference MPS with Qualtran's dense and sparse resource estimates."""

    tensors: tuple[np.ndarray, ...]
    expected_state: np.ndarray
    dense_cost: dict[str, int]
    sparse_cost: dict[str, int]


_QUALTRAN_CASES = pytest.mark.parametrize(
    "case",
    [
        QualtranCase(
            tuple(REFERENCE_MPS_TENSORS), REFERENCE_MPS_EXPECTED_STATE, QUALTRAN_COST_DENSE, QUALTRAN_COST_SPARSE
        ),
        QualtranCase(
            NON_ZERO_SPIN_TENSORS,
            NON_ZERO_SPIN_EXPECTED_STATE,
            QUALTRAN_COST_NON_ZERO_SPIN_DENSE,
            QUALTRAN_COST_NON_ZERO_SPIN_SPARSE,
        ),
    ],
    ids=["standard", "non_zero_spin"],
)

_SINGLE_SITE = np.array([[[1.0], [0.0], [0.0], [0.0]]])


def multiplexed_ry(angles, physical: int, chi: int, target_bit: int, control_bit: int | None = None) -> np.ndarray:
    """Ry rotations of physical qubit ``target_bit`` addressed by the bond state.

    Basis state ``p * chi + a`` holds physical state ``p``, whose bit ``k`` is physical qubit ``k``,
    and bond state ``a``. An optional physical qubit ``control_bit`` must be set.
    """
    result = np.eye(physical * chi)
    for p in range(physical):
        if p >> target_bit & 1 or (control_bit is not None and not p >> control_bit & 1):
            continue
        flipped = p | 1 << target_bit
        for a, angle in enumerate(angles):
            cosine, sine = np.cos(angle / 2), np.sin(angle / 2)
            zero, one = p * chi + a, flipped * chi + a
            result[[zero, zero, one, one], [zero, one, zero, one]] = [cosine, -sine, sine, cosine]
    return result


def controlled_bond_unitary(unitary: np.ndarray, control_bit: int) -> np.ndarray:
    """Apply ``unitary`` to the bond register when physical qubit ``control_bit`` is set."""
    chi = len(unitary)
    result = np.eye(4 * chi)
    for p in range(4):
        if p >> control_bit & 1:
            result[p * chi : (p + 1) * chi, p * chi : (p + 1) * chi] = unitary
    return result


def dense_site_circuit(rotation_angles, mixing_givens, block_givens, chi: int) -> np.ndarray:
    """Unitary of the dense site circuit of ``PrepareSequentialMPS`` (Fig. 5 of Rupprecht & Wölk)."""
    block = reconstruct_givens(*block_givens)
    if len(block) == 2 * chi:
        return block @ multiplexed_ry(rotation_angles[0], 2, chi, 0)
    # CNOT with physical qubit 1 as control and physical qubit 0 as target.
    cnot = np.zeros((4 * chi, 4 * chi))
    for p in range(4):
        mapped = p ^ 1 if p & 2 else p
        cnot[mapped * chi : (mapped + 1) * chi, p * chi : (p + 1) * chi] = np.eye(chi)
    return (
        block
        @ multiplexed_ry(rotation_angles[2], 4, chi, 0, 1)
        @ controlled_bond_unitary(reconstruct_givens(*mixing_givens[1]), 1)
        @ cnot
        @ multiplexed_ry(rotation_angles[1], 4, chi, 1, 0)
        @ controlled_bond_unitary(reconstruct_givens(*mixing_givens[0]), 0)
        @ cnot
        @ multiplexed_ry(rotation_angles[0], 4, chi, 0)
    )


def assert_same_givens(actual: GivensLayerData | None, expected: GivensLayerData | None) -> None:
    """Require identical layer structure and numerically identical angles."""
    if actual is None or expected is None:
        assert actual is None
        assert expected is None
        return
    assert actual.layer_shifted == expected.layer_shifted
    assert actual.phases == expected.phases
    assert len(actual.layer_angles) == len(expected.layer_angles)
    for actual_layer, expected_layer in zip(actual.layer_angles, expected.layer_angles, strict=True):
        np.testing.assert_allclose(actual_layer, expected_layer, atol=1e-12)


def assert_same_preparation_data(
    actual: MatrixProductStatePreparationData, expected: MatrixProductStatePreparationData
) -> None:
    """Require numerically identical decompositions."""
    assert actual.num_sites == expected.num_sites
    assert actual.num_qubits_per_site == expected.num_qubits_per_site
    assert actual.ancilla_bits == expected.ancilla_bits
    np.testing.assert_allclose(actual.initial_state_vec, expected.initial_state_vec, atol=1e-14)
    assert len(actual.sites) == len(expected.sites)
    for actual_site, expected_site in zip(actual.sites, expected.sites, strict=True):
        np.testing.assert_allclose(actual_site.rot_angles, expected_site.rot_angles, atol=1e-12)
        assert_same_givens(actual_site.w0, expected_site.w0)
        assert_same_givens(actual_site.w1, expected_site.w1)
        assert_same_givens(actual_site.u, expected_site.u)


def assert_prepares_state(params: dict, num_sites: int, ancilla_bits: int, target_state: np.ndarray) -> None:
    """Simulate MPS preparation with general synthesis and compare the post-selected state with the target.

    ``target_state`` is indexed like :func:`contract_mps` and is mapped to the blocked
    Jordan-Wigner qubit basis with the site-to-orbital order in ``params``.
    """
    ancilla_zero_prob, prepared = simulate_mps_preparation(_OPERATION, _SITE_STRUCT, params, num_sites, ancilla_bits)
    # Six rotation bits keep statevector simulation small while retaining enough accuracy
    # to detect synthesis regressions.
    assert ancilla_zero_prob > 0.90, f"P(ancilla=0) = {ancilla_zero_prob:.4f} too low"
    target = blocked_jordan_wigner_state(target_state, params["siteToOrbitalOrder"], params["numQubitsPerSite"])
    fidelity = np.abs(np.vdot(target, prepared)) ** 2
    assert fidelity > 0.95, f"Fidelity {fidelity:.4f} too low for num_sites={num_sites}"


def logical_counts(wavefunction: Wavefunction, **settings) -> dict:
    """Return the logical resource counts of the preparation circuit with general synthesis."""
    circuit = create("state_prep", "matrix_product_state", **settings).run(wavefunction)
    return circuit.estimate().logical_counts


class TestDenseSiteSynthesis:
    """Test the native dense site synthesis through its Python binding."""

    @pytest.mark.parametrize("physical", [2, 4])
    def test_site_circuits_reconstruct_chained_sites(self, physical):
        """Each site circuit maps every left-bond state to its column of the site isometry.

        Every site absorbs the right factor of the following site, so its target is the site
        isometry rotated on the right bond by that factor.
        """
        rng = np.random.default_rng(10 * physical)
        bonds, chi = [1, 3, 2, 4, 1] if physical == 4 else [1, 3, 2, 4, 2, 1], 4
        tensors = [
            random_orthogonal(physical * right, rng)[:, :left].reshape(physical, right, left).transpose(2, 0, 1)
            for left, right in itertools.pairwise(bonds)
        ]

        syntheses = decompose_mps(make_mps(tensors), chi)
        tensors = tensors[1:]

        assert len(syntheses) == len(tensors)
        following_factors = [synthesis[3] for synthesis in syntheses[1:]] + [np.eye(bonds[-1])]
        for tensor, synthesis, following_factor in zip(tensors, syntheses, following_factors, strict=True):
            rotation_angles, mixing_givens, block_givens, left_factor = synthesis
            left = tensor.shape[0]
            assert len(rotation_angles) == (3 if physical == 4 else 1)
            assert [len(angles) for angles in rotation_angles] == [chi] * len(rotation_angles)
            assert len(mixing_givens) == (2 if physical == 4 else 0)
            assert all(isinstance(flag, bool) for flag in block_givens[1] + block_givens[2])
            np.testing.assert_allclose(left_factor.T @ left_factor, np.eye(left), atol=1e-11)
            circuit = dense_site_circuit(rotation_angles, mixing_givens, block_givens, chi)
            np.testing.assert_allclose(circuit.T @ circuit, np.eye(physical * chi), atol=1e-11)
            expected = site_isometry(tensor @ following_factor.T, chi) @ left_factor.T
            np.testing.assert_allclose(circuit[:, :left], expected, atol=1e-10)

    def test_site_with_degenerate_csd_spectrum_reconstructs(self):
        """A site whose CSD blocks have twelve zero singular values still synthesizes exactly.

        Regression test: with vectorization, Eigen 3.4's divide-and-conquer SVD returned
        non-finite singular vectors for this site.
        """
        mps = random_mps(5, 16, rng=np.random.default_rng(2))
        sites = mps.sites
        chi = 16
        (rotation_angles, mixing_givens, block_givens, left_factor), (*_, right_factor) = decompose_mps(mps, chi)[2:]

        tensor = dense_site(sites[3])
        circuit = dense_site_circuit(rotation_angles, mixing_givens, block_givens, chi)
        np.testing.assert_allclose(circuit.T @ circuit, np.eye(4 * chi), atol=1e-10)
        expected = site_isometry(tensor @ right_factor.T, chi) @ left_factor.T
        np.testing.assert_allclose(circuit[:, : tensor.shape[0]], expected, atol=1e-10)

    def test_large_site_with_degenerate_csd_spectrum_reconstructs(self):
        """A chi = 256 site whose CSD blocks have 192 zero singular values synthesizes exactly.

        Regression test: Eigen 3.4's divide-and-conquer SVD crashed on this site.
        """
        site = random_mps(12, 256, rng=np.random.default_rng(1)).sites[8]
        chi = 256
        rotation_angles, mixing_givens, block_givens, left_factor = dense_unitary_synthesis(site, chi)

        tensor = dense_site(site)
        circuit = dense_site_circuit(rotation_angles, mixing_givens, block_givens, chi)
        np.testing.assert_allclose(circuit.T @ circuit, np.eye(4 * chi), atol=1e-10)
        expected = site_isometry(tensor, chi) @ left_factor.T
        np.testing.assert_allclose(circuit[:, : tensor.shape[0]], expected, atol=1e-10)

    def test_rejects_invalid_sites(self):
        """Native validation errors surface as ValueError for tensor and container inputs."""
        rng = np.random.default_rng(5)
        tensor = random_orthogonal(8, rng)[:, :2].reshape(4, 2, 2).transpose(2, 0, 1)
        site = make_site(tensor)
        with pytest.raises(ValueError, match="bond"):
            dense_unitary_synthesis(site, 1)
        three_left = random_orthogonal(8, rng)[:, :3].reshape(4, 2, 3).transpose(2, 0, 1)
        with pytest.raises(ValueError, match="incompatible bond spaces"):
            make_mps([np.ones((1, 4, 2)), tensor, three_left, np.ones((2, 4, 1))])
        with pytest.raises(ValueError, match="isometric"):
            dense_unitary_synthesis(make_site(2.0 * tensor), 2)
        with pytest.raises(ValueError, match="real"):
            dense_unitary_synthesis(make_site(tensor.astype(complex)), 2)
        with pytest.raises(ValueError, match="two or four physical states"):
            dense_unitary_synthesis(
                make_site(
                    np.ones((1, 3, 1)) / np.sqrt(3),
                    [Configuration.from_spin_half_string(state) for state in ("0", "u", "d")],
                ),
                2,
            )
        with pytest.raises(ValueError, match="successor factor"):
            dense_unitary_synthesis(site, 2, np.eye(3))
        with pytest.raises(ValueError, match="orthogonal successor"):
            dense_unitary_synthesis(site, 2, np.zeros((2, 2)))
        with pytest.raises(ValueError, match="finite"):
            make_site(np.full((1, 4, 1), np.nan))

    @pytest.mark.parametrize("method", ["general", "block_sparse"])
    def test_container_skips_initial_site_and_rejects_invalid_method(self, method):
        """The initial site need not be an isometry and a one-site MPS has no decompositions."""
        assert decompose_mps(make_mps([2.0 * _SINGLE_SITE]), 2, method) == []
        mps = make_mps([np.ones((1, 4, 2)), np.eye(2, 4).reshape(2, 4, 1)])
        assert len(decompose_mps(mps, 2, method)) == 1
        with pytest.raises(ValueError, match="bond"):
            decompose_mps(mps, 1, method)
        with pytest.raises(ValueError, match="isometric"):
            decompose_mps(make_mps([np.ones((1, 4, 2)), np.ones((2, 4, 1))]), 2, method)
        with pytest.raises(ValueError, match="real"):
            decompose_mps(make_mps([_SINGLE_SITE.astype(complex)]), 2, method)
        with pytest.raises(ValueError, match="must be"):
            decompose_mps(mps, 2, "unknown")

    @pytest.mark.parametrize("method", ["general", "block_sparse"])
    def test_container_and_tensor_preprocessing_agree(self, method):
        """The container wrapper and tensor-level helpers prepare identical states."""
        tensors = right_normalized_tensors(REFERENCE_MPS_TENSORS)
        from_container = generate_matrix_product_state_preparation_data(make_mps(tensors), method)
        from_tensors = generate_matrix_product_state_preparation_data(tensors, method)
        assert from_container.to_qsharp_params(8) == from_tensors.to_qsharp_params(8)


class TestGenerateGeneralPreparationData:
    """Test the full preprocessing pipeline for general synthesis."""

    @pytest.mark.parametrize("num_sites", [2, 3])
    @pytest.mark.parametrize("site_dim", [2, 4])
    def test_data_structure(self, num_sites, site_dim):
        """One site unitary is generated per site after the first."""
        mps = random_mps(num_sites=num_sites, bond_dim=2, site_dim=site_dim, rng=np.random.default_rng(42))
        data = generate_matrix_product_state_preparation_data(mps.sites)

        assert data.num_sites == num_sites
        assert data.num_qubits_per_site == site_dim // 2
        assert data.ancilla_bits >= 1
        assert len(data.sites) == num_sites - 1
        ancilla_dim = 1 << data.ancilla_bits
        assert len(data.initial_state_vec) == site_dim * ancilla_dim
        for site in data.sites:
            num_rotations = 3 if site_dim == 4 else 1
            assert [len(angles) for angles in site.rot_angles] == [ancilla_dim] * num_rotations
            if site_dim == 4:
                assert len(site.w0.phases) == len(site.w1.phases) == ancilla_dim
            else:
                assert site.w0 is None
                assert site.w1 is None
                assert site.to_qsharp()["rot1Angles"] == site.to_qsharp()["w0LayerAngles"] == []
            assert len(site.u.phases) == site_dim * ancilla_dim

    def test_initial_state_normalized(self):
        """The initial state vector is normalized."""
        mps = random_mps(num_sites=3, bond_dim=4, rng=np.random.default_rng(42))
        data = generate_matrix_product_state_preparation_data(mps.sites)
        assert abs(np.linalg.norm(data.initial_state_vec) - 1.0) < 1e-10

    @pytest.mark.parametrize("method", ["general", "block_sparse"])
    @pytest.mark.parametrize("use_container", [False, True])
    def test_only_initial_site_is_exported(self, monkeypatch, method, use_container):
        """Preparation never exports or copies tensors for the synthesized sites."""
        mps = right_normalized_mps(REFERENCE_MPS_TENSORS)
        to_dense = MPSSite.to_dense
        exports = []

        def track_export(site):
            exports.append(site.shape)
            return to_dense(site)

        def fail_tensor_copy(_site):
            pytest.fail("Synthesis must read the immutable site rather than copying its tensor.")

        monkeypatch.setattr(MPSSite, "to_dense", track_export)
        monkeypatch.setattr(MPSSite, "tensor", property(fail_tensor_copy))
        data = generate_matrix_product_state_preparation_data(mps if use_container else mps.sites, method)

        assert exports == [mps.sites[0].shape]
        np.testing.assert_allclose(np.linalg.norm(data.initial_state_vec), 1.0)
        assert len(data.sites) == mps.num_sites - 1

    @pytest.mark.parametrize("method", ["general", "block_sparse"])
    @pytest.mark.parametrize("mismatch", ["symmetry", "sector_order"])
    def test_bare_sites_require_matching_bond_spaces(self, method, mismatch):
        """Bare sites require the same bond-space compatibility as containers."""
        sites = particle_number_blocked_sites(right_normalized_tensors(REFERENCE_MPS_TENSORS))
        index = next(i for i in range(1, len(sites)) if len(sites[i].left_sector_order) > 1)
        site = sites[index]
        if mismatch == "symmetry":
            sites[index] = make_site(dense_site(site))
        else:
            sites[index] = MPSSite(
                site.tensor,
                list(reversed(site.left_sector_order)),
                site.physical_sector_order,
                site.right_sector_order,
                site.physical_basis,
            )
        assert sites[index - 1].right_bond_dimension == sites[index].left_bond_dimension
        with pytest.raises(ValueError, match="incompatible bond spaces"):
            generate_matrix_product_state_preparation_data(sites, method)

    def test_array_and_site_inputs_agree(self):
        """Dense arrays take the same native MPSSite path as explicit sites."""
        tensors = right_normalized_tensors(REFERENCE_MPS_TENSORS)
        assert_same_preparation_data(
            generate_matrix_product_state_preparation_data(tensors),
            generate_matrix_product_state_preparation_data(right_normalized_mps(REFERENCE_MPS_TENSORS).sites),
        )

    def test_particle_number_blocked_sites_match_dense(self):
        """Missing symmetry blocks are zeros in the decomposed target, matching dense input."""
        tensors = right_normalized_tensors(REFERENCE_MPS_TENSORS)
        blocked = particle_number_blocked_sites(tensors)
        assert any(len(site.left_sector_order) > 1 for site in blocked)
        for site, tensor in zip(blocked, tensors, strict=True):
            np.testing.assert_array_equal(dense_site(site), tensor)
        assert_same_preparation_data(
            generate_matrix_product_state_preparation_data(blocked),
            generate_matrix_product_state_preparation_data(tensors),
        )

    def test_spinless_particle_number_blocked_sites_match_dense(self):
        """Blocked sites with the ('0', '1') basis decompose like their dense arrays."""
        tensors = random_particle_number_tensors(5, 2, max_bond=3, site_dim=2, rng=np.random.default_rng(8))
        blocked = particle_number_blocked_sites(tensors)
        assert any(len(site.left_sector_order) > 1 for site in blocked)
        assert all(site.physical_dimension == 2 for site in blocked)
        assert_same_preparation_data(
            generate_matrix_product_state_preparation_data(blocked),
            generate_matrix_product_state_preparation_data(tensors),
        )

    @_QUALTRAN_CASES
    def test_qualtran_tensors_produce_valid_data(self, case):
        """The Qualtran reference MPSs contract to their expected states and decompose."""
        contracted = contract_mps(make_mps(case.tensors, orthogonality_center=None))
        np.testing.assert_allclose(contracted, case.expected_state, atol=1e-3)
        data = generate_matrix_product_state_preparation_data(right_normalized_mps(case.tensors).sites)
        assert data.num_sites == 4
        assert data.ancilla_bits >= 3
        assert len(data.sites) == 3
        assert abs(np.linalg.norm(data.initial_state_vec) - 1.0) < 1e-10

    @pytest.mark.parametrize("value", [0.0, np.nan, np.inf])
    def test_rejects_invalid_initial_state(self, value):
        """Invalid amplitudes cannot be serialized as a quantum state."""
        with pytest.raises(ValueError, match="finite"):
            generate_matrix_product_state_preparation_data([np.full((1, 4, 1), value)])

    def test_rejects_complex_mps(self):
        """The real-valued synthesis path rejects complex tensors explicitly."""
        with pytest.raises(ValueError, match="only real-valued"):
            generate_matrix_product_state_preparation_data([np.array([[[1.0j], [0.0], [0.0], [0.0]]])])

    def test_rejects_invalid_chains(self):
        """The direct preprocessing entry point validates the chain structure."""
        with pytest.raises(ValueError, match="nonempty sites"):
            generate_matrix_product_state_preparation_data([])
        with pytest.raises(ValueError, match="outer bond dimensions"):
            generate_matrix_product_state_preparation_data([np.ones((2, 4, 1))])
        with pytest.raises(ValueError, match="two or four physical states per site"):
            generate_matrix_product_state_preparation_data([np.ones((1, 3, 1))])
        with pytest.raises(ValueError, match="same physical dimension on every site"):
            generate_matrix_product_state_preparation_data([np.ones((1, 2, 2)), np.ones((2, 4, 1))])
        with pytest.raises(ValueError, match="shape"):
            generate_matrix_product_state_preparation_data([np.ones((4, 1))])

    def test_site_to_orbital_order_comes_from_container(self):
        """A container supplies its site order, and bare sites map chain site k to orbital k."""
        tensors = right_normalized_tensors(REFERENCE_MPS_TENSORS)
        assert generate_matrix_product_state_preparation_data(tensors).site_to_orbital_order == [0, 1, 2, 3]
        container = make_mps(tensors, site_to_orbital_order=[2, 0, 3, 1])
        params = generate_matrix_product_state_preparation_data(container).to_qsharp_params(rotation_bits=6)
        assert params["siteToOrbitalOrder"] == [2, 0, 3, 1]


class TestGeneralStatePreparationRun:
    """Test public algorithm validation and circuit construction."""

    def test_run_requires_mps_container(self):
        """Non-MPS wavefunctions are rejected."""
        with pytest.raises(TypeError, match="requires an MPSContainer"):
            create("state_prep", "matrix_product_state").run(create_test_wavefunction())

    def test_run_builds_qsharp_factory_from_container(self):
        """run() forwards the site decomposition, settings, and site order to Q#."""
        site_to_orbital_order = [2, 0, 3, 1]
        tensors = right_normalized_tensors(REFERENCE_MPS_TENSORS)
        wavefunction = Wavefunction(make_mps(tensors, site_to_orbital_order=site_to_orbital_order))
        circuit = create("state_prep", "matrix_product_state", rotation_bits=8).run(wavefunction)

        assert isinstance(circuit, Circuit)
        assert circuit.encoding == "jordan-wigner"
        assert circuit._qsharp_op is not None
        factory = circuit._qsharp_factory
        assert factory.program is QSHARP_UTILS.MPSSequential.MakeMPSSequentialCircuit
        expected = generate_matrix_product_state_preparation_data(wavefunction.get_container()).to_qsharp_params(8)
        assert factory.parameter["siteToOrbitalOrder"] == site_to_orbital_order
        assert list(factory.parameter) == list(expected)
        for key in ("numSites", "numQubitsPerSite", "siteToOrbitalOrder", "rotationBits", "numAncillaQubits"):
            assert factory.parameter[key] == expected[key]
        np.testing.assert_allclose(factory.parameter["initialStateVec"], expected["initialStateVec"])
        assert len(factory.parameter["siteDecompositions"]) == 3

    def test_run_defaults_to_identity_site_order(self):
        """An omitted site order maps chain site k to orbital k."""
        circuit = create("state_prep", "matrix_product_state").run(
            Wavefunction(right_normalized_mps(REFERENCE_MPS_TENSORS))
        )
        assert circuit._qsharp_factory.parameter["siteToOrbitalOrder"] == [0, 1, 2, 3]
        assert circuit._qsharp_factory.parameter["rotationBits"] == 10

    def test_fast_resource_estimation_uses_placeholder_circuit(self, monkeypatch):
        """Fast mode reads only the bond dimensions and never densifies a site."""
        tensors = right_normalized_tensors(REFERENCE_MPS_TENSORS)
        wavefunction = Wavefunction(make_mps(tensors, site_to_orbital_order=[1, 0, 3, 2]))
        ancilla_bits = generate_matrix_product_state_preparation_data(tensors).ancilla_bits

        def fail_to_dense(_site):
            raise AssertionError("fast resource estimation must not densify MPS sites")

        monkeypatch.setattr(MPSSite, "to_dense", fail_to_dense)
        prep = create("state_prep", "matrix_product_state", rotation_bits=7, fast_resource_estimation=True)
        circuit = prep.run(wavefunction)

        assert circuit._qsharp_op is None
        factory = circuit._qsharp_factory
        assert factory.program is QSHARP_UTILS.MPSSequential.MakeMPSSequentialPlaceholderCircuit
        assert list(factory.parameter) == [
            "initialStateVec",
            "numSites",
            "numQubitsPerSite",
            "siteToOrbitalOrder",
            "rotationBits",
            "numAncillaQubits",
        ]
        assert factory.parameter["numSites"] == 4
        assert factory.parameter["numQubitsPerSite"] == 2
        assert factory.parameter["siteToOrbitalOrder"] == [1, 0, 3, 2]
        assert factory.parameter["rotationBits"] == 7
        assert factory.parameter["numAncillaQubits"] == ancilla_bits
        assert len(factory.parameter["initialStateVec"]) == 4 << ancilla_bits
        assert abs(np.linalg.norm(factory.parameter["initialStateVec"]) - 1.0) < 1e-12

    @pytest.mark.parametrize("fast_resource_estimation", [False, True])
    def test_run_requires_one_site_per_orbital(self, fast_resource_estimation):
        """An active-space MPS that omits molecular orbitals is rejected."""
        tensors = right_normalized_tensors(REFERENCE_MPS_TENSORS[2:])
        tensors[0] = tensors[0][:1] / np.linalg.norm(tensors[0][:1])
        orbitals = Orbitals(
            np.eye(3),
            None,
            None,
            create_test_basis_set(3),
            sym.spin_index_set(3, [0, 1], [0, 1]),
            sym.spin_index_set(3, [], []),
        )
        mps = MPSContainer([make_site(tensor) for tensor in tensors], orbitals, orthogonality_center=0)
        prep = create("state_prep", "matrix_product_state", fast_resource_estimation=fast_resource_estimation)
        with pytest.raises(ValueError, match="exactly one MPS site per molecular orbital"):
            prep.run(Wavefunction(mps))

    @pytest.mark.parametrize("fast_resource_estimation", [False, True])
    def test_run_requires_canonical_physical_basis_on_every_site(self, fast_resource_estimation):
        """A permuted local basis on any site is rejected."""
        tensors = right_normalized_tensors(REFERENCE_MPS_TENSORS)
        swapped_basis = [Configuration.from_spin_half_string(state) for state in ("0", "d", "u", "2")]
        sites = [make_site(tensor) for tensor in tensors]
        sites[2] = make_site(tensors[2], swapped_basis)
        prep = create("state_prep", "matrix_product_state", fast_resource_estimation=fast_resource_estimation)
        with pytest.raises(ValueError, match=r"physical basis ordering \('0', 'u', 'd', '2'\)"):
            prep.run(Wavefunction(make_mps(sites)))
        with pytest.raises(ValueError, match=r"physical basis ordering \('0', 'u', 'd', '2'\)"):
            generate_matrix_product_state_preparation_data(sites)

    def test_run_builds_spinless_circuit(self):
        """A spinless MPS maps each site to one qubit."""
        mps = random_mps(num_sites=3, bond_dim=2, site_dim=2, rng=np.random.default_rng(7))
        circuit = create("state_prep", "matrix_product_state", rotation_bits=6).run(Wavefunction(mps))

        params = circuit._qsharp_factory.parameter
        assert params["numSites"] == 3
        assert params["numQubitsPerSite"] == 1
        assert len(params["initialStateVec"]) == 2 << params["numAncillaQubits"]
        counts = circuit.estimate().logical_counts
        assert counts["numQubits"] >= 3 + params["numAncillaQubits"]
        assert counts["cczCount"] > 0

    @pytest.mark.parametrize("fast_resource_estimation", [False, True])
    def test_run_requires_canonical_spinless_basis(self, fast_resource_estimation):
        """A permuted binary local basis is rejected."""
        tensors = random_mps(num_sites=2, bond_dim=2, site_dim=2, rng=np.random.default_rng(3)).sites
        swapped_basis = [Configuration.from_bitstring(state) for state in ("1", "0")]
        sites = [tensors[0], make_site(tensors[1].to_dense().reshape(tensors[1].shape), swapped_basis)]
        prep = create("state_prep", "matrix_product_state", fast_resource_estimation=fast_resource_estimation)
        with pytest.raises(ValueError, match=r"physical basis ordering \('0', '1'\)"):
            prep.run(Wavefunction(make_mps(sites)))

    @pytest.mark.parametrize("fast_resource_estimation", [False, True])
    def test_run_requires_two_or_four_physical_states(self, fast_resource_estimation):
        """Sites with any other local dimension are rejected."""
        three_states = [Configuration.from_spin_half_string(state) for state in ("0", "u", "d")]
        site = make_site(np.array([[[1.0], [0.0], [0.0]]]), three_states)
        prep = create("state_prep", "matrix_product_state", fast_resource_estimation=fast_resource_estimation)
        with pytest.raises(ValueError, match="two or four physical states per site"):
            prep.run(Wavefunction(make_mps([site])))

    @pytest.mark.parametrize("fast_resource_estimation", [False, True])
    def test_run_requires_uniform_physical_dimension(self, fast_resource_estimation):
        """Spinless and spatial-orbital sites cannot be mixed in one chain."""
        sites = [make_site(np.array([[[1.0, 0.0], [0.0, 0.0]]])), make_site(np.ones((2, 4, 1)) / np.sqrt(8))]
        prep = create("state_prep", "matrix_product_state", fast_resource_estimation=fast_resource_estimation)
        with pytest.raises(ValueError, match="same physical dimension on every site"):
            prep.run(Wavefunction(make_mps(sites)))

    @pytest.mark.parametrize("orthogonality_center", [1, None])
    def test_run_requires_right_canonical_mps(self, orthogonality_center):
        """State preparation rejects mixed or unspecified canonicalization."""
        mps = make_mps([_SINGLE_SITE, _SINGLE_SITE], orthogonality_center=orthogonality_center)
        assert mps.orthogonality_center == orthogonality_center
        with pytest.raises(ValueError, match="right-canonical MPS with center zero"):
            create("state_prep", "matrix_product_state").run(Wavefunction(mps))

    def test_run_requires_real_tensors(self):
        """Complex MPS tensors are rejected."""
        tensors = [tensor.astype(complex) for tensor in right_normalized_tensors(REFERENCE_MPS_TENSORS)]
        with pytest.raises(ValueError, match="only real-valued"):
            create("state_prep", "matrix_product_state").run(Wavefunction(make_mps(tensors)))

    def test_base_utility_access_does_not_invalidate_mps_circuit(self):
        """Ordinary utility access preserves callables stored by MPS circuits."""
        circuit = create("state_prep", "matrix_product_state").run(Wavefunction(make_mps([_SINGLE_SITE])))
        base_callable = QSHARP_UTILS.StatePreparation.MakeStatePreparationCircuit

        mps_context = circuit._qsharp_factory.program.__dict__["_qdk_context"]
        assert base_callable.__dict__["_qdk_context"] is mps_context

    @pytest.mark.parametrize("rotation_bits", [1, 63])
    def test_rejects_unsupported_rotation_precision(self, rotation_bits):
        """Settings reject precision values that Q# cannot execute safely."""
        with pytest.raises(ValueError, match="out of allowed range"):
            create("state_prep", "matrix_product_state").settings().update("rotation_bits", rotation_bits)

    def test_rejects_unknown_unitary_synthesis(self):
        """Settings and the data generator accept only the two synthesis methods."""
        with pytest.raises(ValueError, match="out of allowed options"):
            create("state_prep", "matrix_product_state").settings().update("unitary_synthesis", "dense")
        with pytest.raises(ValueError, match="unitary_synthesis must be one of"):
            generate_matrix_product_state_preparation_data(right_normalized_tensors(REFERENCE_MPS_TENSORS), "dense")


class TestGeneralQSharpFidelity:
    """Test that the MPSSequential Q# circuit prepares the target state."""

    @pytest.mark.parametrize(("num_sites", "bond_dim"), [(2, 4), (3, 4), (4, 2)])
    def test_fidelity_random_mps(self, num_sites, bond_dim):
        """Random right-canonical MPSs are prepared with high fidelity."""
        mps = random_mps(num_sites=num_sites, bond_dim=bond_dim, rng=np.random.default_rng(42))
        data = generate_matrix_product_state_preparation_data(mps.sites)
        assert_prepares_state(data.to_qsharp_params(rotation_bits=6), num_sites, data.ancilla_bits, contract_mps(mps))

    @pytest.mark.parametrize(("num_sites", "bond_dim"), [(2, 2), (4, 4), (5, 3)])
    def test_fidelity_random_spinless_mps(self, num_sites, bond_dim):
        """Random right-canonical spinless MPSs are prepared with high fidelity."""
        mps = random_mps(num_sites=num_sites, bond_dim=bond_dim, site_dim=2, rng=np.random.default_rng(11))
        data = generate_matrix_product_state_preparation_data(mps.sites)
        assert data.num_qubits_per_site == 1
        assert_prepares_state(data.to_qsharp_params(rotation_bits=6), num_sites, data.ancilla_bits, contract_mps(mps))

    @pytest.mark.parametrize("site_dim", [2, 4])
    def test_fidelity_particle_number_blocked_mps(self, site_dim):
        """Particle-number-blocked sites, whose dense arrays have zero blocks, are prepared exactly."""
        tensors = random_particle_number_tensors(4, 2, max_bond=3, site_dim=site_dim, rng=np.random.default_rng(4))
        mps = make_mps(particle_number_blocked_sites(tensors))
        data = generate_matrix_product_state_preparation_data(mps.sites)
        assert_prepares_state(data.to_qsharp_params(rotation_bits=6), 4, data.ancilla_bits, contract_mps(mps))

    @_QUALTRAN_CASES
    def test_fidelity_qualtran_mps(self, case):
        """The Qualtran reference MPSs are prepared with high fidelity."""
        data = generate_matrix_product_state_preparation_data(right_normalized_mps(case.tensors).sites)
        target_state = case.expected_state / np.linalg.norm(case.expected_state)
        assert_prepares_state(data.to_qsharp_params(rotation_bits=6), 4, data.ancilla_bits, target_state)

    @pytest.mark.parametrize(
        ("field", "message"),
        [
            ("uLayerShifted", "one shift flag per layer"),
            ("uPhases", "one entry per basis state of the target register"),
        ],
    )
    def test_rejects_givens_data_with_mismatched_lengths(self, field, message):
        """The Givens operations reject shift flags or phases that do not match the layers and register."""
        mps = random_mps(num_sites=2, bond_dim=4, rng=np.random.default_rng(42))
        data = generate_matrix_product_state_preparation_data(mps.sites)
        params = data.to_qsharp_params(rotation_bits=6)
        site = params["siteDecompositions"][0]
        site[field] = site[field][:-1]
        # A Q# runtime failure leaves the interpreter unusable, so run it on a throwaway context.
        with use_qsharp_context(create_qsharp_context()), pytest.raises(QSharpError, match=message):
            simulate_mps_preparation(_OPERATION, _SITE_STRUCT, params, 2, data.ancilla_bits)

    def test_fidelity_permuted_site_order(self):
        """A non-identity site_to_orbital_order places each chain site on its mapped orbital."""
        site_to_orbital_order = [2, 0, 3, 1]
        data = generate_matrix_product_state_preparation_data(
            make_mps(right_normalized_tensors(REFERENCE_MPS_TENSORS), site_to_orbital_order=site_to_orbital_order)
        )
        params = data.to_qsharp_params(rotation_bits=6)
        assert_prepares_state(params, 4, data.ancilla_bits, REFERENCE_MPS_EXPECTED_STATE)

    def test_fidelity_permuted_spinless_site_order(self):
        """Spinless chain sites land on their mapped orbitals with the reordering signs."""
        mps = random_mps(num_sites=4, bond_dim=4, site_dim=2, rng=np.random.default_rng(19))
        data = generate_matrix_product_state_preparation_data(make_mps(mps.sites, site_to_orbital_order=[3, 1, 0, 2]))
        params = data.to_qsharp_params(rotation_bits=6)
        assert_prepares_state(params, 4, data.ancilla_bits, contract_mps(mps))

    @pytest.mark.parametrize(
        ("tensors", "site_to_orbital_order", "expected"),
        [*JORDAN_WIGNER_CONVENTION_CASES, *SPINLESS_JORDAN_WIGNER_CONVENTION_CASES],
    )
    def test_fidelity_follows_blocked_jordan_wigner_convention(self, tensors, site_to_orbital_order, expected):
        """Sites land on blocked Jordan-Wigner qubits with the fermionic reordering signs."""
        data = generate_matrix_product_state_preparation_data(
            make_mps(tensors, site_to_orbital_order=site_to_orbital_order)
        )
        params = data.to_qsharp_params(rotation_bits=6)
        ancilla_zero_prob, prepared = simulate_mps_preparation(_OPERATION, _SITE_STRUCT, params, 2, data.ancilla_bits)
        assert ancilla_zero_prob > 0.90, f"P(ancilla=0) = {ancilla_zero_prob:.4f} too low"
        target = dense_target(expected, 2 * params["numQubitsPerSite"])
        np.testing.assert_allclose(
            blocked_jordan_wigner_state(
                contract_mps(make_mps(tensors)), site_to_orbital_order, data.num_qubits_per_site
            ),
            target,
            atol=1e-12,
        )
        fidelity = np.abs(np.vdot(target, prepared)) ** 2
        assert fidelity > 0.95, f"Fidelity {fidelity:.4f} too low"


class TestGeneralResourceEstimation:
    """Test resource estimates against Qualtran and between the exact and fast modes."""

    @_QUALTRAN_CASES
    def test_resource_estimate_is_consistent_with_qualtran(self, case):
        """Qubit counts are comparable to Qualtran's dense mode and Toffolis to its sparse mode."""
        counts = logical_counts(Wavefunction(right_normalized_mps(case.tensors)))

        assert case.dense_cost["num_qubits"] <= counts["numQubits"] <= 2 * case.dense_cost["num_qubits"]
        # The CCZ count includes every QROAM and Select decomposition, so it exceeds
        # Qualtran's sparse Toffoli count.
        assert 0 < counts["cczCount"] <= 10 * case.sparse_cost["toffoli"]

    @pytest.mark.parametrize(
        "wavefunction_factory",
        [
            pytest.param(lambda: right_normalized_mps(REFERENCE_MPS_TENSORS), id="standard"),
            pytest.param(lambda: right_normalized_mps(NON_ZERO_SPIN_TENSORS), id="non_zero_spin"),
            pytest.param(lambda: random_mps(3, 2, rng=np.random.default_rng(42)), id="random_3_2"),
            pytest.param(lambda: random_mps(3, 4, rng=np.random.default_rng(99)), id="random_3_4"),
            pytest.param(lambda: random_mps(4, 2, rng=np.random.default_rng(7)), id="random_4_2"),
        ],
    )
    def test_fast_estimate_matches_exact_estimate(self, wavefunction_factory):
        """Placeholder site data reproduces the qubit count and Toffoli cost of the exact circuit."""
        wavefunction = Wavefunction(wavefunction_factory())
        exact = logical_counts(wavefunction)
        fast = logical_counts(wavefunction, fast_resource_estimation=True)

        assert fast["numQubits"] == exact["numQubits"]
        ratio = fast["cczCount"] / exact["cczCount"]
        assert 0.9 <= ratio <= 1.1, (
            f"Fast/exact CCZ ratio {ratio:.3f}: fast={fast['cczCount']}, exact={exact['cczCount']}"
        )

    @pytest.mark.parametrize(
        ("num_sites", "bond_dim", "seed", "max_ratio"),
        [(4, 4, 5, 1.5), (6, 8, 6, 1.5), (20, 8, 2, 1.1)],
        ids=["spinless_4_4", "spinless_6_8", "spinless_20_8"],
    )
    def test_fast_estimate_bounds_spinless_exact_estimate(self, num_sites, bond_dim, seed, max_ratio):
        """For spinless sites the placeholder is an upper bound that tightens as full-bond sites dominate."""
        wavefunction = Wavefunction(random_mps(num_sites, bond_dim, site_dim=2, rng=np.random.default_rng(seed)))
        exact = logical_counts(wavefunction)
        fast = logical_counts(wavefunction, fast_resource_estimation=True)

        assert fast["numQubits"] == exact["numQubits"]
        ratio = fast["cczCount"] / exact["cczCount"]
        assert 1.0 <= ratio <= max_ratio, (
            f"Fast/exact CCZ ratio {ratio:.3f}: fast={fast['cczCount']}, exact={exact['cczCount']}"
        )
