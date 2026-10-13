"""Tests for constant-depth MPS state preparation with measurements and feed-forward.

The examples are the GHZ, Z2-symmetric (including the cluster state) and AKLT MPS of Smith et al.,
PRX Quantum 5, 030344 (2024). Their left-canonical tensors are mirrored into the right-canonical
convention of the MPS container by swapping the bonds and reversing the chain.
"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import json

import numpy as np
import pytest
from qdk.qsharp import CircuitGenerationMethod
from qdk.test_utils import dump_operation_on_state

from qdk_chemistry.algorithms import create
from qdk_chemistry.algorithms.state_preparation.constant_depth_matrix_product_state import (
    ConstantDepthMatrixProductStatePreparation,
    _fragment_tensor,
    _pushing_relations,
)
from qdk_chemistry.algorithms.state_preparation.matrix_product_state import MatrixProductStatePreparation
from qdk_chemistry.data import MPSContainer, Orbitals, Wavefunction
from qdk_chemistry.data import symmetry as sym
from qdk_chemistry.utils.qsharp import create_qsharp_context, use_qsharp_context

from .test_helpers import create_test_basis_set, create_test_wavefunction
from .test_state_preparation_mps import (
    blocked_jordan_wigner_state,
    contract_mps,
    dense_site,
    make_mps,
    make_site,
    random_mps,
    reconstruct_givens,
)

_ROTATION_BITS = 6
_SEEDS = (1, 2, 3)

# Every fragment shares one phase gradient register. The state is the same as with one register per
# fragment, but the simulated state stays small.
_SHARED_GRADIENT_OP = """
function MakeSharedGradientOp(params : QDKChemistry.Utils.MPSConstantDepth.MPSConstantDepthParams) : Qubit[] => Unit {
    qs => {
        use phaseGradient = Qubit[params.rotationBitPrecision];
        within {
            QDKChemistry.Utils.PhaseGradient.PreparePhaseGradientState(phaseGradient);
        } apply {
            QDKChemistry.Utils.MPSConstantDepth.PrepareMPSConstantDepth(
                params.fragments, params.numSites, params.siteToOrbitalOrder, params.numBondQubits, qs,
                [phaseGradient, size = Length(params.fragments)]
            );
        }
    }
}
"""


def _mirrored_chain(left_canonical: np.ndarray, num_sites: int, right_boundary: np.ndarray, seed: int) -> list:
    """Repeat a left-canonical tensor (left, physical, right) between boundary sites, mirrored to right-canonical.

    The first site is a random normalized vector on the right bond and ``right_boundary`` closes the
    chain.
    """
    _, physical, bond = left_canonical.shape
    first = np.random.default_rng(seed).normal(size=(1, physical, bond))
    return [first / np.linalg.norm(first)] + [left_canonical.transpose(2, 1, 0)] * (num_sites - 2) + [right_boundary]


def _ghz_tensors(num_sites: int, dim: int = 2) -> list[np.ndarray]:
    """Qudit GHZ state with A^m_ij = δ_ij δ_im (Eq. 45 of Smith et al.), which is mirror symmetric."""
    middle = np.zeros((dim, dim, dim))
    first = np.zeros((1, dim, dim))
    last = np.zeros((dim, dim, 1))
    for index in range(dim):
        middle[index, index, index] = 1.0
        first[0, index, index] = 1 / np.sqrt(dim)
        last[index, index, 0] = 1.0
    return [first] + [middle] * (num_sites - 2) + [last]


def _z2_tensors(num_sites: int, g: float, seed: int) -> list[np.ndarray]:
    """Z2-symmetric family of Eq. (33) of Smith et al.; g = -1 is the cluster state."""
    eta = 1 / np.sqrt(1 + abs(g))
    root = np.sqrt(-g)
    a0 = eta * np.array([[1.0, 0.0], [root, 0.0]])
    a1 = eta * np.array([[0.0, -root], [0.0, 1.0]])
    last = np.zeros((2, 2, 1))
    last[0, 0, 0] = last[1, 1, 0] = 1.0
    return _mirrored_chain(np.stack([a0, a1], axis=1), num_sites, last, seed)


def _aklt_tensors(num_sites: int, seed: int) -> list[np.ndarray]:
    """AKLT state of Eq. (36) of Smith et al., with physical states (0, +, -) on indices (0, 1, 2) of four."""
    raising = np.array([[0.0, 1.0], [0.0, 0.0]])
    tensor = np.zeros((2, 4, 2))
    tensor[:, 0, :] = -np.sqrt(1 / 3) * np.diag([1.0, -1.0])
    tensor[:, 1, :] = np.sqrt(2 / 3) * raising
    tensor[:, 2, :] = -np.sqrt(2 / 3) * raising.T
    last = np.zeros((2, 4, 1))
    last[0, 1, 0] = last[1, 2, 0] = 1.0
    return _mirrored_chain(tensor, num_sites, last, seed)


def _pauli(label: int, num_qubits: int) -> np.ndarray:
    """Return X^x Z^z for the label x | (z << num_qubits), with X^x Z^z |a> = (-1)^{|a & z|} |a ^ x>."""
    dim = 1 << num_qubits
    x, z = label % dim, label >> num_qubits
    matrix = np.zeros((dim, dim))
    for state in range(dim):
        matrix[state ^ x, state] = (-1) ** (state & z).bit_count()
    return matrix


def _preparation_data(mps: MPSContainer, fragment_size: int):
    prep = ConstantDepthMatrixProductStatePreparation()
    prep.settings().update("fragment_size", fragment_size)
    return prep.generate_constant_depth_preparation_data(mps)


def _target_state(mps: MPSContainer) -> np.ndarray:
    qubits_per_site = 1 if mps.sites[0].physical_dimension == 2 else 2
    return blocked_jordan_wigner_state(contract_mps(mps), mps.site_to_orbital_order, qubits_per_site)


def _active_space_mps() -> MPSContainer:
    """Two GHZ sites on three molecular orbitals."""
    orbitals = Orbitals(
        np.eye(3),
        None,
        None,
        create_test_basis_set(3),
        sym.spin_index_set(3, [0, 1], [0, 1]),
        sym.spin_index_set(3, [], []),
    )
    return MPSContainer([make_site(tensor) for tensor in _ghz_tensors(2)], orbitals, orthogonality_center=0)


def _asap_depth(grid: list, num_named_qubits: int) -> int:
    """Return the as-soon-as-possible depth of the leaf operations of a simulated circuit.

    The simulator reuses released scratch qubits, which would chain operations of different
    fragments, so a scratch qubit (index at least ``num_named_qubits``) is a separate wire in each
    call of a fragment-level operation. A correction is chosen from all earlier fusion
    measurements, so its operations wait for them; classical feed-forward is free.
    """
    scoped_operations = {"PrepareFragment", "FuseBonds", "CorrectFragment", "PreparePhaseGradientState"}
    layers: dict = {}
    state = {"fused": 0, "scopes": 0}

    def visit(columns: list, scope: int, correcting: bool, fusing: bool) -> None:
        for column in columns:
            for operation in column["components"]:
                if operation.get("children"):
                    gate = operation["gate"]
                    child_scope = scope
                    if gate in scoped_operations:
                        state["scopes"] += 1
                        child_scope = state["scopes"]
                    visit(
                        operation["children"],
                        child_scope,
                        correcting or gate == "CorrectFragment",
                        fusing or gate == "FuseBonds",
                    )
                    continue
                qubits = [
                    entry["qubit"] for key in ("targets", "controls", "qubits") for entry in operation.get(key, [])
                ]
                wires = {qubit if qubit < num_named_qubits else (qubit, scope) for qubit in qubits}
                layer = 1 + max([layers.get(wire, 0) for wire in wires] + ([state["fused"]] if correcting else []))
                for wire in wires:
                    layers[wire] = layer
                if fusing and operation["kind"] == "measurement":
                    state["fused"] = max(state["fused"], layer)

    visit(grid, 0, False, False)
    return max(layers.values())


def _simulated_circuit(module: str, program: str, params: dict) -> dict:
    """Return the circuit of one simulated run of ``QDKChemistry.Utils.<module>.<program>`` with a fixed seed."""
    context = create_qsharp_context()
    with use_qsharp_context(context):
        context.set_quantum_seed(1)
        entry_point = getattr(getattr(context.code.QDKChemistry.Utils, module), program)
        circuit = context.circuit(entry_point, *params.values(), generation_method=CircuitGenerationMethod.Simulate)
    return json.loads(circuit.json())


# Builders of (MPS, fragment size).
_PREPARATION_CASES = [
    pytest.param(lambda: make_mps(_ghz_tensors(4)), 2, id="ghz"),
    pytest.param(lambda: make_mps(_ghz_tensors(5)), 1, id="ghz-single-site-fragments"),
    pytest.param(lambda: make_mps(_ghz_tensors(3, dim=4)), 1, id="qudit-ghz-two-qubit-fusion"),
    pytest.param(lambda: make_mps(_z2_tensors(6, -1.0, seed=1)), 2, id="cluster-three-fragments"),
    pytest.param(lambda: make_mps(_z2_tensors(5, -0.3, seed=2)), 2, id="z2-uneven-fragments"),
    pytest.param(lambda: make_mps(_aklt_tensors(4, seed=3)), 2, id="aklt"),
    pytest.param(lambda: make_mps(_aklt_tensors(3, seed=4)), 1, id="aklt-single-site-fragments"),
    pytest.param(
        lambda: make_mps(_z2_tensors(4, -0.5, seed=5), site_to_orbital_order=[2, 0, 3, 1]), 2, id="z2-permuted-order"
    ),
    pytest.param(
        lambda: make_mps(_aklt_tensors(4, seed=6), site_to_orbital_order=[1, 3, 0, 2]), 2, id="aklt-permuted-order"
    ),
    pytest.param(lambda: make_mps(_ghz_tensors(3)), 5, id="single-fragment"),
]


class TestPushingRelations:
    """Test the pushing relations that correct the Pauli defects of the fusion measurements."""

    @pytest.mark.parametrize(("build_mps", "fragment_size"), _PREPARATION_CASES)
    def test_corrections_satisfy_pushing_relations(self, build_mps, fragment_size):
        """Every defect B on a fused bond obeys B T^m = sum_n W_mn T^n H with the correction W^T.

        The trivial defect needs no correction, and the last fragment absorbs every defect.
        """
        mps = build_mps()
        data = _preparation_data(mps, fragment_size)

        assert [site for fragment in data.fragments for site in fragment.sites] == list(range(mps.num_sites))
        for fragment in data.fragments[1:]:
            tensor = _fragment_tensor([dense_site(mps.sites[site]) for site in fragment.sites])
            matrices = tensor.transpose(1, 0, 2)
            right_qubits = (tensor.shape[2] - 1).bit_length()
            assert tensor.shape[0] == 1 << fragment.num_fusion_qubits
            assert len(fragment.corrections) == len(fragment.pushed_labels) == 4**fragment.num_fusion_qubits
            for label, (correction, pushed) in enumerate(
                zip(fragment.corrections, fragment.pushed_labels, strict=True)
            ):
                rotation = reconstruct_givens(correction).T
                defect, pushed_defect = _pauli(label, fragment.num_fusion_qubits), _pauli(pushed, right_qubits)
                np.testing.assert_allclose(rotation @ rotation.T, np.eye(len(rotation)), atol=1e-10)
                np.testing.assert_allclose(
                    defect @ matrices, np.einsum("mn,nab,bc->mac", rotation, matrices, pushed_defect), atol=1e-10
                )
            assert fragment.pushed_labels[0] == 0
            np.testing.assert_allclose(reconstruct_givens(fragment.corrections[0]), np.eye(tensor.shape[1]), atol=1e-10)
            if fragment is data.fragments[-1]:
                assert set(fragment.pushed_labels) == {0}

    @pytest.mark.parametrize("dim", [2, 4])
    def test_ghz_pushes_shifts_and_absorbs_clock_defects(self, dim):
        """GHZ defects X^x push through as X^x, and Z^z defects are removed locally (Eq. 46 of Smith et al.)."""
        data = _preparation_data(make_mps(_ghz_tensors(3, dim)), 1)
        assert data.fragments[1].pushed_labels == [label % dim for label in range(dim * dim)]

    def test_aklt_pushes_bell_defects_to_themselves(self):
        """Each Bell-basis defect of the AKLT state pushes to itself, a Z2 x Z2 subgroup of Eq. (37) of Smith et al."""
        data = _preparation_data(make_mps(_aklt_tensors(4, seed=1)), 1)
        for fragment in data.fragments[1:-1]:
            assert fragment.pushed_labels == [0, 1, 2, 3]

    def test_rejects_generic_fragment_tensor(self):
        """A generic tensor has no Pauli that a defect can be pushed to."""
        tensor = np.random.default_rng(7).normal(size=(2, 2, 2))
        with pytest.raises(ValueError, match="no Pauli on the right bond pushes the defect"):
            _pushing_relations(tensor)


class TestValidation:
    """Test the settings and the rejection of unsupported inputs."""

    def test_default_settings(self):
        prep = create("state_prep", "constant_depth_matrix_product_state")
        assert prep.name() == "constant_depth_matrix_product_state"
        assert prep.settings().get("fragment_size") == 2
        assert prep.settings().get("rotation_bit_precision") == 10

    @pytest.mark.parametrize(
        ("build_mps", "fragment_size", "message"),
        [
            # The middle of three fragments must push defects; the last one absorbs any defect.
            pytest.param(
                lambda: random_mps(6, 2, site_dim=2, rng=np.random.default_rng(3)),
                2,
                "every Pauli defect on a fused bond to be pushable",
                id="generic-mps",
            ),
            pytest.param(
                lambda: random_mps(4, 3, site_dim=2, rng=np.random.default_rng(3)),
                2,
                "power-of-two bond dimensions between fragments",
                id="fused-bond-of-three",
            ),
            pytest.param(lambda: make_mps(_ghz_tensors(3)), 0, "fragment size of at least one", id="empty-fragments"),
            pytest.param(
                lambda: make_mps([tensor.astype(complex) for tensor in _ghz_tensors(3)]),
                2,
                "only real-valued",
                id="complex",
            ),
            pytest.param(
                lambda: make_mps(_ghz_tensors(3), orthogonality_center=1),
                2,
                "right-canonical MPS with center zero",
                id="nonzero-center",
            ),
            pytest.param(_active_space_mps, 2, "exactly one MPS site per molecular orbital", id="missing-orbitals"),
        ],
    )
    def test_rejects_unsupported_mps(self, build_mps, fragment_size, message):
        prep = create("state_prep", "constant_depth_matrix_product_state", fragment_size=fragment_size)
        with pytest.raises(ValueError, match=message):
            prep.run(Wavefunction(build_mps()))

    def test_rejects_non_mps_wavefunction(self):
        prep = create("state_prep", "constant_depth_matrix_product_state")
        with pytest.raises(TypeError, match="requires an MPSContainer"):
            prep.run(create_test_wavefunction())


class TestQSharpPreparation:
    """Test that the adaptive Q# circuits prepare the target state for every measurement record."""

    @pytest.mark.parametrize(("build_mps", "fragment_size"), _PREPARATION_CASES)
    def test_prepares_mps_state(self, build_mps, fragment_size):
        """The state matches the MPS in the blocked Jordan-Wigner basis for several measurement records.

        Six rotation bits keep the simulation small while retaining enough accuracy to detect errors.
        """
        mps = build_mps()
        data = _preparation_data(mps, fragment_size)
        params = data.to_qsharp_params(rotation_bit_precision=_ROTATION_BITS)
        target = _target_state(mps)
        context = create_qsharp_context()
        with use_qsharp_context(context):
            context.eval(_SHARED_GRADIENT_OP)
            module = context.code.QDKChemistry.Utils.MPSConstantDepth
            operation = context.code.MakeSharedGradientOp(module.MPSConstantDepthParams(**params))
            for seed in _SEEDS:
                context.set_quantum_seed(seed)
                prepared = dump_operation_on_state(operation, len(target).bit_length() - 1, context=context)
                fidelity = np.abs(np.vdot(target, prepared)) ** 2
                assert fidelity > 0.97, f"Fidelity {fidelity:.4f} too low for seed {seed}"

    def test_run_returns_adaptive_circuit(self):
        """The circuit from ``run`` prepares the MPS with one set of registers per fragment."""
        mps = make_mps(_ghz_tensors(4))
        target = _target_state(mps)
        context = create_qsharp_context()
        with use_qsharp_context(context):
            context.set_quantum_seed(2)
            prep = create(
                "state_prep",
                "constant_depth_matrix_product_state",
                fragment_size=2,
                rotation_bit_precision=_ROTATION_BITS,
            )
            circuit = prep.run(Wavefunction(mps))
            prepared = dump_operation_on_state(circuit._qsharp_op, circuit.num_qubits, context=context)
            counts = circuit.estimate().logical_counts

        assert circuit.encoding == "jordan-wigner"
        assert circuit.num_qubits == 4
        assert np.abs(np.vdot(target, prepared)) ** 2 > 0.97
        # Two phase gradient, bond and angle registers, and one fusion register.
        assert counts["numQubits"] >= 4 + 2 * (2 * _ROTATION_BITS + 1) + 1
        assert counts["measurementCount"] >= 2


class TestCircuitDepth:
    """Test that the circuit depth does not grow with the number of sites."""

    def test_depth_is_independent_of_num_sites(self):
        """Single-site GHZ fragments have equal depth for every chain length, unlike sequential preparation."""
        bits = 2
        depths, num_qubits = [], []
        for num_sites in (3, 5, 7):
            data = _preparation_data(make_mps(_ghz_tensors(num_sites)), 1)
            num_fragments = len(data.fragments)
            # State, phase gradient, bond, fusion and angle registers, in order of allocation.
            num_named_qubits = num_sites + 2 * num_fragments * bits + (2 * num_fragments - 1) * data.bond_bits
            circuit = _simulated_circuit("MPSConstantDepth", "MakeMPSConstantDepthCircuit", data.to_qsharp_params(bits))
            depths.append(_asap_depth(circuit["componentGrid"], num_named_qubits))
            num_qubits.append(len(circuit["qubits"]))

        assert len(set(depths)) == 1, f"depths {depths} grow with the number of sites"
        assert num_qubits[0] < num_qubits[1] < num_qubits[2]

        sequential = MatrixProductStatePreparation()
        sequential_depths = []
        for num_sites in (3, 7):
            params = sequential.generate_matrix_product_state_preparation_data(
                make_mps(_ghz_tensors(num_sites))
            ).to_qsharp_params(bits)
            circuit = _simulated_circuit("MPSSequential", "MakeMPSSequentialCircuit", params)
            sequential_depths.append(_asap_depth(circuit["componentGrid"], len(circuit["qubits"])))
        assert sequential_depths[0] < sequential_depths[1]
        assert depths[0] < sequential_depths[1]
