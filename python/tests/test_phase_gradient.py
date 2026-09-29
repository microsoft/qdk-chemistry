"""Tests for phase gradient states and generalized phase-gradient addition."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import cmath
import math

import numpy as np
import pytest
from qdk.test_utils import dump_operation_on_state

from qdk_chemistry.algorithms.phase_estimation.circuit_builder._gradient_pool import (
    GradientPoolPlan,
    plan_gradient_pool,
)
from qdk_chemistry.data.circuit import PhaseGradient
from qdk_chemistry.utils.qsharp import get_qsharp_context

_PG = "QDKChemistry.Utils.PhaseGradient"


def _state(operation: str, num_qubits: int, amplitudes: list[float] | None = None) -> np.ndarray:
    """Return the state *operation* produces, numbered big-endian."""
    return np.asarray(
        dump_operation_on_state(operation, num_qubits, amplitudes, context=get_qsharp_context()), dtype=complex
    )


def _index(values: list[int], widths: list[int]) -> int:
    """Return the big-endian basis index of little-endian registers holding *values*."""
    index, offset = 0, sum(widths)
    for value, width in zip(values, widths, strict=True):
        for bit in range(width):
            offset -= 1
            index |= ((value >> bit) & 1) << offset
    return index


def _gradient_state(phase: float, num_qubits: int) -> np.ndarray:
    """Return the gradient state sum_k e^{-i phase k}|k> on little-endian qubits, big-endian numbered."""
    state = np.zeros(2**num_qubits, dtype=complex)
    for k in range(2**num_qubits):
        state[_index([k], [num_qubits])] = np.exp(-1j * phase * k)
    return state / np.sqrt(2**num_qubits)


class TestPhaseGradientData:
    """The phase gradient description carried by circuit metadata."""

    def test_binary_gradient_has_the_binary_fraction_phase(self):
        """The binary gradient is the special case phase = 2 pi / 2^n."""
        gradient = PhaseGradient.binary(5)
        assert gradient == PhaseGradient(2 * math.pi / 32, 5)
        assert gradient.is_binary
        assert not PhaseGradient(0.37, 5).is_binary

    @pytest.mark.parametrize(("phase", "num_qubits", "match"), [(0.1, 0, "positive"), (math.inf, 3, "finite")])
    def test_rejects_an_unusable_gradient(self, phase, num_qubits, match):
        """An empty register or a non-finite phase cannot be prepared."""
        with pytest.raises(ValueError, match=match):
            PhaseGradient(phase, num_qubits)


class TestPreparePhaseGradients:
    """Preparation of consecutive gradient registers."""

    @pytest.mark.parametrize("num_qubits", [1, 3, 4])
    def test_binary_and_generalized_preparations_agree(self, num_qubits):
        """The binary gradient is a generalized gradient, whichever way it is prepared."""
        phase = PhaseGradient.binary(num_qubits).phase
        binary = _state(f"qs => {_PG}.PreparePhaseGradients([({phase}, {num_qubits}, true)], qs)", num_qubits)
        general = _state(f"qs => {_PG}.PreparePhaseGradients([({phase}, {num_qubits}, false)], qs)", num_qubits)
        assert np.allclose(binary, _gradient_state(phase, num_qubits), atol=1e-10)
        assert np.allclose(general, binary, atol=1e-10)

    def test_consecutive_registers_hold_their_own_gradients(self):
        """Each entry fills the next slice of the register."""
        gradients = [(0.37, 2, False), (-1.1, 3, False)]
        literal = "[" + ", ".join(f"({p}, {n}, {'true' if b else 'false'})" for p, n, b in gradients) + "]"
        actual = _state(f"qs => {_PG}.PreparePhaseGradients({literal}, qs)", 5)
        assert np.allclose(actual, np.kron(_gradient_state(0.37, 2), _gradient_state(-1.1, 3)), atol=1e-10)


class TestPhaseByGeneralizedGradient:
    """Generalized phase-gradient addition applies e^{i phi w} exactly and returns its catalyst."""

    @staticmethod
    def _operation(phi: float, n: int, controlled: bool) -> str:
        weight, catalyst = (f"qs[1..{n}]", f"qs[{n + 1}...]") if controlled else (f"qs[0..{n - 1}]", f"qs[{n}...]")
        call = f"{_PG}.PhaseByGeneralizedGradient({phi}, {weight}, {catalyst})"
        if controlled:
            call = f"Controlled {_PG}.PhaseByGeneralizedGradient([qs[0]], ({phi}, {weight}, {catalyst}))"
        prepare = f"{_PG}.PrepareGeneralizedPhaseGradient({_PG}.GeneralizedPhaseGradientAngles({phi}, {n}), {catalyst})"
        return f"qs => {{ within {{ {prepare}; }} apply {{ {call}; }} }}"

    @pytest.mark.parametrize(("n", "phi"), [(1, 0.7), (3, 0.37), (3, -2.9), (4, 1.234)])
    def test_phases_every_weight_and_restores_the_catalyst(self, n, phi):
        """Every weight picks up e^{i phi w}; the catalyst ends in |0> after unpreparation."""
        amplitudes = np.zeros(2 ** (2 * n))
        for weight in range(2**n):
            amplitudes[_index([weight, 0], [n, n])] = 2 ** (-n / 2)
        actual = _state(self._operation(phi, n, controlled=False), 2 * n, amplitudes.tolist())

        expected = np.zeros_like(actual)
        for weight in range(2**n):
            expected[_index([weight, 0], [n, n])] = 2 ** (-n / 2) * np.exp(1j * phi * weight)
        assert np.allclose(actual, expected, atol=1e-10)

    @pytest.mark.parametrize(("n", "phi"), [(2, 0.9), (3, -0.41)])
    def test_controlled_addition_acts_only_when_the_control_is_set(self, n, phi):
        """Masking the weight by the control keeps the catalyst an eigenstate on both branches."""
        widths = [1, n, n]
        amplitudes = np.zeros(2 ** (2 * n + 1))
        for control in (0, 1):
            for weight in range(2**n):
                amplitudes[_index([control, weight, 0], widths)] = 2 ** (-(n + 1) / 2)
        actual = _state(self._operation(phi, n, controlled=True), 2 * n + 1, amplitudes.tolist())

        expected = np.zeros_like(actual)
        for control in (0, 1):
            for weight in range(2**n):
                phase = np.exp(1j * phi * weight) if control else 1.0
                expected[_index([control, weight, 0], widths)] = 2 ** (-(n + 1) / 2) * phase
        assert np.allclose(actual, expected, atol=1e-10)


def _assert_plan_serves(requests: list[tuple[PhaseGradient, ...]], plan: GradientPoolPlan) -> None:
    """Every circuit gets distinct qubits, each in the state its gradient needs."""
    for circuit, gradients in enumerate(requests):
        required = [-gradient.phase * 2.0**bit for gradient in gradients for bit in range(gradient.num_qubits)]
        layout, own = plan.layouts[circuit], plan.own_angles[circuit]
        assert len(layout) == len(required)
        assert len(set(layout)) == len(layout), "a circuit must never receive the same qubit twice"
        assert sorted(index for index in layout if index < 0) == [-1 - i for i in reversed(range(len(own)))]
        for index, angle in zip(layout, required, strict=True):
            served = plan.pool_angles[index] if index >= 0 else own[-1 - index]
            assert cmath.isclose(cmath.exp(1j * served), cmath.exp(1j * angle), abs_tol=1e-9)


class TestGradientPoolPlan:
    """Controlled unitaries share every gradient qubit two or more of them need."""

    def test_a_doubled_phase_shares_all_but_one_qubit(self):
        """The gradient for 2 phi is the one for phi without its lowest qubit."""
        requests = [(PhaseGradient(0.74, 3),), (PhaseGradient(0.37, 3),)]
        plan = plan_gradient_pool(requests)
        _assert_plan_serves(requests, plan)
        assert len(plan.pool_angles) == 2
        assert [len(own) for own in plan.own_angles] == [1, 1]

    def test_rescaled_plaquette_catalysts_mostly_share(self):
        """Doubling both the interaction and the hopping angle leaves one own qubit per catalyst."""
        requests = [
            (PhaseGradient(0.3, 9), PhaseGradient(-0.1, 10)),
            (PhaseGradient(0.6, 9), PhaseGradient(-0.2, 10)),
        ]
        plan = plan_gradient_pool(requests)
        _assert_plan_serves(requests, plan)
        assert len(plan.pool_angles) == 17
        assert [len(own) for own in plan.own_angles] == [2, 2]

    def test_angles_equal_modulo_two_pi_are_shared(self):
        """A qubit's state only depends on its angle modulo 2 pi."""
        requests = [(PhaseGradient(0.37, 1),), (PhaseGradient(0.37 + 2 * math.pi, 1),)]
        plan = plan_gradient_pool(requests)
        _assert_plan_serves(requests, plan)
        assert len(plan.pool_angles) == 1

    def test_unrelated_gradients_keep_their_own_qubits(self):
        """Nothing is held for the whole run when no two circuits need the same state."""
        requests = [(PhaseGradient(0.37, 3),), (PhaseGradient(0.5, 3),)]
        plan = plan_gradient_pool(requests)
        _assert_plan_serves(requests, plan)
        assert plan.pool_angles == ()

    def test_an_angle_needed_twice_by_one_circuit_uses_two_qubits(self):
        """A circuit may use both copies at once, so they must not alias."""
        requests = [(PhaseGradient(0.37, 1), PhaseGradient(0.37, 1)), (PhaseGradient(0.37, 1),)]
        plan = plan_gradient_pool(requests)
        _assert_plan_serves(requests, plan)
        assert len(plan.pool_angles) == 1
        assert [len(own) for own in plan.own_angles] == [1, 0]

    def test_a_circuit_without_gradients_draws_nothing(self):
        """Its targets are passed through with the pool stripped."""
        requests = [(PhaseGradient(0.37, 3),), ()]
        plan = plan_gradient_pool(requests)
        _assert_plan_serves(requests, plan)
        assert plan.layouts[1] == ()


class TestPooledGradientControlledOp:
    """The wrapper hands each operation its pooled and own gradient qubits, in the order it expects them."""

    @pytest.mark.parametrize("circuit", [0, 1])
    def test_each_circuit_phases_exactly_on_the_shared_pool(self, circuit):
        """Both GPGA calls read their catalysts from one pool plus a qubit they prepare themselves."""
        n, phi = 3, 0.37
        phases = [2 * phi, phi]
        plan = plan_gradient_pool([(PhaseGradient(phase, n),) for phase in phases])
        pool_size = len(plan.pool_angles)

        op = f"{_PG}.TestControlledGradientPhase({phases[circuit]}, {n}, _, _)"
        wrapped = (
            f"{_PG}.MakePooledGradientControlledOp({pool_size}, {list(plan.layouts[circuit])}, "
            f"{list(plan.own_angles[circuit])}, {op})"
        )
        operation = (
            f"qs => {{ let pool = qs[{n + 1}...]; "
            f"within {{ {_PG}.PrepareGeneralizedPhaseGradient({list(plan.pool_angles)}, pool); }} "
            f"apply {{ ({wrapped})(qs[0], qs[1..{n}] + pool); }} }}"
        )

        widths = [1, n, pool_size]
        amplitudes = np.zeros(2 ** sum(widths))
        for control in (0, 1):
            for weight in range(2**n):
                amplitudes[_index([control, weight, 0], widths)] = 2 ** (-(n + 1) / 2)
        actual = _state(operation, sum(widths), amplitudes.tolist())

        expected = np.zeros_like(actual)
        for control in (0, 1):
            for weight in range(2**n):
                phase = np.exp(1j * phases[circuit] * weight) if control else 1.0
                expected[_index([control, weight, 0], widths)] = 2 ** (-(n + 1) / 2) * phase
        assert np.allclose(actual, expected, atol=1e-10)
