"""Tests for phase gradient states and generalized phase-gradient addition."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import math

import numpy as np
import pytest
from qdk.test_utils import dump_operation_on_state

from qdk_chemistry.data.circuit import PhaseGradient, PhaseGradientPool
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


def _assert_pool_serves(
    requests: list[tuple[PhaseGradient, ...]],
    pool: PhaseGradientPool,
    routes: tuple[tuple[int, ...], ...],
) -> None:
    """Every consumer gets distinct registers, each in the state its gradient needs."""
    assert len(routes) == len(requests)
    for gradients, route in zip(requests, routes, strict=True):
        assert len(route) == len(gradients)
        assert len(set(route)) == len(route), "a consumer must never receive the same pool register twice"
        for index, gradient in zip(route, gradients, strict=True):
            assert pool.gradients[index] == gradient


class TestPhaseGradientPool:
    """Consumers share equal phase-gradient registers, not individual qubits."""

    def test_identical_requests_share_one_register(self):
        """SOSSA and QROM-style consumers requesting the same binary gradient use one register."""
        gradient = PhaseGradient.binary(5)
        requests = [(gradient,), (gradient,)]
        pool, routes = PhaseGradientPool.from_requests(requests)
        _assert_pool_serves(requests, pool, routes)
        assert pool.gradients == (gradient,)
        assert routes == ((0,), (0,))
        assert pool.num_qubits == 5
        assert pool.offsets == (0, 5)

    def test_hubbard_plaquette_consumer_gets_two_registers(self):
        """A plaquette catalyst request has one interaction register and one hopping register."""
        requests = [(PhaseGradient(0.3, 4), PhaseGradient(-0.1, 5))]
        pool, routes = PhaseGradientPool.from_requests(requests)
        _assert_pool_serves(requests, pool, routes)
        assert pool.gradients == requests[0]
        assert routes == ((0, 1),)
        assert pool.num_qubits == 9
        assert pool.offsets == (0, 4, 9)

    def test_rescaled_plaquette_catalysts_do_not_share_registers(self):
        """Register-level pooling does not share the overlapping qubits of rescaled gradients."""
        requests = [
            (PhaseGradient(0.3, 4), PhaseGradient(-0.1, 5)),
            (PhaseGradient(0.6, 4), PhaseGradient(-0.2, 5)),
        ]
        pool, routes = PhaseGradientPool.from_requests(requests)
        _assert_pool_serves(requests, pool, routes)
        assert pool.gradients == requests[0] + requests[1]
        assert routes == ((0, 1), (2, 3))
        assert pool.num_qubits == 18

    def test_a_repeated_gradient_request_uses_distinct_registers(self):
        """One consumer may use equal gradients at the same time, so its routes must not alias."""
        gradient = PhaseGradient(0.37, 1)
        requests = [(gradient, gradient), (gradient,)]
        pool, routes = PhaseGradientPool.from_requests(requests)
        _assert_pool_serves(requests, pool, routes)
        assert pool.gradients == (gradient, gradient)
        assert routes == ((0, 1), (0,))

    def test_a_consumer_without_gradients_draws_nothing(self):
        """Its targets are passed through with no selected pool registers."""
        requests = [(PhaseGradient(0.37, 3),), ()]
        pool, routes = PhaseGradientPool.from_requests(requests)
        _assert_pool_serves(requests, pool, routes)
        assert routes[1] == ()


class TestRoutedGradientControlledOp:
    """The wrapper hands each operation its pooled and own gradient qubits, in the order it expects them."""

    @pytest.mark.parametrize("circuit", [0, 1])
    def test_each_circuit_phases_exactly_on_the_shared_pool(self, circuit):
        """Both GPGA calls read their catalysts from one pool plus a qubit they prepare themselves."""
        n, phi = 3, 0.37
        phases = [2 * phi, phi]
        requests = [(PhaseGradient(phase, n),) for phase in phases]
        pool, routes = PhaseGradientPool.from_requests(requests)
        pool_size = pool.num_qubits
        pool_specs = (
            "["
            + ", ".join(
                f"({gradient.phase}, {gradient.num_qubits}, {'true' if gradient.is_binary else 'false'})"
                for gradient in pool.gradients
            )
            + "]"
        )
        pool_sizes = [gradient.num_qubits for gradient in pool.gradients]

        op = f"{_PG}.TestControlledGradientPhase({phases[circuit]}, {n}, _, _)"
        wrapped = f"{_PG}.MakeRoutedGradientOp({pool_sizes}, {list(routes[circuit])}, {op})"
        operation = (
            f"qs => {{ let pool = qs[{n + 1}...]; "
            f"within {{ {_PG}.PreparePhaseGradients({pool_specs}, pool); }} "
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
