"""Tests for Propagator algorithms."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from __future__ import annotations

import math

import numpy as np
import pytest

from qdk_chemistry.algorithms.propagator import MagnusPropagator
from qdk_chemistry.data import DrivenQubitHamiltonian, QubitHamiltonian


def _make_hamiltonian(labels: list[str], weights: list[float]) -> QubitHamiltonian:
    return QubitHamiltonian(labels, np.array(weights))


class TestTimeAveragedPropagatorDriven:
    """Tests for time-averaged propagator with driven Hamiltonians."""

    def test_constant_drive_returns_h0_plus_h1(self):
        """Constant drive f(t)=1 should give H0 + 1*H1."""
        h0 = _make_hamiltonian(["ZI"], [1.0])
        h1 = _make_hamiltonian(["IX"], [2.0])
        td = DrivenQubitHamiltonian(h0, h1, drive=lambda _t: 1.0)

        propagator = MagnusPropagator()
        result = propagator.run(td, 0.0, 1.0)

        expected = h0 + 1.0 * h1
        np.testing.assert_allclose(result.coefficients, expected.coefficients)

    def test_zero_drive_returns_h0(self):
        """Zero drive f(t)=0 should give H0 + 0*H1 = H0."""
        h0 = _make_hamiltonian(["ZI"], [1.0])
        h1 = _make_hamiltonian(["IX"], [2.0])
        td = DrivenQubitHamiltonian(h0, h1, drive=lambda _t: 0.0)

        propagator = MagnusPropagator()
        result = propagator.run(td, 0.0, 1.0)

        # f_avg = 0, so result should be h0 + 0*h1
        np.testing.assert_allclose(result.coefficients[0], 1.0)

    def test_sinusoidal_drive_averages_correctly(self):
        """sin(t) averaged over [0, pi] should be 2/pi."""
        h0 = _make_hamiltonian(["ZI"], [1.0])
        h1 = _make_hamiltonian(["IX"], [1.0])
        td = DrivenQubitHamiltonian(h0, h1, drive=math.sin)

        propagator = MagnusPropagator()
        result = propagator.run(td, 0.0, math.pi)

        # f_avg = (1/pi) * integral_0^pi sin(t) dt = 2/pi
        f_avg = 2.0 / math.pi
        expected = h0 + f_avg * h1
        np.testing.assert_allclose(result.coefficients, expected.coefficients, atol=1e-12)

    def test_sinusoidal_drive_full_period_averages_to_zero(self):
        """sin(t) averaged over [0, 2*pi] should be 0."""
        h0 = _make_hamiltonian(["ZI"], [1.0])
        h1 = _make_hamiltonian(["IX"], [1.0])
        td = DrivenQubitHamiltonian(h0, h1, drive=math.sin)

        propagator = MagnusPropagator()
        result = propagator.run(td, 0.0, 2.0 * math.pi)

        # f_avg should be ~0 → only H0 terms remain
        def _to_dict(h: QubitHamiltonian) -> dict[str, complex]:
            return {s: c for s, c in zip(h.pauli_strings, h.coefficients, strict=False) if abs(c) > 1e-12}

        expected_dict = _to_dict(h0)
        assert _to_dict(result) == pytest.approx(expected_dict, abs=1e-12)

    def test_linear_ramp_averages_to_midpoint(self):
        """f(t)=t averaged over [0, 2] should be 1."""
        h0 = _make_hamiltonian(["ZI"], [1.0])
        h1 = _make_hamiltonian(["IX"], [1.0])
        td = DrivenQubitHamiltonian(h0, h1, drive=lambda t: t)

        propagator = MagnusPropagator()
        result = propagator.run(td, 0.0, 2.0)

        f_avg = 1.0  # (1/2) * integral_0^2 t dt = (1/2)*2 = 1
        expected = h0 + f_avg * h1
        np.testing.assert_allclose(result.coefficients, expected.coefficients, atol=1e-12)

    def test_step_function_across_discontinuity(self):
        """Step function jumping at t=0.5, interval [0,1]. Average should be 0.5."""
        h0 = _make_hamiltonian(["ZI"], [1.0])
        h1 = _make_hamiltonian(["IX"], [1.0])
        step = lambda t: 0.0 if t < 0.5 else 1.0  # noqa: E731
        td = DrivenQubitHamiltonian(h0, h1, drive=step)

        propagator = MagnusPropagator()
        result = propagator.run(td, 0.0, 1.0)

        f_avg = 0.5
        expected = h0 + f_avg * h1
        np.testing.assert_allclose(result.coefficients, expected.coefficients, atol=1e-10)

    def test_subinterval_averaging(self):
        """Averaging over a subinterval should give the correct local average."""
        h0 = _make_hamiltonian(["ZI"], [1.0])
        h1 = _make_hamiltonian(["IX"], [1.0])
        td = DrivenQubitHamiltonian(h0, h1, drive=math.sin)

        propagator = MagnusPropagator()
        result = propagator.run(td, 0.0, math.pi / 2.0)

        # f_avg = (2/pi) * integral_0^{pi/2} sin(t) dt = (2/pi) * 1 = 2/pi
        f_avg = 2.0 / math.pi
        expected = h0 + f_avg * h1
        np.testing.assert_allclose(result.coefficients, expected.coefficients, atol=1e-12)


class TestTimeAveragedPropagatorValidation:
    """Validation tests for the time-averaged propagator."""

    def test_t_end_before_t_start_raises(self):
        """t_end <= t_start should raise ValueError."""
        h0 = _make_hamiltonian(["ZI"], [1.0])
        h1 = _make_hamiltonian(["IX"], [1.0])
        td = DrivenQubitHamiltonian(h0, h1, drive=lambda _t: 1.0)

        propagator = MagnusPropagator()
        with pytest.raises(ValueError, match="t_end.*must be greater"):
            propagator.run(td, 1.0, 0.5)

    def test_equal_t_start_and_t_end_raises(self):
        """t_start == t_end should raise ValueError."""
        h0 = _make_hamiltonian(["ZI"], [1.0])
        h1 = _make_hamiltonian(["IX"], [1.0])
        td = DrivenQubitHamiltonian(h0, h1, drive=lambda _t: 1.0)

        propagator = MagnusPropagator()
        with pytest.raises(ValueError, match="t_end.*must be greater"):
            propagator.run(td, 1.0, 1.0)


class TestTimeAveragedPropagatorRegistry:
    """Test that the propagator can be created through the registry."""

    def test_create_via_registry(self):
        """The magnus propagator should be available through create()."""
        from qdk_chemistry.algorithms import registry  # noqa: PLC0415

        prop = registry.create("propagator", "magnus")
        assert prop.name() == "magnus"

    def test_create_default(self):
        """Default propagator should be magnus."""
        from qdk_chemistry.algorithms import registry  # noqa: PLC0415

        prop = registry.create("propagator")
        assert prop.name() == "magnus"


class TestMagnusOrder2:
    """Tests for second-order Magnus expansion in the propagator."""

    def test_constant_drive_order2_equals_order1(self):
        """Constant drive: f(t)=c, so f(t1)-f(t2)=0, Ω₂=0."""
        h0 = _make_hamiltonian(["ZI"], [1.0])
        h1 = _make_hamiltonian(["IX"], [1.0])
        td = DrivenQubitHamiltonian(h0, h1, drive=lambda _t: 2.0)

        prop1 = MagnusPropagator()
        result1 = prop1.run(td, 0.0, 1.0)

        prop2 = MagnusPropagator()
        prop2.settings().set("order", 2)
        result2 = prop2.run(td, 0.0, 1.0)

        # Build coefficient dict for comparison (ignore near-zero terms)
        def _to_dict(h: QubitHamiltonian) -> dict[str, complex]:
            return {s: c for s, c in zip(h.pauli_strings, h.coefficients, strict=False) if abs(c) > 1e-12}

        assert _to_dict(result2) == pytest.approx(_to_dict(result1), abs=1e-12)

    def test_linear_drive_order2_nonzero_correction(self):
        """Linear drive f(t)=t on [0,1]: s₂ = -1/6, Ω₂ ∝ [H₁, H₀]."""
        h0 = _make_hamiltonian(["Z"], [1.0])
        h1 = _make_hamiltonian(["X"], [1.0])
        td = DrivenQubitHamiltonian(h0, h1, drive=lambda t: t)

        prop = MagnusPropagator()
        prop.settings().set("order", 2)
        result = prop.run(td, 0.0, 1.0)

        # Order 1: Ω₁^(H)/dt = Z + 0.5*X
        # Ω₂^(H) has coefficient -iY*T³/6 on the Y term
        # Phase factor for n=2: (-i)^1 = -i
        # H_eff correction = (-i) * (-iY/6) = -Y/6 (real)
        assert "Y" in result.pauli_strings
        y_idx = result.pauli_strings.index("Y")
        np.testing.assert_allclose(result.coefficients[y_idx], -1.0 / 6.0, atol=1e-10)

    def test_commuting_h0_h1_order2_equals_order1(self):
        """When [H₁, H₀]=0, order 2 should equal order 1."""
        h0 = _make_hamiltonian(["ZI"], [1.0])
        h1 = _make_hamiltonian(["IZ"], [1.0])  # commutes with h0
        td = DrivenQubitHamiltonian(h0, h1, drive=lambda t: t)

        prop1 = MagnusPropagator()
        result1 = prop1.run(td, 0.0, 1.0)

        prop2 = MagnusPropagator()
        prop2.settings().set("order", 2)
        result2 = prop2.run(td, 0.0, 1.0)

        def _to_dict(h: QubitHamiltonian) -> dict[str, complex]:
            return {s: c for s, c in zip(h.pauli_strings, h.coefficients, strict=False) if abs(c) > 1e-12}

        assert _to_dict(result2) == pytest.approx(_to_dict(result1), abs=1e-12)

    def test_order3_constant_drive_equals_order1(self):
        """Constant drive: all higher-order corrections vanish."""
        h0 = _make_hamiltonian(["Z"], [1.0])
        h1 = _make_hamiltonian(["X"], [1.0])
        td = DrivenQubitHamiltonian(h0, h1, drive=lambda _t: 2.0)

        prop1 = MagnusPropagator()
        result1 = prop1.run(td, 0.0, 1.0)

        prop3 = MagnusPropagator()
        prop3.settings().set("order", 3)
        result3 = prop3.run(td, 0.0, 1.0)

        def _to_dict(h: QubitHamiltonian) -> dict[str, complex]:
            return {s: c for s, c in zip(h.pauli_strings, h.coefficients, strict=False) if abs(c) > 1e-12}

        assert _to_dict(result3) == pytest.approx(_to_dict(result1), abs=1e-12)

    def test_order3_linear_drive_has_correction(self):
        """Order 3 with linear drive should differ from order 2."""
        h0 = _make_hamiltonian(["Z"], [1.0])
        h1 = _make_hamiltonian(["X"], [1.0])
        td = DrivenQubitHamiltonian(h0, h1, drive=lambda t: t)

        prop2 = MagnusPropagator()
        prop2.settings().set("order", 2)
        result2 = prop2.run(td, 0.0, 1.0)

        prop3 = MagnusPropagator()
        prop3.settings().set("order", 3)
        result3 = prop3.run(td, 0.0, 1.0)

        # Order 3 should add terms beyond order 2 (since [H0,[H1,H0]] and [H1,[H1,H0]] are nonzero)
        def _to_dict(h: QubitHamiltonian) -> dict[str, complex]:
            return {s: c for s, c in zip(h.pauli_strings, h.coefficients, strict=False) if abs(c) > 1e-12}

        d2 = _to_dict(result2)
        d3 = _to_dict(result3)
        # They should differ (order 3 correction is nonzero for non-commuting H0, H1 with linear drive)
        assert d2 != pytest.approx(d3, abs=1e-8)

    def test_commuting_h0_h1_high_order_equals_order1(self):
        """When [H₁, H₀]=0, all higher orders equal order 1."""
        h0 = _make_hamiltonian(["ZI"], [1.0])
        h1 = _make_hamiltonian(["IZ"], [1.0])
        td = DrivenQubitHamiltonian(h0, h1, drive=lambda t: t)

        prop1 = MagnusPropagator()
        result1 = prop1.run(td, 0.0, 1.0)

        prop4 = MagnusPropagator()
        prop4.settings().set("order", 4)
        result4 = prop4.run(td, 0.0, 1.0)

        def _to_dict(h: QubitHamiltonian) -> dict[str, complex]:
            return {s: c for s, c in zip(h.pauli_strings, h.coefficients, strict=False) if abs(c) > 1e-12}

        assert _to_dict(result4) == pytest.approx(_to_dict(result1), abs=1e-12)

    def test_order_setting_via_registry(self):
        """Order should be configurable through the registry create kwargs."""
        from qdk_chemistry.algorithms import registry  # noqa: PLC0415

        prop = registry.create("propagator", "magnus", order=2)
        assert prop.settings().get("order") == 2


# ---------------------------------------------------------------
# Magnus convergence test against exact matrix exponentiation
# ---------------------------------------------------------------

# Pauli matrices
_I = np.eye(2, dtype=complex)
_X = np.array([[0, 1], [1, 0]], dtype=complex)
_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
_Z = np.array([[1, 0], [0, -1]], dtype=complex)
_PAULI = {"I": _I, "X": _X, "Y": _Y, "Z": _Z}


def _qh_to_matrix(h: QubitHamiltonian) -> np.ndarray:
    """Convert a QubitHamiltonian to a dense matrix."""
    n = h.num_qubits
    dim = 2**n
    mat = np.zeros((dim, dim), dtype=complex)
    for label, coeff in zip(h.pauli_strings, h.coefficients, strict=True):
        term = np.array([[1.0]], dtype=complex)
        for char in reversed(label):
            term = np.kron(_PAULI[char], term)
        mat += complex(coeff) * term
    return mat


def _exact_propagator(
    h0_mat: np.ndarray,
    h1_mat: np.ndarray,
    drive,
    t_start: float,
    t_end: float,
    steps: int = 2000,
) -> np.ndarray:
    """Compute U = T exp(-i ∫ H(t) dt) via fine-grained product."""
    from scipy.linalg import expm  # noqa: PLC0415

    dt = (t_end - t_start) / steps
    dim = h0_mat.shape[0]
    u = np.eye(dim, dtype=complex)
    for k in range(steps):
        t_mid = t_start + (k + 0.5) * dt
        h_t = h0_mat + drive(t_mid) * h1_mat
        u = expm(-1j * h_t * dt) @ u
    return u


class TestMagnusConvergence:
    """Verify that higher Magnus orders converge to the exact propagator."""

    @staticmethod
    def _propagator_error(h0, h1, drive, t_start, t_end, order):
        """Frobenius-norm error between Magnus and exact propagator."""
        from scipy.linalg import expm  # noqa: PLC0415

        td = DrivenQubitHamiltonian(h0, h1, drive=drive)
        dt = t_end - t_start

        prop = MagnusPropagator()
        prop.settings().set("order", order)
        h_eff = prop.run(td, t_start, t_end)

        h_eff_mat = _qh_to_matrix(h_eff)
        u_magnus = expm(-1j * h_eff_mat * dt)

        h0_mat = _qh_to_matrix(h0)
        h1_mat = _qh_to_matrix(h1)
        u_exact = _exact_propagator(h0_mat, h1_mat, drive, t_start, t_end)

        return np.linalg.norm(u_magnus - u_exact, "fro")

    def test_higher_orders_reduce_error_linear_drive(self):
        """For H=Z+t*X on [0,0.3], error should decrease with Magnus order."""
        h0 = _make_hamiltonian(["Z"], [1.0])
        h1 = _make_hamiltonian(["X"], [1.0])
        drive = lambda t: t  # noqa: E731

        errors = []
        for order in range(1, 5):
            err = self._propagator_error(h0, h1, drive, 0.0, 0.3, order)
            errors.append(err)

        # Each order should improve (or at least not worsen significantly)
        for i in range(1, len(errors)):
            assert errors[i] < errors[i - 1] + 1e-10, (
                f"Order {i + 1} error ({errors[i]:.2e}) not less than order {i} ({errors[i - 1]:.2e})"
            )
        # Order 4 should be very accurate for this small dt
        assert errors[3] < 1e-4, f"Order 4 error {errors[3]:.2e} too large"

    def test_higher_orders_reduce_error_sinusoidal_drive(self):
        """For H=Z+sin(t)*X on [0,0.5], error should decrease with order."""
        h0 = _make_hamiltonian(["Z"], [1.0])
        h1 = _make_hamiltonian(["X"], [1.0])
        drive = math.sin

        errors = []
        for order in range(1, 5):
            err = self._propagator_error(h0, h1, drive, 0.0, 0.5, order)
            errors.append(err)

        for i in range(1, len(errors)):
            assert errors[i] < errors[i - 1] + 1e-10, (
                f"Order {i + 1} error ({errors[i]:.2e}) not less than order {i} ({errors[i - 1]:.2e})"
            )
        assert errors[3] < 1e-4, f"Order 4 error {errors[3]:.2e} too large"

    def test_convergence_two_qubit(self):
        """Two-qubit H=ZI+t*IX on [0,0.2]: orders 1-4 should converge."""
        h0 = _make_hamiltonian(["ZI"], [1.0])
        h1 = _make_hamiltonian(["IX"], [1.0])
        drive = lambda t: t  # noqa: E731

        errors = []
        for order in range(1, 5):
            err = self._propagator_error(h0, h1, drive, 0.0, 0.2, order)
            errors.append(err)

        for i in range(1, len(errors)):
            assert errors[i] < errors[i - 1] + 1e-10
        assert errors[3] < 1e-5
