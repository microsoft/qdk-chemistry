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
    """Tests that higher-order Magnus raises NotImplementedError."""

    def test_order2_raises_not_implemented(self):
        """Order 2 should raise NotImplementedError."""
        h0 = _make_hamiltonian(["ZI"], [1.0])
        h1 = _make_hamiltonian(["IX"], [1.0])
        td = DrivenQubitHamiltonian(h0, h1, drive=lambda _t: 2.0)

        prop = MagnusPropagator()
        prop.settings().set("order", 2)
        with pytest.raises(NotImplementedError, match="not yet implemented"):
            prop.run(td, 0.0, 1.0)

    def test_order3_raises_not_implemented(self):
        """Order 3 should raise NotImplementedError."""
        h0 = _make_hamiltonian(["Z"], [1.0])
        h1 = _make_hamiltonian(["X"], [1.0])
        td = DrivenQubitHamiltonian(h0, h1, drive=lambda t: t)

        prop = MagnusPropagator()
        prop.settings().set("order", 3)
        with pytest.raises(NotImplementedError, match="not yet implemented"):
            prop.run(td, 0.0, 1.0)

    def test_order_setting_via_registry(self):
        """Order should be configurable through the registry create kwargs."""
        from qdk_chemistry.algorithms import registry  # noqa: PLC0415

        prop = registry.create("propagator", "magnus", order=2)
        assert prop.settings().get("order") == 2


# Higher-order Magnus convergence tests removed — order > 1 is not yet implemented.
