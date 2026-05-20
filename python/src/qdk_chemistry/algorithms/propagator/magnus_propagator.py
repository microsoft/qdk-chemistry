r"""Time-averaged propagator with Magnus expansion.

Computes the effective Hamiltonian for a time interval via the Magnus
expansion truncated at a configurable order.

**Order 1** (default) is the time-averaged Hamiltonian:

.. math::

    \Omega_1 = \int_{t_1}^{t_2} H(t')\,\mathrm{d}t'

Higher orders add commutator corrections computed recursively via

.. math::

    \dot\Omega_n
    = \sum_{k=1}^{n-1} \frac{B_k}{k!}
      \sum_{j_1+\cdots+j_k=n-1}
      \mathrm{ad}_{\Omega_{j_1}} \cdots
      \mathrm{ad}_{\Omega_{j_k}}(H(t))

where :math:`B_k` are Bernoulli numbers.

For the driven case :math:`H(t) = H_0 + f(t)\,H_1` every Magnus term
reduces to a linear combination of nested commutators of :math:`H_0`
and :math:`H_1` with scalar coefficients that are iterated integrals
over the drive function.  The propagator returns
:math:`\Omega / \delta t` so that the builder (which multiplies by
``dt``) recovers the full exponent.
"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from __future__ import annotations

from math import factorial
from typing import TYPE_CHECKING

import numpy as np
from scipy import integrate
from scipy.special import bernoulli as _scipy_bernoulli

from qdk_chemistry.data import Settings
from qdk_chemistry.data.time_dependent_qubit_hamiltonian.containers.driven import DrivenContainer

from .base import Propagator

if TYPE_CHECKING:
    from collections.abc import Callable

    from qdk_chemistry.data import QubitHamiltonian, TimeDependentQubitHamiltonian

__all__: list[str] = ["MagnusPropagator", "MagnusPropagatorSettings"]

_EPS: float = float(np.finfo(np.float64).eps)


def _bernoulli(k: int) -> float:
    """Return the *k*-th Bernoulli number (B_1 = -1/2 convention)."""
    return float(_scipy_bernoulli(k)[k])


# ---------------------------------------------------------------
# Symbolic representation of a Magnus term for driven H(t)
# ---------------------------------------------------------------
# Each dΩ_n/dt is a list of _MagnusTerm objects.  A _MagnusTerm
# pairs a fixed QubitHamiltonian (nested commutator of H0, H1)
# with a scalar time-dependent function g(t).  The integrated
# version Ω_n(t) = Σ_i op_i * ∫_{t_start}^t g_i(t') dt'.


class _MagnusTerm:
    """A single term: ``operator * g(t)`` (represents a dΩ/dt contribution)."""

    __slots__ = ("g", "operator")

    def __init__(self, operator: QubitHamiltonian, g: Callable[[float], float]) -> None:
        self.operator = operator
        self.g = g


def _integrate_running(g: Callable[[float], float], t_start: float, t: float) -> float:
    """Compute ∫_{t_start}^t g(t') dt'."""
    if abs(t - t_start) < _EPS:
        return 0.0
    val, _ = integrate.quad(g, t_start, t)
    return val


def _ad_omega_with_h(
    omega_dot_terms: list[_MagnusTerm],
    h0: QubitHamiltonian,
    h1: QubitHamiltonian,
    drive: Callable[[float], float],
    t_start: float,
) -> list[_MagnusTerm]:
    r"""Compute :math:`[\Omega(t), H(t)]` as new dΩ/dt terms.

    Here Ω(t) = Σ_i op_i * G_i(t) where G_i(t) = ∫_{t_start}^t g_i(t')dt'.
    H(t) = H0 + f(t)*H1.

    Returns terms for [Ω(t), H(t)] = Σ_i G_i(t)*[op_i, H0] + G_i(t)*f(t)*[op_i, H1].
    """
    from qdk_chemistry.utils.pauli_commutation import commutator  # noqa: PLC0415

    result: list[_MagnusTerm] = []
    for term in omega_dot_terms:
        comm_h0 = commutator(term.operator, h0)
        if np.any(np.abs(comm_h0.coefficients) > _EPS):
            g_dot = term.g
            result.append(
                _MagnusTerm(
                    comm_h0,
                    lambda t, _g=g_dot, _ts=t_start: _integrate_running(_g, _ts, t),  # type: ignore[misc]
                )
            )

        comm_h1 = commutator(term.operator, h1)
        if np.any(np.abs(comm_h1.coefficients) > _EPS):
            g_dot = term.g
            result.append(
                _MagnusTerm(
                    comm_h1,
                    lambda t, _g=g_dot, _ts=t_start, _f=drive: _integrate_running(_g, _ts, t) * _f(t),  # type: ignore[misc]
                )
            )

    return result


def _ad_omega_with_terms(
    omega_dot_terms: list[_MagnusTerm],
    target_terms: list[_MagnusTerm],
    t_start: float,
) -> list[_MagnusTerm]:
    r"""Compute :math:`[\Omega(t), X(t)]` where both sides are term-lists.

    Ω(t) uses the integrated scalar G(t) = ∫ g(t')dt'; X(t) uses its
    raw scalar function directly.
    """
    from qdk_chemistry.utils.pauli_commutation import commutator  # noqa: PLC0415

    result: list[_MagnusTerm] = []
    for o_term in omega_dot_terms:
        for x_term in target_terms:
            comm = commutator(o_term.operator, x_term.operator)
            if np.any(np.abs(comm.coefficients) > _EPS):
                g_o_dot = o_term.g
                g_x = x_term.g
                result.append(
                    _MagnusTerm(
                        comm,
                        lambda t, _go=g_o_dot, _gx=g_x, _ts=t_start: _integrate_running(_go, _ts, t) * _gx(t),  # type: ignore[misc]
                    )
                )
    return result


def _nested_ad(
    h_terms: list[_MagnusTerm],
    omega_dot_list: list[list[_MagnusTerm]],
    j_indices: list[int],
    t_start: float,
) -> list[_MagnusTerm]:
    r"""Apply ``ad_{Ω_{j_1}} ad_{Ω_{j_2}} … ad_{Ω_{j_k}}(H(t))`` right to left.

    *h_terms* represents H(t).  Each ``ad_{Ω_j}`` wraps the running
    result in a commutator with the integrated Ω_j(t).
    """
    current = h_terms
    for j in reversed(j_indices):
        current = _ad_omega_with_terms(omega_dot_list[j - 1], current, t_start)
    return current


def _ordered_partitions(n: int, k: int) -> list[list[int]]:
    """Generate all ordered tuples of *k* positive integers summing to *n*."""
    if k == 1:
        return [[n]]
    result: list[list[int]] = []
    for first in range(1, n - k + 2):
        for rest in _ordered_partitions(n - first, k - 1):
            result.append([first, *rest])
    return result


class MagnusPropagatorSettings(Settings):
    """Settings for the time-averaged propagator."""

    def __init__(self):
        """Initialize settings with default Magnus expansion order."""
        super().__init__()
        self._set_default("order", "int", 1, "Magnus expansion order (1, 2, 3, ...).")


class MagnusPropagator(Propagator):
    r"""Time-averaged propagator with recursive Magnus expansion.

    Evaluates the effective Hamiltonian for an interval :math:`[t_1, t_2]`
    using the Magnus expansion truncated at the configured ``order``.

    For :class:`~qdk_chemistry.data.time_dependent_qubit_hamiltonian.containers.driven.DrivenContainer`
    Hamiltonians all integrals reduce to scalar quadratures over the
    drive function.  Higher orders are built recursively via nested
    commutators of :math:`H_0` and :math:`H_1`.

    For other container types only order 1 (midpoint evaluation) is
    supported.

    """

    def __init__(self):
        """Initialize the time-averaged propagator."""
        super().__init__()
        self._settings = MagnusPropagatorSettings()

    def _run_impl(
        self,
        hamiltonian: TimeDependentQubitHamiltonian,
        t_start: float,
        t_end: float,
    ) -> QubitHamiltonian:
        r"""Compute the effective Hamiltonian over :math:`[t_1, t_2]`.

        Args:
            hamiltonian: Time-dependent Hamiltonian.
            t_start: Start of the interval.
            t_end: End of the interval.

        Returns:
            Effective time-independent qubit Hamiltonian for the interval.

        Raises:
            ValueError: If *t_end* is not greater than *t_start*.

        """
        if t_end <= t_start:
            raise ValueError(f"t_end ({t_end}) must be greater than t_start ({t_start}).")

        order: int = self._settings.get("order")
        container = hamiltonian.get_container()
        if not isinstance(container, DrivenContainer):
            raise NotImplementedError(
                f"Magnus propagator is not yet implemented for {type(container).__name__} containers."
            )
        return self._magnus_driven(container, t_start, t_end, order)

    # ------------------------------------------------------------------
    # Driven Magnus expansion (arbitrary order)
    # ------------------------------------------------------------------

    @staticmethod
    def _magnus_driven(
        container: DrivenContainer,
        t_start: float,
        t_end: float,
        order: int,
    ) -> QubitHamiltonian:
        r"""Recursive Magnus expansion for driven Hamiltonians.

        Builds :math:`\Omega = \sum_{n=1}^{\mathrm{order}} \Omega_n` via
        the recursion

        .. math::

            \dot\Omega_n = \sum_{k=1}^{n-1} \frac{B_k}{k!}
            \sum_{j_1+\cdots+j_k=n-1} \mathrm{ad}_{\Omega_{j_1}}
            \cdots \mathrm{ad}_{\Omega_{j_k}}(H(t))

        where :math:`\mathrm{ad}_{\Omega_j}` uses the integrated
        :math:`\Omega_j(t) = \int_{t_\text{start}}^t \dot\Omega_j(t')\,dt'`.

        Returns :math:`\Omega / \delta t`.

        """
        dt = t_end - t_start
        h0, h1, drive = container.base_hamiltonian, container.drive_hamiltonian, container.drive

        # H(t) as symbolic terms: H0*1 + H1*f(t)
        h_terms: list[_MagnusTerm] = [
            _MagnusTerm(h0, lambda _t: 1.0),
            _MagnusTerm(h1, drive),
        ]

        # omega_dot_list[n-1] holds the dΩ_n/dt terms
        omega_dot_list: list[list[_MagnusTerm]] = []

        # Accumulate all (label → coefficient) contributions and merge duplicates
        merged: dict[str, complex] = {}

        for n in range(1, order + 1):
            if n == 1:
                # dΩ₁/dt = H(t)
                omega_n_dot = h_terms
            else:
                # dΩ_n/dt = Σ_{k=1}^{n-1} (B_k / k!) Σ_{j1+...+jk=n-1} ad_{Ω_j1}...ad_{Ω_jk}(H(t))
                omega_n_dot = []
                for k in range(1, n):
                    bk = _bernoulli(k)
                    if abs(bk) < _EPS:
                        continue
                    coeff = bk / factorial(k)

                    for partition in _ordered_partitions(n - 1, k):
                        ad_result = _nested_ad(h_terms, omega_dot_list, partition, t_start)
                        for term in ad_result:
                            g_orig = term.g
                            omega_n_dot.append(_MagnusTerm(term.operator, lambda t, _g=g_orig, _c=coeff: _c * _g(t)))  # type: ignore[misc]

            omega_dot_list.append(omega_n_dot)

            # Phase factor: H_eff = (1/dt) Σ_n (-i)^{n-1} Ω_n^{(H)}
            # because the builder exponentiates as exp(-i H_eff dt) but
            # the physical propagator is exp(Σ_n (-i)^n Ω_n^{(H)}).
            phase: complex = (-1j) ** (n - 1)

            # Integrate each dΩ_n/dt term over [t_start, t_end] and merge into coefficient dict
            for term in omega_n_dot:
                scalar_real, _ = integrate.quad(term.g, t_start, t_end)
                scalar: complex = phase * scalar_real / dt
                if abs(scalar) < _EPS:
                    continue
                for label, c in zip(term.operator.pauli_strings, term.operator.coefficients, strict=True):
                    merged[label] = merged.get(label, 0.0) + scalar * complex(c)

        # Build final QubitHamiltonian from merged terms
        from qdk_chemistry.data import QubitHamiltonian as _QubitHamiltonian  # noqa: PLC0415

        # Filter near-zero terms
        merged = {k: v for k, v in merged.items() if abs(v) > _EPS}

        if not merged:
            return _QubitHamiltonian(["I" * container.num_qubits], np.array([0.0]))

        labels = list(merged.keys())
        coeffs = np.array([merged[k] for k in labels])
        return _QubitHamiltonian(labels, coeffs)

    def name(self) -> str:
        """Return ``magnus`` as the algorithm name."""
        return "magnus"
