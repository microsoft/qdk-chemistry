r"""Time-averaged propagator with Magnus expansion.

Computes the effective (time-independent) Hamiltonian for a time interval
:math:`[t_1, t_2]` via the Magnus expansion of the time-ordered
propagator :math:`U(t_2, t_1) = \exp(\Omega(t_2, t_1))`, where

.. math::

    \Omega = \Omega_1 + \Omega_2 + \Omega_3 + \cdots

is a series of nested time integrals of :math:`H(t)`.  The leading
(order-1) term is the time-averaged Hamiltonian

.. math::

    \Omega_1 = \int_{t_1}^{t_2} H(t')\,\mathrm{d}t',

while higher orders add nested-commutator corrections, e.g.

.. math::

    \Omega_2 = -\frac{1}{2}
        \int_{t_1}^{t_2}\!\mathrm{d}t'
        \int_{t_1}^{t'}\!\mathrm{d}t''\,[H(t'), H(t'')],

and in general the :math:`\Omega_n` follow the recursion

.. math::

    \dot\Omega_n
    = \sum_{k=1}^{n-1} \frac{B_k}{k!}
      \sum_{j_1+\cdots+j_k=n-1}
      \mathrm{ad}_{\Omega_{j_1}} \cdots
      \mathrm{ad}_{\Omega_{j_k}}(H(t)),

where :math:`B_k` are Bernoulli numbers.

Implementation status
---------------------
Only the leading-order (order-1) term is currently implemented, and only
for
:class:`~qdk_chemistry.data.time_dependent_qubit_hamiltonian.containers.driven.DrivenContainer`
Hamiltonians of the form :math:`H(t) = H_0 + f(t)\,H_1`.  Requesting an
``order`` greater than 1, or passing any other container type, raises
:class:`NotImplementedError`.

Accuracy
--------
The order-1 propagator gives :math:`O(\Delta t^2)` per-step accuracy.
"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from __future__ import annotations

from typing import TYPE_CHECKING

from scipy import integrate

from qdk_chemistry.data import Settings
from qdk_chemistry.data.time_dependent_qubit_hamiltonian.containers.driven import DrivenContainer

from .base import Propagator

if TYPE_CHECKING:
    from qdk_chemistry.data import QubitOperator, TimeDependentQubitHamiltonian

__all__: list[str] = ["MagnusPropagator", "MagnusPropagatorSettings"]


class MagnusPropagatorSettings(Settings):
    """Settings for the Magnus propagator.

    Attributes:
        order (int): Magnus expansion order (default = 1). Only the
            leading-order term (order 1, time averaging) is currently
            implemented; any larger value raises
            :class:`NotImplementedError` when the propagator runs.

    """

    def __init__(self):
        """Initialize settings with default Magnus expansion order."""
        super().__init__()
        self._set_default("order", "int", 1, "Magnus expansion order (currently only 1 is supported).")


class MagnusPropagator(Propagator):
    r"""Magnus propagator for time-dependent Hamiltonian simulation.

    Evaluates the effective Hamiltonian for an interval :math:`[t_1, t_2]`
    via the Magnus expansion.  Currently only the leading-order
    (**order 1**, time averaging) term is implemented, and only for
    :class:`~qdk_chemistry.data.time_dependent_qubit_hamiltonian.containers.driven.DrivenContainer`
    Hamiltonians :math:`H(t) = H_0 + f(t)\,H_1`, for which the drive
    integral reduces to a scalar quadrature and

    .. math::

        H_\text{eff} = H_0 + \bar f\,H_1,
        \qquad
        \bar f = \frac{1}{\delta t}\int_{t_1}^{t_2} f(t')\,\mathrm{d}t'.

    The propagator returns this
    :math:`H_\text{eff} = \Omega_1 / \delta t` (the time-averaged exponent
    divided by :math:`\delta t = t_2 - t_1`) so that a time-stepping
    integrator that multiplies by :math:`\delta t` recovers the full
    exponent :math:`\Omega_1`.  This gives :math:`O(\Delta t^2)` per-step accuracy.

    Requesting an ``order`` greater than 1, or passing any other container
    type, raises :class:`NotImplementedError`.

    """

    def __init__(self):
        """Initialize the Magnus propagator."""
        super().__init__()
        self._settings = MagnusPropagatorSettings()

    def _run_impl(
        self,
        hamiltonian: TimeDependentQubitHamiltonian,
        t_start: float,
        t_end: float,
    ) -> QubitOperator:
        r"""Compute the effective Hamiltonian over :math:`[t_1, t_2]`.

        Args:
            hamiltonian: Time-dependent Hamiltonian.
            t_start: Start of the interval.
            t_end: End of the interval.

        Returns:
            Effective time-independent qubit Hamiltonian for the interval.

        Raises:
            ValueError: If *t_end* is not greater than *t_start*.
            NotImplementedError: If the Hamiltonian container is not a
                :class:`~qdk_chemistry.data.time_dependent_qubit_hamiltonian.containers.driven.DrivenContainer`,
                or if the requested Magnus expansion ``order`` is greater
                than 1.

        """
        if t_end <= t_start:
            raise ValueError(f"t_end ({t_end}) must be greater than t_start ({t_start}).")

        order: int = self._settings.get("order")
        container = hamiltonian.get_container()
        if not isinstance(container, DrivenContainer):
            raise NotImplementedError(
                f"Magnus propagator is not yet implemented for {type(container).__name__} containers."
            )
        if order > 1:
            raise NotImplementedError(
                f"Magnus expansion order {order} is not yet implemented. Only order 1 (time averaging) is supported."
            )
        return self._magnus_driven(container, t_start, t_end)

    # ------------------------------------------------------------------
    # Order-1 Magnus: time-averaged Hamiltonian
    # ------------------------------------------------------------------

    @staticmethod
    def _magnus_driven(
        container: DrivenContainer,
        t_start: float,
        t_end: float,
    ) -> QubitOperator:
        r"""Order-1 Magnus expansion (time averaging) for driven Hamiltonians.

        For the driven case :math:`H(t) = H_0 + f(t)\,H_1` the order-1
        (leading) Magnus term is the time-averaged Hamiltonian

        .. math::

            \Omega_1 = \int_{t_1}^{t_2} H(t')\,\mathrm{d}t'
                = \delta t\,H_0
                + \left(\int_{t_1}^{t_2} f(t')\,\mathrm{d}t'\right) H_1,

        because :math:`H_0` is constant and only the scalar drive
        :math:`f(t)` carries the time dependence.  The drive integral is
        evaluated by numerical quadrature.

        The propagator returns
        :math:`H_\text{eff} = \Omega_1 / \delta t = H_0 + \bar f\,H_1`,
        the exponent divided by :math:`\delta t = t_2 - t_1`, so that a
        time-stepping integrator (which multiplies by ``dt``) recovers the
        full exponent :math:`\Omega_1`.

        Only the leading-order term is computed; this gives
        :math:`O(\delta t^2)` per-step accuracy.

        Partitions from :math:`H_0` and :math:`H_1` are preserved via
        :meth:`~qdk_chemistry.data.QubitOperator.__mul__` and
        :meth:`~qdk_chemistry.data.QubitOperator.__add__`.

        Returns the effective time-independent Hamiltonian
        :math:`H_\text{eff}`.

        """
        dt = t_end - t_start
        h0, h1, drive = container.base_hamiltonian, container.drive_hamiltonian, container.drive
        f_avg = integrate.quad(drive, t_start, t_end)[0] / dt

        return h0 + f_avg * h1

    def name(self) -> str:
        """Return ``magnus`` as the algorithm name."""
        return "magnus"
