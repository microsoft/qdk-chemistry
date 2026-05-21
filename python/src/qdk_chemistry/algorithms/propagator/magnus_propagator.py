r"""Time-averaged propagator via first-order Magnus expansion.

Computes the effective Hamiltonian for a time interval as the time average:

.. math::

    H_\text{eff} = \frac{1}{\delta t} \int_{t_1}^{t_2} H(t')\,\mathrm{d}t'

For the driven case :math:`H(t) = H_0 + f(t)\,H_1` this reduces to
:math:`H_\text{eff} = H_0 + \bar f\,H_1` where :math:`\bar f` is the
time-averaged drive.
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
    from qdk_chemistry.data import QubitHamiltonian, TimeDependentQubitHamiltonian

__all__: list[str] = ["MagnusPropagator", "MagnusPropagatorSettings"]


class MagnusPropagatorSettings(Settings):
    """Settings for the Magnus propagator."""

    def __init__(self):
        """Initialize settings with default Magnus expansion order."""
        super().__init__()
        self._set_default("order", "int", 1, "Magnus expansion order (currently only 1 is supported).")


class MagnusPropagator(Propagator):
    r"""First-order Magnus propagator (time averaging).

    Evaluates the effective Hamiltonian for an interval :math:`[t_1, t_2]`
    as the time average :math:`H_\text{eff} = H_0 + \bar f\,H_1`.

    For :class:`~qdk_chemistry.data.time_dependent_qubit_hamiltonian.containers.driven.DrivenContainer`
    Hamiltonians the drive integral reduces to a scalar quadrature.

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
    ) -> QubitHamiltonian:
        r"""Order-1 Magnus expansion (time averaging) for driven Hamiltonians.

        Computes :math:`H_\text{eff} = \frac{1}{\delta t} \int_{t_1}^{t_2} H(t')\,\mathrm{d}t'`
        where :math:`H(t) = H_0 + f(t)\,H_1`.

        Partitions from :math:`H_0` and :math:`H_1` are preserved via
        :meth:`~qdk_chemistry.data.QubitHamiltonian.__mul__` and
        :meth:`~qdk_chemistry.data.QubitHamiltonian.__add__`.

        Returns :math:`H_\text{eff}`.

        """
        dt = t_end - t_start
        h0, h1, drive = container.base_hamiltonian, container.drive_hamiltonian, container.drive
        f_avg = integrate.quad(drive, t_start, t_end)[0] / dt

        return h0 + f_avg * h1

    def name(self) -> str:
        """Return ``magnus`` as the algorithm name."""
        return "magnus"
