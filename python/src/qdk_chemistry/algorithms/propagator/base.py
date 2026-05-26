"""QDK/Chemistry propagator abstractions.

A propagator evaluates a time-dependent Hamiltonian over a time interval
and returns a single effective (time-independent) qubit Hamiltonian that
approximates the average interaction during that interval.
"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from abc import abstractmethod

from qdk_chemistry.algorithms.base import Algorithm, AlgorithmFactory
from qdk_chemistry.data import QubitHamiltonian, TimeDependentQubitHamiltonian

__all__: list[str] = ["Propagator", "PropagatorFactory"]


class Propagator(Algorithm):
    r"""Abstract base for propagator algorithms.

    A propagator maps a time-dependent Hamiltonian and a time interval
    :math:`[t_1, t_2]` to an effective time-independent
    :class:`~qdk_chemistry.data.QubitHamiltonian`:

    .. math::

        H_{\mathrm{eff}} = \frac{1}{\delta t}
        \int_{t_1}^{t_2} H(t')\,\mathrm{d}t'

    Concrete implementations may compute the integral analytically,
    numerically, or via other approximation schemes.
    """

    def __init__(self):
        """Initialize the Propagator."""
        super().__init__()

    def type_name(self) -> str:
        """Return ``propagator`` as the algorithm type name."""
        return "propagator"

    @abstractmethod
    def _run_impl(
        self,
        hamiltonian: TimeDependentQubitHamiltonian,
        t_start: float,
        t_end: float,
    ) -> QubitHamiltonian:
        """Evaluate the effective Hamiltonian over a time interval.

        Args:
            hamiltonian: Time-dependent Hamiltonian.
            t_start: Start of the interval.
            t_end: End of the interval.

        Returns:
            Effective time-independent qubit Hamiltonian for the interval.

        """


class PropagatorFactory(AlgorithmFactory):
    """Factory class for creating Propagator instances."""

    def algorithm_type_name(self) -> str:
        """Return ``propagator`` as the algorithm type name."""
        return "propagator"

    def default_algorithm_name(self) -> str:
        """Return ``magnus`` as the default algorithm name."""
        return "magnus"
