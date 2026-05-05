"""QDK/Chemistry resource estimator abstractions.

This module defines the abstract base class for resource estimator algorithms
that estimate quantum resources required to execute a circuit.
"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from abc import abstractmethod

from qdk_chemistry.algorithms.base import Algorithm, AlgorithmFactory
from qdk_chemistry.data import Circuit
from qdk_chemistry.data.resource_estimator_data import ResourceEstimatorData

__all__: list[str] = ["ResourceEstimator", "ResourceEstimatorFactory"]


class ResourceEstimator(Algorithm):
    """Abstract base class for quantum resource estimator algorithms."""

    def __init__(self):
        """Initialize the ResourceEstimator with default settings."""
        super().__init__()

    def type_name(self) -> str:
        """Return the algorithm type name as resource_estimator."""
        return "resource_estimator"

    @abstractmethod
    def _run_impl(
        self,
        circuit: Circuit,
    ) -> ResourceEstimatorData:
        """Estimate the quantum resources required for the given circuit.

        Estimation parameters are provided via ``self.settings()``.

        Args:
            circuit: The quantum circuit to estimate resources for.

        Returns:
            ResourceEstimatorData: The estimated resources.

        """


class ResourceEstimatorFactory(AlgorithmFactory):
    """Factory class for creating ResourceEstimator instances."""

    def __init__(self):
        """Initialize the ResourceEstimatorFactory."""
        super().__init__()

    def algorithm_type_name(self) -> str:
        """Return the algorithm type name as resource_estimator."""
        return "resource_estimator"

    def default_algorithm_name(self) -> str:
        """Return the qdk_qre_v1 as default algorithm name."""
        return "qdk_qre_v1"
