"""QDK/Chemistry Resource Estimator implementation using QDK.

This module provides a ResourceEstimator implementation that uses the QDK
resource estimation backend to estimate quantum resources required for a circuit.
It supports circuits provided as Q# factory data or QASM.
"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------
import qsharp
import qsharp.estimator
import qsharp.openqasm

from qdk_chemistry.algorithms.resource_estimator.base import ResourceEstimator
from qdk_chemistry.data import Circuit, Settings
from qdk_chemistry.data.resource_estimator_data import (
    ErrorBudget,
    EstimationConfig,
    LogicalCounts,
    LogicalQubit,
    PhysicalCounts,
    ResourceEstimatorData,
)
from qdk_chemistry.utils import Logger

__all__: list[str] = ["QdkQreV1", "QdkQreV1Settings"]


class QdkQreV1Settings(Settings):
    """Settings for the QDK QRE v1 Resource Estimator."""

    def __init__(self) -> None:
        """Initialize QDK QRE v1 settings."""
        Logger.trace_entering()
        super().__init__()
        self._set_default(
            "error_budget", "double", 0.001,
            "Total error budget for the estimation"
        )


class QdkQreV1(ResourceEstimator):
    """QDK QRE v1 Resource Estimator algorithm implementation.

    Uses the Q# resource estimator (v1/v2 API) to estimate physical
    resources required for a quantum circuit.
    """

    def __init__(self) -> None:
        """Initialize the QDK QRE v1 Resource Estimator."""
        Logger.trace_entering()
        super().__init__()
        self._settings = QdkQreV1Settings()

    def name(self) -> str:
        """Return the algorithm name as qdk_qre_v1."""
        return "qdk_qre_v1"

    def _run_impl(
        self,
        circuit: Circuit,
    ) -> ResourceEstimatorData:
        """Estimate the quantum resources required for the given circuit.

        Estimation parameters are taken from ``self.settings()``.

        Args:
            circuit: The quantum circuit to estimate resources for.

        Returns:
            ResourceEstimatorData: The estimated resources.

        Raises:
            RuntimeError: If no suitable circuit representation is available for estimation.

        """
        Logger.trace_entering()

        params = {"errorBudget": self._settings.get("error_budget")}

        if circuit._qsharp_factory is not None:
            result = qsharp.estimate(
                circuit._qsharp_factory.program,
                params,
                *circuit._qsharp_factory.parameter.values(),
            )
        elif circuit.qasm is not None:
            result = qsharp.openqasm.estimate(circuit.qasm, params)
        else:
            raise RuntimeError(
                "Cannot estimate resources: no Q# factory data or QASM representation is available."
            )

        # Convert the EstimatorResult (dict subclass) to typed data
        raw = dict(result) if isinstance(result, dict) else result

        lc_raw = raw.get("logicalCounts", {})
        pc_raw = raw.get("physicalCounts", {})
        bd_raw = pc_raw.get("breakdown", {})
        lq_raw = raw.get("logicalQubit", {})
        eb_raw = raw.get("errorBudget", {})
        jp_raw = raw.get("jobParams", {})

        # Build provenance config from jobParams
        qp = jp_raw.get("qubitParams", {})
        qec = jp_raw.get("qecScheme", {})
        config = EstimationConfig(
            qubit_model=qp.get("name", qp.get("instructionSet", "")),
            qec_scheme=qec.get("name", ""),
            error_budget=float(jp_raw.get("errorBudget", 0.0)),
        )

        return ResourceEstimatorData(
            logical_counts=LogicalCounts(
                num_qubits=lc_raw.get("numQubits", 0),
                t_count=lc_raw.get("tCount", 0),
                rotation_count=lc_raw.get("rotationCount", 0),
                rotation_depth=lc_raw.get("rotationDepth", 0),
                ccz_count=lc_raw.get("cczCount", 0),
                ccix_count=lc_raw.get("ccixCount", 0),
                measurement_count=lc_raw.get("measurementCount", 0),
            ),
            physical_counts=PhysicalCounts(
                physical_qubits=pc_raw.get("physicalQubits", 0),
                runtime=pc_raw.get("runtime", 0),
                runtime_unit="ns",
                rqops=pc_raw.get("rqops", 0),
                algorithm_qubits=bd_raw.get("physicalQubitsForAlgorithm", 0),
                factory_qubits=bd_raw.get("physicalQubitsForTfactories", 0),
                algorithmic_logical_depth=bd_raw.get("algorithmicLogicalDepth", 0),
                logical_depth=bd_raw.get("logicalDepth", 0),
            ),
            logical_qubit=LogicalQubit(
                code_distance=lq_raw.get("codeDistance", 0),
                logical_cycle_time=lq_raw.get("logicalCycleTime", 0),
                logical_error_rate=lq_raw.get("logicalErrorRate", 0.0),
                physical_qubits=lq_raw.get("physicalQubits", 0),
            ),
            error_budget=ErrorBudget(
                logical=eb_raw.get("logical", 0.0),
                rotations=eb_raw.get("rotations", 0.0),
                tstates=eb_raw.get("tstates", 0.0),
            ),
            estimator=self.name(),
            status=raw.get("status", "unknown"),
            error=eb_raw.get("logical", 0.0) + eb_raw.get("rotations", 0.0) + eb_raw.get("tstates", 0.0),
            config=config,
        )
