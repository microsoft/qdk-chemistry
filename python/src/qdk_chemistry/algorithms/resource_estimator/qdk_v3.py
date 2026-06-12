"""QDK/Chemistry Resource Estimator v3 using the QRE Pareto-front API.

This module provides a ResourceEstimator implementation that uses the QDK
QRE v3 estimation backend (``qsharp.qre.estimate``) to produce a
Pareto-optimal frontier of resource estimation results exploring the
qubits-vs-runtime tradeoff space.
"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------
from qsharp.qre import PSSPC, LatticeSurgery
from qsharp.qre import estimate as qre_estimate
from qsharp.qre.application import OpenQASMApplication, QSharpApplication
from qsharp.qre.models import GateBased, Majorana, RoundBasedFactory, SurfaceCode

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

__all__: list[str] = ["QdkQreV3", "QdkQreV3Settings"]


class QdkQreV3Settings(Settings):
    """Settings for the QDK QRE v3 Resource Estimator (Pareto front)."""

    def __init__(self) -> None:
        """Initialize QDK QRE v3 settings."""
        Logger.trace_entering()
        super().__init__()
        self._set_default("error_budget", "double", 0.01, "Maximum total error budget for the estimation")
        self._set_default(
            "qubit_model",
            "string",
            "gate_based",
            "Qubit technology model: 'gate_based' or 'majorana'",
        )
        self._set_default(
            "qec_scheme",
            "string",
            "surface_code",
            "QEC scheme: 'surface_code'",
        )
        self._set_default(
            "factory",
            "string",
            "round_based",
            "Magic state factory model: 'round_based'",
        )
        self._set_default(
            "gate_time",
            "int",
            50,
            "Single-qubit gate time in nanoseconds (gate_based only)",
        )
        self._set_default(
            "measurement_time",
            "int",
            100,
            "Measurement time in nanoseconds (gate_based only)",
        )
        self._set_default(
            "use_graph",
            "bool",
            False,
            "Use graph-based pruning (faster but may miss some Pareto points)",
        )
        self._set_default(
            "slow_down_factors",
            "vector<double>",
            [1.0, 2.0, 3.0, 5.0, 8.0, 10.0, 15.0, 20.0, 25.0, 30.0],
            "LatticeSurgery slow-down factors for Pareto exploration",
        )


class QdkQreV3(ResourceEstimator):
    """QDK QRE v3 Resource Estimator producing a Pareto frontier.

    Uses the Q# QRE v3 API (``qsharp.qre.estimate``) to explore the full
    space of QEC code distances and factory configurations, returning
    all Pareto-optimal (qubits, runtime) tradeoff points.
    """

    def __init__(self) -> None:
        """Initialize the QDK QRE v3 Resource Estimator."""
        Logger.trace_entering()
        super().__init__()
        self._settings = QdkQreV3Settings()

    def name(self) -> str:
        """Return the algorithm name as qdk_qre_v3."""
        return "qdk_qre_v3"

    def _run_impl(
        self,
        circuit: Circuit,
    ) -> list[ResourceEstimatorData]:
        """Estimate the Pareto-optimal quantum resource frontier for the circuit.

        Uses the QRE v3 estimation engine to explore the full design space
        of QEC configurations and returns a list of Pareto-optimal results
        sorted by ascending physical qubits (descending runtime).

        Args:
            circuit: The quantum circuit to estimate resources for.

        Returns:
            list[ResourceEstimatorData]: Pareto-optimal estimation results.

        Raises:
            RuntimeError: If no suitable circuit representation is available.

        """
        Logger.trace_entering()

        max_error = self._settings.get("error_budget")
        use_graph = self._settings.get("use_graph")
        qubit_model_name = self._settings.get("qubit_model")
        qec_scheme_name = self._settings.get("qec_scheme")
        factory_name = self._settings.get("factory")

        # Build the Application from the circuit
        app = self._make_application(circuit)

        # Build the Architecture
        if qubit_model_name == "majorana":
            arch = Majorana()
        else:
            arch = GateBased(
                gate_time=int(self._settings.get("gate_time")),
                measurement_time=int(self._settings.get("measurement_time")),
            )

        # Build the ISA query (QEC + factory)
        isa_query = SurfaceCode.q() * RoundBasedFactory.q()

        # Build the trace query with varied slow-down factors for
        # Pareto exploration of the qubits-vs-runtime tradeoff.
        slow_down_factors = self._settings.get("slow_down_factors")
        trace_query = PSSPC.q() * LatticeSurgery.q(slow_down_factor=slow_down_factors)

        # Run the estimation — returns an EstimationTable (Pareto front)
        table = qre_estimate(
            application=app,
            architecture=arch,
            isa_query=isa_query,
            trace_query=trace_query,
            max_error=max_error,
            use_graph=use_graph,
        )

        # Convert each Pareto-optimal entry to a ResourceEstimatorData
        results: list[ResourceEstimatorData] = []
        for entry in table:
            config = EstimationConfig(
                qubit_model=qubit_model_name,
                qec_scheme=qec_scheme_name,
                error_budget=max_error,
                description=str(entry.source) if hasattr(entry, "source") else "",
                gate_time_ns=int(self._settings.get("gate_time")) if qubit_model_name == "gate_based" else 0,
                measurement_time_ns=int(self._settings.get("measurement_time"))
                if qubit_model_name == "gate_based"
                else 0,
                factory=factory_name,
            )

            results.append(
                ResourceEstimatorData(
                    logical_counts=LogicalCounts(
                        num_qubits=0,  # Not available per-entry in QRE v3
                    ),
                    physical_counts=PhysicalCounts(
                        physical_qubits=entry.qubits,
                        runtime=entry.runtime,
                        runtime_unit="ns",
                    ),
                    logical_qubit=LogicalQubit(),
                    error_budget=ErrorBudget(),
                    estimator=self.name(),
                    status="success",
                    error=entry.error,
                    config=config,
                )
            )

        # Sort by ascending qubits (the table may already be sorted but ensure)
        results.sort(key=lambda r: r.physical_counts.physical_qubits)

        return results

    @staticmethod
    def _make_application(circuit: Circuit):
        """Create a qsharp.qre Application from a qdk_chemistry Circuit.

        Args:
            circuit: The circuit to wrap.

        Returns:
            A qsharp.qre.Application instance.

        Raises:
            RuntimeError: If no suitable representation is available.

        """
        if circuit._qsharp_factory is not None:  # noqa: SLF001
            return QSharpApplication(
                circuit._qsharp_factory.program,  # noqa: SLF001
                args=tuple(circuit._qsharp_factory.parameter.values()),  # noqa: SLF001
            )
        if circuit.qasm is not None:
            return OpenQASMApplication(circuit.qasm)
        if circuit.qir is not None:
            # Convert QIR → QASM so the OpenQASM application can process it
            qasm_str = circuit.get_qasm()
            return OpenQASMApplication(qasm_str)
        raise RuntimeError("Cannot estimate resources: no Q# factory data, QASM, or QIR representation is available.")
