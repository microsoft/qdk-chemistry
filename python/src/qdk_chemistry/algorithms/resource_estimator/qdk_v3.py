"""QDK/Chemistry Resource Estimator v3 using the QRE Pareto-front API.

This module provides a ResourceEstimator implementation that uses the QDK
QRE v3 estimation backend to produce a Pareto-optimal frontier of resource
estimation results exploring the qubits-vs-runtime tradeoff space.
"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------
import random
import time
from typing import Any

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
    """Settings for the QDK QRE v3 Resource Estimator."""

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
        self._set_default("qec_scheme", "string", "surface_code", "QEC scheme: 'surface_code'")
        self._set_default("factory", "string", "round_based", "Magic state factory model: 'round_based'")
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
    """QDK QRE v3 Resource Estimator producing a Pareto frontier."""

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
        """Estimate the Pareto-optimal quantum resource frontier for the circuit."""
        Logger.trace_entering()

        from qdk.qre import PSSPC, LatticeSurgery  # noqa: PLC0415
        from qdk.qre import estimate as qre_estimate  # noqa: PLC0415
        from qdk.qre.models import GateBased, Majorana, RoundBasedFactory, SurfaceCode  # noqa: PLC0415

        max_error = self._settings.get("error_budget")
        use_graph = self._settings.get("use_graph")
        qubit_model_name = self._settings.get("qubit_model")
        qec_scheme_name = self._settings.get("qec_scheme")
        factory_name = self._settings.get("factory")

        app = self._make_application(circuit)

        if qubit_model_name == "majorana":
            arch = Majorana()
        else:
            arch = GateBased(
                gate_time=int(self._settings.get("gate_time")),
                measurement_time=int(self._settings.get("measurement_time")),
            )

        if qec_scheme_name != "surface_code":
            raise ValueError(f"Unsupported QRE v3 QEC scheme: {qec_scheme_name}")
        if factory_name != "round_based":
            raise ValueError(f"Unsupported QRE v3 factory model: {factory_name}")

        isa_query = SurfaceCode.q() * RoundBasedFactory.q()
        trace_query = PSSPC.q() * LatticeSurgery.q(slow_down_factor=self._settings.get("slow_down_factors"))

        table = qre_estimate(
            application=app,
            architecture=arch,
            isa_query=isa_query,
            trace_query=trace_query,
            max_error=max_error,
            use_graph=use_graph,
        )

        results = [
            self._to_resource_estimator_data(
                entry,
                qubit_model_name=qubit_model_name,
                qec_scheme_name=qec_scheme_name,
                factory_name=factory_name,
                max_error=max_error,
            )
            for entry in table
        ]
        results.sort(key=lambda result: result.physical_counts.physical_qubits)
        return results

    @staticmethod
    def _make_application(circuit: Circuit) -> Any:
        """Create a qdk.qre Application from a qdk_chemistry Circuit."""
        if circuit._qsharp_factory is not None:  # noqa: SLF001
            return QdkQreV3._make_qsharp_application(
                circuit._qsharp_factory.program,  # noqa: SLF001
                args=tuple(circuit._qsharp_factory.parameter.values()),  # noqa: SLF001
            )
        if circuit.qasm is not None:
            return QdkQreV3._make_openqasm_application(circuit.qasm)
        if circuit.qir is not None:
            return QdkQreV3._make_openqasm_application(circuit.get_qasm())

        raise RuntimeError("Cannot estimate resources: no Q# factory data, QASM, or QIR representation is available.")

    @staticmethod
    def _make_qsharp_application(program: Any, *, args: tuple[Any, ...]) -> Any:
        """Create a lightweight QRE application for Q# entry expressions."""
        from qdk.qre._application import Application  # noqa: PLC0415

        class QSharpApplication(Application[None]):
            def __init__(self, entry_expr: Any, args: tuple[Any, ...]) -> None:
                self.entry_expr = entry_expr
                self.args = args

            def get_trace(self, _parameters: None = None) -> Any:
                return QdkQreV3._trace_from_entry_expr(self.entry_expr, *self.args)

        return QSharpApplication(program, args)

    @staticmethod
    def _make_openqasm_application(program: str) -> Any:
        """Create a lightweight QRE application for OpenQASM programs."""
        from qdk.qre._application import Application  # noqa: PLC0415

        class OpenQASMApplication(Application[None]):
            def __init__(self, program: str) -> None:
                self.program = program

            def get_trace(self, _parameters: None = None) -> Any:
                from qdk import code  # noqa: PLC0415
                from qdk.openqasm import ProgramType, import_openqasm  # noqa: PLC0415

                program = self.program
                if isinstance(program, str):
                    for _ in range(1_000):
                        name = f"openqasm{random.randint(0, 1_000_000)}"
                        if not hasattr(code, "qasm_import") or not hasattr(code.qasm_import, name):
                            break
                    else:
                        raise RuntimeError("Failed to find a unique name for the OpenQASM program.")

                    import_openqasm(program, name=name, program_type=ProgramType.File)
                    program = getattr(code.qasm_import, name)

                return QdkQreV3._trace_from_entry_expr(program)

        return OpenQASMApplication(program)

    @staticmethod
    def _trace_from_entry_expr(entry_expr: Any, *args: Any) -> Any:
        """Convert QDK logical counts into a QRE trace."""
        from qdk._interpreter import logical_counts  # noqa: PLC0415
        from qdk.estimator import LogicalCounts as QdkLogicalCounts  # noqa: PLC0415
        from qdk.qre._qre import Trace  # noqa: PLC0415
        from qdk.qre.instruction_ids import CCX, MEAS_Z, READ_FROM_MEMORY, RZ, WRITE_TO_MEMORY, T  # noqa: PLC0415
        from qdk.qre.property_keys import (  # noqa: PLC0415
            ALGORITHM_COMPUTE_QUBITS,
            ALGORITHM_MEMORY_QUBITS,
            EVALUATION_TIME,
        )

        start = time.time_ns()
        counts = entry_expr if isinstance(entry_expr, QdkLogicalCounts) else logical_counts(entry_expr, *args)
        evaluation_time = time.time_ns() - start

        ccx_count = counts.get("cczCount", 0) + counts.get("ccixCount", 0)
        num_qubits = counts.get("numQubits", 0)
        compute_qubits = counts.get("numComputeQubits", num_qubits)
        memory_qubits = num_qubits - compute_qubits
        trace = Trace(compute_qubits)

        rotation_count = counts.get("rotationCount", 0)
        rotation_depth = counts.get("rotationDepth", rotation_count)
        if rotation_count != 0 and rotation_depth != 0:
            for count, depth in QdkQreV3._bucketize_rotation_counts(rotation_count, rotation_depth):
                block = trace.add_block(repetitions=depth)
                for index in range(count):
                    block.add_operation(RZ, [index])

        if t_count := counts.get("tCount", 0):
            block = trace.add_block(repetitions=t_count)
            block.add_operation(T, [0])

        if ccx_count:
            block = trace.add_block(repetitions=ccx_count)
            block.add_operation(CCX, [0, 1, 2])

        if meas_count := counts.get("measurementCount", 0):
            block = trace.add_block(repetitions=meas_count)
            block.add_operation(MEAS_Z, [0])

        if memory_qubits != 0:
            trace.memory_qubits = memory_qubits

            if rfm_count := counts.get("readFromMemoryCount", 0):
                block = trace.add_block(repetitions=rfm_count)
                block.add_operation(READ_FROM_MEMORY, [0, compute_qubits])

            if wtm_count := counts.get("writeToMemoryCount", 0):
                block = trace.add_block(repetitions=wtm_count)
                block.add_operation(WRITE_TO_MEMORY, [0, compute_qubits])

        trace.set_property(EVALUATION_TIME, evaluation_time)
        trace.set_property(ALGORITHM_COMPUTE_QUBITS, compute_qubits)
        trace.set_property(ALGORITHM_MEMORY_QUBITS, memory_qubits)
        return trace

    @staticmethod
    def _bucketize_rotation_counts(rotation_count: int, rotation_depth: int) -> list[tuple[int, int]]:
        """Return rotation count/depth buckets for QRE trace construction."""
        if rotation_depth == 0:
            return []

        base = rotation_count // rotation_depth
        extra = rotation_count % rotation_depth

        result: list[tuple[int, int]] = []
        if extra > 0:
            result.append((base + 1, extra))
        if rotation_depth - extra > 0:
            result.append((base, rotation_depth - extra))
        return result

    def _to_resource_estimator_data(
        self,
        entry: Any,
        *,
        qubit_model_name: str,
        qec_scheme_name: str,
        factory_name: str,
        max_error: float,
    ) -> ResourceEstimatorData:
        """Convert one QRE v3 Pareto entry into ResourceEstimatorData."""
        return ResourceEstimatorData(
            logical_counts=LogicalCounts(num_qubits=0),
            physical_counts=PhysicalCounts(
                physical_qubits=int(getattr(entry, "qubits", 0)),
                runtime=int(getattr(entry, "runtime", 0)),
                runtime_unit="ns",
            ),
            logical_qubit=LogicalQubit(),
            error_budget=ErrorBudget(),
            estimator=self.name(),
            status="success",
            error=float(getattr(entry, "error", 0.0)),
            config=EstimationConfig(
                qubit_model=qubit_model_name,
                qec_scheme=qec_scheme_name,
                error_budget=max_error,
                description=str(getattr(entry, "source", "")),
                gate_time_ns=int(self._settings.get("gate_time")) if qubit_model_name == "gate_based" else 0,
                measurement_time_ns=(
                    int(self._settings.get("measurement_time")) if qubit_model_name == "gate_based" else 0
                ),
                factory=factory_name,
            ),
        )
