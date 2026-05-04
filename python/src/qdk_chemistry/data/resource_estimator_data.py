"""QDK/Chemistry Resource Estimator Data module."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from typing import Any

import h5py

from qdk_chemistry.data.base import DataClass

__all__: list[str] = []


def _typed_obj_to_dict(obj: object) -> dict[str, Any]:
    """Serialize a __slots__-based object to a dict."""
    return {s: getattr(obj, s) for s in obj.__slots__}


def _typed_obj_to_hdf5(obj: object, group: h5py.Group) -> None:
    """Write a __slots__-based object as HDF5 attributes."""
    for s in obj.__slots__:
        group.attrs[s] = getattr(obj, s)


def _make_eq(cls):
    """Add __eq__ and __repr__ based on __slots__."""
    def __eq__(self, other):
        if not isinstance(other, cls):
            return NotImplemented
        return all(getattr(self, s) == getattr(other, s) for s in self.__slots__)

    def __repr__(self):
        fields = ", ".join(f"{s}={getattr(self, s)}" for s in self.__slots__)
        return f"{cls.__name__}({fields})"

    cls.__eq__ = __eq__
    cls.__repr__ = __repr__
    return cls


# ---------------------------------------------------------------------------
# Circuit-level metrics (backend-agnostic, pre-QEC)
# ---------------------------------------------------------------------------
@_make_eq
class CircuitCounts:
    """Circuit-level gate and depth counts.

    These metrics are available from any backend (Qiskit, Cirq, Q#)
    without requiring a QEC model.
    """

    __slots__ = (
        "depth",
        "num_gates",
        "num_single_qubit_clifford",
        "num_two_qubit_clifford",
        "num_non_clifford",
    )

    def __init__(
        self,
        depth: int = 0,
        num_gates: int = 0,
        num_single_qubit_clifford: int = 0,
        num_two_qubit_clifford: int = 0,
        num_non_clifford: int = 0,
    ) -> None:
        self.depth = depth
        self.num_gates = num_gates
        self.num_single_qubit_clifford = num_single_qubit_clifford
        self.num_two_qubit_clifford = num_two_qubit_clifford
        self.num_non_clifford = num_non_clifford


# ---------------------------------------------------------------------------
# Logical-level counts (application profile, QEC-agnostic)
# ---------------------------------------------------------------------------
@_make_eq
class LogicalCounts:
    """Logical resource counts from a quantum resource estimation."""

    __slots__ = (
        "num_qubits",
        "t_count",
        "rotation_count",
        "rotation_depth",
        "ccz_count",
        "ccix_count",
        "measurement_count",
    )

    def __init__(
        self,
        num_qubits: int = 0,
        t_count: int = 0,
        rotation_count: int = 0,
        rotation_depth: int = 0,
        ccz_count: int = 0,
        ccix_count: int = 0,
        measurement_count: int = 0,
    ) -> None:
        self.num_qubits = num_qubits
        self.t_count = t_count
        self.rotation_count = rotation_count
        self.rotation_depth = rotation_depth
        self.ccz_count = ccz_count
        self.ccix_count = ccix_count
        self.measurement_count = measurement_count


# ---------------------------------------------------------------------------
# Physical-level counts (post-QEC, architecture-dependent)
# ---------------------------------------------------------------------------
@_make_eq
class PhysicalCounts:
    """Physical resource counts from a quantum resource estimation."""

    __slots__ = (
        "physical_qubits",
        "runtime",
        "runtime_unit",
        "rqops",
        "algorithm_qubits",
        "factory_qubits",
        "algorithmic_logical_depth",
        "logical_depth",
    )

    def __init__(
        self,
        physical_qubits: int = 0,
        runtime: int = 0,
        runtime_unit: str = "ns",
        rqops: int = 0,
        algorithm_qubits: int = 0,
        factory_qubits: int = 0,
        algorithmic_logical_depth: int = 0,
        logical_depth: int = 0,
    ) -> None:
        self.physical_qubits = physical_qubits
        self.runtime = runtime
        self.runtime_unit = runtime_unit
        self.rqops = rqops
        self.algorithm_qubits = algorithm_qubits
        self.factory_qubits = factory_qubits
        self.algorithmic_logical_depth = algorithmic_logical_depth
        self.logical_depth = logical_depth


# ---------------------------------------------------------------------------
# Logical qubit properties
# ---------------------------------------------------------------------------
@_make_eq
class LogicalQubit:
    """Logical qubit properties from a quantum resource estimation."""

    __slots__ = (
        "code_distance",
        "logical_cycle_time",
        "logical_error_rate",
        "physical_qubits",
    )

    def __init__(
        self,
        code_distance: int = 0,
        logical_cycle_time: int = 0,
        logical_error_rate: float = 0.0,
        physical_qubits: int = 0,
    ) -> None:
        self.code_distance = code_distance
        self.logical_cycle_time = logical_cycle_time
        self.logical_error_rate = logical_error_rate
        self.physical_qubits = physical_qubits


# ---------------------------------------------------------------------------
# Error budget
# ---------------------------------------------------------------------------
@_make_eq
class ErrorBudget:
    """Error budget breakdown from a quantum resource estimation."""

    __slots__ = ("logical", "rotations", "tstates")

    def __init__(
        self,
        logical: float = 0.0,
        rotations: float = 0.0,
        tstates: float = 0.0,
    ) -> None:
        self.logical = logical
        self.rotations = rotations
        self.tstates = tstates


# ---------------------------------------------------------------------------
# Estimation configuration / provenance
# ---------------------------------------------------------------------------
@_make_eq
class EstimationConfig:
    """Configuration that produced a resource estimation result.

    Captures the architecture/QEC combination so that individual results
    within a Pareto set (or across different backends) can be traced back
    to the parameters that generated them.
    """

    __slots__ = (
        "qubit_model",
        "qec_scheme",
        "error_budget",
        "description",
    )

    def __init__(
        self,
        qubit_model: str = "",
        qec_scheme: str = "",
        error_budget: float = 0.0,
        description: str = "",
    ) -> None:
        self.qubit_model = qubit_model
        self.qec_scheme = qec_scheme
        self.error_budget = error_budget
        self.description = description


# ---------------------------------------------------------------------------
# Top-level DataClass
# ---------------------------------------------------------------------------
class ResourceEstimatorData(DataClass):
    """Resource estimation results from quantum resource estimation algorithms.

    The :attr:`config` describing the provenance.
    """

    # Class attribute for filename validation
    _data_type_name = "resource_estimator_data"

    # Serialization version for this class
    _serialization_version = "0.1.0"

    def __init__(
        self,
        logical_counts: LogicalCounts,
        physical_counts: PhysicalCounts,
        logical_qubit: LogicalQubit,
        error_budget: ErrorBudget,
        estimator: str,
        status: str = "success",
        error: float = 0.0,
        circuit_counts: CircuitCounts | None = None,
        config: EstimationConfig | None = None,
    ) -> None:
        """Initialize resource estimator data.

        Args:
            logical_counts: Logical resource counts (qubits, T-gates, rotations, etc.).
            physical_counts: Physical resource counts (physical qubits, runtime, RQOPS).
            logical_qubit: Logical qubit properties (code distance, cycle time, error rate).
            error_budget: Error budget breakdown (logical, rotations, T-states).
            estimator: Name of the estimator algorithm that produced this result.
            status: Status of the estimation (e.g. ``"success"``).
            error: Achieved total error probability of the estimation.
            circuit_counts: Optional circuit-level gate/depth counts.
            config: Optional estimation configuration describing the
                architecture/QEC combination that produced this result.

        """
        self.logical_counts = logical_counts
        self.physical_counts = physical_counts
        self.logical_qubit = logical_qubit
        self.error_budget = error_budget
        self.estimator = estimator
        self.status = status
        self.error = error
        self.circuit_counts = circuit_counts
        self.config = config
        super().__init__()

    # DataClass interface implementation

    def get_summary(self) -> str:
        """Get a human-readable summary of the resource estimation.

        Returns:
            str: Summary string describing the estimation results.

        """
        lc = self.logical_counts
        pc = self.physical_counts
        lines = [
            f"Resource Estimator Data (estimator: {self.estimator})",
            f"  Status: {self.status}",
            f"  Error: {self.error}",
            f"  Logical qubits: {lc.num_qubits}",
            f"  T-count: {lc.t_count}",
            f"  Physical qubits: {pc.physical_qubits}",
            f"  Runtime: {pc.runtime} {pc.runtime_unit}",
        ]
        if self.config is not None:
            cfg = self.config
            lines.append(f"  Qubit model: {cfg.qubit_model}")
            lines.append(f"  QEC scheme: {cfg.qec_scheme}")
        if self.circuit_counts is not None:
            cc = self.circuit_counts
            lines.append(f"  Circuit depth: {cc.depth}")
            lines.append(f"  Circuit gates: {cc.num_gates}")
        return "\n".join(lines)

    def to_json(self) -> dict[str, Any]:
        """Convert resource estimator data to a dictionary for JSON serialization.

        Returns:
            dict[str, Any]: Dictionary representation of the estimation data.

        """
        result: dict[str, Any] = {
            "estimator": self.estimator,
            "status": self.status,
            "error": self.error,
            "logical_counts": _typed_obj_to_dict(self.logical_counts),
            "physical_counts": _typed_obj_to_dict(self.physical_counts),
            "logical_qubit": _typed_obj_to_dict(self.logical_qubit),
            "error_budget": _typed_obj_to_dict(self.error_budget),
        }
        if self.circuit_counts is not None:
            result["circuit_counts"] = _typed_obj_to_dict(self.circuit_counts)
        if self.config is not None:
            result["config"] = _typed_obj_to_dict(self.config)
        return self._add_json_version(result)

    def to_hdf5(self, group: h5py.Group) -> None:
        """Save the estimation data to an HDF5 group.

        Args:
            group: HDF5 group or file to write the estimation data to.

        """
        self._add_hdf5_version(group)
        group.attrs["estimator"] = self.estimator
        group.attrs["status"] = self.status
        group.attrs["error"] = self.error

        _typed_obj_to_hdf5(self.logical_counts, group.create_group("logical_counts"))
        _typed_obj_to_hdf5(self.physical_counts, group.create_group("physical_counts"))
        _typed_obj_to_hdf5(self.logical_qubit, group.create_group("logical_qubit"))
        _typed_obj_to_hdf5(self.error_budget, group.create_group("error_budget"))

        if self.circuit_counts is not None:
            _typed_obj_to_hdf5(self.circuit_counts, group.create_group("circuit_counts"))
        if self.config is not None:
            _typed_obj_to_hdf5(self.config, group.create_group("config"))

    @classmethod
    def from_json(cls, json_data: dict[str, Any]) -> "ResourceEstimatorData":
        """Create resource estimator data from a JSON dictionary.

        Args:
            json_data: Dictionary containing the serialized data.

        Returns:
            ResourceEstimatorData: New instance reconstructed from JSON data.

        """
        cls._validate_json_version(cls._serialization_version, json_data)
        lc_d = json_data.get("logical_counts", {})
        pc_d = json_data.get("physical_counts", {})
        lq_d = json_data.get("logical_qubit", {})
        eb_d = json_data.get("error_budget", {})
        cc_d = json_data.get("circuit_counts")
        cfg_d = json_data.get("config")

        return cls(
            logical_counts=LogicalCounts(
                num_qubits=lc_d.get("num_qubits", 0),
                t_count=lc_d.get("t_count", 0),
                rotation_count=lc_d.get("rotation_count", 0),
                rotation_depth=lc_d.get("rotation_depth", 0),
                ccz_count=lc_d.get("ccz_count", 0),
                ccix_count=lc_d.get("ccix_count", 0),
                measurement_count=lc_d.get("measurement_count", 0),
            ),
            physical_counts=PhysicalCounts(
                physical_qubits=pc_d.get("physical_qubits", 0),
                runtime=pc_d.get("runtime", 0),
                runtime_unit=pc_d.get("runtime_unit", "ns"),
                rqops=pc_d.get("rqops", 0),
                algorithm_qubits=pc_d.get("algorithm_qubits", 0),
                factory_qubits=pc_d.get("factory_qubits", 0),
                algorithmic_logical_depth=pc_d.get("algorithmic_logical_depth", 0),
                logical_depth=pc_d.get("logical_depth", 0),
            ),
            logical_qubit=LogicalQubit(
                code_distance=lq_d.get("code_distance", 0),
                logical_cycle_time=lq_d.get("logical_cycle_time", 0),
                logical_error_rate=lq_d.get("logical_error_rate", 0.0),
                physical_qubits=lq_d.get("physical_qubits", 0),
            ),
            error_budget=ErrorBudget(
                logical=eb_d.get("logical", 0.0),
                rotations=eb_d.get("rotations", 0.0),
                tstates=eb_d.get("tstates", 0.0),
            ),
            estimator=json_data.get("estimator", ""),
            status=json_data.get("status", "unknown"),
            error=json_data.get("error", 0.0),
            circuit_counts=CircuitCounts(
                depth=cc_d.get("depth", 0),
                num_gates=cc_d.get("num_gates", 0),
                num_single_qubit_clifford=cc_d.get("num_single_qubit_clifford", 0),
                num_two_qubit_clifford=cc_d.get("num_two_qubit_clifford", 0),
                num_non_clifford=cc_d.get("num_non_clifford", 0),
            ) if cc_d is not None else None,
            config=EstimationConfig(
                qubit_model=cfg_d.get("qubit_model", ""),
                qec_scheme=cfg_d.get("qec_scheme", ""),
                error_budget=cfg_d.get("error_budget", 0.0),
                description=cfg_d.get("description", ""),
            ) if cfg_d is not None else None,
        )

    @classmethod
    def from_hdf5(cls, group: h5py.Group) -> "ResourceEstimatorData":
        """Load resource estimator data from an HDF5 group.

        Args:
            group: HDF5 group or file containing the data.

        Returns:
            ResourceEstimatorData: New instance reconstructed from HDF5 data.

        """
        cls._validate_hdf5_version(cls._serialization_version, group)

        lc_grp = group["logical_counts"]
        pc_grp = group["physical_counts"]
        lq_grp = group["logical_qubit"]
        eb_grp = group["error_budget"]

        circuit_counts = None
        if "circuit_counts" in group:
            cc_grp = group["circuit_counts"]
            circuit_counts = CircuitCounts(
                depth=int(cc_grp.attrs["depth"]),
                num_gates=int(cc_grp.attrs["num_gates"]),
                num_single_qubit_clifford=int(cc_grp.attrs["num_single_qubit_clifford"]),
                num_two_qubit_clifford=int(cc_grp.attrs["num_two_qubit_clifford"]),
                num_non_clifford=int(cc_grp.attrs["num_non_clifford"]),
            )

        config = None
        if "config" in group:
            cfg_grp = group["config"]
            config = EstimationConfig(
                qubit_model=str(cfg_grp.attrs["qubit_model"]),
                qec_scheme=str(cfg_grp.attrs["qec_scheme"]),
                error_budget=float(cfg_grp.attrs["error_budget"]),
                description=str(cfg_grp.attrs["description"]),
            )

        return cls(
            logical_counts=LogicalCounts(
                num_qubits=int(lc_grp.attrs["num_qubits"]),
                t_count=int(lc_grp.attrs["t_count"]),
                rotation_count=int(lc_grp.attrs["rotation_count"]),
                rotation_depth=int(lc_grp.attrs["rotation_depth"]),
                ccz_count=int(lc_grp.attrs["ccz_count"]),
                ccix_count=int(lc_grp.attrs["ccix_count"]),
                measurement_count=int(lc_grp.attrs["measurement_count"]),
            ),
            physical_counts=PhysicalCounts(
                physical_qubits=int(pc_grp.attrs["physical_qubits"]),
                runtime=int(pc_grp.attrs["runtime"]),
                runtime_unit=str(pc_grp.attrs["runtime_unit"]),
                rqops=int(pc_grp.attrs["rqops"]),
                algorithm_qubits=int(pc_grp.attrs["algorithm_qubits"]),
                factory_qubits=int(pc_grp.attrs["factory_qubits"]),
                algorithmic_logical_depth=int(pc_grp.attrs["algorithmic_logical_depth"]),
                logical_depth=int(pc_grp.attrs["logical_depth"]),
            ),
            logical_qubit=LogicalQubit(
                code_distance=int(lq_grp.attrs["code_distance"]),
                logical_cycle_time=int(lq_grp.attrs["logical_cycle_time"]),
                logical_error_rate=float(lq_grp.attrs["logical_error_rate"]),
                physical_qubits=int(lq_grp.attrs["physical_qubits"]),
            ),
            error_budget=ErrorBudget(
                logical=float(eb_grp.attrs["logical"]),
                rotations=float(eb_grp.attrs["rotations"]),
                tstates=float(eb_grp.attrs["tstates"]),
            ),
            estimator=str(group.attrs.get("estimator", "")),
            status=str(group.attrs.get("status", "unknown")),
            error=float(group.attrs.get("error", 0.0)),
            circuit_counts=circuit_counts,
            config=config,
        )
