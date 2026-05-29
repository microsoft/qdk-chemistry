# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

r"""Utilities for transpiling qiskit quantum circuits.

This module provides various custom transformation passes for optimizing circuits, including merging Z-basis
rotations, substituting Clifford Rz gates, and removing Z-basis operations on qubits in the :math:`\lvert 0 \rangle`
state. It also includes functions to create custom pass managers based on preset configurations and custom passes.
"""

import numpy as np
from qiskit.circuit import ParameterExpression
from qiskit.circuit.library import (
    CZGate,
    IGate,
    RXGate,
    RXXGate,
    RYGate,
    RYYGate,
    RZGate,
    RZZGate,
    SdgGate,
    SGate,
    TdgGate,
    TGate,
    XGate,
    YGate,
    ZGate,
)
from qiskit.dagcircuit import DAGCircuit
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.passes.optimization import Optimize1qGatesDecomposition
from qiskit.transpiler.passes.optimization.light_cone import LightCone

from qdk_chemistry.algorithms.hamiltonian_unitary_builder.base import HamiltonianUnitaryBuilder
from qdk_chemistry.data import Settings
from qdk_chemistry.definitions import DIAGONAL_Z_1Q_GATES
from qdk_chemistry.utils import Logger

__all__ = [
    "CommuteDiagonalThroughMeasurement",
    "DecomposeRzzToCliffordCz",
    "ExtractCliffordFromRotation",
    "MergeZBasisRotations",
    "ReduceToLightCone",
    "RemoveZBasisOnZeroState",
    "SubstituteCliffordRz",
    "SubstitutePauliRotation",
]


class MergeZBasisRotations(TransformationPass):
    r"""Transformation pass to merge consecutive Z-basis rotations into a single Rz gate and remove identity gates.

    This pass identifies sequences of single-qubit gates in the Z-basis,
    specifically Rz(θ), Z, S, and Sdg, and combines them into a single Rz(θ_new)
    operation whenever possible. These gates all correspond to rotations around the
    Z-axis of the Bloch sphere and can be represented in a unified form.

    Gates:

    * Rz(θ): Arbitrary rotation by angle θ.
    * Z: Equivalent to Rz(π).
    * S: Equivalent to Rz(π/2).
    * Sdg: Equivalent to Rz(-π/2).
    * Id: Equivalent to Rz(0) (no effect, removed).

    Behavior:

    * Does not merge across non-Z-basis gates (e.g., X, H, CX).
    * Removes Id gates entirely since they have no effect.
    * Respects circuit boundaries and barriers.

    Example:
        Input sequence:
            :math:`S \rightarrow R_z(π/3) \rightarrow S^\dagger \rightarrow Z`
        Output:
            :math:`R_z(π/2 + π/3 - π/2 + π) = R_z(π + π/3)`

    Note:
        * Useful for simplifying circuits before basis gate decomposition.
        * Reduces gate count and improves optimization opportunities downstream.

    """

    def __init__(self):
        """Use Optimize1qGatesDecomposition to handle gate optimization to merge Z basis rotations."""
        Logger.trace_entering()
        super().__init__()
        self._optimize1q_decomposition = Optimize1qGatesDecomposition(basis=["rz", "rx"])

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        """Run the pass on the given ``DAGCircuit``.

        Args:
            dag: The input ``DAGCircuit`` to transform.

        Returns:
            The transformed ``DAGCircuit`` with merged Z-basis rotations.

        """
        Logger.trace_entering()
        Logger.debug("Running MergeZBasisRotations pass.")
        new_dag = DAGCircuit()
        for qreg in dag.qregs.values():
            new_dag.add_qreg(qreg)
        for creg in dag.cregs.values():
            new_dag.add_creg(creg)

        current_region = DAGCircuit()
        for qreg in dag.qregs.values():
            current_region.add_qreg(qreg)
        for creg in dag.cregs.values():
            current_region.add_creg(creg)

        for node in dag.topological_op_nodes():
            name = node.op.name

            is_z_basis_gate = name in {"rz", "z", "s", "sdg"}
            is_id_gate = name == "id"

            if is_id_gate:
                # Remove Id gates (no effect on state)
                continue

            if is_z_basis_gate:
                # Add Z-basis gate to current merge region
                current_region.apply_operation_back(node.op, node.qargs, node.cargs)

            else:
                # Non-Z-basis gate: process current region first
                if current_region.size() > 0:
                    optimized_region = self._optimize1q_decomposition.run(current_region)
                    new_dag.compose(optimized_region, inplace=True)
                    current_region = DAGCircuit()
                    for qreg in dag.qregs.values():
                        current_region.add_qreg(qreg)
                    for creg in dag.cregs.values():
                        current_region.add_creg(creg)

                # Add this non-Z gate directly (acts as boundary)
                new_dag.apply_operation_back(node.op, node.qargs, node.cargs)

        # Process any remaining region
        if current_region.size() > 0:
            optimized_region = self._optimize1q_decomposition.run(current_region)
            new_dag.compose(optimized_region, inplace=True)

        return new_dag


class SubstituteCliffordRzSettings(Settings):
    """Settings configuration for SubstituteCliffordRz.

    SubstituteCliffordRz-specific settings:
        equivalent_gate_set (vector<string>, default=["id", "t", "s", "sdg", "tdg", "z"]): Equivalent gate set to use.
        tolerance (double, default=float(np.finfo(np.float64).eps)): Float comparison tolerance to use.

    """

    def __init__(self):
        """Initialize SubstituteCliffordRzSettings."""
        Logger.trace_entering()
        super().__init__()
        self._set_default("equivalent_gate_set", "vector<string>", ["id", "t", "s", "z", "sdg", "tdg"])
        self._set_default("tolerance", "double", float(np.finfo(np.float64).eps))

    def set(self, key: str, value):
        """Override set to ensure 'id' is always in equivalent_gate_set and duplicates are removed.

        Args:
            key (str): Setting key to set.
            value: Value to set.

        """
        # Ensure 'id' is present in equivalent_gate_set and remove duplicates
        if key == "equivalent_gate_set" and isinstance(value, list):
            value = list({*value, "id"})
        Logger.trace_entering()
        super().set(key, value)

    def update(self, settings_dict: dict):
        """Override update to ensure 'id' is always in equivalent_gate_set and duplicates are removed.

        Args:
            settings_dict (dict): Dictionary of settings to update.

        """
        # Ensure 'id' is present in equivalent_gate_set and remove duplicates
        if "equivalent_gate_set" in settings_dict and isinstance(settings_dict["equivalent_gate_set"], list):
            settings_dict = {
                **settings_dict,
                "equivalent_gate_set": list({*settings_dict["equivalent_gate_set"], "id"}),
            }
        Logger.trace_entering()
        super().update(settings_dict)


class SubstituteCliffordRz(TransformationPass):
    """Transformation pass to substitute Rz(θ) gates with equivalent Clifford+T gates for special angles.

    This pass replaces Rz(θ) gates with one of the following gates:

    * Identity (Id)
    * T gate (T)
    * Phase gate (S)
    * Pauli-Z (Z)
    * Inverse Phase gate (Sdg)
    * T-dagger gate (Tdg)

    Substitution rules:

    +--------------------+--------------------------+
    | Rz angle (θ)       | Equivalent gate          |
    +====================+==========================+
    | 0                  | Id                       |
    +--------------------+--------------------------+
    | π/4                | T                        |
    +--------------------+--------------------------+
    | π/2                | S                        |
    +--------------------+--------------------------+
    | π                  | Z                        |
    +--------------------+--------------------------+
    | -π/2 or 3π/2       | Sdg                      |
    +--------------------+--------------------------+
    | -π/4 or 7π/4       | Tdg                      |
    +--------------------+--------------------------+

    Note:
        * Only substitutes gates whose angle is non-parameterized and matches
          one of the above special phases within the specified tolerance.
        * Leaves parameterized Rz gates untouched to preserve symbolic expressions.
        * Ignores gates not in the user-specified ``equivalent_gate_set``

    """

    def __init__(
        self,
        equivalent_gate_set: list[str] | None = None,
        tolerance: float = float(np.finfo(np.float64).eps),
    ):
        """Initialize the SubstituteCliffordRz transformation pass.

        Args:
            equivalent_gate_set (list[str] | None): List of gates to substitute rz with special
                angles. Default is None, which means ['id', 't', 's', 'z', 'sdg', 'tdg'].
            tolerance (float): Angle comparison tolerance. Default is np.finfo(np.float64).eps.

        """
        Logger.trace_entering()
        super().__init__()
        self._settings = SubstituteCliffordRzSettings()
        if equivalent_gate_set is not None:
            if not isinstance(equivalent_gate_set, list):
                raise TypeError("equivalent_gate_set must be a list of gate names or None")
            self._settings.set("equivalent_gate_set", equivalent_gate_set)
        self._settings.set("tolerance", tolerance)
        self._factor_to_gate: tuple[tuple[int, str, type], ...] = (
            (0, "id", IGate),
            (1, "t", TGate),
            (2, "s", SGate),
            (4, "z", ZGate),
            (6, "sdg", SdgGate),
            (7, "tdg", TdgGate),
        )
        self._build_lookup()

    def _build_lookup(self):
        """Precompute the mod-8 lookup table from current settings."""
        gate_set = set(self._settings.get("equivalent_gate_set"))
        tolerance = self._settings.get("tolerance")
        # Precompute (mod8_target, gate_instance) pairs for enabled gates
        self._lookup: list[tuple[float, object]] = []
        for mod8_val, name, cls in self._factor_to_gate:
            if name in gate_set:
                self._lookup.append((float(mod8_val), cls()))
        self._tolerance = tolerance
        self._inv_quarter_pi = 4.0 / np.pi  # precompute reciprocal

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        """Run the pass on the given ``DAGCircuit``.

        Args:
            dag: The input ``DAGCircuit`` to transform.

        Returns:
            The transformed ``DAGCircuit`` with Rz substitutions.

        """
        Logger.trace_entering()
        tolerance = self._tolerance
        lookup = self._lookup
        inv_quarter_pi = self._inv_quarter_pi

        for node in dag.op_nodes():
            if node.op.name != "rz":
                continue

            angle = node.op.params[0]

            # Skip parameterized rotations
            if isinstance(angle, ParameterExpression):
                continue

            # Compute mod-8 factor: angle = factor * (π/4)
            factor = angle * inv_quarter_pi
            mod8 = factor % 8.0

            for target, gate in lookup:
                if abs(mod8 - target) <= tolerance or abs(mod8 - target - 8.0) <= tolerance:
                    dag.substitute_node(node, gate, inplace=True)
                    break
        return dag

    def settings(self) -> Settings:
        """Get the settings for SubstituteCliffordRz.

        Returns:
            The settings object associated with SubstituteCliffordRz.

        """
        Logger.trace_entering()
        return self._settings


class SubstitutePauliRotationSettings(Settings):
    """Settings configuration for SubstitutePauliRotation.

    SubstitutePauliRotation-specific settings:
        equivalent_gate_set (vector<string>, default=["id", "x", "y", "z"]): Equivalent gate set to use.
        tolerance (double, default=float(np.finfo(np.float64).eps)): Float comparison tolerance to use.

    """

    def __init__(self):
        """Initialize SubstitutePauliRotationSettings."""
        Logger.trace_entering()
        super().__init__()
        self._set_default("equivalent_gate_set", "vector<string>", ["id", "x", "y", "z"])
        self._set_default("tolerance", "double", float(np.finfo(np.float64).eps))

    def set(self, key: str, value):
        """Override set to ensure 'id' is always in equivalent_gate_set and duplicates are removed.

        Args:
            key (str): Setting key to set.
            value: Value to set.

        """
        if key == "equivalent_gate_set" and isinstance(value, list):
            value = list({*value, "id"})
        Logger.trace_entering()
        super().set(key, value)

    def update(self, settings_dict: dict):
        """Override update to ensure 'id' is always in equivalent_gate_set and duplicates are removed.

        Args:
            settings_dict (dict): Dictionary of settings to update.

        """
        if "equivalent_gate_set" in settings_dict and isinstance(settings_dict["equivalent_gate_set"], list):
            settings_dict = {
                **settings_dict,
                "equivalent_gate_set": list({*settings_dict["equivalent_gate_set"], "id"}),
            }
        Logger.trace_entering()
        super().update(settings_dict)


class SubstitutePauliRotation(TransformationPass):
    r"""Substitute Pauli rotation gates at integer multiples of π.

    Handles both 1-qubit (Rx, Ry, Rz) and 2-qubit (Rxx, Ryy, Rzz) rotations:

    * :math:`R_P(k\pi)` where *k* is even → removed (identity up to global phase).
    * :math:`R_P(k\pi)` where *k* is odd  → the corresponding Pauli gate(s).

    For 2-qubit rotations the Pauli is applied independently to each qubit.
    """

    def __init__(
        self,
        equivalent_gate_set: list[str] | None = None,
        tolerance: float = float(np.finfo(np.float64).eps),
    ):
        """Initialize SubstitutePauliRotation.

        Args:
            equivalent_gate_set: List of gates to allow substitution with, or None for
                defaults (``["id", "x", "y", "z"]``).
            tolerance: Angle comparison tolerance.

        """
        Logger.trace_entering()
        super().__init__()
        self._rotation_gates = {
            "rx": (XGate, RXGate),
            "ry": (YGate, RYGate),
            "rz": (ZGate, RZGate),
            "rxx": (XGate, RXXGate),
            "ryy": (YGate, RYYGate),
            "rzz": (ZGate, RZZGate),
        }
        self._settings = SubstitutePauliRotationSettings()
        if equivalent_gate_set is not None:
            if not isinstance(equivalent_gate_set, list):
                raise TypeError("equivalent_gate_set must be a list of gate names or None")
            self._settings.set("equivalent_gate_set", equivalent_gate_set)
        self._settings.set("tolerance", tolerance)

    def substitute_pauli_rotations(
        self,
        dag: DAGCircuit,
        rotation_gates: dict[str, tuple],
        equivalent_gate_set: list[str],
        tolerance: float,
    ) -> DAGCircuit:
        r"""Replace rotation gates at integer multiples of π with Pauli gates or identity.

        For a rotation gate :math:`R_P(\theta)`:

        * Even multiples of π (0, 2π, …) → identity (gate removed).
        * Odd multiples of π (π, 3π, …) → the corresponding Pauli gate(s).

        For 2-qubit rotations the Pauli is applied independently to each qubit.

        Args:
            dag: The input ``DAGCircuit`` to transform.
            rotation_gates: Gate-name → (Pauli class, Rotation class) mapping.
            equivalent_gate_set: List of gate names allowed for substitution.
            tolerance: Angle comparison tolerance.

        Returns:
            The transformed ``DAGCircuit`` with Pauli substitutions.

        """
        if "id" not in equivalent_gate_set:
            raise ValueError("Gate 'id' is missing in equivalent_gate_set.")
        if len(equivalent_gate_set) != len(set(equivalent_gate_set)):
            raise ValueError(f"Gates in equivalent_gate_set ({equivalent_gate_set}) are not unique.")

        for node in dag.op_nodes():
            if node.op.name not in rotation_gates:
                continue

            angle = node.op.params[0]
            if isinstance(angle, ParameterExpression):
                continue

            pauli_cls, _ = rotation_gates[node.op.name]
            pauli_name = pauli_cls().name  # e.g. "x", "y", "z"

            k = round(float(angle) / np.pi)
            if abs(float(angle) - k * np.pi) > tolerance:
                continue

            if k % 2 == 0 and "id" in equivalent_gate_set:
                # Even multiple of π → identity; remove the gate
                dag.remove_op_node(node)
            elif k % 2 != 0 and pauli_name in equivalent_gate_set:
                # Odd multiple of π → Pauli gate(s)
                if len(node.qargs) == 1:
                    dag.substitute_node(node, pauli_cls(), inplace=True)
                else:
                    replacement = DAGCircuit()
                    replacement.add_qubits(node.qargs)
                    for qubit in node.qargs:
                        replacement.apply_operation_back(pauli_cls(), qargs=[qubit])
                    dag.substitute_node_with_dag(node, replacement)

        return dag

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        """Run the pass on the given ``DAGCircuit``.

        Args:
            dag: The input ``DAGCircuit`` to transform.

        Returns:
            The transformed ``DAGCircuit`` with Pauli substitutions.

        """
        Logger.trace_entering()
        Logger.debug("Running SubstitutePauliRotation pass.")
        return self.substitute_pauli_rotations(
            dag,
            self._rotation_gates,
            self._settings.get("equivalent_gate_set"),
            self._settings.get("tolerance"),
        )

    def settings(self) -> Settings:
        """Get the settings for this pass.

        Returns:
            The settings object associated with this pass.

        """
        Logger.trace_entering()
        return self._settings


class RemoveZBasisOnZeroState(TransformationPass):
    r"""Transformation pass to remove Z-basis operations on qubits that are in the :math:`\lvert 0 \rangle` state.

    This optimization eliminates gates that apply only a global phase to the qubit,
    which has no effect on observable outcomes (measurement probabilities) or
    downstream quantum operations. Specifically, diagonal gates in the computational
    basis (e.g., Rz(θ), Z, S, Sdg) act trivially on the :math:`\lvert 0 \rangle` state:

    * :math:`R_z(θ) \lvert 0 \rangle = e^{-iθ/2} \lvert 0 \rangle`
    * :math:`Z \lvert 0 \rangle = +1 \lvert 0 \rangle`
    * :math:`S \lvert 0 \rangle = +1 \lvert 0 \rangle`
    * :math:`S^\dagger \lvert 0 \rangle = +1 \lvert 0 \rangle`

    These gates only introduce a global phase factor, which is physically unobservable.

    This transformation must not be applied to qubits in superposition or entangled
    states, since Z-basis rotations there modify relative phases between basis states.
    """

    def __init__(self):
        """Initialize the ``RemoveZBasisOnZeroState`` transformation pass."""
        Logger.trace_entering()
        super().__init__()
        self._z_basis_gates = {"rz", "z", "s", "sdg"}

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        """Run the pass on the given ``DAGCircuit``.

        Args:
            dag: The input ``DAGCircuit`` to transform.

        Returns:
            The transformed ``DAGCircuit`` with Z-basis gates removed.

        """
        Logger.trace_entering()
        Logger.debug("Running RemoveZBasisOnZeroState pass.")

        # Track qubits still in |0⟩ (True means untouched)
        zero_state_qubits = dict.fromkeys(dag.qubits, True)

        nodes_to_process = list(dag.topological_op_nodes())
        for node in nodes_to_process:
            name = node.op.name
            qubits = node.qargs

            # Check if Z-basis gate and qubit still in |0⟩
            if name in self._z_basis_gates:
                remove_gate = all(zero_state_qubits.get(q, False) for q in qubits)
                if remove_gate:
                    Logger.debug(f"Removing {name} on qubit {qubits} (still |0⟩)")
                    dag.remove_op_node(node)
                    continue  # Skip to next node

            # Mark qubits as no longer |0⟩ for non-diagonal gates
            if name not in self._z_basis_gates and not self._is_diagonal(name):
                for q in qubits:
                    zero_state_qubits[q] = False

        return dag

    def _is_diagonal(self, gate_name: str) -> bool:
        """Determine if a gate is diagonal in computational basis.

        Args:
            gate_name: Name of the gate.

        Returns:
            bool: True if the gate is diagonal, False otherwise.

        Note:
            The gate classification logic depends on the ``DIAGONAL_Z_1Q_GATES`` defined in ``definitions.py``.

        """
        Logger.trace_entering()
        return gate_name in DIAGONAL_Z_1Q_GATES


class DecomposeRzzToCliffordCz(TransformationPass):
    """Transformation pass to decompose RZZ(±π/2) gates into CZ gates with residual single-qubit Clifford gates.

    Uses the identity ``exp(iπ/4 Z_i Z_j) = exp(-iπ/4) · Rz_i(-π/2) · Rz_j(-π/2) · CZ_ij`` to replace each
    RZZ(π/2) gate with a CZ gate and Rz(-π/2) on each qubit (and analogously for RZZ(-π/2) with Rz(π/2)).
    After decomposition, accumulated Rz rotations are merged and simplified to named Clifford gates.

    The residual single-qubit gates depend on the vertex degree (number of RZZ edges per qubit).
    For RZZ(π/2), each edge contributes Rz(π/2) per qubit:

    +--------------+------------------+--------------------+
    | degree mod 4 | accumulated Rz   | residual gate      |
    +==============+==================+====================+
    | 0            | 0                | I (identity)       |
    +--------------+------------------+--------------------+
    | 1            | π/2              | S                  |
    +--------------+------------------+--------------------+
    | 2            | π                | Z                  |
    +--------------+------------------+--------------------+
    | 3            | 3π/2             | Sdg                |
    +--------------+------------------+--------------------+

    For RZZ(-π/2), the signs flip: S ↔ Sdg (Z and I are unaffected).

    Examples:
        * **1D chain** (degree 2): each interior qubit accumulates Rz(-π) → Z gate per qubit, plus CZ on each edge.
        * **2D torus** (degree 4): each qubit accumulates Rz(-2π) → identity, leaving only CZ gates.

    Note:
        This pass only decomposes RZZ gates whose angle is exactly ±π/2 within the specified tolerance.
        Other RZZ angles are left untouched. After decomposition, :class:`SubstituteCliffordRz`
        is applied automatically to simplify the residual single-qubit gates.

    """

    def __init__(self, tolerance: float = float(np.finfo(np.float64).eps)):
        """Initialize the DecomposeRzzToCliffordCz transformation pass.

        Args:
            tolerance (float): Angle comparison tolerance for identifying ±π/2. Default is np.finfo(np.float64).eps.

        """
        Logger.trace_entering()
        super().__init__()
        self._tolerance = tolerance

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        """Run the DecomposeRzzToCliffordCz pass on the DAG circuit.

        Each RZZ(±π/2) is replaced by a CZ gate.  The residual Rz rotations
        are accumulated per qubit and flushed at the boundary of each
        diagonal-Z layer — i.e. just before the first non-diagonal gate
        (Rx, H, …) on that qubit's wire.  Within a layer the Rz commutes
        freely past CZ, so the accumulation is exact.

        On even-degree lattices the per-layer Rz sums to a multiple of 2π
        and is dropped entirely, leaving only CZ gates.

        Args:
            dag (DAGCircuit): The input DAG circuit.

        Returns:
            DAGCircuit: The transformed DAG circuit.

        """
        Logger.trace_entering()
        target_angle = np.pi / 2
        tolerance = self._tolerance

        # Build a new DAG, accumulating Rz per qubit across diagonal-Z regions
        new_dag = DAGCircuit()
        for qreg in dag.qregs.values():
            new_dag.add_qreg(qreg)
        for creg in dag.cregs.values():
            new_dag.add_creg(creg)

        accumulated: dict = {}  # qubit -> float

        for node in dag.topological_op_nodes():
            # Check if this is an RZZ(±π/2) to decompose
            if node.op.name == "rzz" and not isinstance(node.op.params[0], ParameterExpression):
                angle = float(node.op.params[0])
                angle_mod = angle % (2 * np.pi)
                if np.isclose(angle_mod, target_angle, atol=tolerance) or np.isclose(
                    angle_mod, 2 * np.pi - target_angle, atol=tolerance
                ):
                    q0, q1 = node.qargs
                    accumulated[q0] = accumulated.get(q0, 0.0) + angle
                    accumulated[q1] = accumulated.get(q1, 0.0) + angle
                    new_dag.apply_operation_back(CZGate(), [q0, q1])
                    continue

            # For non-diagonal gates, flush accumulated Rz on touched qubits
            if node.op.name not in DIAGONAL_Z_1Q_GATES and node.op.name not in _DIAGONAL_Z_2Q_GATES:
                for qubit in node.qargs:
                    if qubit in accumulated:
                        self._flush_rz(new_dag, qubit, accumulated.pop(qubit), tolerance)

            new_dag.apply_operation_back(node.op, node.qargs, node.cargs)

        # Flush any remaining accumulated Rz
        for qubit, angle in accumulated.items():
            self._flush_rz(new_dag, qubit, angle, tolerance)

        return new_dag

    @staticmethod
    def _flush_rz(dag: DAGCircuit, qubit, angle: float, tolerance: float) -> None:
        """Emit an Rz gate if the accumulated angle is non-trivial mod 2π."""
        total_mod = angle % (2 * np.pi)
        if np.isclose(total_mod, 0.0, atol=tolerance) or np.isclose(total_mod, 2 * np.pi, atol=tolerance):
            return
        dag.apply_operation_back(RZGate(angle), [qubit])


class ExtractCliffordFromRotation(TransformationPass):
    r"""Split an Rz gate into a Clifford part and a small-angle residual.

    For each ``Rz(θ)`` in the circuit, this pass finds the nearest Clifford
    angle :math:`c = n\pi/2` and, when the residual :math:`\theta - c` is
    non-trivial, replaces the gate with ``Rz(c) · Rz(θ - c)``.  A
    subsequent :class:`SubstituteCliffordRz` pass can then convert ``Rz(c)``
    to the equivalent noiseless Clifford gate (S, Sdg, Z, or Id).

    This is useful in noise models where Clifford gates are cheaper (or
    noiseless) while arbitrary-angle rotations carry injected-rotation
    noise.  Without this pass, an ``Rz(θ)`` that is *close* to a Clifford
    angle pays the full rotation-noise cost even though most of the angle
    could have been implemented as a noiseless Clifford.

    Example::

        Rz(0.3 - π) → Rz(-π) · Rz(0.3)
                     → Z · Rz(0.3)        (after SubstituteCliffordRz)

    """

    _QUARTER_PI = np.pi / 4.0

    def __init__(self, tolerance: float = 1e-10):
        """Initialize ExtractCliffordFromRotation.

        Args:
            tolerance: Angles within this tolerance of a Clifford multiple are left for SubstituteCliffordRz.

        """
        super().__init__()
        self._tolerance = tolerance

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        """Run the pass on the given ``DAGCircuit``.

        Args:
            dag: The input ``DAGCircuit`` to transform.

        Returns:
            The transformed ``DAGCircuit`` with Clifford parts extracted.

        """
        for node in dag.op_nodes():
            if node.op.name != "rz":
                continue

            angle = node.op.params[0]
            if isinstance(angle, ParameterExpression):
                continue

            theta = float(angle)

            # Find nearest multiple of π/2
            n = round(theta / (np.pi / 2.0))
            clifford_angle = n * (np.pi / 2.0)
            residual = theta - clifford_angle

            # Skip if the residual is essentially zero (pure Clifford)
            if abs(residual) <= self._tolerance:
                continue

            # Skip if the Clifford part is zero (already a pure residual)
            if n == 0:
                continue

            # Replace Rz(θ) with Rz(clifford) · Rz(residual)
            replacement = DAGCircuit()
            replacement.add_qubits(node.qargs)
            replacement.apply_operation_back(RZGate(clifford_angle), [node.qargs[0]])
            replacement.apply_operation_back(RZGate(residual), [node.qargs[0]])
            dag.substitute_node_with_dag(node, replacement)

        return dag


class ReduceToLightCone(TransformationPass):
    """Remove gates outside the backward light cone of an observable or circuit measurement.

    Wraps Qiskit's :class:`~qiskit.transpiler.passes.optimization.LightCone` pass with support
    for :class:`~qdk_chemistry.data.QubitHamiltonian` observables. Gates that cannot affect the
    expectation value of the observable (or measurement outcome) are removed.

    The observable can be specified as:

    * ``None`` (default) — uses the measurements already present in the circuit.
    * A :class:`~qdk_chemistry.data.QubitHamiltonian` — non-trivial Pauli terms and qubit
      indices are extracted automatically.

    Example::

        from qdk_chemistry.data import QubitHamiltonian
        from qdk_chemistry.plugins.qiskit._interop.transpiler import ReduceToLightCone
        from qiskit.transpiler import PassManager

        obs = QubitHamiltonian(pauli_strings=["IIZIIII"], coefficients=[1.0])
        pm = PassManager([ReduceToLightCone(observable=obs)])
        reduced = pm.run(circuit)

    Note:
        For multi-term observables where different terms have different Pauli types on the same
        qubit, the last encountered Pauli type is used. The light cone result is correct regardless
        of which Pauli type is chosen, since any non-identity operator prevents gate removal.

    """

    def __init__(self, observable=None):
        """Initialize the ReduceToLightCone transformation pass.

        Args:
            observable: Observable for light cone computation. ``None`` uses circuit measurements.

        """
        Logger.trace_entering()
        super().__init__()
        self._observable = observable

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        """Run the ReduceToLightCone pass on the DAG circuit.

        Args:
            dag (DAGCircuit): The input DAG circuit.

        Returns:
            DAGCircuit: The reduced DAG containing only gates within the light cone.

        """
        Logger.trace_entering()
        if self._observable is None:
            # Extract measured qubit indices from the circuit and use explicit
            # Z-basis light cone so the reduction is as aggressive as when an
            # observable is provided.  Qiskit's LightCone does not allow both
            # measurements and explicit indices, so temporarily strip them.
            measure_nodes = [node for node in dag.op_nodes() if node.op.name == "measure"]
            measured_qubits = sorted({dag.find_bit(qarg).index for node in measure_nodes for qarg in node.qargs})
            if measured_qubits:
                for node in measure_nodes:
                    dag.remove_op_node(node)
                bit_terms = "Z" * len(measured_qubits)
                dag = LightCone(bit_terms=bit_terms, indices=measured_qubits).run(dag)
                # Re-add measurement gates on the surviving qubits
                from qiskit.circuit.library import Measure  # noqa: PLC0415

                for idx in measured_qubits:
                    qubit = dag.qubits[idx]
                    if qubit in {q for node in dag.op_nodes() for q in node.qargs} or True:
                        # Find the classical bit that was originally paired with this qubit
                        if dag.clbits:
                            clbit = dag.clbits[measured_qubits.index(idx) % len(dag.clbits)]
                            dag.apply_operation_back(Measure(), [qubit], [clbit])
                return dag
            return LightCone().run(dag)

        pauli_map: dict[int, str] = {}
        for pauli_str in self._observable.pauli_strings:
            pauli_map |= HamiltonianUnitaryBuilder._pauli_label_to_map(pauli_str)  # noqa: SLF001
        if not pauli_map:
            return dag
        indices = sorted(pauli_map)
        bit_terms = "".join(pauli_map[i] for i in indices)
        return LightCone(bit_terms=bit_terms, indices=indices).run(dag)


# Two-qubit gates that are diagonal in the computational (Z) basis.
_DIAGONAL_Z_2Q_GATES = frozenset({"cz", "rzz", "cp", "ccz"})


class CommuteDiagonalThroughMeasurement(TransformationPass):
    r"""Remove Z-diagonal gates at the end of the circuit that commute with Z-basis measurement.

    Any gate whose matrix is diagonal in the computational basis commutes
    with a Z-basis measurement and therefore cannot affect the measurement
    outcome.  This pass iteratively removes such gates working backward
    from each measurement.

    A multi-qubit diagonal gate (e.g. CZ) is only removed when **all** of
    its qubit wires are still in the "diagonal tail" — i.e. every qubit of
    the gate has nothing but other diagonal gates (or the measurement
    itself) between it and the end of the circuit.

    Recognized diagonal gates:

    * **1-qubit:** Rz, S, Sdg, T, Tdg, Z, Id
    * **2-qubit:** CZ, RZZ, CP, CCZ

    Example::

        Before:  ──[H]──[CZ]──[Rz]──M──
                       │
                  ──[X]──[CZ]──────────

        After:   ──[H]──M──
                  ──[X]────

        (CZ is removed because both its qubits have only diagonal gates
        or measurements after it.  Rz is removed because it's a 1-qubit
        diagonal gate right before the measurement.)

    """

    def __init__(self):
        """Initialize CommuteDiagonalThroughMeasurement."""
        super().__init__()

    @staticmethod
    def _is_diagonal_z(op) -> bool:
        """Return True if the operation is diagonal in the Z basis."""
        return op.name in DIAGONAL_Z_1Q_GATES or op.name in _DIAGONAL_Z_2Q_GATES

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        """Run the pass on the given ``DAGCircuit``.

        Args:
            dag: The input ``DAGCircuit`` to transform.

        Returns:
            The transformed ``DAGCircuit`` with trailing diagonal gates removed.

        """
        # Identify measured qubits
        measured_qubits: set = set()
        for node in dag.op_nodes():
            if node.op.name == "measure":
                measured_qubits.add(node.qargs[0])

        if not measured_qubits:
            return dag

        # Track which qubits are still in the "diagonal tail" (eligible for removal)
        in_tail: set = set(measured_qubits)

        # Iteratively peel diagonal gates from the end.
        # Re-fetch wire ops each iteration since removal mutates the DAG.
        changed = True
        while changed:
            changed = False
            for qubit in list(in_tail):
                # Walk backward on this qubit's wire to find the frontier gate
                for node in reversed(list(dag.nodes_on_wire(qubit, only_ops=True))):
                    if node.op.name == "measure":
                        continue

                    # This is the last non-measure gate on this qubit
                    if self._is_diagonal_z(node.op) and all(q in in_tail for q in node.qargs):
                        dag.remove_op_node(node)
                        changed = True
                    else:
                        in_tail.discard(qubit)
                    break

        return dag
