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
from qiskit.circuit.library import IGate, RZGate, SdgGate, SGate, ZGate
from qiskit.dagcircuit import DAGCircuit
from qiskit.transpiler.basepasses import TransformationPass

from qdk_chemistry.data import Settings
from qdk_chemistry.definitions import DIAGONAL_Z_1Q_GATES
from qdk_chemistry.utils import Logger

__all__ = [
    "MergeZBasisRotations",
    "RemoveZBasisOnZeroState",
    "SubstituteCliffordRz",
]


class MergeZBasisRotationsSettings(Settings):
    """Settings configuration for MergeZBasisRotations.

    MergeZBasisRotations-specific settings:
        tolerance (double, default=float(np.finfo(np.float64).eps)): Float comparison tolerance to use.

    """

    def __init__(self):
        """Initialize MergeZBasisRotationsSettings."""
        Logger.trace_entering()
        super().__init__()
        self._set_default("tolerance", "double", float(np.finfo(np.float64).eps))


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

    def __init__(self, tolerance: float = float(np.finfo(np.float64).eps)):
        """Initialize MergeZBasisRotations."""
        Logger.trace_entering()
        super().__init__()
        self._settings = MergeZBasisRotationsSettings()
        self._settings.set("tolerance", tolerance)
        # Fixed angle contributions for Z-basis Clifford gates
        self._z_fixed_angles: dict[str, float] = {"z": np.pi, "s": np.pi / 2, "sdg": -np.pi / 2}
        self._z_basis_gates = frozenset({"rz", "z", "s", "sdg"})

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        """Run the pass on the given ``DAGCircuit``.

        Operates in-place on the DAG using per-qubit angle accumulators, avoiding
        intermediate DAGCircuit creation and external optimization passes.

        Args:
            dag: The input ``DAGCircuit`` to transform.

        Returns:
            The transformed ``DAGCircuit`` with merged Z-basis rotations.

        """
        Logger.trace_entering()

        # Per-qubit state: (accumulated_angle, list_of_nodes_in_run)
        accumulators: dict[object, tuple[float, list]] = {}
        nodes_to_remove: list = []
        substitutions: list[tuple] = []  # (node, new_gate)

        for node in dag.topological_op_nodes():
            name = node.op.name

            if name == "id":
                nodes_to_remove.append(node)
                continue

            if name in self._z_basis_gates:
                qubit = node.qargs[0]  # Z-basis gates are always 1-qubit

                # Get angle for this gate
                if name == "rz":
                    angle = node.op.params[0]
                    # Skip parameterized rotations - they act as boundaries
                    if isinstance(angle, ParameterExpression):
                        # Flush any pending accumulator for this qubit first
                        if qubit in accumulators:
                            acc_angle, acc_nodes = accumulators.pop(qubit)
                            self._flush(acc_angle, acc_nodes, nodes_to_remove, substitutions)
                        # Leave parameterized gate as-is
                        continue
                else:
                    angle = self._z_fixed_angles[name]

                # Accumulate
                if qubit in accumulators:
                    acc = accumulators[qubit]
                    accumulators[qubit] = (acc[0] + angle, acc[1] + [node])
                else:
                    accumulators[qubit] = (angle, [node])

            else:
                # Non-Z gate: flush accumulators for all affected qubits
                for qubit in node.qargs:
                    if qubit in accumulators:
                        acc_angle, acc_nodes = accumulators.pop(qubit)
                        self._flush(acc_angle, acc_nodes, nodes_to_remove, substitutions)

        # Flush remaining accumulators
        for acc_angle, acc_nodes in accumulators.values():
            self._flush(acc_angle, acc_nodes, nodes_to_remove, substitutions)

        # Apply substitutions before removals
        for node, gate in substitutions:
            dag.substitute_node(node, gate, inplace=True)

        # Remove merged/identity nodes
        for node in nodes_to_remove:
            dag.remove_op_node(node)

        return dag

    @staticmethod
    def _flush(
        angle: float,
        nodes: list,
        nodes_to_remove: list,
        substitutions: list,
        tol: float = float(np.finfo(np.float64).eps),
    ):
        """Flush an accumulated Z-rotation run into a single gate or removal.

        Args:
            angle: The accumulated rotation angle.
            nodes: The list of nodes in the accumulated run.
            nodes_to_remove: List to append nodes that should be removed.
            substitutions: List to append (node, new_gate) pairs for substitution.
            tol: Tolerance for floating-point comparison to zero.

        """
        # Normalize angle to [0, 2π)
        norm = angle % (2 * np.pi)
        if norm < tol or (2 * np.pi - norm) < tol:
            # Net rotation is effectively zero: remove all nodes
            nodes_to_remove.extend(nodes)
        elif len(nodes) == 1:
            # Single gate: just substitute if angle changed
            if abs(angle - nodes[0].op.params[0]) > tol if nodes[0].op.name == "rz" else True:
                substitutions.append((nodes[0], RZGate(angle)))
        else:
            # Multiple gates: keep last node, substitute with merged Rz, remove rest
            substitutions.append((nodes[-1], RZGate(angle)))
            nodes_to_remove.extend(nodes[:-1])


class SubstituteCliffordRzSettings(Settings):
    """Settings configuration for SubstituteCliffordRz.

    SubstituteCliffordRz-specific settings:
        equivalent_gate_set (vector<string>, default=["id", "s", "sdg", "z"]): Equivalent gate set to use.
        tolerance (double, default=float(np.finfo(np.float64).eps)): Float comparison tolerance to use.

    """

    def __init__(self):
        """Initialize SubstituteCliffordRzSettings."""
        Logger.trace_entering()
        super().__init__()
        self._set_default("equivalent_gate_set", "vector<string>", ["id", "s", "sdg", "z"])
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
    """Transformation pass to substitute Rz(θ) gates with equivalent Clifford gates for special angles.

    This pass replaces Rz(θ) gates with one of the following Clifford gates:

    * Identity (Id)
    * Phase gate (S)
    * Inverse Phase gate (Sdg)
    * Pauli-Z (Z)

    Substitution rules:

    +--------------------+--------------------------+
    | Rz angle (θ)       | Equivalent Clifford gate |
    +====================+==========================+
    | 0                  | Id                       |
    +--------------------+--------------------------+
    | π/2                | S                        |
    +--------------------+--------------------------+
    | π                  | Z                        |
    +--------------------+--------------------------+
    | -π/2 or 3π/2       | Sdg                      |
    +--------------------+--------------------------+

    Note:
        * Only substitutes gates whose angle is non-parameterized and matches
          one of the above special Clifford phases within the specified tolerance.
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
                angles. Default is None, which means ['id', 's', 'z', 'sdg'].
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
            (2, "s", SGate),
            (4, "z", ZGate),
            (6, "sdg", SdgGate),
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
        self._z_basis_gates = frozenset({"rz", "z", "s", "sdg"})
        self._diagonal_gates = DIAGONAL_Z_1Q_GATES

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        """Run the pass on the given ``DAGCircuit``.

        Args:
            dag: The input ``DAGCircuit`` to transform.

        Returns:
            The transformed ``DAGCircuit`` with Z-basis gates removed.

        """
        Logger.trace_entering()

        # Track qubits still in |0⟩ — use a set for fast membership
        zero_state_qubits = set(dag.qubits)
        z_basis = self._z_basis_gates
        diagonal = self._diagonal_gates
        nodes_to_remove = []

        for node in dag.topological_op_nodes():
            name = node.op.name
            qubits = node.qargs

            # Check if Z-basis gate and qubit still in |0⟩ (Z-basis gates are 1-qubit)
            if name in z_basis and qubits[0] in zero_state_qubits:
                nodes_to_remove.append(node)
                continue

            # Mark qubits as no longer |0⟩ for non-diagonal gates
            if name not in diagonal:
                for q in qubits:
                    zero_state_qubits.discard(q)

        # Batch removal
        for node in nodes_to_remove:
            dag.remove_op_node(node)

        return dag
