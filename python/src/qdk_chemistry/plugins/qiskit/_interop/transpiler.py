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
from qiskit.circuit.library import IGate, SdgGate, SGate, ZGate
from qiskit.dagcircuit import DAGCircuit
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.passes.optimization import Optimize1qGatesDecomposition

from qdk_chemistry.data import Settings
from qdk_chemistry.definitions import DIAGONAL_Z_1Q_GATES
from qdk_chemistry.utils import Logger

__all__ = [
    "MergeZBasisRotations",
    "RemoveZBasisOnZeroState",
    "SubstituteCliffordRz",
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
                angles. Default is None, which means ['id', 's', 'sdg', 'z'].
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

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        """Run the pass on the given ``DAGCircuit``.

        Args:
            dag: The input ``DAGCircuit`` to transform.

        Returns:
            The transformed ``DAGCircuit`` with Rz substitutions.

        """
        Logger.trace_entering()
        equivalent_gate_set = self._settings.get("equivalent_gate_set")
        tolerance = self._settings.get("tolerance")

        if "id" not in equivalent_gate_set:
            raise ValueError("Gate 'id' is missing in equivalent_gate_set.")
        if len(equivalent_gate_set) != len(set(equivalent_gate_set)):
            raise ValueError(f"Gates in equivalent_gate_set ({equivalent_gate_set}) are not unique.")

        Logger.debug("SubstituteCliffordRz pass: simplification logic needs careful review.")

        for node in dag.op_nodes():
            if node.op.name == "rz":
                angle = node.op.params[0]

                # Skip parameterized rotations
                if isinstance(angle, ParameterExpression):
                    Logger.debug("Skipping parameterized Rz.")
                    continue

                factor = 2 * angle / np.pi
                mod4_factor = np.mod(factor, 4)
                Logger.debug(f"Rz({angle:.4f}) = {factor:.4f} * π/2 (mod 4 = {mod4_factor:.2f})")

                replacement_gate = None
                if np.isclose(mod4_factor, 0, atol=tolerance) and "id" in equivalent_gate_set:
                    Logger.debug(f"Substituting Rz({angle:.4f}) with Id.")
                    replacement_gate = IGate()
                elif np.isclose(mod4_factor, 1, atol=tolerance) and "s" in equivalent_gate_set:
                    Logger.debug(f"Substituting Rz({angle:.4f}) with S.")
                    replacement_gate = SGate()
                elif np.isclose(mod4_factor, 2, atol=tolerance) and "z" in equivalent_gate_set:
                    Logger.debug(f"Substituting Rz({angle:.4f}) with Z.")
                    replacement_gate = ZGate()
                elif np.isclose(mod4_factor, 3, atol=tolerance) and "sdg" in equivalent_gate_set:
                    Logger.debug(f"Substituting Rz({angle:.4f}) with Sdg.")
                    replacement_gate = SdgGate()

                if replacement_gate:
                    dag.substitute_node(node, replacement_gate, inplace=True)
                else:
                    Logger.debug(f"Keeping original Rz({angle:.4f}).")

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
