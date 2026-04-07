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
    IGate,
    RXGate,
    RXXGate,
    RYGate,
    RYYGate,
    RZGate,
    RZZGate,
    SdgGate,
    SGate,
    XGate,
    YGate,
    ZGate,
)
from qiskit.dagcircuit import DAGCircuit
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.passes.optimization import Optimize1qGatesDecomposition

from qdk_chemistry.data import Settings
from qdk_chemistry.definitions import DIAGONAL_Z_1Q_GATES
from qdk_chemistry.utils import Logger

__all__ = [
    "FactorCliffordFromRz",
    "FactorPauliFromRotation",
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


# Clifford angles indexed by their multiples of π/2 (mod 4).
_CLIFFORD_TABLE: list[tuple[float, type]] = [
    (0.0, IGate),  # 0 · π/2
    (np.pi / 2, SGate),  # 1 · π/2
    (np.pi, ZGate),  # 2 · π/2
    (3 * np.pi / 2, SdgGate),  # 3 · π/2  (≡ −π/2 mod 2π)
]


class FactorCliffordFromRz(TransformationPass):
    r"""Factor out the nearest Clifford rotation from an arbitrary Rz gate.

    For every ``Rz(θ)`` in the circuit this pass finds the Clifford angle
    ``θ_c ∈ {0, π/2, π, 3π/2}`` that is closest to ``θ (mod 2π)`` and
    replaces the single gate with the two-gate sequence::

        Clifford(θ_c) · Rz(θ − θ_c)

    When ``θ`` is already an exact Clifford angle (within *tolerance*) the
    residual ``Rz`` vanishes and only the Clifford gate is emitted.
    Parameterized ``Rz`` gates are left untouched.

    This is useful as a pre-processing step: by pulling out the Clifford
    component the remaining ``Rz`` rotations are bounded to ``|θ_rem| ≤ π/4``,
    which can improve subsequent synthesis or error-mitigation passes.

    Example:
        ``Rz(1.6)`` is closest to ``S = Rz(π/2 ≈ 1.5708)``, so it becomes
        ``S · Rz(0.0292)``.

    """

    def __init__(self, tolerance: float = float(np.finfo(np.float64).eps)):
        """Initialize the FactorCliffordFromRz transformation pass.

        Args:
            tolerance: Angle comparison tolerance used to decide whether
                the residual rotation is negligible (i.e. the original gate
                is already Clifford).  Default is ``np.finfo(np.float64).eps``.

        """
        Logger.trace_entering()
        super().__init__()
        self._tolerance = tolerance

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        """Run the pass on the given ``DAGCircuit``.

        Args:
            dag: The input ``DAGCircuit`` to transform.

        Returns:
            The transformed ``DAGCircuit`` with factored Clifford rotations.

        """
        Logger.trace_entering()
        Logger.debug("Running FactorCliffordFromRz pass.")

        for node in dag.op_nodes():
            if node.op.name != "rz":
                continue

            angle = node.op.params[0]
            if isinstance(angle, ParameterExpression):
                continue

            # Normalise to [0, 2π)
            angle_mod = float(angle) % (2 * np.pi)

            # Find the nearest Clifford angle
            best_cliff_angle, best_gate_cls = min(
                _CLIFFORD_TABLE,
                key=lambda entry: min(abs(angle_mod - entry[0]), 2 * np.pi - abs(angle_mod - entry[0])),
            )

            remainder = angle_mod - best_cliff_angle
            # Wrap remainder into (−π, π]
            remainder = (remainder + np.pi) % (2 * np.pi) - np.pi

            if abs(remainder) <= self._tolerance:
                # Exact Clifford — replace with the single Clifford gate
                dag.substitute_node(node, best_gate_cls(), inplace=True)
            else:
                # Factor: Clifford · Rz(remainder)
                replacement = DAGCircuit()
                replacement.add_qubits(node.qargs)
                if not isinstance(best_gate_cls(), IGate):
                    replacement.apply_operation_back(best_gate_cls(), qargs=node.qargs)
                replacement.apply_operation_back(RZGate(remainder), qargs=node.qargs)
                dag.substitute_node_with_dag(node, replacement)

        return dag


# Mapping from rotation gate name to (single-qubit Pauli class, rotation gate class).
_ROTATION_GATE_TABLE: dict[str, tuple[type, type]] = {
    "rx": (XGate, RXGate),
    "ry": (YGate, RYGate),
    "rz": (ZGate, RZGate),
    "rxx": (XGate, RXXGate),
    "ryy": (YGate, RYYGate),
    "rzz": (ZGate, RZZGate),
}


class FactorPauliFromRotation(TransformationPass):
    r"""Factor out multiples of π from 1- and 2-qubit Pauli rotation gates.

    Every rotation gate :math:`R_P(\theta) = \exp(-i \theta / 2\; P)` satisfies:

    * :math:`R_P(\pi) = -i P` — equivalent to applying the Pauli operator (up to
      global phase).
    * :math:`R_P(2\pi) = -I` — identity (up to global phase).

    This pass finds the nearest integer multiple :math:`k` of :math:`\pi` in the
    gate angle and decomposes::

        R_P(θ)  →  P^(k mod 2)  ·  R_P(θ − kπ)

    where the Pauli part is emitted only when *k* is odd.  When the residual
    vanishes (the angle was already an exact multiple of :math:`\pi`) only the
    Pauli gates (or nothing) are emitted.

    Supported gates: ``rx``, ``ry``, ``rz``, ``rxx``, ``ryy``, ``rzz``.
    Parameterized gates are left untouched.

    For 2-qubit gates such as :math:`R_{XX}(\pi)` the Pauli contribution is
    the tensor product :math:`X \otimes X`, so the pass emits the
    corresponding single-qubit Pauli on each qubit independently.

    Example:
        ``Rzz(π + 0.1)`` → ``Z · Z · Rzz(0.1)`` (Z on each qubit, then small
        residual rotation).

    """

    def __init__(self, tolerance: float = float(np.finfo(np.float64).eps)):
        """Initialize the FactorPauliFromRotation transformation pass.

        Args:
            tolerance: Angle comparison tolerance used to decide whether
                the residual rotation is negligible.  Default is
                ``np.finfo(np.float64).eps``.

        """
        Logger.trace_entering()
        super().__init__()
        self._tolerance = tolerance

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        """Run the pass on the given ``DAGCircuit``.

        Args:
            dag: The input ``DAGCircuit`` to transform.

        Returns:
            The transformed ``DAGCircuit`` with factored Pauli rotations.

        """
        Logger.trace_entering()
        Logger.debug("Running FactorPauliFromRotation pass.")

        for node in dag.op_nodes():
            if node.op.name not in _ROTATION_GATE_TABLE:
                continue

            angle = node.op.params[0]
            if isinstance(angle, ParameterExpression):
                continue

            # Normalise to [0, 2π)
            angle_f = float(angle) % (2 * np.pi)
            pauli_cls, rot_cls = _ROTATION_GATE_TABLE[node.op.name]

            # Nearest integer multiple of π
            k = round(angle_f / np.pi)
            remainder = angle_f - k * np.pi

            need_pauli = (k % 2) != 0
            need_residual = abs(remainder) > self._tolerance

            if not need_pauli and not need_residual:
                # Even multiple of π → identity; remove the gate
                dag.remove_op_node(node)
                continue

            if not need_pauli and need_residual:
                # Even multiple of π + small residual → just the reduced rotation
                dag.substitute_node(node, rot_cls(remainder), inplace=True)
                continue

            # Odd multiple of π → emit Pauli gate(s), possibly followed by residual
            replacement = DAGCircuit()
            replacement.add_qubits(node.qargs)
            for qubit in node.qargs:
                replacement.apply_operation_back(pauli_cls(), qargs=[qubit])
            if need_residual:
                replacement.apply_operation_back(rot_cls(remainder), qargs=node.qargs)
            dag.substitute_node_with_dag(node, replacement)

        return dag


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
