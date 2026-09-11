"""QDK/Chemistry Global, constant definitions module.

The properties in this module are intended to remain constant within QDK/Chemistry, although they are
not strictly immutable from a physical perspective (in contrast to the properties in `constants.py`).

This module provides definitions for quantum simulation configuration settings.
"""
# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

# General gate categories including compositional (untranspiled) gates
SINGLE_QUBIT_CLIFFORD_GATES = frozenset(["h", "s", "sdg", "x", "y", "z", "sx"])
TWO_QUBIT_CLIFFORD_GATES = frozenset(["cx", "cz", "swap"])
NON_CLIFFORD_GATES = frozenset(["t", "tdg", "rx", "ry", "rz", "u", "u1", "u2", "u3", "ccx"])
SUPERPOSITION_1Q_GATES = frozenset(["h", "rx", "ry", "sx", "sxdg", "u"])
BI_DIRECTIONAL_2Q_GATES = frozenset(["swap"])
UNI_DIRECTIONAL_2Q_CLIFFORD_GATES = frozenset(["cx", "cz"])
DIAGONAL_Z_1Q_GATES = frozenset(["rz", "t", "tdg", "s", "sdg", "z", "id"])

# Bounds for the deterministic/random accuracy split in the partially randomized
# time-evolution builder. The split fraction s sets two error sub-budgets: the
# Trotter step count grows like 1/sqrt(s) and the qDRIFT sample count like
# 1/sqrt(1 - s), so s = 0 or s = 1 would diverge. Clamping s to
# [ACCURACY_SPLIT_MIN, ACCURACY_SPLIT_MAX] keeps both counts finite (worst case
# ~1000x the baseline cost).
ACCURACY_SPLIT_MIN = 1e-6
ACCURACY_SPLIT_MAX = 1.0 - 1e-6

__all__ = [
    "ACCURACY_SPLIT_MAX",
    "ACCURACY_SPLIT_MIN",
    "BI_DIRECTIONAL_2Q_GATES",
    "DIAGONAL_Z_1Q_GATES",
    "NON_CLIFFORD_GATES",
    "SINGLE_QUBIT_CLIFFORD_GATES",
    "SUPERPOSITION_1Q_GATES",
    "TWO_QUBIT_CLIFFORD_GATES",
    "UNI_DIRECTIONAL_2Q_CLIFFORD_GATES",
]
