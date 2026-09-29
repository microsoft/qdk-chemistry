"""Plan which phase gradient qubits controlled unitaries can share."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import math
from collections.abc import Sequence
from dataclasses import dataclass

from qdk_chemistry.data.circuit import PhaseGradient

__all__: list[str] = ["GradientPoolPlan", "plan_gradient_pool"]

_ANGLE_TOLERANCE = 1e-12


@dataclass(frozen=True)
class GradientPoolPlan:
    """Which gradient qubits several circuits share, and which each prepares for itself."""

    pool_angles: tuple[float, ...]
    """Angle of each shared qubit, prepared once and appended to every circuit's targets."""

    layouts: tuple[tuple[int, ...], ...]
    """Per circuit, where each of its gradient qubits comes from: a pool index, or ``-1 - i`` for own qubit ``i``."""

    own_angles: tuple[tuple[float, ...], ...]
    """Per circuit, the angles of the qubits only it needs, prepared around each call."""


def plan_gradient_pool(requests: Sequence[tuple[PhaseGradient, ...]]) -> GradientPoolPlan:
    r"""Share every gradient qubit that more than one circuit needs.

    A gradient :math:`\sum_k e^{-i\varphi k}|k\rangle` is the product over its qubits ``j`` of
    :math:`(|0\rangle + e^{-i\varphi 2^j}|1\rangle)/\sqrt{2}`, and its consumers return it unchanged. Each qubit is
    therefore a reusable resource fixed by one angle modulo :math:`2\pi`, and circuits needing the same angle can share
    it. This recovers, for example, that the gradient for :math:`2\varphi` is the gradient for :math:`\varphi` without
    its lowest qubit. Only qubits two or more circuits need are pooled and held throughout; a qubit only one circuit
    needs is prepared around that circuit's call. A circuit that needs an angle twice gets two distinct qubits, so it
    never receives the same qubit twice.

    Args:
        requests: The gradients each circuit declares, in register order.

    Returns:
        GradientPoolPlan: The shared pool and how each circuit draws from it.

    """
    circuit_slots = []
    angles_by_slot: dict[tuple[int, int], float] = {}
    for gradients in requests:
        occurrences: dict[int, int] = {}
        slots = []
        for gradient in gradients:
            for bit in range(gradient.num_qubits):
                angle = -gradient.phase * 2.0**bit
                key = _angle_key(angle)
                slot = (key, occurrences.get(key, 0))
                occurrences[key] = slot[1] + 1
                angles_by_slot.setdefault(slot, angle)
                slots.append(slot)
        circuit_slots.append(slots)

    users: dict[tuple[int, int], int] = {}
    for slots in circuit_slots:
        for slot in set(slots):
            users[slot] = users.get(slot, 0) + 1

    pool_index: dict[tuple[int, int], int] = {}
    for slots in circuit_slots:
        for slot in slots:
            if users[slot] > 1 and slot not in pool_index:
                pool_index[slot] = len(pool_index)

    layouts, own_angles = [], []
    for slots in circuit_slots:
        layout: list[int] = []
        own: list[float] = []
        for slot in slots:
            if slot in pool_index:
                layout.append(pool_index[slot])
            else:
                layout.append(-1 - len(own))
                own.append(angles_by_slot[slot])
        layouts.append(tuple(layout))
        own_angles.append(tuple(own))

    return GradientPoolPlan(
        pool_angles=tuple(angles_by_slot[slot] for slot in pool_index),
        layouts=tuple(layouts),
        own_angles=tuple(own_angles),
    )


def _angle_key(angle: float) -> int:
    """Bucket an angle modulo 2π; angles in different buckets are only ever not shared, never confused."""
    buckets = round(2.0 * math.pi / _ANGLE_TOLERANCE)
    return round((angle % (2.0 * math.pi)) / _ANGLE_TOLERANCE) % buckets
