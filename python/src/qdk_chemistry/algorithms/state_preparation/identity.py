"""Identity state preparation — leaves the initial state unchanged."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from qdk_chemistry.data import Circuit
from qdk_chemistry.data.circuit import QsharpFactoryData
from qdk_chemistry.utils.qsharp import QSHARP_UTILS

__all__: list[str] = ["identity_state_prep"]


def identity_state_prep(num_qubits: int) -> Circuit:
    """Create an identity state-preparation circuit that leaves the initial state unchanged.

    Useful as a trivial state-prep when evolving from a computational
    basis state (e.g. ``|0...0>``) without any additional preparation.

    Args:
        num_qubits: Number of qubits in the circuit.

    Returns:
        A ``Circuit`` representing the identity operation on *num_qubits* qubits.

    """
    params = {"pauliExponents": [], "pauliCoefficients": [], "repetitions": 1}
    targets = list(range(num_qubits))
    return Circuit(
        qsharp_op=QSHARP_UTILS.PauliExp.MakeRepPauliExpOp(params),
        qsharp_factory=QsharpFactoryData(
            program=QSHARP_UTILS.PauliExp.MakeRepPauliExpCircuit,
            parameter={"evo_params": params, "target_indices": targets},
        ),
    )
