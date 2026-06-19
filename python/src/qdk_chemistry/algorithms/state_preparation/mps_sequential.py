"""MPS Berry state preparation via sequential site unitaries.

Implements the Matrix Product State (MPS) sequential state preparation algorithm
based on Berry et al. (arXiv:2409.11748). Each site unitary is decomposed using
the 7-matrix CSD construction (Appendix B) and synthesized into a quantum circuit
via Givens rotation layers with QROAM angle loading and phase gradient rotations.

Attribution
-----------
The Berry decomposition and Givens rotation circuit synthesis is based on code
originally published by Felix Rupprecht (DLR) on Zenodo:
    https://zenodo.org/records/15587498
The implementation has been rewritten and adapted for integration into the
QDK Chemistry library.

References
----------
- Berry, Tong, et al. arXiv:2409.11748
- Rupprecht & Wölk (2025), Zenodo: https://zenodo.org/records/15587498
"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from __future__ import annotations

from pathlib import Path

from qdk_chemistry.data.circuit import Circuit, QsharpFactoryData
from qdk_chemistry.data.mps_wavefunction import MPSWavefunction

from .mps_berry_preprocessing import prepare_gate_based_data
from .state_preparation import StatePreparation

__all__: list[str] = ["MPSSequentialStatePreparation"]

# Q# source files for MPS Berry state preparation (loaded via utils/qsharp)
_MPS_QS_DIR = Path(__file__).parents[2] / "utils" / "qsharp"

_MPS_QS_FILES = [
    _MPS_QS_DIR / "PhaseGradient.qs",
    _MPS_QS_DIR / "QroamStatePrep.qs",
    _MPS_QS_DIR / "GivensDecomposition.qs",
    _MPS_QS_DIR / "MPSPreparationBerry.qs",
]


def _get_mps_berry_op():
    """Lazily load the MPS Berry Q# operations."""
    import qdk  # noqa: PLC0415
    from qdk import qsharp  # noqa: PLC0415

    try:
        return qdk.code.MPSPreparationBerry
    except AttributeError:
        code = "\n".join(f.read_text() for f in _MPS_QS_FILES)
        qsharp.eval(code)
        return qdk.code.MPSPreparationBerry


class MPSSequentialStatePreparation(StatePreparation):
    r"""MPS-based sequential state preparation using the Berry decomposition.

    Implements the sequential site unitary construction from Berry et al.
    (arXiv:2409.11748) for preparing quantum states from Matrix Product State
    representations. Each site unitary is decomposed into:

    - Givens rotation layers (via QROAM + phase gradient)
    - Uniformly controlled Y-rotations (via QROAM + phase gradient)
    - Block-diagonal unitaries (via merged Givens layers)

    The algorithm prepares the state qubit-by-qubit (2 qubits per site), using
    an ancilla register that stores the virtual bond dimension.

    Attribution
    -----------
    Based on code originally published by Felix Rupprecht (DLR) on Zenodo:
        https://zenodo.org/records/15587498
    Rewritten and adapted for integration into the QDK Chemistry library.
    """

    def __init__(self):
        """Initialize the MPS sequential state preparation algorithm."""
        super().__init__()

    def name(self) -> str:
        """Return the algorithm name.

        Returns:
            str: The name ``"mps_sequential_state"``
        """
        return "mps_sequential_state"

    def _run_impl(self, wavefunction: MPSWavefunction) -> Circuit:
        """Prepare a quantum circuit from an MPS Wavefunction.

        Args:
            wavefunction: An MPSWavefunction containing MPS tensors.

        Returns:
            Circuit: A Circuit object implementing the MPS state preparation.

        Raises:
            TypeError: If wavefunction is not an MPSWavefunction instance.
        """
        if not isinstance(wavefunction, MPSWavefunction):
            raise TypeError(
                f"MPSSequentialStatePreparation requires an MPSWavefunction, "
                f"got {type(wavefunction).__name__}."
            )

        # Compute the gate-based decomposition data
        data = prepare_gate_based_data(wavefunction.tensors)

        # Build Q# factory parameters
        params = {
            "initialStateVec": data["initial_state_vec"],
            "numSites": data["num_sites"],
            "bRot": 10,  # default phase gradient precision
            "siteVLayerAngles": data["site_v_layer_angles"],
            "siteVLayerShifted": data["site_v_layer_shifted"],
            "siteVPhases": data["site_v_phases"],
            "siteRot0Angles": data["site_rot0_angles"],
            "siteRot1Angles": data["site_rot1_angles"],
            "siteRot2Angles": data["site_rot2_angles"],
            "siteW0LayerAngles": data["site_w0_layer_angles"],
            "siteW0LayerShifted": data["site_w0_layer_shifted"],
            "siteW0Phases": data["site_w0_phases"],
            "siteW1LayerAngles": data["site_w1_layer_angles"],
            "siteW1LayerShifted": data["site_w1_layer_shifted"],
            "siteW1Phases": data["site_w1_phases"],
            "siteULayerAngles": data["site_u_layer_angles"],
            "siteULayerShifted": data["site_u_layer_shifted"],
            "siteUPhases": data["site_u_phases"],
        }

        mps_ops = _get_mps_berry_op()

        qsharp_factory = QsharpFactoryData(
            program=mps_ops.MPSPreparationBerry,
            parameter=params,
        )

        return Circuit(qsharp_factory=qsharp_factory, encoding="jordan-wigner")
