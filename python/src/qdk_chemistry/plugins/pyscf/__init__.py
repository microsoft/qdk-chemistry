"""QDK/Chemistry-PySCF Bindings."""
# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------


# Import necessary modules
def load():
    """Load the Pyscf plugin into QDK/Chemistry."""
    from qdk_chemistry.algorithms import register  # noqa: PLC0415
    from qdk_chemistry.plugins.pyscf.active_space_avas import PyscfAVAS  # noqa: PLC0415
    from qdk_chemistry.plugins.pyscf.coupled_cluster import PyscfCoupledClusterCalculator  # noqa: PLC0415
    from qdk_chemistry.plugins.pyscf.localization import PyscfLocalizer  # noqa: PLC0415
    from qdk_chemistry.plugins.pyscf.mcscf import PyscfMcscfCalculator  # noqa: PLC0415
    from qdk_chemistry.plugins.pyscf.scf_solver import PyscfScfSolver  # noqa: PLC0415
    from qdk_chemistry.plugins.pyscf.stability import PyscfStabilityChecker  # noqa: PLC0415

    register(lambda: PyscfAVAS())
    register(lambda: PyscfCoupledClusterCalculator())
    register(lambda: PyscfLocalizer())
    register(lambda: PyscfMcscfCalculator())
    register(lambda: PyscfScfSolver())
    register(lambda: PyscfStabilityChecker())
