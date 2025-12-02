"""QDK/Chemistry-PySCF Bindings."""
# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------


# Import necessary modules
def load():
    """Load the Pyscf plugin into QDK/Chemistry."""
    import qdk_chemistry.plugins.pyscf.active_space_avas  # noqa: PLC0415
    import qdk_chemistry.plugins.pyscf.coupled_cluster  # noqa: PLC0415
    import qdk_chemistry.plugins.pyscf.localization  # noqa: PLC0415
    import qdk_chemistry.plugins.pyscf.mcscf  # noqa: PLC0415
    import qdk_chemistry.plugins.pyscf.scf_solver  # noqa: PLC0415
    import qdk_chemistry.plugins.pyscf.stability  # noqa: PLC0415
