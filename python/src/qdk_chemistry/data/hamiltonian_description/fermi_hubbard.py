"""QDK/Chemistry Fermi-Hubbard model Hamiltonian description module."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from qdk_chemistry._core.data import Hamiltonian, LatticeGeometry, LatticeGraph
from qdk_chemistry._core.utils.model_hamiltonians import create_hubbard_hamiltonian
from qdk_chemistry.data.hamiltonian_description.model_hamiltonian import ModelHamiltonianDescription

__all__: list[str] = ["FermiHubbardModelHamiltonianDescription"]


class FermiHubbardModelHamiltonianDescription(ModelHamiltonianDescription):
    r"""The Fermi-Hubbard model on the nearest-neighbor bonds of a lattice.

    .. math::

        H = \sum_{i,\sigma} \epsilon\, n_{i\sigma}
          - t \sum_{\langle i,j \rangle, \sigma} (a^\dagger_{i\sigma} a_{j\sigma} + \text{h.c.})
          + U \sum_i n_{i\uparrow} n_{i\downarrow}
    """

    @staticmethod
    def data_type_name() -> str:
        """Return the wire-format identifier for a Fermi-Hubbard model description.

        Returns:
            ``"fermi_hubbard_model_hamiltonian_description"``.

        """
        return "fermi_hubbard_model_hamiltonian_description"

    def __init__(self, lattice: LatticeGeometry, t: float, u: float, epsilon: float = 0.0) -> None:
        """Initialize a Fermi-Hubbard model description.

        Args:
            lattice: The lattice the model is defined on.
            t: The nearest-neighbor hopping integral.
            u: The on-site Coulomb repulsion.
            epsilon: The on-site orbital energy.

        """
        super().__init__(lattice, {"t": t, "u": u, "epsilon": epsilon})

    def materialize(self) -> Hamiltonian:
        """Build the Fermi-Hubbard Hamiltonian with ``create_hubbard_hamiltonian``.

        Returns:
            Hamiltonian: The Fermi-Hubbard Hamiltonian on the nearest-neighbor bonds of the lattice.

        """
        return create_hubbard_hamiltonian(
            LatticeGraph.from_geometry(self.lattice),
            epsilon=self.parameters["epsilon"],
            t=self.parameters["t"],
            U=self.parameters["u"],
        )
