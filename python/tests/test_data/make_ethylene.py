"""Generate ethylene Hamiltonian and run SCI calculation to verify."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from pathlib import Path

from qdk_chemistry.algorithms import create
from qdk_chemistry.data import Configuration, Orbitals, Structure

if __name__ == "__main__":
    ethylene = Structure.from_xyz_file(Path("ethylene.structure.xyz"))
    hf = create("scf_solver", "qdk", method="hf", basis_set="def2-svp")
    hf_energy, hf_wfn = hf.run(ethylene, 0, 1)

    print("HF energy and determinant:", hf_energy, hf_wfn.get_active_num_electrons(), hf_wfn.get_determinants()[0])

    active_space_orbitals = Orbitals(
        hf_wfn.get_orbitals().get_coefficients()[0],
        hf_wfn.get_orbitals().get_energies()[0],
        hf_wfn.get_orbitals().get_overlap_matrix(),
        hf_wfn.get_orbitals().get_basis_set(),
        (
            [5, 7, 8, 9],  # active orbitals
            [0, 1, 2, 3, 4, 6],  # inactive orbitals
        ),
    )

    h_gen = create("hamiltonian_constructor", "qdk")
    active_space_h = h_gen.run(active_space_orbitals)
    json_file = Path("ethylene_4e4o_2det.hamiltonian.json")
    if json_file.exists():
        json_file.rename(json_file.with_suffix(".json.bak"))
    active_space_h.to_json_file(json_file)

    sci = create("multi_configuration_calculator", "macis_cas")
    sci_energy, sci_wfn = sci.run(active_space_h, 2, 2)

    pmc = create("projected_multi_configuration_calculator", "macis_pmc")
    pmc_energy, pmc_wfn = pmc.run(active_space_h, [Configuration("2200"), Configuration("2020")])
    print("PMC energy:", pmc_energy)
    print("PMC correction:", pmc_energy - active_space_h.get_core_energy())
    print("Leading configurations and coefficients:")
    for det, coeff in zip(pmc_wfn.get_determinants(), pmc_wfn.get_coefficients(), strict=True):
        if abs(coeff) > 0.001:
            print(det, coeff)
