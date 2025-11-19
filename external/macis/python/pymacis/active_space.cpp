// Copyright (c) Microsoft Corporation.

#include "active_space.hpp"

#include <macis/mcscf/fock_matrices.hpp>

Hamiltonian compute_active_hamiltonian(size_t nactive, size_t ninactive,
                                       const Hamiltonian &ham) {
  size_t norb = ham.norb;
  if (norb == 0) {
    throw std::runtime_error("Number of orbitals must be greater than 0");
  }
  if (nactive == 0) {
    throw std::runtime_error(
        "Number of active orbitals must be greater than 0");
  }
  if (nactive + ninactive > norb) {
    throw std::runtime_error(
        "Total number of active and inactive orbitals exceeds number of "
        "orbitals");
  }

  if (ninactive == 0)
    return ham;  // No inactive orbitals, return original Hamiltonian

  auto *t_ptr = ham._T.data();
  auto *v_ptr = ham._V.data();

  // Output active Hamiltonian
  Hamiltonian active_ham;
  active_ham.norb = nactive;
  active_ham.nbasis = ham.nbasis;
  active_ham._T.resize(nactive * nactive);
  active_ham._V.resize(nactive * nactive * nactive * nactive);
  active_ham._F_inactive.resize(ham.nbasis * ham.nbasis);
  auto *ta_ptr = active_ham._T.data();
  auto *va_ptr = active_ham._V.data();
  auto *fi_ptr = active_ham._F_inactive.data();

  // Perform linear active space downfolding
  macis::active_hamiltonian(macis::NumOrbital(norb), macis::NumActive(nactive),
                            macis::NumInactive(ninactive), t_ptr, norb, v_ptr,
                            norb, fi_ptr, norb, ta_ptr, nactive, va_ptr,
                            nactive);

  // Compute Core Energy
  active_ham.core_energy = ham.core_energy;
  active_ham.core_energy += macis::inactive_energy(
      macis::NumInactive(ninactive), t_ptr, norb, fi_ptr, norb);

  return active_ham;
}

void export_active_space_pybind(pybind11::module &m) {
  // Active space operations
  m.def("compute_active_hamiltonian", &compute_active_hamiltonian,
        R"pbdoc(
            Compute active space Hamiltonian from full system.

            Transforms a full molecular Hamiltonian to an active space representation
            by incorporating the mean-field effects of inactive (doubly occupied) orbitals
            into the one-electron part of the active space Hamiltonian.

            Args:
                nactive (int): Number of active orbitals to include in CI calculation.
                ninactive (int): Number of inactive (doubly occupied) orbitals.
                H (Hamiltonian): Full system Hamiltonian containing all molecular orbitals.

            Returns:
                Hamiltonian: Active space Hamiltonian with:
                    - norb = nactive
                    - Modified one-electron integrals including inactive orbital contributions
                    - Two-electron integrals for active orbitals only
                    - Updated core energy including inactive orbital energy

            Note:
                The total number of orbitals in the input Hamiltonian should be
                nactive + ninactive. The active orbitals are assumed to be the
                first nactive orbitals in the orbital ordering.

            Example:
                >>> # Full system with 10 orbitals, use 6 active + 2 inactive
                >>> full_ham = pymacis.read_fcidump("system.fcidump")
                >>> active_ham = pymacis.compute_active_hamiltonian(6, 2, full_ham)
                >>> # Now ready for CASCI with 6 active orbitals
                >>> result = pymacis.casci(nalpha=5, nbeta=5, H=active_ham)
        )pbdoc",
        py::arg("nactive"), py::arg("ninactive"), py::arg("H"));
}
