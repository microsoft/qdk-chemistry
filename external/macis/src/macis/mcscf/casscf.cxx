/*
 * MACIS Copyright (c) 2023, The Regents of the University of California,
 * through Lawrence Berkeley National Laboratory (subject to receipt of
 * any required approvals from the U.S. Dept. of Energy). All rights reserved.
 * Portions Copyright (c) Microsoft Corporation.
 *
 * See LICENSE.txt for details
 */

#include <macis/hamiltonian_generator/double_loop.hpp>
#include <macis/mcscf/cas.hpp>
#include <macis/mcscf/mcscf.hpp>
#include <macis/mcscf/mcscf_impl.hpp>

namespace macis {

double casscf_diis(MCSCFSettings settings, NumElectron nalpha,
                   NumElectron nbeta, NumOrbital norb, NumInactive ninact,
                   NumActive nact, NumVirtual num_virtual_orbitals,
                   double E_core, double* T, size_t LDT, double* V, size_t LDV,
                   double* A1RDM, size_t LDD1, double* A2RDM,
                   size_t LDD2 MACIS_MPI_CODE(, MPI_Comm comm)) {
  using generator_t = DoubleLoopHamiltonianGenerator<wfn_t<64>>;
  using functor_t = CASRDMFunctor<generator_t>;
  functor_t op;
  return mcscf_impl<functor_t>(op, settings, nalpha, nbeta, norb, ninact, nact,
                               num_virtual_orbitals, E_core, T, LDT, V, LDV,
                               A1RDM, LDD1, A2RDM, LDD2 MACIS_MPI_CODE(, comm));
}

}  // namespace macis
