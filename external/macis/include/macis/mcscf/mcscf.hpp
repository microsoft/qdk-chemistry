/*
 * MACIS Copyright (c) 2023, The Regents of the University of California,
 * through Lawrence Berkeley National Laboratory (subject to receipt of
 * any required approvals from the U.S. Dept. of Energy). All rights reserved.
 *
 * See LICENSE.txt for details
 */

#pragma once
#include <macis/types.hpp>
#include <macis/util/mpi.hpp>

namespace macis {

struct MCSCFSettings {
  size_t max_macro_iter = 100;
  double max_orbital_step = 0.5;
  double orb_grad_tol_mcscf = 5e-6;

  bool enable_diis = true;
  size_t diis_start_iter = 3;
  size_t diis_nkeep = 10;

  // size_t max_bfgs_iter      = 100;
  // double orb_grad_tol_bfgs  = 5e-7;

  double ci_res_tol = 1e-8;
  size_t ci_max_subspace = 200;
  double ci_matel_tol = std::numeric_limits<double>::epsilon();
};

double casscf_diis(MCSCFSettings settings, NumElectron nalpha,
                   NumElectron nbeta, NumOrbital norb, NumInactive ninact,
                   NumActive nact, NumVirtual num_virtual_orbitals,
                   double E_core, double* T, size_t LDT, double* V, size_t LDV,
                   double* A1RDM, size_t LDD1, double* A2RDM,
                   size_t LDD2 MACIS_MPI_CODE(, MPI_Comm comm));

}  // namespace macis
