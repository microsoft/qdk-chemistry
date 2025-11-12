// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once

#include <qdk/chemistry/scf/core/basis_set.h>
#include <qdk/chemistry/scf/core/scf.h>

namespace qdk::chemistry::scf {
/**
 * @brief Compute Schwarz integral estimates for shell pairs
 *
 * Calculates upper bounds (μν|μν) for screening ERI integrals.
 *
 * @param obs Orbital basis set
 * @param mpi MPI configuration
 * @param res Output array of Schwarz estimates
 */
void schwarz_integral(const BasisSet* obs, const ParallelConfig& mpi,
                      double* res);

/**
 * @brief Compute shell-pair density norms
 *
 * Calculates max_μ,ν in shell |D_μν| for density screening.
 *
 * @param obs Orbital basis set
 * @param D Density matrix
 * @param res Output array of shell norms
 */
void compute_shell_norm(const BasisSet* obs, const double* D, double* res);
}  // namespace qdk::chemistry::scf
