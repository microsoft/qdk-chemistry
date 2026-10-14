// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once

#include <Eigen/Core>
#include <memory>

namespace qdk::chemistry::scf {
class BasisSet;
struct Molecule;
struct ParallelConfig;

namespace detail {

/** @brief Internal decontracted basis and exact recontraction matrix. */
struct DecontractedBasis {
  std::shared_ptr<BasisSet> basis;
  Eigen::MatrixXd contraction;
};

/**
 * @brief Decontract a basis, coalescing only exactly repeated primitives.
 * @param contracted_basis Non-null contracted basis with molecular data.
 * @return Primitive basis and exact recontraction matrix, with primitive AO
 * rows and original contracted AO columns; recontraction is C transpose H C.
 */
DecontractedBasis decontract_basis(
    const std::shared_ptr<BasisSet>& contracted_basis);

}  // namespace detail

/**
 * @brief Build the spin-free X2C-1e one-electron AO Hamiltonian.
 * @param internal_basis_set Non-null internal spherical all-electron basis.
 * @param mpi Parallel configuration used for integral evaluation.
 * @param decontract Whether to decontract before X2C and recontract afterward.
 * @param nuclei Current nuclear positions and charges supplying V and pVp;
 * AO centers remain those of internal_basis_set.
 * @return Square AO matrix in Hartree in the original supplied basis.
 * @throws std::invalid_argument for non-finite, Cartesian, or ECP basis inputs.
 * @throws std::runtime_error for multi-rank execution, failed numerical
 * construction, or screening that loses the supplied contracted space.
 */
Eigen::MatrixXd build_x2c_one_body_ao(
    const std::shared_ptr<BasisSet>& internal_basis_set,
    const ParallelConfig& mpi, bool decontract, const Molecule& nuclei);

}  // namespace qdk::chemistry::scf
