// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once

#include <qdk/chemistry/scf/core/types.h>

#include <memory>

namespace qdk::chemistry::scf {
class BasisSet;
class OneBodyIntegral;
struct ParallelConfig;

namespace detail {

/** @brief Internal decontracted basis and exact recontraction matrix. */
struct DecontractedBasis {
  std::shared_ptr<BasisSet> basis;
  Eigen::MatrixXd contraction;
};

DecontractedBasis decontract_basis(
    const std::shared_ptr<BasisSet>& contracted_basis);

}  // namespace detail

/**
 * @brief Build the nonrelativistic one-electron AO Hamiltonian.
 * @param basis_set Internal AO basis.
 * @param integrals One-electron integral evaluator for the same basis.
 * @return Kinetic, nuclear-attraction, and ECP integrals in the AO basis.
 */
RowMajorMatrix build_nonrelativistic_one_body_ao(const BasisSet& basis_set,
                                                 OneBodyIntegral& integrals);

/**
 * @brief Build the spin-free X2C-1e one-electron AO Hamiltonian.
 * @param internal_basis_set Internal spherical all-electron basis.
 * @param mpi Parallel configuration used for integral evaluation.
 * @param decontract Whether to decontract before X2C and recontract afterward.
 * @return X2C-1e one-electron matrix in the supplied basis.
 */
RowMajorMatrix build_x2c_one_body_ao(
    const std::shared_ptr<BasisSet>& internal_basis_set,
    const ParallelConfig& mpi, bool decontract);

}  // namespace qdk::chemistry::scf
