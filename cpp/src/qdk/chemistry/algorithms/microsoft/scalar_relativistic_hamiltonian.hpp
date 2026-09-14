// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once

#include <Eigen/Core>
#include <memory>

namespace qdk::chemistry::scf {
class BasisSet;
}

namespace qdk::chemistry::algorithms::microsoft {

namespace detail {

/** @brief Internal decontracted basis and exact recontraction matrix. */
struct DecontractedBasis {
  std::shared_ptr<qdk::chemistry::scf::BasisSet> basis;
  Eigen::MatrixXd contraction;
};

/**
 * @brief Decontract a basis while coalescing only exactly repeated primitives.
 * @param contracted_basis Internal contracted basis representation.
 * @return Uncontracted basis and its exact recontraction matrix.
 */
DecontractedBasis decontract_basis(
    const std::shared_ptr<qdk::chemistry::scf::BasisSet>& contracted_basis);

/**
 * @brief Build the spin-free X2C-1e AO Hamiltonian.
 * @param internal_basis_set Internal spherical all-electron basis.
 * @param decontract Whether to decontract before X2C and recontract afterward.
 * @return X2C-1e one-electron matrix in the supplied basis.
 */
Eigen::MatrixXd build_x2c_one_body_ao(
    const std::shared_ptr<qdk::chemistry::scf::BasisSet>& internal_basis_set,
    bool decontract);

}  // namespace detail

}  // namespace qdk::chemistry::algorithms::microsoft
