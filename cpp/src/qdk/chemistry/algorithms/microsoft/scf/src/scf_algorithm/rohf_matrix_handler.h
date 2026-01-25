// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once

#include <qdk/chemistry/scf/core/types.h>

#include <memory>

namespace qdk::chemistry::scf {

namespace impl {
class ROHFMatrixHandler;
}  // namespace impl

class ROHFMatrixHandler {
 public:
  explicit ROHFMatrixHandler();

  /**
   * @brief Destructor
   */
  ~ROHFMatrixHandler() noexcept;

  /**
   * @brief Build ROHF effective Fock matrix and total Density matrix
   *
   * @param[in] F Spin-blocked Fock matrix
   * @param[in] C Orbital coefficients matrix
   * @param[in] P Spin-blocked density matrix
   * @param[in] nd Number of doubly occupied orbitals
   * @param[in] ns Number of singly occupied orbitals
   */
  void build_ROHF_F_P_matrix(const RowMajorMatrix& F, const RowMajorMatrix& C,
                             RowMajorMatrix& P, int nd, int ns);

  /**
   * @brief Get reference to Fock matrix
   *
   * @return Reference to Fock matrix
   */
  const RowMajorMatrix& get_fock_matrix();

  /**
   * @brief Get reference to Density matrix
   *
   * @return Reference to Density matrix
   */
  RowMajorMatrix& get_density_matrix();

  /**
   * @brief Update spin-blocked density matrices from total density matrix
   *
   * @param[out] P Spin-blocked density matrices to update
   * @param[in] C Orbital coefficients matrix
   */
  void update_spin_density_matrices(RowMajorMatrix& P, const RowMajorMatrix& C);

 private:
  std::unique_ptr<impl::ROHFMatrixHandler> handler_impl_;
};

}  // namespace qdk::chemistry::scf
