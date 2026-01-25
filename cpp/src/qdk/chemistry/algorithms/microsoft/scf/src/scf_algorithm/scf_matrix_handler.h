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
   * @brief Receive and store Fock and Density matrices
   */
  void receive_F_P_matrices(const RowMajorMatrix& F, RowMajorMatrix& P);

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

 private:
  std::unique_ptr<impl::ROHFMatrixHandler> handler_impl_;
};

}  // namespace qdk::chemistry::scf
