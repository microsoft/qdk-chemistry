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

class SCFMatrixHandler {
 public:
 /**
   * @brief Virtual receiver for Fock and Density matrices
   */
  virtual void receive_F_P_matrices(const RowMajorMatrix& F, RowMajorMatrix& P) = 0;

  /**
   * @brief Virtual destructor
   */
  virtual ~SCFMatrixHandler() = default;

  /**
   * @brief Get reference to Fock matrix
   *
   * @return Reference to Fock matrix
   */
  virtual const RowMajorMatrix& get_fock_matrix() = 0;

  /**
   * @brief Get reference to Density matrix
   *
   * @return Reference to Density matrix
   */
  virtual RowMajorMatrix& get_density_matrix() = 0;

 protected:
  SCFMatrixHandler() {};
};

class RHFUHFMatrixHandler : public SCFMatrixHandler {
 public:
  explicit RHFUHFMatrixHandler()
      : SCFMatrixHandler() {};

  void receive_F_P_matrices(const RowMajorMatrix& F, RowMajorMatrix& P) override {
    Fock_matrix_ = F;
    Density_matrix_ = P;
  }

  /**
   * @brief Get reference to Fock matrix
   *
   * @return Reference to Fock matrix
   */
  const RowMajorMatrix& get_fock_matrix() override { return Fock_matrix_; };

  /**
   * @brief Get reference to Density matrix
   *
   * @return Reference to Density matrix
   */
  RowMajorMatrix& get_density_matrix() override { return Density_matrix_; }

 private:
  RowMajorMatrix& Fock_matrix_;
  RowMajorMatrix& Density_matrix_;
};

class ROHFMatrixHandler : public SCFMatrixHandler {
 public:
  explicit ROHFMatrixHandler()
      : SCFMatrixHandler(), handler_impl_() {};

  void receive_F_P_matrices(const RowMajorMatrix& F, RowMajorMatrix& P) override;

  /**
   * @brief Get reference to Fock matrix
   *
   * @return Reference to Fock matrix
   */
  const RowMajorMatrix& get_fock_matrix() override;

  /**
   * @brief Get reference to Density matrix
   *
   * @return Reference to Density matrix
   */
  RowMajorMatrix& get_density_matrix() override;

 private:
  std::unique_ptr<impl::ROHFMatrixHandler> handler_impl_;
};

}  // namespace qdk::chemistry::scf
