// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once

#include <btas/tensor.h>

#include <SeQuant/core/container.hpp>
#include <SeQuant/core/eval/result.hpp>
#include <qdk/chemistry/utils/cutensor/engine.hpp>

namespace qdk::chemistry::utils::cutensor {

/**
 * @brief SeQuant evaluation result held as a device tensor and evaluated with
 *        cuTENSOR on the calling thread's Engine.
 *
 * It reads the index-hash annotations of sequant::EvalExprBTAS, so evaluation
 * trees binarized for BTAS run unchanged once their leaves are device tensors.
 * Contractions (including batch modes), permutations, sums, and scaling run on
 * the device; repeated labels within one operand, labels summed within a single
 * operand, and (anti)symmetrization fall back to the BTAS backend on the host.
 */
class ResultTensorCUTENSOR final : public sequant::Result {
 public:
  using annot_t = sequant::container::svector<long>;

  explicit ResultTensorCUTENSOR(DeviceTensor tensor)
      : Result{std::move(tensor)} {}

  [[nodiscard]] sequant::ResultPtr sum(
      Result const& other, std::array<std::any, 3> const& annot) const override;
  [[nodiscard]] sequant::ResultPtr prod(Result const& other,
                                        std::array<std::any, 3> const& annot,
                                        sequant::DeNest flag) const override;
  [[nodiscard]] sequant::ResultPtr permute(
      std::array<std::any, 2> const& annot) const override;
  [[nodiscard]] sequant::ResultPtr adjoint(
      std::array<std::any, 2> const& annot) const override;
  void add_inplace(Result const& other) override;
  [[nodiscard]] double norm2() const override;
  [[nodiscard]] sequant::ResultPtr clone() const override;
  [[nodiscard]] sequant::ResultPtr symmetrize() const override;
  [[nodiscard]] sequant::ResultPtr antisymmetrize(
      std::size_t bra_rank) const override;
  [[nodiscard]] sequant::ResultPtr mult_by_phase(
      std::int8_t factor) const override;
  [[nodiscard]] std::size_t size_in_bytes() const override;

 private:
  [[nodiscard]] id_t type_id() const noexcept override {
    return id_for_type<ResultTensorCUTENSOR>();
  }
};

/// Uploads a row-major host tensor with the current thread's Engine.
sequant::ResultPtr to_device(const btas::Tensor<double>& host);

/// Uploads row-major host data of the given extents with the current Engine.
sequant::ResultPtr to_device(const double* host, Extents extents);

/// Downloads a device tensor into a row-major host tensor (waits for it).
btas::Tensor<double> to_host(const DeviceTensor& tensor);

}  // namespace qdk::chemistry::utils::cutensor
