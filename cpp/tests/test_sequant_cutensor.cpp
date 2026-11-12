// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <gtest/gtest.h>

#include <SeQuant/core/eval/backends/btas/result.hpp>
#include <any>
#include <array>
#include <optional>
#include <qdk/chemistry/utils/cutensor/sequant_result.hpp>
#include <random>
#include <vector>

namespace {

namespace gpu = qdk::chemistry::utils::cutensor;
using Annot = sequant::container::svector<long>;
using HostTensor = btas::Tensor<double>;
using HostResult = sequant::ResultTensorBTAS<HostTensor>;
using Scalar = sequant::ResultScalar<double>;
using sequant::eval_result;
using sequant::ResultPtr;

// Each operation must match the BTAS backend on the same operands.
class SeQuantCutensorTest : public ::testing::Test {
 protected:
  void SetUp() override {
    if (gpu::device_count() == 0) GTEST_SKIP() << "no CUDA device";
    scope_.emplace(gpu::Engine::for_worker(0));
  }

  HostTensor random(std::vector<std::size_t> extents) {
    HostTensor tensor{btas::Range{extents}};
    std::uniform_real_distribution<double> value(-1.0, 1.0);
    for (auto& x : tensor) x = value(rng_);
    return tensor;
  }

  static ResultPtr host(HostTensor tensor) {
    return eval_result<HostResult>(std::move(tensor));
  }

  static void expect_same(const ResultPtr& expected, const ResultPtr& actual) {
    if (expected->is<Scalar>()) {
      ASSERT_TRUE(actual->is<Scalar>());
      EXPECT_NEAR(actual->get<double>(), expected->get<double>(), 1e-11);
      return;
    }
    ASSERT_TRUE(actual->is<gpu::ResultTensorCUTENSOR>());
    const auto& want = expected->get<HostTensor>();
    const auto got = gpu::to_host(actual->get<gpu::DeviceTensor>());
    ASSERT_EQ(got.rank(), want.rank());
    for (std::size_t i = 0; i < want.rank(); ++i)
      ASSERT_EQ(got.extent(i), want.extent(i));
    for (std::size_t i = 0; i < want.size(); ++i)
      EXPECT_NEAR(got.data()[i], want.data()[i], 1e-11) << "element " << i;
  }

  // Applies `op` to the BTAS and the cuTENSOR form of the same operands.
  template <typename Op>
  void check(const std::vector<HostTensor>& operands, Op&& op) {
    std::vector<ResultPtr> on_host, on_device;
    for (const auto& operand : operands) {
      on_host.push_back(host(operand));
      on_device.push_back(gpu::to_device(operand));
    }
    expect_same(op(on_host), op(on_device));
  }

  static std::array<std::any, 3> annot(Annot left, Annot right, Annot result) {
    return {std::any{left}, std::any{right}, std::any{result}};
  }

 private:
  std::mt19937 rng_{20261105};
  std::optional<gpu::EngineScope> scope_;
};

TEST_F(SeQuantCutensorTest, Contraction) {
  check({random({3, 4, 5}), random({5, 6, 4})}, [](const auto& t) {
    return t[0]->prod(*t[1], annot({1, 2, 3}, {3, 4, 2}, {4, 1}),
                      sequant::DeNest::False);
  });
}

TEST_F(SeQuantCutensorTest, BatchModes) {
  check({random({2, 3, 4}), random({2, 4, 5})}, [](const auto& t) {
    return t[0]->prod(*t[1], annot({7, 1, 3}, {7, 3, 2}, {1, 7, 2}),
                      sequant::DeNest::False);
  });
}

TEST_F(SeQuantCutensorTest, OuterProduct) {
  check({random({3, 2}), random({4})}, [](const auto& t) {
    return t[0]->prod(*t[1], annot({1, 2}, {3}, {3, 1, 2}),
                      sequant::DeNest::False);
  });
}

TEST_F(SeQuantCutensorTest, FullContractionToScalar) {
  check({random({3, 4, 5}), random({5, 3, 4})}, [](const auto& t) {
    return t[0]->prod(*t[1], annot({1, 2, 3}, {3, 1, 2}, {}),
                      sequant::DeNest::False);
  });
}

TEST_F(SeQuantCutensorTest, ScaledPermutation) {
  check({random({3, 4, 5})}, [](const auto& t) {
    const Scalar scale{-0.75};
    return scale.prod(*t[0], annot({}, {1, 2, 3}, {3, 1, 2}),
                      sequant::DeNest::False);
  });
}

TEST_F(SeQuantCutensorTest, PermutedSum) {
  check({random({3, 4, 5}), random({5, 3, 4})}, [](const auto& t) {
    return t[0]->sum(*t[1], annot({1, 2, 3}, {3, 1, 2}, {2, 3, 1}));
  });
}

TEST_F(SeQuantCutensorTest, PermuteAndAdjoint) {
  const std::array<std::any, 2> modes{std::any{Annot{1, 2, 3, 4}},
                                      std::any{Annot{3, 4, 1, 2}}};
  check({random({2, 3, 4, 5})},
        [&](const auto& t) { return t[0]->permute(modes); });
  check({random({2, 3, 4, 5})},
        [&](const auto& t) { return t[0]->adjoint(modes); });
}

TEST_F(SeQuantCutensorTest, AddInplaceCloneAndPhase) {
  check({random({4, 5}), random({4, 5})}, [](const auto& t) {
    auto copy = t[0]->clone();
    copy->add_inplace(*t[1]);
    copy->add_inplace(*t[1]);
    return copy->mult_by_phase(-1);
  });
}

TEST_F(SeQuantCutensorTest, CloneOwnsItsStorage) {
  check({random({4, 5}), random({4, 5})}, [](const auto& t) {
    auto copy = t[0]->clone();
    copy->add_inplace(*t[1]);
    return t[0];
  });
}

TEST_F(SeQuantCutensorTest, Antisymmetrize) {
  check({random({3, 3, 4, 4})},
        [](const auto& t) { return t[0]->antisymmetrize(2); });
  check({random({3, 3, 4, 4})},
        [](const auto& t) { return t[0]->symmetrize(); });
}

}  // namespace
