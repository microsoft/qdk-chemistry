// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <btas/btas.h>
#include <btas/tensor_func.h>
#include <gtest/gtest.h>

#include <SeQuant/core/context.hpp>
#include <SeQuant/core/eval/backends/btas/eval_expr.hpp>
#include <SeQuant/core/eval/backends/btas/result.hpp>
#include <SeQuant/core/eval/eval.hpp>
#include <SeQuant/core/expr.hpp>
#include <SeQuant/domain/mbpt/context.hpp>
#include <SeQuant/domain/mbpt/convention.hpp>
#include <SeQuant/domain/mbpt/op.hpp>
#include <cstddef>
#include <random>
#include <utility>
#include <vector>

#include "ut_common.hpp"

namespace {

using BTensor = btas::Tensor<double>;

BTensor random_tensor(std::vector<std::size_t> const& extents,
                      std::mt19937& engine) {
  std::uniform_real_distribution<double> distribution(-1.0, 1.0);
  BTensor x{btas::Range(extents)};
  x.generate([&] { return distribution(engine); });
  return x;
}

// Antisymmetric within the bra pair and within the ket pair.
BTensor antisymmetrize(BTensor const& x) {
  BTensor y = x;
  y -= BTensor{btas::permute(x, {1, 0, 2, 3})};
  y -= BTensor{btas::permute(x, {0, 1, 3, 2})};
  y += BTensor{btas::permute(x, {1, 0, 3, 2})};
  return y;
}

}  // namespace

TEST(SeQuantBtasTest, EvaluatesWickDerivedCcsdEnergy) {
  using namespace sequant;

  auto context = set_scoped_default_context(
      {.index_space_registry_shared_ptr =
           mbpt::make_min_sr_spaces(mbpt::SpinConvention::None),
       .vacuum = Vacuum::SingleProduct});
  auto mbpt_context = mbpt::set_scoped_default_mbpt_context(
      {.op_registry_ptr = mbpt::make_minimal_registry()});

  // With a normal-ordered two-body H, <0|H exp(T)|0> stops at second order.
  // Each factor needs its own T: tensor-level operators carry fixed indices.
  using mbpt::tensor::H;
  using mbpt::tensor::T;
  auto energy = mbpt::tensor::vac_av(
      H(2) * (T(2) + ex<Constant>(rational{1, 2}) * T(2) * T(2)));

  constexpr std::size_t nocc = 2;
  constexpr std::size_t nvirt = 3;
  constexpr std::size_t nmo = nocc + nvirt;
  std::mt19937 engine(42);
  // Full-space data with the symmetries SeQuant's canonicalization assumes.
  BTensor f = random_tensor({nmo, nmo}, engine);
  f += BTensor{btas::permute(f, {1, 0})};
  BTensor g = antisymmetrize(random_tensor({nmo, nmo, nmo, nmo}, engine));
  g += BTensor{btas::permute(g, {2, 3, 0, 1})};
  BTensor const t1 = random_tensor({nmo, nmo}, engine);
  BTensor const t2 =
      antisymmetrize(random_tensor({nmo, nmo, nmo, nmo}, engine));

  auto const occupied =
      get_default_context().index_space_registry()->retrieve(L"i");
  auto const leaf = [&](EvalNodeBTAS const& node) -> ResultPtr {
    if (node->result_type() != ResultType::Tensor) {
      return eval_result<ResultScalar<double>>(
          node->as_constant().value<double>());
    }
    auto const& tensor = node->as_tensor();
    BTensor const& full = tensor.label() == L"f"   ? f
                          : tensor.label() == L"g" ? g
                          : tensor.bra_rank() == 1 ? t1
                                                   : t2;
    std::vector<std::size_t> offsets;
    std::vector<std::size_t> extents;
    for (auto const& index : tensor.const_braket_indices()) {
      bool const is_occupied = index.space() == occupied;
      offsets.push_back(is_occupied ? 0 : nocc);
      extents.push_back(is_occupied ? nocc : nvirt);
    }
    BTensor block{btas::Range(extents)};
    for (auto const& idx : block.range()) {
      auto full_idx = idx;
      for (std::size_t k = 0; k < offsets.size(); ++k) {
        full_idx[k] += offsets[k];
      }
      block(idx) = full(full_idx);
    }
    return eval_result<ResultTensorBTAS<BTensor>>(std::move(block));
  };
  auto const tree = binarize<EvalExprBTAS>(ResultExpr(Variable(L"E"), energy));
  auto const computed = evaluate(tree, leaf)->get<double>();

  // Spin-orbital CCSD energy: f_ia t_ai + <ij||ab> (t_abij/4 + t_ai t_bj/2).
  double expected = 0.0;
  for (std::size_t i = 0; i < nocc; ++i) {
    for (std::size_t a = nocc; a < nmo; ++a) {
      expected += f(i, a) * t1(a, i);
      for (std::size_t j = 0; j < nocc; ++j) {
        for (std::size_t b = nocc; b < nmo; ++b) {
          expected += g(i, j, a, b) *
                      (0.25 * t2(a, b, i, j) + 0.5 * t1(a, i) * t1(b, j));
        }
      }
    }
  }
  EXPECT_NEAR(computed, expected, testing::numerical_zero_tolerance);
}
