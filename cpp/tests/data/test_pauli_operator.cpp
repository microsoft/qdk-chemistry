// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <gtest/gtest.h>

#include <qdk/chemistry/data/pauli_operator.hpp>

#include "ut_common.hpp"

using namespace qdk::chemistry::data;

// PauliOperatorExpression Tests

TEST(PauliOperatorExpressionTest, PauliOperatorConstruction) {
  PauliOperator opX = PauliOperator::X(0);
  EXPECT_EQ(opX.get_operator_type(), 1);
  EXPECT_EQ(opX.get_qubit_index(), 0);
  EXPECT_TRUE(opX.is_pauli_operator());
  EXPECT_FALSE(opX.is_product_expression());
  EXPECT_FALSE(opX.is_sum_expression());

  PauliOperator opY = PauliOperator::Y(1);
  EXPECT_EQ(opY.get_operator_type(), 2);
  EXPECT_EQ(opY.get_qubit_index(), 1);

  PauliOperator opZ = PauliOperator::Z(2);
  EXPECT_EQ(opZ.get_operator_type(), 3);
  EXPECT_EQ(opZ.get_qubit_index(), 2);

  PauliOperator opI = PauliOperator::I(3);
  EXPECT_EQ(opI.get_operator_type(), 0);
  EXPECT_EQ(opI.get_qubit_index(), 3);
}

TEST(PauliOperatorExpressionTest, PauliOperatorToString) {
  PauliOperator opX = PauliOperator::X(0);
  EXPECT_EQ(opX.to_string(), "X(0)");

  PauliOperator opY = PauliOperator::Y(1);
  EXPECT_EQ(opY.to_string(), "Y(1)");

  PauliOperator opZ = PauliOperator::Z(2);
  EXPECT_EQ(opZ.to_string(), "Z(2)");

  PauliOperator opI = PauliOperator::I(3);
  EXPECT_EQ(opI.to_string(), "I(3)");
}

TEST(PauliOperatorExpressionTest, PauliOperatorClone) {
  PauliOperator opX = PauliOperator::X(0);
  auto opX_clone = opX.clone();
  auto& cloned_opX = dynamic_cast<PauliOperator&>(*opX_clone);
  EXPECT_EQ(cloned_opX.get_operator_type(), 1);
  EXPECT_EQ(cloned_opX.get_qubit_index(), 0);
  EXPECT_NE(opX_clone.get(), &opX);
}

// ProductPauliOperatorExpression Tests

TEST(PauliOperatorExpressionTest, ProductPauliOperatorConstruction) {
  // OP = 1.0
  ProductPauliOperatorExpression prod;
  EXPECT_EQ(prod.get_coefficient(), std::complex<double>(1.0, 0.0));
  EXPECT_TRUE(prod.get_factors().empty());
  EXPECT_TRUE(prod.is_product_expression());
  EXPECT_FALSE(prod.is_pauli_operator());
  EXPECT_FALSE(prod.is_sum_expression());

  // OP = (2.0 - i)
  ProductPauliOperatorExpression prod_with_coeff(
      std::complex<double>(2.0, -1.0));
  EXPECT_EQ(prod_with_coeff.get_coefficient(), std::complex<double>(2.0, -1.0));
  EXPECT_TRUE(prod_with_coeff.get_factors().empty());

  // OP = X(0) * Y(1)
  PauliOperator opX = PauliOperator::X(0);
  PauliOperator opY = PauliOperator::Y(1);
  ProductPauliOperatorExpression prod_from_ops(opX, opY);
  EXPECT_EQ(prod_from_ops.get_coefficient(), std::complex<double>(1.0, 0.0));
  EXPECT_EQ(prod_from_ops.get_factors().size(), 2);

  // OP = 3.0 * X(0)
  ProductPauliOperatorExpression prod_with_coeff_and_op(
      std::complex<double>(3.0, 0.0), opX);
  EXPECT_EQ(prod_with_coeff_and_op.get_coefficient(),
            std::complex<double>(3.0, 0.0));
  EXPECT_EQ(prod_with_coeff_and_op.get_factors().size(), 1);
  EXPECT_EQ(
      dynamic_cast<PauliOperator&>(*prod_with_coeff_and_op.get_factors()[0])
          .get_operator_type(),
      1);

  // Copy constructor test
  ProductPauliOperatorExpression prod_copy(prod_with_coeff_and_op);
  EXPECT_EQ(prod_copy.get_coefficient(), std::complex<double>(3.0, 0.0));
  EXPECT_EQ(prod_copy.get_factors().size(), 1);
  EXPECT_EQ(dynamic_cast<PauliOperator&>(*prod_copy.get_factors()[0])
                .get_operator_type(),
            1);
}

TEST(PauliOperatorExpressionTest, ProductPauliOperatorToString) {
  // OP = 2 * X(0) * Y(1)
  auto prod = std::make_unique<ProductPauliOperatorExpression>(
      std::complex<double>(2.0, 0.0));
  prod->add_factor(std::make_unique<PauliOperator>(PauliOperator::X(0)));
  prod->add_factor(std::make_unique<PauliOperator>(PauliOperator::Y(1)));

  EXPECT_EQ(prod->to_string(), "2 * X(0) * Y(1)");

  // Product with sum
  // OP = 2 * X(0) * Y(1) * (Z(2) + I(3))
  auto sum = std::make_unique<SumPauliOperatorExpression>(PauliOperator::Z(2),
                                                          PauliOperator::I(3));
  prod->add_factor(std::move(sum));

  EXPECT_EQ(prod->to_string(), "2 * X(0) * Y(1) * (Z(2) + I(3))");
}

TEST(PauliOperatorExpressionTest, ProductPauliOperatorClone) {
  auto prod = std::make_unique<ProductPauliOperatorExpression>(
      std::complex<double>(2.0, 0.0));
  prod->add_factor(std::make_unique<PauliOperator>(PauliOperator::X(0)));
  prod->add_factor(std::make_unique<PauliOperator>(PauliOperator::Y(1)));

  auto prod_clone = prod->clone();
  auto& cloned_prod =
      dynamic_cast<ProductPauliOperatorExpression&>(*prod_clone);
  EXPECT_EQ(cloned_prod.get_coefficient(), std::complex<double>(2.0, 0.0));
  EXPECT_EQ(cloned_prod.get_factors().size(), 2);
  EXPECT_NE(prod_clone.get(), prod.get());
}

TEST(PauliOperatorExpressionTest, ProductPauliOperatorMultiplyCoefficient) {
  ProductPauliOperatorExpression prod(std::complex<double>(2.0, 0.0));
  EXPECT_EQ(prod.get_coefficient(), std::complex<double>(2.0, 0.0));

  prod.multiply_coefficient(std::complex<double>(0.5, -1.0));
  EXPECT_EQ(prod.get_coefficient(), std::complex<double>(1.0, -2.0));
}

// SumPauliOperatorExpression Tests

TEST(PauliOperatorExpressionTest, SumPauliOperatorConstruction) {
  // OP = 0
  SumPauliOperatorExpression default_sum;
  EXPECT_TRUE(default_sum.get_terms().empty());
  EXPECT_TRUE(default_sum.is_sum_expression());
  EXPECT_FALSE(default_sum.is_pauli_operator());
  EXPECT_FALSE(default_sum.is_product_expression());

  // OP = X(0) + Y(1)
  PauliOperator opX = PauliOperator::X(0);
  PauliOperator opY = PauliOperator::Y(1);
  SumPauliOperatorExpression sum(opX, opY);
  EXPECT_EQ(sum.get_terms().size(), 2);
}

TEST(PauliOperatorExpressionTest, SumPauliOperatorToString) {
  // OP = 0
  SumPauliOperatorExpression empty_sum;
  EXPECT_EQ(empty_sum.to_string(), "0");

  // OP = X(0) + Y(1)
  auto sum = std::make_unique<SumPauliOperatorExpression>();
  sum->add_term(std::make_unique<PauliOperator>(PauliOperator::X(0)));
  sum->add_term(std::make_unique<PauliOperator>(PauliOperator::Y(1)));

  EXPECT_EQ(sum->to_string(), "X(0) + Y(1)");

  // Sum with scaled product
  // OP = X(0) + Y(1) + 2 * Z(2) * Y(0) * X(1)
  auto prod = std::make_unique<ProductPauliOperatorExpression>(
      std::complex<double>(2.0, 0.0), PauliOperator::Z(2));
  prod->add_factor(std::make_unique<ProductPauliOperatorExpression>(
      PauliOperator::Y(0), PauliOperator::X(1)));

  sum->add_term(std::move(prod));

  EXPECT_EQ(sum->to_string(), "X(0) + Y(1) + 2 * Z(2) * Y(0) * X(1)");

  // Copy constructor test
  SumPauliOperatorExpression sum_copy(*sum);
  EXPECT_EQ(sum_copy.to_string(), "X(0) + Y(1) + 2 * Z(2) * Y(0) * X(1)");
}

TEST(PauliOperatorExpressionTest, SumPauliOperatorClone) {
  auto sum = std::make_unique<SumPauliOperatorExpression>();
  sum->add_term(std::make_unique<PauliOperator>(PauliOperator::X(0)));
  sum->add_term(std::make_unique<PauliOperator>(PauliOperator::Y(1)));

  auto sum_clone = sum->clone();
  auto& cloned_sum = dynamic_cast<SumPauliOperatorExpression&>(*sum_clone);
  EXPECT_EQ(cloned_sum.get_terms().size(), 2);
  for (auto i = 0; i < 2; ++i) {
    auto& original_term = *sum->get_terms()[i];
    auto& cloned_term = *cloned_sum.get_terms()[i];
    auto& original_pauli = dynamic_cast<PauliOperator&>(original_term);
    auto& cloned_pauli = dynamic_cast<PauliOperator&>(cloned_term);
    EXPECT_EQ(cloned_pauli.get_operator_type(),
              original_pauli.get_operator_type());
    EXPECT_EQ(cloned_pauli.get_qubit_index(), original_pauli.get_qubit_index());
  }
  EXPECT_NE(sum_clone.get(), sum.get());
}

// Distribute Tests

TEST(PauliOperatorExpressionTest, PauliOperatorDistribute) {
  // Simple Pauli operator, distribution is trivial
  // OP = Y(1)
  PauliOperator opY = PauliOperator::Y(1);
  auto sum_expr = opY.distribute();
  EXPECT_EQ(sum_expr->to_string(), "Y(1)");
}

TEST(PauliOperatorExpressionTest, ProductPauliOperatorDistribute) {
  // Simple, single factor product, distribution is trivial
  // OP = 2 * X(0)
  auto prod = std::make_unique<ProductPauliOperatorExpression>(
      std::complex<double>(2.0, 0.0));
  prod->add_factor(std::make_unique<PauliOperator>(PauliOperator::X(0)));

  auto sum_expr = prod->distribute();
  EXPECT_EQ(sum_expr->to_string(), "2 * X(0)");

  // Product with sum factor
  // OP = 2 * X(0) * (Y(1) + Z(2))
  auto sum = std::make_unique<SumPauliOperatorExpression>(PauliOperator::Y(1),
                                                          PauliOperator::Z(2));
  prod->add_factor(std::move(sum));
  // Distribute over sum
  sum_expr = prod->distribute();
  EXPECT_EQ(sum_expr->to_string(), "2 * X(0) * Y(1) + 2 * X(0) * Z(2)");

  // Product with multiple sum factors
  // OP = 2 * X(0) * (Y(1) + Z(2)) * (I(3) + X(4))
  auto sum2 = std::make_unique<SumPauliOperatorExpression>(PauliOperator::I(3),
                                                           PauliOperator::X(4));
  prod->add_factor(std::move(sum2));
  // Distribute over sums
  sum_expr = prod->distribute();
  EXPECT_EQ(sum_expr->to_string(),
            "2 * X(0) * Y(1) * I(3) + 2 * X(0) * Y(1) * X(4) + 2 * X(0) * Z(2) "
            "* I(3) + 2 * X(0) * Z(2) * X(4)");
}

TEST(PauliOperatorExpressionTest, SumPauliOperatorDistribute) {
  // Simple sum, distribution is trivial
  // OP = X(0) + Y(1)
  auto sum = std::make_unique<SumPauliOperatorExpression>();
  sum->add_term(std::make_unique<PauliOperator>(PauliOperator::X(0)));
  sum->add_term(std::make_unique<PauliOperator>(PauliOperator::Y(1)));

  auto sum_expr = sum->distribute();
  EXPECT_EQ(sum_expr->to_string(), "X(0) + Y(1)");
}

// Math Tests

TEST(PauliOperatorExpressionTest, ScalingOfPauliOperators) {
  PauliOperator opX = PauliOperator::X(0);

  auto prod = std::complex<double>(3.0, 0.0) * opX;
  EXPECT_EQ(prod.to_string(), "3 * X(0)");
  auto prod2 = opX * std::complex<double>(2.0, 0.0);
  EXPECT_EQ(prod2.to_string(), "2 * X(0)");
  auto prod3 = -1 * opX;
  EXPECT_EQ(prod3.to_string(), "-X(0)");
  auto prod4 = opX * -0.5;
  EXPECT_EQ(prod4.to_string(), "-0.5 * X(0)");
}

TEST(PauliOperatorExpressionTest, ScalingOfSumPauliOperatorExpressions) {
  PauliOperator opX = PauliOperator::X(0);
  PauliOperator opY = PauliOperator::Y(1);
  SumPauliOperatorExpression sum(opX, opY);

  auto prod = std::complex<double>(2.0, 0.0) * sum;
  EXPECT_EQ(prod.to_string(), "2 * (X(0) + Y(1))");
  auto prod2 = sum * std::complex<double>(-1.0, 0.0);
  EXPECT_EQ(prod2.to_string(), "-(X(0) + Y(1))");
  auto prod3 = -0.5 * sum;
  EXPECT_EQ(prod3.to_string(), "-0.5 * (X(0) + Y(1))");
  auto prod4 = sum * 3.0;
  EXPECT_EQ(prod4.to_string(), "3 * (X(0) + Y(1))");
}

TEST(PauliOperatorExpressionTest, ScalingOfProductPauliOperatorExpressions) {
  PauliOperator opX = PauliOperator::X(0);
  PauliOperator opY = PauliOperator::Y(1);
  ProductPauliOperatorExpression prod(opX, opY);

  auto prod1 = std::complex<double>(2.0, 0.0) * prod;
  EXPECT_EQ(prod1.to_string(), "2 * X(0) * Y(1)");
  auto prod2 = prod * std::complex<double>(-1.0, 0.0);
  EXPECT_EQ(prod2.to_string(), "-X(0) * Y(1)");
  auto prod3 = -0.5 * prod;
  EXPECT_EQ(prod3.to_string(), "-0.5 * X(0) * Y(1)");
  auto prod4 = prod * 3.0;
  EXPECT_EQ(prod4.to_string(), "3 * X(0) * Y(1)");
}

TEST(PauliOperatorExpressionTest,
     ScalarMultiplicationOnProductUpdatesCoefficient) {
  PauliOperator opX = PauliOperator::X(0);
  PauliOperator opZ = PauliOperator::Z(2);

  auto prod = 2.0 * opX;
  EXPECT_EQ(prod.get_coefficient(), std::complex<double>(2.0, 0.0));
  EXPECT_EQ(prod.to_string(), "2 * X(0)");

  auto scaled = 0.5 * prod;
  EXPECT_EQ(scaled.get_coefficient(), std::complex<double>(1.0, 0.0));
  EXPECT_EQ(scaled.to_string(), "X(0)");
  EXPECT_EQ(scaled.get_factors().size(), prod.get_factors().size());

  auto flat_prod = (2.0 * opX) * opZ;
  EXPECT_EQ(flat_prod.get_coefficient(), std::complex<double>(2.0, 0.0));
  EXPECT_EQ(flat_prod.get_factors().size(), 2);
  EXPECT_EQ(flat_prod.to_string(), "2 * X(0) * Z(2)");

  auto scaled_flat = 0.5 * flat_prod;
  EXPECT_EQ(scaled_flat.get_coefficient(), std::complex<double>(1.0, 0.0));
  EXPECT_EQ(scaled_flat.get_factors().size(), 2);
  EXPECT_EQ(scaled_flat.to_string(), "X(0) * Z(2)");
}

TEST(PauliOperatorExpressionTest, AddPauliOperators) {
  PauliOperator opX = PauliOperator::X(0);
  PauliOperator opY = PauliOperator::Y(1);

  SumPauliOperatorExpression sum = opX + opY;
  EXPECT_EQ(sum.to_string(), "X(0) + Y(1)");
}

TEST(PauliOperatorExpressionTest, AddPauliOperatorAndProduct) {
  PauliOperator opX = PauliOperator::X(0);
  PauliOperator opY = PauliOperator::Y(1);
  ProductPauliOperatorExpression prod(std::complex<double>(2.0, 0.0), opY);
  prod.add_factor(std::make_unique<PauliOperator>(PauliOperator::Y(3)));

  SumPauliOperatorExpression sum1 = opX + prod;
  EXPECT_EQ(sum1.to_string(), "X(0) + 2 * Y(1) * Y(3)");

  SumPauliOperatorExpression sum2 = prod + opX;
  EXPECT_EQ(sum2.to_string(), "2 * Y(1) * Y(3) + X(0)");
}

TEST(PauliOperatorExpressionTest, AddPauliOperatorAndSum) {
  PauliOperator opX = PauliOperator::X(0);
  PauliOperator opY = PauliOperator::Y(1);
  SumPauliOperatorExpression sum(opY, PauliOperator::Z(2));

  SumPauliOperatorExpression sum1 = opX + sum;
  EXPECT_EQ(sum1.to_string(), "X(0) + Y(1) + Z(2)");

  SumPauliOperatorExpression sum2 = sum + opX;
  EXPECT_EQ(sum2.to_string(), "Y(1) + Z(2) + X(0)");
}

TEST(PauliOperatorExpressionTest, AddSumPauliOperatorExpressions) {
  PauliOperator opX = PauliOperator::X(0);
  PauliOperator opY = PauliOperator::Y(1);
  SumPauliOperatorExpression sum1(opX, opY);

  PauliOperator opZ = PauliOperator::Z(2);
  SumPauliOperatorExpression sum2(opZ, PauliOperator::I(3));

  SumPauliOperatorExpression total_sum = sum1 + sum2;
  EXPECT_EQ(total_sum.to_string(), "X(0) + Y(1) + Z(2) + I(3)");
}

TEST(PauliOperatorExpressionTest, AddSumAndProductPauliOperatorExpressions) {
  PauliOperator opX = PauliOperator::X(0);
  PauliOperator opY = PauliOperator::Y(1);
  SumPauliOperatorExpression sum(opX, opY);

  PauliOperator opZ = PauliOperator::Z(2);
  ProductPauliOperatorExpression prod(std::complex<double>(3.0, 0.0), opZ);
  prod.add_factor(std::make_unique<PauliOperator>(PauliOperator::I(3)));

  SumPauliOperatorExpression total_sum = sum + prod;
  EXPECT_EQ(total_sum.to_string(), "X(0) + Y(1) + 3 * Z(2) * I(3)");
}

TEST(PauliOperatorExpressionTest, AddProductPauliOperatorExpressions) {
  PauliOperator opX = PauliOperator::X(0);
  PauliOperator opY = PauliOperator::Y(1);
  ProductPauliOperatorExpression prod1(std::complex<double>(2.0, 0.0), opX);
  prod1.add_factor(std::make_unique<PauliOperator>(PauliOperator::Z(2)));

  PauliOperator opZ = PauliOperator::Z(2);
  ProductPauliOperatorExpression prod2(std::complex<double>(-1.0, 0.0), opY);
  prod2.add_factor(std::make_unique<PauliOperator>(PauliOperator::I(3)));

  SumPauliOperatorExpression total_sum = prod1 + prod2;
  EXPECT_EQ(total_sum.to_string(), "2 * X(0) * Z(2) - Y(1) * I(3)");
}

TEST(PauliOperatorExpressionTest, MultiplyPauliOperators) {
  PauliOperator opX = PauliOperator::X(0);
  PauliOperator opY = PauliOperator::Y(1);

  ProductPauliOperatorExpression prod = 2 * opX * opY;
  EXPECT_EQ(prod.to_string(), "2 * X(0) * Y(1)");
}

TEST(PauliOperatorExpressionTest, MultiplyPauliOperatorAndProduct) {
  PauliOperator opX = PauliOperator::X(0);
  PauliOperator opY = PauliOperator::Y(1);
  ProductPauliOperatorExpression prod(std::complex<double>(3.0, 0.0), opY);
  prod.add_factor(std::make_unique<PauliOperator>(PauliOperator::Z(2)));

  ProductPauliOperatorExpression prod_result = 2 * opX * prod;
  EXPECT_EQ(prod_result.get_coefficient(), std::complex<double>(6.0, 0.0));
  EXPECT_EQ(prod_result.get_factors().size(), 3);
  EXPECT_EQ(prod_result.to_string(), "6 * X(0) * Y(1) * Z(2)");
}

TEST(PauliOperatorExpressionTest, MultiplyPauliOperatorAndSum) {
  PauliOperator opX = PauliOperator::X(0);
  PauliOperator opY = PauliOperator::Y(1);
  SumPauliOperatorExpression sum(opY, PauliOperator::Z(2));

  auto prod_result = 2 * opX * sum;
  EXPECT_EQ(prod_result.to_string(), "2 * X(0) * (Y(1) + Z(2))");

  auto prod_result2 = sum * opX * -1;
  EXPECT_EQ(prod_result2.to_string(), "-(Y(1) + Z(2)) * X(0)");
}

TEST(PauliOperatorExpressionTest, MultiplySumPauliOperatorExpressions) {
  PauliOperator opX = PauliOperator::X(0);
  PauliOperator opY = PauliOperator::Y(1);
  SumPauliOperatorExpression sum1(opX, opY);

  PauliOperator opZ = PauliOperator::Z(2);
  SumPauliOperatorExpression sum2(opZ, PauliOperator::I(3));

  auto prod_result = 3 * sum1 * sum2;
  EXPECT_EQ(prod_result.to_string(), "3 * (X(0) + Y(1)) * (Z(2) + I(3))");
}

TEST(PauliOperatorExpressionTest,
     MultiplySumAndProductPauliOperatorExpressions) {
  PauliOperator opX = PauliOperator::X(0);
  PauliOperator opY = PauliOperator::Y(1);
  SumPauliOperatorExpression sum(opX, opY);

  PauliOperator opZ = PauliOperator::Z(2);
  ProductPauliOperatorExpression prod(std::complex<double>(2.0, 0.0), opZ);
  prod.add_factor(std::make_unique<PauliOperator>(PauliOperator::I(3)));

  auto prod_result = -1 * sum * prod;
  EXPECT_EQ(prod_result.to_string(), "-2 * (X(0) + Y(1)) * Z(2) * I(3)");
}

TEST(PauliOperatorExpressionTest, MultiplyProductPauliOperatorExpressions) {
  PauliOperator opX = PauliOperator::X(0);
  PauliOperator opY = PauliOperator::Y(1);
  ProductPauliOperatorExpression prod1(std::complex<double>(2.0, 0.0), opX);
  prod1.add_factor(std::make_unique<PauliOperator>(PauliOperator::Z(2)));

  PauliOperator opZ = PauliOperator::Z(2);
  ProductPauliOperatorExpression prod2(std::complex<double>(-1.0, 0.0), opY);
  prod2.add_factor(std::make_unique<PauliOperator>(PauliOperator::I(3)));

  auto prod_result = 0.5 * prod1 * prod2;
  EXPECT_EQ(prod_result.get_coefficient(), std::complex<double>(-1.0, 0.0));
  EXPECT_EQ(prod_result.get_factors().size(), 4);
  EXPECT_EQ(prod_result.to_string(), "-X(0) * Z(2) * Y(1) * I(3)");
}

// Simplify Tests

TEST(PauliOperatorExpressionTest, PauliOperatorSimplify) {
  // Simple Pauli operator, simplification is trivial
  // OP = Z(1)
  PauliOperator opZ = PauliOperator::Z(1);
  auto simplified_expr = opZ.simplify();
  EXPECT_EQ(simplified_expr->to_string(), "Z(1)");
}

TEST(PauliOperatorExpressionTest, ProductPauliOperatorSimplify) {
  // Simple, single factor product, simplification is trivial
  // OP = 3 * Y(0)
  auto prod = 3 * PauliOperator::Y(0);

  auto simplified_expr = prod.simplify();
  EXPECT_EQ(simplified_expr->to_string(), "3 * Y(0)");

  // Product with multiple factors that need to be reordered
  // OP = 4 * X(0) * Z(2) * Y(1)
  auto prod2 =
      4 * PauliOperator::X(0) * PauliOperator::Z(2) * PauliOperator::Y(1);

  simplified_expr = prod2.simplify();
  EXPECT_EQ(simplified_expr->to_string(), "4 * X(0) * Y(1) * Z(2)");
  EXPECT_EQ(dynamic_cast<ProductPauliOperatorExpression&>(*simplified_expr)
                .get_factors()
                .size(),
            3);

  // Product with sum factor
  // OP = 2 * Y(0) * (X(1) + I(2)) -> 2*Y(0)*X(1) + 2*Y(0)
  // After simplify: 2*Y(0)*X(1) + 2*Y(0) (I(2) stripped)
  auto prod3 =
      2 * PauliOperator::Y(0) * (PauliOperator::X(1) + PauliOperator::I(2));
  simplified_expr = prod3.simplify();
  EXPECT_EQ(simplified_expr->to_string(), "2 * Y(0) * X(1) + 2 * Y(0)");

  // Product of multiple products with different coefficients that need to be
  // combined
  // OP = (2 * X(0) * Y(1)) * (3 * Z(2) * (I(3) + Y(5)))
  // After distribute and simplify: 6*X(0)*Y(1)*Z(2) + 6*X(0)*Y(1)*Z(2)*Y(5)
  auto prod4 =
      (2 * PauliOperator::X(0) * PauliOperator::Y(1)) *
      (3 * PauliOperator::Z(2) * (PauliOperator::I(3) + PauliOperator::Y(5)));
  simplified_expr = prod4.simplify();
  EXPECT_EQ(simplified_expr->to_string(),
            "6 * X(0) * Y(1) * Z(2) + 6 * X(0) * Y(1) * Z(2) * Y(5)");

  // Products with operators on the same qubit that need to be combined
  // P * P = I for any Pauli P
  // OP = X(0) * X(0) -> 1 (pure scalar, identity stripped)
  auto prod5 = PauliOperator::X(0) * PauliOperator::X(0);
  simplified_expr = prod5.simplify();
  EXPECT_EQ(simplified_expr->to_string(), "1");

  // OP = 2 * Y(1) * Y(1) -> 2 (pure scalar, identity stripped)
  auto prod6 = 2 * PauliOperator::Y(1) * PauliOperator::Y(1);
  simplified_expr = prod6.simplify();
  EXPECT_EQ(simplified_expr->to_string(), "2");

  // X * Y = iZ
  // OP = X(0) * Y(0) -> i * Z(0)
  auto prod7 = PauliOperator::X(0) * PauliOperator::Y(0);
  simplified_expr = prod7.simplify();
  EXPECT_EQ(simplified_expr->to_string(), "i * Z(0)");

  // Y * X = -iZ
  // OP = Y(0) * X(0) -> -i * Z(0)
  auto prod8 = PauliOperator::Y(0) * PauliOperator::X(0);
  simplified_expr = prod8.simplify();
  EXPECT_EQ(simplified_expr->to_string(), "-i * Z(0)");

  // Y * Z = iX
  // OP = 3 * Y(2) * Z(2) -> 3i * X(2)
  auto prod9 = 3 * PauliOperator::Y(2) * PauliOperator::Z(2);
  simplified_expr = prod9.simplify();
  EXPECT_EQ(simplified_expr->to_string(), "3i * X(2)");

  // Z * X = iY
  // OP = Z(0) * X(0) -> i * Y(0)
  auto prod10 = PauliOperator::Z(0) * PauliOperator::X(0);
  simplified_expr = prod10.simplify();
  EXPECT_EQ(simplified_expr->to_string(), "i * Y(0)");

  // Multiple operators on the same qubit with reordering
  // OP = X(0) * Z(1) * Y(0) -> i * Z(0) * Z(1)  (X * Y = iZ)
  auto prod11 = PauliOperator::X(0) * PauliOperator::Z(1) * PauliOperator::Y(0);
  simplified_expr = prod11.simplify();
  EXPECT_EQ(simplified_expr->to_string(), "i * Z(0) * Z(1)");

  // Three operators on same qubit: X * Y * Z = iZ * Z = i * I = i
  // OP = X(0) * Y(0) * Z(0) -> i (pure scalar, identity stripped)
  auto prod12 = PauliOperator::X(0) * PauliOperator::Y(0) * PauliOperator::Z(0);
  simplified_expr = prod12.simplify();
  EXPECT_EQ(simplified_expr->to_string(), "i");

  // I * P = P
  // OP = I(0) * X(0) -> X(0)
  auto prod13 = PauliOperator::I(0) * PauliOperator::X(0);
  simplified_expr = prod13.simplify();
  EXPECT_EQ(simplified_expr->to_string(), "X(0)");
}

TEST(PauliOperatorExpressionTest, SumPauliOperatorSimplify) {
  // Simple sum, simplification is trivial
  // OP = X(0) + Z(1)
  auto sum = PauliOperator::X(0) + PauliOperator::Z(1);

  auto simplified_expr = sum.simplify();
  EXPECT_EQ(simplified_expr->to_string(), "X(0) + Z(1)");

  // Sum with multiple terms (already distributed)
  // OP = Y(0) + 2 * X(1) * Z(2) + 3 * I(3)
  // Note: I(3) gets stripped, leaving just the scalar 3
  auto sum2 = PauliOperator::Y(0) +
              (2 * PauliOperator::X(1) * PauliOperator::Z(2)) +
              (3 * PauliOperator::I(3));

  simplified_expr = sum2.simplify();
  EXPECT_EQ(simplified_expr->to_string(), "Y(0) + 2 * X(1) * Z(2) + 3");
}

TEST(PauliOperatorExpressionTest, TermCollection) {
  // Test that like terms are combined
  // OP = X(0) + X(0) -> 2 * X(0)
  auto sum1 = PauliOperator::X(0) + PauliOperator::X(0);
  auto simplified_expr = sum1.simplify();
  EXPECT_EQ(simplified_expr->to_string(), "2 * X(0)");

  // OP = X(0) - X(0) -> 0 (cancellation)
  auto sum2 = PauliOperator::X(0) - PauliOperator::X(0);
  simplified_expr = sum2.simplify();
  EXPECT_EQ(simplified_expr->to_string(), "0");

  // OP = 2*X(0)*Y(1) + 3*Y(1)*X(0) -> 5*X(0)*Y(1)
  auto sum3 = (2 * PauliOperator::X(0) * PauliOperator::Y(1)) +
              (3 * PauliOperator::Y(1) * PauliOperator::X(0));
  simplified_expr = sum3.simplify();
  EXPECT_EQ(simplified_expr->to_string(), "5 * X(0) * Y(1)");

  // OP = X(0) + Y(1) + X(0) -> 2*X(0) + Y(1)
  auto sum4 = PauliOperator::X(0) + PauliOperator::Y(1) + PauliOperator::X(0);
  simplified_expr = sum4.simplify();
  EXPECT_EQ(simplified_expr->to_string(), "2 * X(0) + Y(1)");

  // Test with complex coefficients: X(0) + i*X(0) -> (1+i)*X(0)
  auto sum5 = PauliOperator::X(0) +
              (std::complex<double>(0.0, 1.0) * PauliOperator::X(0));
  simplified_expr = sum5.simplify();
  EXPECT_EQ(simplified_expr->to_string(), "(1+1i) * X(0)");
}

TEST(PauliOperatorExpressionTest, SimplifyComputesDistributed) {
  // Test that simplify()  distributes when expression is not distributed
  auto prod = PauliOperator::X(0) * (PauliOperator::Y(1) + PauliOperator::Z(2));
  // The product contains a sum, so it's not distributed
  EXPECT_FALSE(prod.is_distributed());

  // Create a sum containing the undistributed product
  SumPauliOperatorExpression sum;
  sum.add_term(prod.clone());

  // simplify() should distribute the product within the sum
  auto simplified_expr = sum.simplify();
  EXPECT_TRUE(simplified_expr->is_distributed());
  EXPECT_EQ(simplified_expr->to_string(), "X(0) * Y(1) + X(0) * Z(2)");
}

TEST(PauliOperatorExpressionTest, PruneThreshold) {
  // Create a sum with terms of varying coefficient magnitudes
  // OP = 1e-5 * X(0) + 0.5 * Y(1) + 1e-12 * Z(2) + 2.0 * X(3)
  auto sum = (1e-5 * PauliOperator::X(0)) + (0.5 * PauliOperator::Y(1)) +
             (1e-12 * PauliOperator::Z(2)) + (2.0 * PauliOperator::X(3));

  // Threshold at 1e-10: should remove only Z(2)
  auto thresholded1 = sum.prune_threshold(1e-10);
  EXPECT_EQ(thresholded1->get_terms().size(), 3);

  // Threshold at 1e-4: should remove X(0) and Z(2)
  auto thresholded2 = sum.prune_threshold(1e-4);
  EXPECT_EQ(thresholded2->get_terms().size(), 2);

  // Threshold at 1.0: should remove X(0), Y(1), and Z(2), leaving only X(3)
  auto thresholded3 = sum.prune_threshold(1.0);
  EXPECT_EQ(thresholded3->get_terms().size(), 1);
  EXPECT_EQ(thresholded3->to_string(), "2 * X(3)");

  // Threshold at 0: should keep all terms
  auto thresholded4 = sum.prune_threshold(0.0);
  EXPECT_EQ(thresholded4->get_terms().size(), 4);

  // Threshold at very large value: should remove all terms
  auto thresholded5 = sum.prune_threshold(100.0);
  EXPECT_EQ(thresholded5->get_terms().size(), 0);
  EXPECT_EQ(thresholded5->to_string(), "0");

  // Test that prune_threshold is accessible from base class pointer
  PauliOperator pauli = PauliOperator::X(0);
  auto pauli_pruned = pauli.prune_threshold(0.5);
  EXPECT_EQ(pauli_pruned->get_terms().size(), 1);

  auto pauli_pruned2 = pauli.prune_threshold(2.0);
  EXPECT_EQ(pauli_pruned2->get_terms().size(), 0);

  // Test on ProductPauliOperatorExpression
  auto prod = 0.1 * PauliOperator::Y(1);
  auto prod_pruned = prod.prune_threshold(0.05);
  EXPECT_EQ(prod_pruned->get_terms().size(), 1);

  auto prod_pruned2 = prod.prune_threshold(0.5);
  EXPECT_EQ(prod_pruned2->get_terms().size(), 0);
}

TEST(PauliOperatorExpressionTest, UnaryNegation) {
  // Test unary negation of PauliOperator
  auto opX = PauliOperator::X(0);
  auto neg_opX = -opX;
  EXPECT_EQ(neg_opX.to_string(), "-X(0)");
  EXPECT_EQ(neg_opX.get_coefficient(), std::complex<double>(-1.0, 0.0));
  EXPECT_EQ(neg_opX.get_factors().size(), 1);

  // Test unary negation of ProductPauliOperatorExpression
  // Scaling a product should multiply its coefficient directly
  auto prod = std::complex<double>(2.0, 1.0) * PauliOperator::Y(1);
  auto neg_prod = -prod;
  // The coefficient should be negated directly: -(2+i) = (-2-i)
  EXPECT_EQ(neg_prod.get_coefficient(), std::complex<double>(-2.0, -1.0));
  EXPECT_EQ(neg_prod.get_factors().size(), 1);

  // Test unary negation of SumPauliOperatorExpression
  auto sum = PauliOperator::X(0) + PauliOperator::Y(1);
  auto neg_sum = -sum;
  // The negated sum wraps the original sum with coefficient -1
  EXPECT_EQ(neg_sum.get_factors().size(), 1);
  EXPECT_EQ(neg_sum.get_coefficient(), std::complex<double>(-1.0, 0.0));

  // Test double negation should give coefficient 1
  auto double_neg = -(-opX);
  EXPECT_EQ(double_neg.get_coefficient(), std::complex<double>(1.0, 0.0));

  // Test scaling a product by a scalar
  auto scaled = std::complex<double>(3.0, 0.0) * prod;
  // 3 * (2+i)*Y(1) = (6+3i)*Y(1)
  EXPECT_EQ(scaled.get_coefficient(), std::complex<double>(6.0, 3.0));
  EXPECT_EQ(scaled.get_factors().size(), 1);
}

TEST(PauliOperatorExpressionTest, PauliOperatorToChar) {
  EXPECT_EQ(PauliOperator::I(0).to_char(), 'I');
  EXPECT_EQ(PauliOperator::X(1).to_char(), 'X');
  EXPECT_EQ(PauliOperator::Y(2).to_char(), 'Y');
  EXPECT_EQ(PauliOperator::Z(3).to_char(), 'Z');
}

TEST(PauliOperatorExpressionTest, ProductCanonicalString) {
  // X(0)*Z(2) on 4 qubits should be "XIZI"
  auto prod = PauliOperator::X(0) * PauliOperator::Z(2);
  auto simplified = prod.simplify();
  auto* prod_ptr = simplified->as_product_expression();
  ASSERT_NE(prod_ptr, nullptr);
  EXPECT_EQ(prod_ptr->to_canonical_string(4), "XIZI");

  // Y(1) on 3 qubits should be "IYI"
  auto prod2 = 1.0 * PauliOperator::Y(1);
  EXPECT_EQ(prod2.to_canonical_string(3), "IYI");

  // Empty product (scalar) on 2 qubits should be "II"
  ProductPauliOperatorExpression scalar_prod(std::complex<double>(2.0, 0.0));
  EXPECT_EQ(scalar_prod.to_canonical_string(2), "II");

  // Test with min_qubit/max_qubit range
  // X(2)*Y(4) with range [2,5] should be "XIYI"
  auto prod3 = PauliOperator::X(2) * PauliOperator::Y(4);
  auto simplified3 = prod3.simplify();
  auto* prod3_ptr = simplified3->as_product_expression();
  EXPECT_EQ(prod3_ptr->to_canonical_string(2, 5), "XIYI");
}

TEST(PauliOperatorExpressionTest, ProductQubitRange) {
  auto prod = PauliOperator::X(2) * PauliOperator::Z(5);
  auto simplified = prod.simplify();
  auto* prod_ptr = simplified->as_product_expression();

  EXPECT_EQ(prod_ptr->min_qubit_index(), 2);
  EXPECT_EQ(prod_ptr->max_qubit_index(), 5);
  EXPECT_EQ(prod_ptr->num_qubits(), 4);  // 5 - 2 + 1 = 4

  // Empty product should throw
  ProductPauliOperatorExpression empty_prod;
  EXPECT_THROW(empty_prod.min_qubit_index(), std::logic_error);
  EXPECT_THROW(empty_prod.max_qubit_index(), std::logic_error);
  EXPECT_EQ(empty_prod.num_qubits(), 0);
}

TEST(PauliOperatorExpressionTest, SumCanonicalString) {
  // X(0) + Z(1) on 2 qubits
  auto sum = PauliOperator::X(0) + PauliOperator::Z(1);
  auto simplified = sum.simplify();
  auto* sum_ptr = simplified->as_sum_expression();
  ASSERT_NE(sum_ptr, nullptr);

  // Not a single product term, should throw
  EXPECT_THROW(sum_ptr->to_canonical_string(2), std::logic_error);

  // Single term after simplification: X(0) + X(0) -> 2*X(0)
  auto sum3 = PauliOperator::X(0) + PauliOperator::X(0);
  auto simplified3 = sum3.simplify();
  auto* sum3_ptr = simplified3->as_sum_expression();
  ASSERT_NE(sum3_ptr, nullptr);
  EXPECT_EQ(sum3_ptr->to_canonical_string(3), "XII");
}

TEST(PauliOperatorExpressionTest, SumQubitRange) {
  auto sum = PauliOperator::X(1) + PauliOperator::Z(4);
  auto simplified = sum.simplify();
  auto* sum_ptr = simplified->as_sum_expression();

  EXPECT_EQ(sum_ptr->min_qubit_index(), 1);
  EXPECT_EQ(sum_ptr->max_qubit_index(), 4);
  EXPECT_EQ(sum_ptr->num_qubits(), 4);  // 4 - 1 + 1 = 4

  // Empty sum should throw
  SumPauliOperatorExpression empty_sum;
  EXPECT_THROW(empty_sum.min_qubit_index(), std::logic_error);
  EXPECT_THROW(empty_sum.max_qubit_index(), std::logic_error);
  EXPECT_EQ(empty_sum.num_qubits(), 0);
}

TEST(PauliOperatorExpressionTest, CanonicalTerms) {
  // 2*X(0) + 3*Y(1)
  auto sum = (2.0 * PauliOperator::X(0)) + (3.0 * PauliOperator::Y(1));
  auto simplified = sum.simplify();
  auto* sum_ptr = simplified->as_sum_expression();

  auto terms = sum_ptr->to_canonical_terms(2);
  EXPECT_EQ(terms.size(), 2);

  // Check that we have the expected terms (order may vary based on input)
  bool found_X = false, found_Y = false;
  for (const auto& [coeff, str] : terms) {
    if (str == "XI") {
      EXPECT_EQ(coeff, std::complex<double>(2.0, 0.0));
      found_X = true;
    } else if (str == "IY") {
      EXPECT_EQ(coeff, std::complex<double>(3.0, 0.0));
      found_Y = true;
    }
  }
  EXPECT_TRUE(found_X);
  EXPECT_TRUE(found_Y);
}

// ============================================================================
// Edge Case Tests for PauliOperator
// ============================================================================

TEST(PauliOperatorExpressionTest, PauliOperatorCanonicalString) {
  // Single operator on single qubit
  auto opX0 = PauliOperator::X(0);
  EXPECT_EQ(opX0.to_canonical_string(1), "X");
  EXPECT_EQ(opX0.to_canonical_string(3), "XII");

  // Operator on qubit 0 should be at position 0 (leftmost)
  auto opZ0 = PauliOperator::Z(0);
  EXPECT_EQ(opZ0.to_canonical_string(4), "ZIII");

  // Operator on higher qubit
  auto opY2 = PauliOperator::Y(2);
  EXPECT_EQ(opY2.to_canonical_string(4), "IIYI");

  // Identity operator
  auto opI5 = PauliOperator::I(5);
  EXPECT_EQ(opI5.to_canonical_string(6), "IIIIII");

  // Test with range that excludes the operator
  auto opX3 = PauliOperator::X(3);
  EXPECT_EQ(opX3.to_canonical_string(0, 2),
            "III");  // Range [0,2] excludes qubit 3

  // Test with range that includes the operator
  EXPECT_EQ(opX3.to_canonical_string(2, 4),
            "IXI");  // Range [2,4], X at position 1
}

TEST(PauliOperatorExpressionTest, PauliOperatorQubitRangeEdgeCases) {
  // Qubit index 0
  auto opX0 = PauliOperator::X(0);
  EXPECT_EQ(opX0.min_qubit_index(), 0);
  EXPECT_EQ(opX0.max_qubit_index(), 0);
  EXPECT_EQ(opX0.num_qubits(), 1);

  // Large qubit index
  auto opZ100 = PauliOperator::Z(100);
  EXPECT_EQ(opZ100.min_qubit_index(), 100);
  EXPECT_EQ(opZ100.max_qubit_index(), 100);
  EXPECT_EQ(opZ100.num_qubits(), 1);

  // Canonical string for large qubit index with appropriate range
  EXPECT_EQ(opZ100.to_canonical_string(100, 102), "ZII");
}

TEST(PauliOperatorExpressionTest, BaseClassVirtualMethods) {
  // Test that virtual methods work through base class pointer
  std::unique_ptr<PauliOperatorExpression> expr =
      std::make_unique<PauliOperator>(PauliOperator::X(2));

  EXPECT_EQ(expr->min_qubit_index(), 2);
  EXPECT_EQ(expr->max_qubit_index(), 2);
  EXPECT_EQ(expr->num_qubits(), 1);
  EXPECT_EQ(expr->to_canonical_string(4), "IIXI");
  EXPECT_EQ(expr->to_canonical_string(1, 3), "IXI");

  // Test with ProductPauliOperatorExpression through base class
  auto prod = PauliOperator::X(0) * PauliOperator::Z(2);
  std::unique_ptr<PauliOperatorExpression> prod_simplified = prod.simplify();
  EXPECT_EQ(prod_simplified->min_qubit_index(), 0);
  EXPECT_EQ(prod_simplified->max_qubit_index(), 2);
  EXPECT_EQ(prod_simplified->num_qubits(), 3);
  EXPECT_EQ(prod_simplified->to_canonical_string(3), "XIZ");

  // Test with SumPauliOperatorExpression through base class
  auto sum = PauliOperator::X(1) + PauliOperator::Y(3);
  auto sum_dist = sum.distribute();
  std::unique_ptr<PauliOperatorExpression> sum_simplified =
      sum_dist->simplify();
  EXPECT_EQ(sum_simplified->min_qubit_index(), 1);
  EXPECT_EQ(sum_simplified->max_qubit_index(), 3);
  EXPECT_EQ(sum_simplified->num_qubits(), 3);
}

// ============================================================================
// Edge Case Tests for ProductPauliOperatorExpression
// ============================================================================

TEST(PauliOperatorExpressionTest, ProductCanonicalStringEdgeCases) {
  // Single qubit product
  auto prod1 = 1.0 * PauliOperator::X(0);
  EXPECT_EQ(prod1.to_canonical_string(1), "X");

  // Qubit index 0 at edge
  auto prod2 = PauliOperator::X(0) * PauliOperator::Y(0);
  auto s2 = prod2.simplify();
  auto* p2 = s2->as_product_expression();
  // X*Y = iZ on same qubit
  EXPECT_EQ(p2->to_canonical_string(1), "Z");

  // Large gap between qubit indices
  auto prod3 = PauliOperator::X(0) * PauliOperator::Z(10);
  auto s3 = prod3.simplify();
  auto* p3 = s3->as_product_expression();
  EXPECT_EQ(p3->to_canonical_string(11), "XIIIIIIIIIZ");
  EXPECT_EQ(p3->to_canonical_string(0, 10), "XIIIIIIIIIZ");

  // Range that truncates (doesn't include all qubits)
  EXPECT_EQ(p3->to_canonical_string(0, 5), "XIIIII");   // Only first 6 qubits
  EXPECT_EQ(p3->to_canonical_string(5, 10), "IIIIIZ");  // Only last 6 qubits

  // All identity Pauli operators after simplification
  auto prod4 = PauliOperator::X(0) * PauliOperator::X(0);  // X*X = I
  auto s4 = prod4.simplify();
  auto* p4 = s4->as_product_expression();
  EXPECT_EQ(p4->to_canonical_string(2), "II");  // Pure scalar, all identities
}

TEST(PauliOperatorExpressionTest, ProductWithComplexCoefficients) {
  // Complex coefficient shouldn't affect canonical string (only Pauli ops)
  auto prod = std::complex<double>(0.5, 0.5) * PauliOperator::X(0) *
              PauliOperator::Y(1);
  auto s = prod.simplify();
  auto* p = s->as_product_expression();
  EXPECT_EQ(p->to_canonical_string(2), "XY");

  // Canonical string is independent of coefficient
  auto prod2 = std::complex<double>(-1.0, 2.0) * PauliOperator::X(0) *
               PauliOperator::Y(1);
  auto s2 = prod2.simplify();
  auto* p2 = s2->as_product_expression();
  EXPECT_EQ(p2->to_canonical_string(2), "XY");
}

// ============================================================================
// Edge Case Tests for SumPauliOperatorExpression
// ============================================================================

TEST(PauliOperatorExpressionTest, SumQubitRangeEdgeCases) {
  // Sum with all terms on same qubit
  auto sum1 = PauliOperator::X(5) + PauliOperator::Y(5);
  auto s1 = sum1.simplify();
  auto* sum1_ptr = s1->as_sum_expression();
  EXPECT_EQ(sum1_ptr->min_qubit_index(), 5);
  EXPECT_EQ(sum1_ptr->max_qubit_index(), 5);
  EXPECT_EQ(sum1_ptr->num_qubits(), 1);

  // Sum with large qubit range
  auto sum2 = PauliOperator::X(0) + PauliOperator::Z(50);
  auto s2 = sum2.simplify();
  auto* sum2_ptr = s2->as_sum_expression();
  EXPECT_EQ(sum2_ptr->min_qubit_index(), 0);
  EXPECT_EQ(sum2_ptr->max_qubit_index(), 50);
  EXPECT_EQ(sum2_ptr->num_qubits(), 51);
}

TEST(PauliOperatorExpressionTest, CanonicalTermsEdgeCases) {
  // Single term
  auto sum1 = 1.0 * PauliOperator::X(0);
  SumPauliOperatorExpression single_sum;
  single_sum.add_term(sum1.clone());
  auto terms1 = single_sum.to_canonical_terms(2);
  EXPECT_EQ(terms1.size(), 1);
  EXPECT_EQ(terms1[0].second, "XI");

  // Terms with complex coefficients
  auto sum2 = (std::complex<double>(1.0, 2.0) * PauliOperator::X(0)) +
              (std::complex<double>(-1.0, 0.5) * PauliOperator::Y(1));
  auto s2 = sum2.simplify();
  auto* sum2_ptr = s2->as_sum_expression();
  auto terms2 = sum2_ptr->to_canonical_terms(2);
  EXPECT_EQ(terms2.size(), 2);

  // Test auto-detected range version
  auto terms2_auto = sum2_ptr->to_canonical_terms();
  EXPECT_EQ(terms2_auto.size(), 2);
}

TEST(PauliOperatorExpressionTest, EmptyExpressionCanonicalString) {
  // Empty product should give all I's
  ProductPauliOperatorExpression empty_prod;
  EXPECT_EQ(empty_prod.to_canonical_string(3), "III");
  EXPECT_EQ(empty_prod.to_canonical_string(0, 2), "III");

  // Empty sum should return "0"
  SumPauliOperatorExpression empty_sum;
  EXPECT_EQ(empty_sum.to_canonical_string(2), "0");
}

TEST(PauliOperatorExpressionTest, ZeroQubitCanonicalString) {
  // Edge case: 0 qubits (should return empty string)
  auto opX = PauliOperator::X(0);
  // Note: to_canonical_string(0) would mean range [0, -1] which is invalid
  // We test with num_qubits=1 as minimum sensible value
  EXPECT_EQ(opX.to_canonical_string(1), "X");

  ProductPauliOperatorExpression empty_prod;
  EXPECT_EQ(empty_prod.to_canonical_string(1), "I");
}

// ============================================================================
// Tests for to_canonical_terms via base class
// ============================================================================

TEST(PauliOperatorExpressionTest, PauliOperatorToCanonicalTerms) {
  // Test PauliOperator::to_canonical_terms directly
  auto opX = PauliOperator::X(2);
  auto terms = opX.to_canonical_terms(4);
  EXPECT_EQ(terms.size(), 1);
  EXPECT_EQ(terms[0].first, std::complex<double>(1.0, 0.0));
  EXPECT_EQ(terms[0].second, "IIXI");

  // Test auto-detected range (includes from 0 to qubit_index)
  auto terms_auto = opX.to_canonical_terms();
  EXPECT_EQ(terms_auto.size(), 1);
  EXPECT_EQ(terms_auto[0].first, std::complex<double>(1.0, 0.0));
  EXPECT_EQ(terms_auto[0].second, "IIX");  // 3 qubits (0, 1, 2)

  // Test via base class pointer
  std::unique_ptr<PauliOperatorExpression> expr =
      std::make_unique<PauliOperator>(PauliOperator::Y(1));
  auto base_terms = expr->to_canonical_terms(3);
  EXPECT_EQ(base_terms.size(), 1);
  EXPECT_EQ(base_terms[0].first, std::complex<double>(1.0, 0.0));
  EXPECT_EQ(base_terms[0].second, "IYI");
}

TEST(PauliOperatorExpressionTest, ProductToCanonicalTerms) {
  // Test ProductPauliOperatorExpression::to_canonical_terms
  auto prod = 2.5 * PauliOperator::X(0) * PauliOperator::Z(2);
  auto simplified = prod.simplify();
  auto* prod_ptr = simplified->as_product_expression();
  ASSERT_NE(prod_ptr, nullptr);

  auto terms = prod_ptr->to_canonical_terms(4);
  EXPECT_EQ(terms.size(), 1);
  EXPECT_EQ(terms[0].first, std::complex<double>(2.5, 0.0));
  EXPECT_EQ(terms[0].second, "XIZI");

  // Test auto-detected range
  auto terms_auto = prod_ptr->to_canonical_terms();
  EXPECT_EQ(terms_auto.size(), 1);
  EXPECT_EQ(terms_auto[0].first, std::complex<double>(2.5, 0.0));
  EXPECT_EQ(terms_auto[0].second, "XIZ");  // 3 qubits (0, 1, 2)

  // Test with complex coefficient
  auto prod2 = std::complex<double>(1.0, -2.0) * PauliOperator::Y(1);
  auto terms2 = prod2.to_canonical_terms(2);
  EXPECT_EQ(terms2.size(), 1);
  EXPECT_EQ(terms2[0].first, std::complex<double>(1.0, -2.0));
  EXPECT_EQ(terms2[0].second, "IY");

  // Test via base class pointer
  std::unique_ptr<PauliOperatorExpression> base_ptr = prod.simplify();
  auto base_terms = base_ptr->to_canonical_terms(3);
  EXPECT_EQ(base_terms.size(), 1);
  EXPECT_EQ(base_terms[0].second, "XIZ");
}

TEST(PauliOperatorExpressionTest, EmptyProductToCanonicalTerms) {
  // Empty product (pure scalar)
  ProductPauliOperatorExpression scalar(std::complex<double>(3.0, 1.0));
  auto terms = scalar.to_canonical_terms(2);
  EXPECT_EQ(terms.size(), 1);
  EXPECT_EQ(terms[0].first, std::complex<double>(3.0, 1.0));
  EXPECT_EQ(terms[0].second, "II");

  // Auto-detected for empty product returns single identity
  auto terms_auto = scalar.to_canonical_terms();
  EXPECT_EQ(terms_auto.size(), 1);
  EXPECT_EQ(terms_auto[0].first, std::complex<double>(3.0, 1.0));
  EXPECT_EQ(terms_auto[0].second, "I");
}

TEST(PauliOperatorExpressionTest, BaseClassToCanonicalTermsPolymorphism) {
  // Test that to_canonical_terms works polymorphically for all types
  std::vector<std::unique_ptr<PauliOperatorExpression>> expressions;

  // Add PauliOperator
  expressions.push_back(std::make_unique<PauliOperator>(PauliOperator::X(0)));

  // Add ProductPauliOperatorExpression
  auto prod = 2.0 * PauliOperator::Y(1);
  expressions.push_back(std::make_unique<ProductPauliOperatorExpression>(prod));

  // Add SumPauliOperatorExpression
  auto sum = PauliOperator::X(0) + PauliOperator::Z(1);
  auto sum_dist = sum.distribute();
  expressions.push_back(sum_dist->simplify());

  // All should be callable through base class
  for (const auto& expr : expressions) {
    auto terms = expr->to_canonical_terms(2);
    EXPECT_GE(terms.size(), 1);  // Each should have at least one term
    for (const auto& [coeff, str] : terms) {
      EXPECT_EQ(str.size(), 2);  // All should be 2 characters for 2 qubits
    }
  }
}

// ============================================================================
// PauliTermAccumulator Tests
// ============================================================================

TEST(PauliTermAccumulatorTest, BasicAccumulation) {
  PauliTermAccumulator acc;

  // Accumulate identity term
  acc.accumulate({}, std::complex<double>(1.0, 0.0));

  // Accumulate X(0)
  SparsePauliWord x0 = {{0, 1}};  // X on qubit 0
  acc.accumulate(x0, std::complex<double>(0.5, 0.0));

  // Accumulate Z(1)
  SparsePauliWord z1 = {{1, 3}};  // Z on qubit 1
  acc.accumulate(z1, std::complex<double>(0.25, 0.0));

  EXPECT_EQ(acc.size(), 3);  // I, X(0), Z(1)

  // Get terms
  auto terms = acc.get_terms(0.0);
  EXPECT_EQ(terms.size(), 3);
}

TEST(PauliTermAccumulatorTest, CoefficientCombining) {
  PauliTermAccumulator acc;

  SparsePauliWord x0 = {{0, 1}};

  // Add the same term multiple times
  acc.accumulate(x0, std::complex<double>(0.5, 0.0));
  acc.accumulate(x0, std::complex<double>(0.3, 0.0));
  acc.accumulate(x0, std::complex<double>(0.2, 0.0));

  EXPECT_EQ(acc.size(), 1);  // Should combine into single term

  auto terms = acc.get_terms(0.0);
  EXPECT_EQ(terms.size(), 1);
  EXPECT_NEAR(terms[0].first.real(), 1.0, testing::wf_tolerance);
  EXPECT_NEAR(terms[0].first.imag(), 0.0, testing::wf_tolerance);
}

TEST(PauliTermAccumulatorTest, CancellationToZero) {
  PauliTermAccumulator acc;

  SparsePauliWord x0 = {{0, 1}};

  // Add terms that cancel
  acc.accumulate(x0, std::complex<double>(1.0, 0.0));
  acc.accumulate(x0, std::complex<double>(-1.0, 0.0));

  // The term should still exist (with zero coefficient)
  EXPECT_EQ(acc.size(), 1);

  // But should be filtered out when threshold > 0
  auto terms = acc.get_terms(testing::numerical_zero_tolerance);
  EXPECT_EQ(terms.size(), 0);
}

TEST(PauliTermAccumulatorTest, AccumulateProduct) {
  PauliTermAccumulator acc;

  SparsePauliWord x0 = {{0, 1}};  // X(0)
  SparsePauliWord y0 = {{0, 2}};  // Y(0)

  // X(0) * Y(0) = i*Z(0)
  acc.accumulate_product(x0, y0, std::complex<double>(1.0, 0.0));

  EXPECT_EQ(acc.size(), 1);

  auto terms = acc.get_terms(0.0);
  EXPECT_EQ(terms.size(), 1);

  // Check the result is Z(0) with coefficient i
  EXPECT_NEAR(terms[0].first.real(), 0.0, testing::wf_tolerance);
  EXPECT_NEAR(terms[0].first.imag(), 1.0, testing::wf_tolerance);
  EXPECT_EQ(terms[0].second.size(), 1);
  EXPECT_EQ(terms[0].second[0].first, 0);   // qubit 0
  EXPECT_EQ(terms[0].second[0].second, 3);  // Z
}

TEST(PauliTermAccumulatorTest, ProductWithDifferentQubits) {
  PauliTermAccumulator acc;

  SparsePauliWord x0 = {{0, 1}};  // X(0)
  SparsePauliWord y1 = {{1, 2}};  // Y(1)

  // X(0) * Y(1) = X(0)Y(1) with no phase
  acc.accumulate_product(x0, y1, std::complex<double>(2.0, 0.0));

  EXPECT_EQ(acc.size(), 1);

  auto terms = acc.get_terms(0.0);
  EXPECT_EQ(terms.size(), 1);
  EXPECT_NEAR(terms[0].first.real(), 2.0, testing::wf_tolerance);
  EXPECT_EQ(terms[0].second.size(), 2);  // X(0) and Y(1)
}

TEST(PauliTermAccumulatorTest, Multiply) {
  PauliTermAccumulator acc;
  SparsePauliWord x0 = {{0, 1}};  // X(0)
  SparsePauliWord y0 = {{0, 2}};  // Y(0)

  auto [phase, result] = acc.multiply(x0, y0);

  // X * Y = iZ
  EXPECT_NEAR(phase.real(), 0.0, testing::wf_tolerance);
  EXPECT_NEAR(phase.imag(), 1.0, testing::wf_tolerance);
  EXPECT_EQ(result.size(), 1);
  EXPECT_EQ(result[0].first, 0);   // qubit 0
  EXPECT_EQ(result[0].second, 3);  // Z
}

TEST(PauliTermAccumulatorTest, MultiplyPauliAlgebra) {
  PauliTermAccumulator acc;
  // Test all Pauli multiplication rules
  SparsePauliWord x = {{0, 1}};
  SparsePauliWord y = {{0, 2}};
  SparsePauliWord z = {{0, 3}};

  // X * Y = iZ
  auto [xy_phase, xy_result] = acc.multiply(x, y);
  EXPECT_NEAR(xy_phase.imag(), 1.0, testing::wf_tolerance);
  EXPECT_EQ(xy_result[0].second, 3);

  // Y * Z = iX
  auto [yz_phase, yz_result] = acc.multiply(y, z);
  EXPECT_NEAR(yz_phase.imag(), 1.0, testing::wf_tolerance);
  EXPECT_EQ(yz_result[0].second, 1);

  // Z * X = iY
  auto [zx_phase, zx_result] = acc.multiply(z, x);
  EXPECT_NEAR(zx_phase.imag(), 1.0, testing::wf_tolerance);
  EXPECT_EQ(zx_result[0].second, 2);

  // Y * X = -iZ
  auto [yx_phase, yx_result] = acc.multiply(y, x);
  EXPECT_NEAR(yx_phase.imag(), -1.0, testing::wf_tolerance);
  EXPECT_EQ(yx_result[0].second, 3);

  // X * X = I (empty word)
  auto [xx_phase, xx_result] = acc.multiply(x, x);
  EXPECT_NEAR(xx_phase.real(), 1.0, testing::wf_tolerance);
  EXPECT_EQ(xx_result.size(), 0);
}

TEST(PauliTermAccumulatorTest, GetTermsAsStrings) {
  PauliTermAccumulator acc;

  acc.accumulate({}, std::complex<double>(1.0, 0.0));         // I
  acc.accumulate({{0, 1}}, std::complex<double>(0.5, 0.0));   // X(0)
  acc.accumulate({{1, 3}}, std::complex<double>(0.25, 0.0));  // Z(1)

  auto terms = acc.get_terms_as_strings(4, 0.0);
  EXPECT_EQ(terms.size(), 3);

  // Find each term
  std::map<std::string, std::complex<double>> term_map;
  for (const auto& [coeff, str] : terms) {
    term_map[str] = coeff;
  }

  EXPECT_EQ(term_map.count("IIII"), 1);
  EXPECT_EQ(term_map.count("XIII"), 1);
  EXPECT_EQ(term_map.count("IZII"), 1);
  EXPECT_NEAR(term_map["IIII"].real(), 1.0, testing::wf_tolerance);
  EXPECT_NEAR(term_map["XIII"].real(), 0.5, testing::wf_tolerance);
  EXPECT_NEAR(term_map["IZII"].real(), 0.25, testing::wf_tolerance);
}

TEST(PauliTermAccumulatorTest, ThresholdFiltering) {
  PauliTermAccumulator acc;

  acc.accumulate({}, std::complex<double>(1.0, 0.0));
  acc.accumulate({{0, 1}}, std::complex<double>(testing::integral_tolerance,
                                                0.0));  // Below numerical_zero
  acc.accumulate({{1, 3}},
                 std::complex<double>(testing::numerical_zero_tolerance * 10,
                                      0.0));  // Above numerical_zero

  // With no threshold, get all terms
  auto terms_all = acc.get_terms(0.0);
  EXPECT_EQ(terms_all.size(), 3);

  // With numerical_zero_tolerance threshold, filter small terms
  auto terms_filtered = acc.get_terms(testing::numerical_zero_tolerance);
  EXPECT_EQ(terms_filtered.size(), 2);

  // String version should also filter
  auto str_terms_filtered =
      acc.get_terms_as_strings(4, testing::numerical_zero_tolerance);
  EXPECT_EQ(str_terms_filtered.size(), 2);
}

TEST(PauliTermAccumulatorTest, Clear) {
  PauliTermAccumulator acc;

  acc.accumulate({}, std::complex<double>(1.0, 0.0));
  acc.accumulate({{0, 1}}, std::complex<double>(0.5, 0.0));

  EXPECT_EQ(acc.size(), 2);

  acc.clear();
  EXPECT_EQ(acc.size(), 0);

  auto terms = acc.get_terms(0.0);
  EXPECT_EQ(terms.size(), 0);
}

TEST(PauliTermAccumulatorTest, CacheOperations) {
  PauliTermAccumulator acc;

  // Fresh accumulator has empty cache
  EXPECT_EQ(acc.cache_size(), 0);

  // Perform some multiplications
  SparsePauliWord x0 = {{0, 1}};
  SparsePauliWord y0 = {{0, 2}};

  acc.multiply(x0, y0);
  EXPECT_GE(acc.cache_size(), 1);

  // Clear cache
  acc.clear_cache();
  EXPECT_EQ(acc.cache_size(), 0);

  // Verify cache doesn't persist across instances
  acc.multiply(x0, y0);
  EXPECT_GE(acc.cache_size(), 1);

  PauliTermAccumulator acc2;
  EXPECT_EQ(acc2.cache_size(), 0);  // New instance has fresh cache
}

TEST(PauliTermAccumulatorTest, ComplexCoefficients) {
  PauliTermAccumulator acc;

  SparsePauliWord x0 = {{0, 1}};
  acc.accumulate(x0, std::complex<double>(1.0, 2.0));
  acc.accumulate(x0, std::complex<double>(3.0, -1.0));

  auto terms = acc.get_terms(0.0);
  EXPECT_EQ(terms.size(), 1);
  EXPECT_NEAR(terms[0].first.real(), 4.0, testing::wf_tolerance);
  EXPECT_NEAR(terms[0].first.imag(), 1.0, testing::wf_tolerance);
}

// ============================================================================
// Excitation Term Computation Tests
// ============================================================================

TEST(PauliTermAccumulatorTest, ComputeAllJWExcitationTerms) {
  auto terms = PauliTermAccumulator::compute_all_jw_excitation_terms(4);

  // Should have 16 entries (4x4 grid)
  EXPECT_EQ(terms.size(), 16);

  // Check diagonal element (0,0) - number operator: 0.5*I - 0.5*Z(0)
  auto it = terms.find({0, 0});
  ASSERT_NE(it, terms.end());
  EXPECT_EQ(it->second.size(), 2);

  // Check off-diagonal element (0,1) - should have 4 terms
  auto it01 = terms.find({0, 1});
  ASSERT_NE(it01, terms.end());
  EXPECT_EQ(it01->second.size(), 4);  // XX, YY, XY, YX terms
}

TEST(PauliTermAccumulatorTest, JWNumberOperator) {
  auto terms = PauliTermAccumulator::compute_all_jw_excitation_terms(2);

  // E_00 = a†_0 a_0 = 0.5 * I - 0.5 * Z(0)
  auto it = terms.find({0, 0});
  ASSERT_NE(it, terms.end());

  std::map<SparsePauliWord, std::complex<double>> term_map;
  for (const auto& [coeff, word] : it->second) {
    term_map[word] = coeff;
  }

  // Identity term
  EXPECT_EQ(term_map.count({}), 1);
  EXPECT_NEAR(term_map[{}].real(), 0.5, testing::wf_tolerance);

  // Z(0) term
  SparsePauliWord z0 = {{0, 3}};
  EXPECT_EQ(term_map.count(z0), 1);
  EXPECT_NEAR(term_map[z0].real(), -0.5, testing::wf_tolerance);
}

TEST(PauliTermAccumulatorTest, ComputeAllBKExcitationTerms) {
  // Set up BK index sets for 4 qubits
  std::unordered_map<std::uint64_t, std::vector<std::uint64_t>> parity_sets = {
      {0, {}}, {1, {0}}, {2, {}}, {3, {2}}};
  std::unordered_map<std::uint64_t, std::vector<std::uint64_t>> update_sets = {
      {0, {1, 3}}, {1, {3}}, {2, {3}}, {3, {}}};
  std::unordered_map<std::uint64_t, std::vector<std::uint64_t>> remainder_sets =
      {{0, {}}, {1, {0}}, {2, {}}, {3, {2}}};

  auto terms = PauliTermAccumulator::compute_all_bk_excitation_terms(
      4, parity_sets, update_sets, remainder_sets);

  // Should have 16 entries
  EXPECT_EQ(terms.size(), 16);

  // Check that all entries exist
  for (std::uint64_t p = 0; p < 4; ++p) {
    for (std::uint64_t q = 0; q < 4; ++q) {
      EXPECT_NE(terms.find({p, q}), terms.end());
    }
  }
}

TEST(PauliTermAccumulatorTest, ExcitationTermsHaveCorrectCoefficientMagnitude) {
  auto jw_terms = PauliTermAccumulator::compute_all_jw_excitation_terms(4);

  // For diagonal terms: coefficients should be 0.5
  for (std::uint64_t p = 0; p < 4; ++p) {
    auto it = jw_terms.find({p, p});
    ASSERT_NE(it, jw_terms.end());
    for (const auto& [coeff, word] : it->second) {
      EXPECT_NEAR(std::abs(coeff), 0.5, testing::wf_tolerance);
    }
  }

  // For off-diagonal terms: coefficients should be 0.25
  for (std::uint64_t p = 0; p < 4; ++p) {
    for (std::uint64_t q = 0; q < 4; ++q) {
      if (p != q) {
        auto it = jw_terms.find({p, q});
        ASSERT_NE(it, jw_terms.end());
        for (const auto& [coeff, word] : it->second) {
          EXPECT_NEAR(std::abs(coeff), 0.25, testing::wf_tolerance);
        }
      }
    }
  }
}
