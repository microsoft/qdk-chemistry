// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <gtest/gtest.h>

#include <qdk/chemistry/data/pauli_operator.hpp>

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
  EXPECT_EQ(prod_result.to_string(), "2 * X(0) * 3 * Y(1) * Z(2)");
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
  EXPECT_EQ(prod_result.to_string(), "-(X(0) + Y(1)) * 2 * Z(2) * I(3)");
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
  EXPECT_EQ(prod_result.to_string(), "0.5 * 2 * X(0) * Z(2) * -Y(1) * I(3)");
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

TEST(PauliOperatorExpressionTest, SimplifyRequiresDistributed) {
  // Test that simplify() throws when expression is not distributed
  auto prod = PauliOperator::X(0) * (PauliOperator::Y(1) + PauliOperator::Z(2));
  // The product contains a sum, so it's not distributed
  EXPECT_FALSE(prod.is_distributed());

  // Create a sum containing the undistributed product
  SumPauliOperatorExpression sum;
  sum.add_term(prod.clone());

  // simplify() should throw
  EXPECT_THROW(sum.simplify(), std::logic_error);
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
