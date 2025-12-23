"""Tests for PauliOperator and Pauli operator expression math.

These tests mirror the C++ tests in cpp/tests/test_pauli_operator.cpp.
The Python API only directly exposes PauliOperator; Product and Sum expressions
are created implicitly through arithmetic operations.
"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from qdk_chemistry.data import PauliOperator


class TestPauliOperatorConstruction:
    """Test cases for PauliOperator construction."""

    def test_pauli_x_construction(self):
        """Test creating a Pauli X operator."""
        op_x = PauliOperator.X(0)
        assert op_x.is_pauli_operator()
        assert not op_x.is_product_expression()
        assert not op_x.is_sum_expression()
        assert op_x.qubit_index == 0

    def test_pauli_y_construction(self):
        """Test creating a Pauli Y operator."""
        op_y = PauliOperator.Y(1)
        assert op_y.is_pauli_operator()
        assert op_y.qubit_index == 1

    def test_pauli_z_construction(self):
        """Test creating a Pauli Z operator."""
        op_z = PauliOperator.Z(2)
        assert op_z.is_pauli_operator()
        assert op_z.qubit_index == 2

    def test_pauli_i_construction(self):
        """Test creating a Pauli I (identity) operator."""
        op_i = PauliOperator.I(3)
        assert op_i.is_pauli_operator()
        assert op_i.qubit_index == 3


class TestPauliOperatorToString:
    """Test cases for PauliOperator string representation."""

    def test_pauli_x_to_string(self):
        """Test X operator string representation."""
        op_x = PauliOperator.X(0)
        assert str(op_x) == "X(0)"

    def test_pauli_y_to_string(self):
        """Test Y operator string representation."""
        op_y = PauliOperator.Y(1)
        assert str(op_y) == "Y(1)"

    def test_pauli_z_to_string(self):
        """Test Z operator string representation."""
        op_z = PauliOperator.Z(2)
        assert str(op_z) == "Z(2)"

    def test_pauli_i_to_string(self):
        """Test I operator string representation."""
        op_i = PauliOperator.I(3)
        assert str(op_i) == "I(3)"


class TestProductPauliOperatorExpression:
    """Test cases for product expressions created via multiplication."""

    def test_product_from_pauli_operators(self):
        """Test creating a product by multiplying two Pauli operators."""
        op_x = PauliOperator.X(0)
        op_y = PauliOperator.Y(1)
        prod = op_x * op_y
        assert prod.is_product_expression()
        assert not prod.is_pauli_operator()
        assert not prod.is_sum_expression()

    def test_product_to_string(self):
        """Test product expression string representation."""
        op_x = PauliOperator.X(0)
        op_y = PauliOperator.Y(1)
        prod = 2 * op_x * op_y
        assert str(prod) == "2 * X(0) * Y(1)"

    def test_product_with_sum(self):
        """Test product containing a sum."""
        op_x = PauliOperator.X(0)
        op_y = PauliOperator.Y(1)
        op_z = PauliOperator.Z(2)
        op_i = PauliOperator.I(3)
        sum_expr = op_z + op_i
        prod = 2 * op_x * op_y * sum_expr
        assert str(prod) == "2 * X(0) * Y(1) * (Z(2) + I(3))"

    def test_product_coefficient(self):
        """Test accessing the product coefficient."""
        op_x = PauliOperator.X(0)
        prod = 2.5 * op_x
        assert prod.coefficient == 2.5


class TestSumPauliOperatorExpression:
    """Test cases for sum expressions created via addition."""

    def test_sum_from_pauli_operators(self):
        """Test creating a sum by adding two Pauli operators."""
        op_x = PauliOperator.X(0)
        op_y = PauliOperator.Y(1)
        sum_expr = op_x + op_y
        assert sum_expr.is_sum_expression()
        assert not sum_expr.is_pauli_operator()
        assert not sum_expr.is_product_expression()

    def test_sum_to_string(self):
        """Test sum expression string representation."""
        op_x = PauliOperator.X(0)
        op_y = PauliOperator.Y(1)
        sum_expr = op_x + op_y
        assert str(sum_expr) == "X(0) + Y(1)"

    def test_sum_with_scaled_product(self):
        """Test sum containing a scaled product."""
        op_x = PauliOperator.X(0)
        op_y = PauliOperator.Y(1)
        op_z = PauliOperator.Z(2)
        op_y0 = PauliOperator.Y(0)
        op_x1 = PauliOperator.X(1)

        sum_expr = op_x + op_y + 2 * op_z * op_y0 * op_x1
        assert str(sum_expr) == "X(0) + Y(1) + 2 * Z(2) * Y(0) * X(1)"


class TestDistribute:
    """Test cases for the distribute() method."""

    def test_pauli_operator_distribute(self):
        """Simple Pauli operator, distribution is trivial."""
        op_y = PauliOperator.Y(1)
        sum_expr = op_y.distribute()
        assert str(sum_expr) == "Y(1)"

    def test_product_distribute_single_factor(self):
        """Simple, single factor product, distribution is trivial."""
        op_x = PauliOperator.X(0)
        prod = 2 * op_x
        sum_expr = prod.distribute()
        assert str(sum_expr) == "2 * X(0)"

    def test_product_distribute_with_sum(self):
        """Product with sum factor distributes over sum."""
        op_x = PauliOperator.X(0)
        op_y = PauliOperator.Y(1)
        op_z = PauliOperator.Z(2)
        prod = 2 * op_x * (op_y + op_z)
        sum_expr = prod.distribute()
        assert str(sum_expr) == "2 * X(0) * Y(1) + 2 * X(0) * Z(2)"

    def test_product_distribute_with_multiple_sums(self):
        """Product with multiple sum factors distributes over all sums."""
        op_x = PauliOperator.X(0)
        op_y = PauliOperator.Y(1)
        op_z = PauliOperator.Z(2)
        op_i = PauliOperator.I(3)
        op_x4 = PauliOperator.X(4)

        prod = 2 * op_x * (op_y + op_z) * (op_i + op_x4)
        sum_expr = prod.distribute()
        expected = "2 * X(0) * Y(1) * I(3) + 2 * X(0) * Y(1) * X(4) + 2 * X(0) * Z(2) * I(3) + 2 * X(0) * Z(2) * X(4)"
        assert str(sum_expr) == expected

    def test_sum_distribute_trivial(self):
        """Simple sum, distribution is trivial."""
        op_x = PauliOperator.X(0)
        op_y = PauliOperator.Y(1)
        sum_expr = op_x + op_y
        distributed = sum_expr.distribute()
        assert str(distributed) == "X(0) + Y(1)"


class TestScaling:
    """Test cases for scaling Pauli operators with scalars."""

    def test_scaling_pauli_operators(self):
        """Test scaling Pauli operators with various coefficients."""
        op_x = PauliOperator.X(0)

        prod = 3 * op_x
        assert str(prod) == "3 * X(0)"

        prod2 = op_x * 2
        assert str(prod2) == "2 * X(0)"

        prod3 = (-1) * op_x
        assert str(prod3) == "-X(0)"

        prod4 = op_x * (-0.5)
        assert str(prod4) == "-0.5 * X(0)"

    def test_scaling_sum_expressions(self):
        """Test scaling sum expressions."""
        op_x = PauliOperator.X(0)
        op_y = PauliOperator.Y(1)
        sum_expr = op_x + op_y

        prod = 2 * sum_expr
        assert str(prod) == "2 * (X(0) + Y(1))"

        prod2 = sum_expr * (-1)
        assert str(prod2) == "-(X(0) + Y(1))"

        prod3 = (-0.5) * sum_expr
        assert str(prod3) == "-0.5 * (X(0) + Y(1))"

        prod4 = sum_expr * 3
        assert str(prod4) == "3 * (X(0) + Y(1))"

    def test_scaling_product_expressions(self):
        """Test scaling product expressions."""
        op_x = PauliOperator.X(0)
        op_y = PauliOperator.Y(1)
        prod = op_x * op_y

        scaled1 = 2 * prod
        assert str(scaled1) == "2 * X(0) * Y(1)"

        scaled2 = prod * (-1)
        assert str(scaled2) == "-X(0) * Y(1)"

        scaled3 = (-0.5) * prod
        assert str(scaled3) == "-0.5 * X(0) * Y(1)"

        scaled4 = prod * 3
        assert str(scaled4) == "3 * X(0) * Y(1)"


class TestAddition:
    """Test cases for adding Pauli operator expressions."""

    def test_add_pauli_operators(self):
        """Test adding two Pauli operators."""
        op_x = PauliOperator.X(0)
        op_y = PauliOperator.Y(1)
        sum_expr = op_x + op_y
        assert str(sum_expr) == "X(0) + Y(1)"

    def test_add_pauli_operator_and_product(self):
        """Test adding Pauli operator and product."""
        op_x = PauliOperator.X(0)
        op_y = PauliOperator.Y(1)
        op_y3 = PauliOperator.Y(3)
        prod = 2 * op_y * op_y3

        sum1 = op_x + prod
        assert str(sum1) == "X(0) + 2 * Y(1) * Y(3)"

        sum2 = prod + op_x
        assert str(sum2) == "2 * Y(1) * Y(3) + X(0)"

    def test_add_pauli_operator_and_sum(self):
        """Test adding Pauli operator and sum."""
        op_x = PauliOperator.X(0)
        op_y = PauliOperator.Y(1)
        op_z = PauliOperator.Z(2)
        sum_expr = op_y + op_z

        sum1 = op_x + sum_expr
        assert str(sum1) == "X(0) + Y(1) + Z(2)"

        sum2 = sum_expr + op_x
        assert str(sum2) == "Y(1) + Z(2) + X(0)"

    def test_add_sum_expressions(self):
        """Test adding two sum expressions."""
        op_x = PauliOperator.X(0)
        op_y = PauliOperator.Y(1)
        op_z = PauliOperator.Z(2)
        op_i = PauliOperator.I(3)
        sum1 = op_x + op_y
        sum2 = op_z + op_i

        total_sum = sum1 + sum2
        assert str(total_sum) == "X(0) + Y(1) + Z(2) + I(3)"

    def test_add_sum_and_product(self):
        """Test adding sum and product expressions."""
        op_x = PauliOperator.X(0)
        op_y = PauliOperator.Y(1)
        op_z = PauliOperator.Z(2)
        op_i = PauliOperator.I(3)
        sum_expr = op_x + op_y
        prod = 3 * op_z * op_i

        total_sum = sum_expr + prod
        assert str(total_sum) == "X(0) + Y(1) + 3 * Z(2) * I(3)"

    def test_add_product_expressions(self):
        """Test adding two product expressions."""
        op_x = PauliOperator.X(0)
        op_y = PauliOperator.Y(1)
        op_z = PauliOperator.Z(2)
        op_i = PauliOperator.I(3)

        prod1 = 2 * op_x * op_z
        prod2 = (-1) * op_y * op_i

        total_sum = prod1 + prod2
        assert str(total_sum) == "2 * X(0) * Z(2) - Y(1) * I(3)"


class TestMultiplication:
    """Test cases for multiplying Pauli operator expressions."""

    def test_multiply_pauli_operators(self):
        """Test multiplying two Pauli operators."""
        op_x = PauliOperator.X(0)
        op_y = PauliOperator.Y(1)
        prod = 2 * op_x * op_y
        assert str(prod) == "2 * X(0) * Y(1)"

    def test_multiply_pauli_operator_and_product(self):
        """Test multiplying Pauli operator and product."""
        op_x = PauliOperator.X(0)
        op_y = PauliOperator.Y(1)
        op_z = PauliOperator.Z(2)
        prod = 3 * op_y * op_z

        prod_result = 2 * op_x * prod
        assert str(prod_result) == "6 * X(0) * Y(1) * Z(2)"

    def test_multiply_pauli_operator_and_sum(self):
        """Test multiplying Pauli operator and sum."""
        op_x = PauliOperator.X(0)
        op_y = PauliOperator.Y(1)
        op_z = PauliOperator.Z(2)
        sum_expr = op_y + op_z

        prod_result = 2 * op_x * sum_expr
        assert str(prod_result) == "2 * X(0) * (Y(1) + Z(2))"

        prod_result2 = sum_expr * op_x * (-1)
        assert str(prod_result2) == "-(Y(1) + Z(2)) * X(0)"

    def test_multiply_sum_expressions(self):
        """Test multiplying two sum expressions."""
        op_x = PauliOperator.X(0)
        op_y = PauliOperator.Y(1)
        op_z = PauliOperator.Z(2)
        op_i = PauliOperator.I(3)
        sum1 = op_x + op_y
        sum2 = op_z + op_i

        prod_result = 3 * sum1 * sum2
        assert str(prod_result) == "3 * (X(0) + Y(1)) * (Z(2) + I(3))"

    def test_multiply_sum_and_product(self):
        """Test multiplying sum and product expressions."""
        op_x = PauliOperator.X(0)
        op_y = PauliOperator.Y(1)
        op_z = PauliOperator.Z(2)
        op_i = PauliOperator.I(3)
        sum_expr = op_x + op_y
        prod = 2 * op_z * op_i

        prod_result = (-1) * sum_expr * prod
        assert str(prod_result) == "-2 * (X(0) + Y(1)) * Z(2) * I(3)"

    def test_multiply_product_expressions(self):
        """Test multiplying two product expressions."""
        op_x = PauliOperator.X(0)
        op_y = PauliOperator.Y(1)
        op_z = PauliOperator.Z(2)
        op_i = PauliOperator.I(3)

        prod1 = 2 * op_x * op_z
        prod2 = (-1) * op_y * op_i

        prod_result = 0.5 * prod1 * prod2
        assert str(prod_result) == "-X(0) * Z(2) * Y(1) * I(3)"

        simplified = prod_result.simplify()
        assert str(simplified) == "-X(0) * Y(1) * Z(2)"


class TestSimplify:
    """Test cases for the simplify() method."""

    def test_pauli_operator_simplify(self):
        """Simple Pauli operator, simplification is trivial."""
        op_z = PauliOperator.Z(1)
        simplified_expr = op_z.simplify()
        assert str(simplified_expr) == "Z(1)"

    def test_product_simplify_single_factor(self):
        """Simple, single factor product, simplification is trivial."""
        prod = 3 * PauliOperator.Y(0)
        simplified_expr = prod.simplify()
        assert str(simplified_expr) == "3 * Y(0)"

    def test_product_simplify_reorder_factors(self):
        """Product with multiple factors that need to be reordered."""
        prod = 4 * PauliOperator.X(0) * PauliOperator.Z(2) * PauliOperator.Y(1)
        simplified_expr = prod.simplify()
        assert str(simplified_expr) == "4 * X(0) * Y(1) * Z(2)"

    def test_product_simplify_with_sum(self):
        """Product with sum factor distributes during simplification."""
        prod = 2 * PauliOperator.Y(0) * (PauliOperator.X(1) + PauliOperator.I(2))
        simplified_expr = prod.simplify()
        # After simplify: 2*Y(0)*X(1) + 2*Y(0) (I(2) stripped)
        assert str(simplified_expr) == "2 * Y(0) * X(1) + 2 * Y(0)"

    def test_product_simplify_same_qubit_pp_equals_i(self):
        """P * P = I for any Pauli P."""
        # X(0) * X(0) -> 1 (pure scalar, identity stripped)
        prod = PauliOperator.X(0) * PauliOperator.X(0)
        simplified_expr = prod.simplify()
        assert str(simplified_expr) == "1"

    def test_product_simplify_same_qubit_scaled(self):
        """2 * Y(1) * Y(1) -> 2 (pure scalar, identity stripped)."""
        prod = 2 * PauliOperator.Y(1) * PauliOperator.Y(1)
        simplified_expr = prod.simplify()
        assert str(simplified_expr) == "2"

    def test_product_simplify_xy_equals_iz(self):
        """X * Y = iZ."""
        prod = PauliOperator.X(0) * PauliOperator.Y(0)
        simplified_expr = prod.simplify()
        assert str(simplified_expr) == "i * Z(0)"

    def test_product_simplify_yx_equals_minus_iz(self):
        """Y * X = -iZ."""
        prod = PauliOperator.Y(0) * PauliOperator.X(0)
        simplified_expr = prod.simplify()
        assert str(simplified_expr) == "-i * Z(0)"

    def test_product_simplify_yz_equals_ix(self):
        """Y * Z = iX."""
        prod = 3 * PauliOperator.Y(2) * PauliOperator.Z(2)
        simplified_expr = prod.simplify()
        assert str(simplified_expr) == "3i * X(2)"

    def test_product_simplify_zx_equals_iy(self):
        """Z * X = iY."""
        prod = PauliOperator.Z(0) * PauliOperator.X(0)
        simplified_expr = prod.simplify()
        assert str(simplified_expr) == "i * Y(0)"

    def test_product_simplify_multiple_same_qubit(self):
        """Multiple operators on the same qubit with reordering."""
        # X(0) * Z(1) * Y(0) -> i * Z(0) * Z(1) (X * Y = iZ)
        prod = PauliOperator.X(0) * PauliOperator.Z(1) * PauliOperator.Y(0)
        simplified_expr = prod.simplify()
        assert str(simplified_expr) == "i * Z(0) * Z(1)"

    def test_product_simplify_three_same_qubit(self):
        """Three operators on same qubit: X * Y * Z = iZ * Z = i * I = i."""
        prod = PauliOperator.X(0) * PauliOperator.Y(0) * PauliOperator.Z(0)
        simplified_expr = prod.simplify()
        assert str(simplified_expr) == "i"

    def test_product_simplify_i_times_p(self):
        """I * P = P."""
        prod = PauliOperator.I(0) * PauliOperator.X(0)
        simplified_expr = prod.simplify()
        assert str(simplified_expr) == "X(0)"

    def test_sum_simplify_trivial(self):
        """Simple sum, simplification is trivial."""
        sum_expr = PauliOperator.X(0) + PauliOperator.Z(1)
        simplified_expr = sum_expr.simplify()
        assert str(simplified_expr) == "X(0) + Z(1)"

    def test_sum_simplify_with_identity(self):
        """Sum with identity term."""
        # I(3) gets stripped, leaving just the scalar 3
        sum_expr = PauliOperator.Y(0) + 2 * PauliOperator.X(1) * PauliOperator.Z(2) + 3 * PauliOperator.I(3)
        simplified_expr = sum_expr.simplify()
        assert str(simplified_expr) == "Y(0) + 2 * X(1) * Z(2) + 3"


class TestTermCollection:
    """Test cases for combining like terms."""

    def test_collect_same_terms(self):
        """Test that like terms are combined."""
        # X(0) + X(0) -> 2 * X(0)
        sum_expr = PauliOperator.X(0) + PauliOperator.X(0)
        simplified_expr = sum_expr.simplify()
        assert str(simplified_expr) == "2 * X(0)"

    def test_collect_canceling_terms(self):
        """Test that terms cancel when subtracted."""
        # X(0) - X(0) -> 0 (cancellation)
        sum_expr = PauliOperator.X(0) - PauliOperator.X(0)
        simplified_expr = sum_expr.simplify()
        assert str(simplified_expr) == "0"

    def test_collect_reordered_products(self):
        """Test that products with different orderings are combined."""
        # 2*X(0)*Y(1) + 3*Y(1)*X(0) -> 5*X(0)*Y(1)
        sum_expr = (2 * PauliOperator.X(0) * PauliOperator.Y(1)) + (3 * PauliOperator.Y(1) * PauliOperator.X(0))
        simplified_expr = sum_expr.simplify()
        assert str(simplified_expr) == "5 * X(0) * Y(1)"

    def test_collect_multiple_same_term(self):
        """Test collecting multiple occurrences of same term."""
        # X(0) + Y(1) + X(0) -> 2*X(0) + Y(1)
        sum_expr = PauliOperator.X(0) + PauliOperator.Y(1) + PauliOperator.X(0)
        simplified_expr = sum_expr.simplify()
        assert str(simplified_expr) == "2 * X(0) + Y(1)"

    def test_collect_complex_coefficients(self):
        """Test with complex coefficients: X(0) + i*X(0) -> (1+i)*X(0)."""
        sum_expr = PauliOperator.X(0) + (1j * PauliOperator.X(0))
        simplified_expr = sum_expr.simplify()
        assert str(simplified_expr) == "(1+1i) * X(0)"


class TestPruneThreshold:
    """Test cases for the prune_threshold() method."""

    def test_prune_threshold_keeps_large_terms(self):
        """Test that terms above threshold are kept."""
        # Create a sum with terms of varying coefficient magnitudes
        sum_expr = (
            (1e-5 * PauliOperator.X(0))
            + (0.5 * PauliOperator.Y(1))
            + (1e-12 * PauliOperator.Z(2))
            + (2.0 * PauliOperator.X(3))
        )

        # Threshold at 1e-10: should remove only Z(2)
        thresholded = sum_expr.prune_threshold(1e-10)
        assert str(thresholded).count("+") == 2  # 3 terms, 2 plus signs

    def test_prune_threshold_removes_small_terms(self):
        """Test that terms below threshold are removed."""
        sum_expr = (
            (1e-5 * PauliOperator.X(0))
            + (0.5 * PauliOperator.Y(1))
            + (1e-12 * PauliOperator.Z(2))
            + (2.0 * PauliOperator.X(3))
        )

        # Threshold at 1e-4: should remove X(0) and Z(2)
        thresholded = sum_expr.prune_threshold(1e-4)
        assert str(thresholded).count("+") == 1  # 2 terms, 1 plus sign

    def test_prune_threshold_leaves_one_term(self):
        """Test pruning that leaves only one term."""
        sum_expr = (
            (1e-5 * PauliOperator.X(0))
            + (0.5 * PauliOperator.Y(1))
            + (1e-12 * PauliOperator.Z(2))
            + (2.0 * PauliOperator.X(3))
        )

        # Threshold at 1.0: should remove X(0), Y(1), and Z(2), leaving only X(3)
        thresholded = sum_expr.prune_threshold(1.0)
        assert str(thresholded) == "2 * X(3)"

    def test_prune_threshold_zero_keeps_all(self):
        """Test that threshold 0 keeps all terms."""
        sum_expr = (
            (1e-5 * PauliOperator.X(0))
            + (0.5 * PauliOperator.Y(1))
            + (1e-12 * PauliOperator.Z(2))
            + (2.0 * PauliOperator.X(3))
        )

        thresholded = sum_expr.prune_threshold(0.0)
        assert str(thresholded).count("+") == 3  # 4 terms, 3 plus signs

    def test_prune_threshold_large_removes_all(self):
        """Test that very large threshold removes all terms."""
        sum_expr = (
            (1e-5 * PauliOperator.X(0))
            + (0.5 * PauliOperator.Y(1))
            + (1e-12 * PauliOperator.Z(2))
            + (2.0 * PauliOperator.X(3))
        )

        thresholded = sum_expr.prune_threshold(100.0)
        assert str(thresholded) == "0"

    def test_prune_threshold_on_pauli_operator(self):
        """Test prune_threshold on single Pauli operator."""
        pauli = PauliOperator.X(0)
        pruned = pauli.prune_threshold(0.5)
        assert "X(0)" in str(pruned)

        pruned2 = pauli.prune_threshold(2.0)
        assert str(pruned2) == "0"

    def test_prune_threshold_on_product(self):
        """Test prune_threshold on product expression."""
        prod = 0.1 * PauliOperator.Y(1)
        pruned = prod.prune_threshold(0.05)
        assert "Y(1)" in str(pruned)

        pruned2 = prod.prune_threshold(0.5)
        assert str(pruned2) == "0"


class TestUnaryNegation:
    """Test cases for unary negation operator."""

    def test_negate_pauli_operator(self):
        """Test unary negation of PauliOperator."""
        op_x = PauliOperator.X(0)
        neg_op_x = -op_x
        assert str(neg_op_x) == "-X(0)"
        assert neg_op_x.coefficient == -1

    def test_negate_product_expression(self):
        """Test unary negation of ProductPauliOperatorExpression."""
        prod = complex(2, 1) * PauliOperator.Y(1)
        neg_prod = -prod
        # The coefficient should be negated directly: -(2+i) = (-2-i)
        assert neg_prod.coefficient == complex(-2, -1)

    def test_negate_sum_expression(self):
        """Test unary negation of SumPauliOperatorExpression."""
        sum_expr = PauliOperator.X(0) + PauliOperator.Y(1)
        neg_sum = -sum_expr
        # The negated sum wraps the original sum with coefficient -1
        assert neg_sum.coefficient == -1

    def test_double_negation(self):
        """Test double negation should give coefficient 1."""
        op_x = PauliOperator.X(0)
        neg_op_x = -op_x
        double_neg = -neg_op_x
        assert double_neg.coefficient == 1

    def test_scaling_after_negation(self):
        """Test scaling a product by a scalar after negation."""
        prod = complex(2, 1) * PauliOperator.Y(1)
        scaled = 3 * prod
        # 3 * (2+i)*Y(1) = (6+3i)*Y(1)
        assert scaled.coefficient == complex(6, 3)


class TestSubtraction:
    """Test cases for subtraction operations."""

    def test_subtract_pauli_operators(self):
        """Test subtracting two Pauli operators."""
        op_x = PauliOperator.X(0)
        op_y = PauliOperator.Y(1)
        diff = op_x - op_y
        assert str(diff) == "X(0) - Y(1)"

    def test_subtract_product_from_pauli(self):
        """Test subtracting a product from a Pauli operator."""
        op_x = PauliOperator.X(0)
        prod = 2 * PauliOperator.Y(1)
        diff = op_x - prod
        assert str(diff) == "X(0) - 2 * Y(1)"

    def test_subtract_sum_from_pauli(self):
        """Test subtracting a sum from a Pauli operator."""
        op_x = PauliOperator.X(0)
        sum_expr = PauliOperator.Y(1) + PauliOperator.Z(2)
        diff = op_x - sum_expr
        assert str(diff) == "X(0) - (Y(1) + Z(2))"


class TestComplexCoefficients:
    """Test cases for complex number coefficients."""

    def test_complex_scalar_multiplication(self):
        """Test multiplying by complex scalars."""
        op_x = PauliOperator.X(0)

        prod = 1j * op_x
        assert str(prod) == "i * X(0)"

        prod2 = complex(1, 1) * op_x
        assert str(prod2) == "(1+1i) * X(0)"

    def test_complex_coefficient_access(self):
        """Test accessing complex coefficients."""
        op_x = PauliOperator.X(0)
        prod = complex(2, 3) * op_x
        coeff = prod.coefficient
        assert coeff.real == 2
        assert coeff.imag == 3


class TestIsDistributed:
    """Test cases for the is_distributed() method."""

    def test_pauli_operator_is_distributed(self):
        """Single Pauli operator is considered distributed."""
        op_x = PauliOperator.X(0)
        assert op_x.is_distributed()

    def test_simple_product_is_distributed(self):
        """Simple product without nested sums is distributed."""
        prod = PauliOperator.X(0) * PauliOperator.Y(1)
        assert prod.is_distributed()

    def test_product_with_sum_not_distributed(self):
        """Product containing a sum is not distributed."""
        prod = PauliOperator.X(0) * (PauliOperator.Y(1) + PauliOperator.Z(2))
        assert not prod.is_distributed()

    def test_sum_of_products_is_distributed(self):
        """Sum of products (no nested sums in factors) is distributed."""
        sum_expr = (PauliOperator.X(0) * PauliOperator.Y(1)) + (PauliOperator.Z(2) * PauliOperator.I(3))
        assert sum_expr.is_distributed()


class TestPauliOperatorToChar:
    """Test cases for the PauliOperator.to_char() method."""

    def test_identity_to_char(self):
        """Test I operator returns 'I'."""
        assert PauliOperator.I(0).to_char() == "I"

    def test_pauli_x_to_char(self):
        """Test X operator returns 'X'."""
        assert PauliOperator.X(1).to_char() == "X"

    def test_pauli_y_to_char(self):
        """Test Y operator returns 'Y'."""
        assert PauliOperator.Y(2).to_char() == "Y"

    def test_pauli_z_to_char(self):
        """Test Z operator returns 'Z'."""
        assert PauliOperator.Z(3).to_char() == "Z"


class TestPauliOperatorQubitRange:
    """Test cases for qubit range methods on PauliOperator."""

    def test_qubit_index_zero(self):
        """Test qubit range for operator at index 0."""
        op = PauliOperator.X(0)
        assert op.min_qubit_index() == 0
        assert op.max_qubit_index() == 0
        assert op.num_qubits() == 1

    def test_qubit_index_nonzero(self):
        """Test qubit range for operator at higher index."""
        op = PauliOperator.Y(5)
        assert op.min_qubit_index() == 5
        assert op.max_qubit_index() == 5
        assert op.num_qubits() == 1

    def test_large_qubit_index(self):
        """Test qubit range for large qubit index."""
        op = PauliOperator.Z(100)
        assert op.min_qubit_index() == 100
        assert op.max_qubit_index() == 100
        assert op.num_qubits() == 1


class TestProductQubitRange:
    """Test cases for qubit range methods on ProductPauliOperatorExpression."""

    def test_product_qubit_range(self):
        """Test qubit range for product of operators."""
        prod = PauliOperator.X(2) * PauliOperator.Z(5)
        simplified = prod.simplify()
        assert simplified.min_qubit_index() == 2
        assert simplified.max_qubit_index() == 5
        assert simplified.num_qubits() == 4  # 5 - 2 + 1 = 4

    def test_product_single_operator(self):
        """Test qubit range for single-operator product."""
        prod = 2.0 * PauliOperator.Y(3)
        assert prod.min_qubit_index() == 3
        assert prod.max_qubit_index() == 3
        assert prod.num_qubits() == 1

    def test_product_adjacent_qubits(self):
        """Test qubit range for adjacent qubits."""
        prod = PauliOperator.X(0) * PauliOperator.Y(1)
        simplified = prod.simplify()
        assert simplified.min_qubit_index() == 0
        assert simplified.max_qubit_index() == 1
        assert simplified.num_qubits() == 2


class TestSumQubitRange:
    """Test cases for qubit range methods on SumPauliOperatorExpression."""

    def test_sum_qubit_range(self):
        """Test qubit range for sum of operators."""
        sum_expr = PauliOperator.X(1) + PauliOperator.Z(4)
        simplified = sum_expr.distribute().simplify()
        assert simplified.min_qubit_index() == 1
        assert simplified.max_qubit_index() == 4
        assert simplified.num_qubits() == 4  # 4 - 1 + 1 = 4

    def test_sum_same_qubit(self):
        """Test qubit range for sum on same qubit."""
        sum_expr = PauliOperator.X(5) + PauliOperator.Y(5)
        simplified = sum_expr.distribute().simplify()
        assert simplified.min_qubit_index() == 5
        assert simplified.max_qubit_index() == 5
        assert simplified.num_qubits() == 1

    def test_sum_large_range(self):
        """Test qubit range for sum with large qubit spread."""
        sum_expr = PauliOperator.X(0) + PauliOperator.Z(50)
        simplified = sum_expr.distribute().simplify()
        assert simplified.min_qubit_index() == 0
        assert simplified.max_qubit_index() == 50
        assert simplified.num_qubits() == 51


class TestPauliOperatorCanonicalString:
    """Test cases for PauliOperator.to_canonical_string()."""

    def test_single_qubit_canonical_string(self):
        """Test canonical string for single qubit."""
        op = PauliOperator.X(0)
        assert op.to_canonical_string(1) == "X"

    def test_canonical_string_with_padding(self):
        """Test canonical string with identity padding."""
        op = PauliOperator.X(0)
        assert op.to_canonical_string(3) == "XII"

    def test_canonical_string_qubit_zero(self):
        """Test X at qubit 0 is leftmost."""
        op = PauliOperator.Z(0)
        assert op.to_canonical_string(4) == "ZIII"

    def test_canonical_string_higher_qubit(self):
        """Test operator at higher qubit index."""
        op = PauliOperator.Y(2)
        assert op.to_canonical_string(4) == "IIYI"

    def test_canonical_string_identity(self):
        """Test identity operator canonical string."""
        op = PauliOperator.I(5)
        assert op.to_canonical_string(6) == "IIIIII"

    def test_canonical_string_with_range(self):
        """Test canonical string with min/max range."""
        op = PauliOperator.X(3)
        # Range [2,4] includes qubit 3, X at position 1 in that range
        assert op.to_canonical_string(2, 4) == "IXI"

    def test_canonical_string_range_excludes_operator(self):
        """Test canonical string with range that excludes operator."""
        op = PauliOperator.X(3)
        # Range [0,2] excludes qubit 3
        assert op.to_canonical_string(0, 2) == "III"


class TestProductCanonicalString:
    """Test cases for ProductPauliOperatorExpression.to_canonical_string()."""

    def test_product_canonical_string(self):
        """Test canonical string for product."""
        prod = PauliOperator.X(0) * PauliOperator.Z(2)
        simplified = prod.simplify()
        assert simplified.to_canonical_string(4) == "XIZI"

    def test_product_single_operator(self):
        """Test canonical string for single-operator product."""
        prod = 1.0 * PauliOperator.Y(1)
        assert prod.to_canonical_string(3) == "IYI"

    def test_product_single_qubit(self):
        """Test canonical string for single qubit product."""
        prod = 1.0 * PauliOperator.X(0)
        assert prod.to_canonical_string(1) == "X"

    def test_product_same_qubit_multiplication(self):
        """Test X*Y = iZ on same qubit."""
        prod = PauliOperator.X(0) * PauliOperator.Y(0)
        simplified = prod.simplify()
        assert simplified.to_canonical_string(1) == "Z"

    def test_product_large_gap(self):
        """Test canonical string with large gap between qubits."""
        prod = PauliOperator.X(0) * PauliOperator.Z(10)
        simplified = prod.simplify()
        assert simplified.to_canonical_string(11) == "XIIIIIIIIIZ"
        assert simplified.to_canonical_string(0, 10) == "XIIIIIIIIIZ"

    def test_product_truncating_range(self):
        """Test canonical string with truncating range."""
        prod = PauliOperator.X(0) * PauliOperator.Z(10)
        simplified = prod.simplify()
        assert simplified.to_canonical_string(0, 5) == "XIIIII"  # Only first 6 qubits
        assert simplified.to_canonical_string(5, 10) == "IIIIIZ"  # Only last 6 qubits

    def test_product_xx_is_identity(self):
        """Test X*X = I gives all identities."""
        prod = PauliOperator.X(0) * PauliOperator.X(0)
        simplified = prod.simplify()
        assert simplified.to_canonical_string(2) == "II"

    def test_product_complex_coefficient(self):
        """Test canonical string is independent of complex coefficient."""
        prod = complex(0.5, 0.5) * PauliOperator.X(0) * PauliOperator.Y(1)
        simplified = prod.simplify()
        assert simplified.to_canonical_string(2) == "XY"


class TestSumCanonicalString:
    """Test cases for SumPauliOperatorExpression.to_canonical_string()."""

    def test_sum_canonical_string(self):
        """Test canonical string for sum."""
        sum_expr = PauliOperator.X(0) + PauliOperator.Z(1)
        simplified = sum_expr.distribute().simplify()
        canonical = simplified.to_canonical_string(2)
        # Should contain both XI and IZ terms
        assert "XI" in canonical
        assert "IZ" in canonical

    def test_sum_single_term(self):
        """Test canonical string for single-term sum."""
        # Create a proper single-term sum by using addition
        single_sum = 1.0 * PauliOperator.X(0)
        # Wrap in sum via distribute
        distributed = single_sum.distribute()
        assert "XI" in distributed.to_canonical_string(2) or distributed.to_canonical_string(2) == "XI"

    def test_sum_same_qubit_operators(self):
        """Test sum of operators on same qubit."""
        sum_expr = PauliOperator.X(0) + PauliOperator.Y(0) + PauliOperator.Z(0)
        simplified = sum_expr.distribute().simplify()
        canonical = simplified.to_canonical_string(1)
        assert "X" in canonical
        assert "Y" in canonical
        assert "Z" in canonical


class TestPauliOperatorCanonicalTerms:
    """Test cases for PauliOperator.to_canonical_terms()."""

    def test_canonical_terms_basic(self):
        """Test canonical terms for single operator."""
        op = PauliOperator.X(2)
        terms = op.to_canonical_terms(4)
        assert len(terms) == 1
        coeff, pauli_str = terms[0]
        assert coeff == complex(1.0, 0.0)
        assert pauli_str == "IIXI"

    def test_canonical_terms_auto_range(self):
        """Test canonical terms with auto-detected range."""
        op = PauliOperator.X(2)
        terms = op.to_canonical_terms()
        assert len(terms) == 1
        coeff, pauli_str = terms[0]
        assert coeff == complex(1.0, 0.0)
        assert pauli_str == "IIX"  # 3 qubits (0, 1, 2)


class TestProductCanonicalTerms:
    """Test cases for ProductPauliOperatorExpression.to_canonical_terms()."""

    def test_product_canonical_terms(self):
        """Test canonical terms for product."""
        prod = 2.5 * PauliOperator.X(0) * PauliOperator.Z(2)
        simplified = prod.simplify()
        terms = simplified.to_canonical_terms(4)
        assert len(terms) == 1
        coeff, pauli_str = terms[0]
        assert coeff == complex(2.5, 0.0)
        assert pauli_str == "XIZI"

    def test_product_canonical_terms_auto_range(self):
        """Test canonical terms with auto-detected range."""
        prod = 2.5 * PauliOperator.X(0) * PauliOperator.Z(2)
        simplified = prod.simplify()
        terms = simplified.to_canonical_terms()
        assert len(terms) == 1
        coeff, pauli_str = terms[0]
        assert coeff == complex(2.5, 0.0)
        assert pauli_str == "XIZ"  # 3 qubits (0, 1, 2)

    def test_product_complex_coefficient_terms(self):
        """Test canonical terms with complex coefficient."""
        prod = complex(1.0, -2.0) * PauliOperator.Y(1)
        terms = prod.to_canonical_terms(2)
        assert len(terms) == 1
        coeff, pauli_str = terms[0]
        assert coeff == complex(1.0, -2.0)
        assert pauli_str == "IY"


class TestSumCanonicalTerms:
    """Test cases for SumPauliOperatorExpression.to_canonical_terms()."""

    def test_sum_canonical_terms(self):
        """Test canonical terms for sum."""
        sum_expr = (2.0 * PauliOperator.X(0)) + (3.0 * PauliOperator.Y(1))
        simplified = sum_expr.distribute().simplify()
        terms = simplified.to_canonical_terms(2)
        assert len(terms) == 2

        # Check that we have both expected terms
        term_dict = {pauli_str: coeff for coeff, pauli_str in terms}
        assert "XI" in term_dict
        assert "IY" in term_dict
        assert term_dict["XI"] == complex(2.0, 0.0)
        assert term_dict["IY"] == complex(3.0, 0.0)

    def test_sum_canonical_terms_auto_range(self):
        """Test canonical terms with auto-detected range."""
        sum_expr = (2.0 * PauliOperator.X(0)) + (3.0 * PauliOperator.Y(1))
        simplified = sum_expr.distribute().simplify()
        terms = simplified.to_canonical_terms()
        assert len(terms) == 2

        term_dict = {pauli_str: coeff for coeff, pauli_str in terms}
        assert "XI" in term_dict
        assert "IY" in term_dict

    def test_sum_complex_coefficients_terms(self):
        """Test canonical terms with complex coefficients."""
        sum_expr = (complex(1.0, 2.0) * PauliOperator.X(0)) + (complex(-1.0, 0.5) * PauliOperator.Y(1))
        simplified = sum_expr.distribute().simplify()
        terms = simplified.to_canonical_terms(2)
        assert len(terms) == 2


class TestCanonicalEdgeCases:
    """Edge case tests for canonical string and terms."""

    def test_single_qubit_all_operators(self):
        """Test canonical string for all Pauli types on single qubit."""
        assert PauliOperator.I(0).to_canonical_string(1) == "I"
        assert PauliOperator.X(0).to_canonical_string(1) == "X"
        assert PauliOperator.Y(0).to_canonical_string(1) == "Y"
        assert PauliOperator.Z(0).to_canonical_string(1) == "Z"

    def test_large_qubit_index_canonical_string(self):
        """Test canonical string for large qubit index."""
        op = PauliOperator.Z(100)
        # With range [100, 102], Z at position 0
        assert op.to_canonical_string(100, 102) == "ZII"

    def test_product_with_range_matching_operators(self):
        """Test canonical string when range exactly matches operators."""
        prod = PauliOperator.X(0) * PauliOperator.Z(2)
        simplified = prod.simplify()
        # Use num_qubits to get exact range (0 to 2 = 3 qubits)
        assert simplified.to_canonical_string(3) == "XIZ"
        # Or use min/max qubit range
        assert simplified.to_canonical_string(0, 2) == "XIZ"
