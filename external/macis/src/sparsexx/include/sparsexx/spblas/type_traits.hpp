/*
 * MACIS Copyright (c) 2023, The Regents of the University of California,
 * through Lawrence Berkeley National Laboratory (subject to receipt of
 * any required approvals from the U.S. Dept. of Energy). All rights reserved.
 *
 * See LICENSE.txt for details
 */

#pragma once

#include <sparsexx/matrix_types/csr_matrix.hpp>
#include <sparsexx/matrix_types/type_traits.hpp>
#include <sparsexx/sparsexx_config.hpp>

namespace sparsexx::spblas::detail {

/**
 * @brief Type trait to check if ALPHA and BETA scalar types are convertible to
 * matrix value type.
 *
 * This trait verifies that the scaling factors ALPHA and BETA can be safely
 * converted to the sparse matrix's value type, ensuring type compatibility for
 * sparse matrix operations.
 *
 * @tparam SpMatType The sparse matrix type
 * @tparam ALPHAT Type of the ALPHA scaling factor
 * @tparam BETAT Type of the BETA scaling factor
 */
template <typename SpMatType, typename ALPHAT, typename BETAT>
struct are_alpha_beta_convertible {
  inline static constexpr bool value =
      std::is_convertible_v<ALPHAT, typename SpMatType::value_type> and
      std::is_convertible_v<BETAT, typename SpMatType::value_type>;
};

/**
 * @brief Convenience variable template for are_alpha_beta_convertible trait.
 *
 * @tparam SpMatType The sparse matrix type
 * @tparam ALPHAT Type of the ALPHA scaling factor
 * @tparam BETAT Type of the BETA scaling factor
 */
template <typename SpMatType, typename ALPHAT, typename BETAT>
inline constexpr bool are_alpha_beta_convertible_v =
    are_alpha_beta_convertible<SpMatType, ALPHAT, BETAT>::value;

/**
 * @brief Type trait to determine if sparse matrix-vector multiplication should
 * use generic CSR implementation.
 *
 * This trait checks if the sparse matrix is in CSR (Compressed Sparse Row)
 * format, the scalar types are convertible. When
 * these conditions are met, the generic CSR implementation will be selected.
 *
 * @tparam SpMatType The sparse matrix type
 * @tparam ALPHAT Type of the ALPHA scaling factor
 * @tparam BETAT Type of the BETA scaling factor
 */
template <typename SpMatType, typename ALPHAT, typename BETAT>
struct spmbv_uses_generic_csr {
  inline static constexpr bool value =
      are_alpha_beta_convertible_v<SpMatType, ALPHAT, BETAT> and
      sparsexx::detail::is_csr_matrix_v<SpMatType>;
};

/**
 * @brief Convenience variable template for spmbv_uses_generic_csr trait.
 *
 * @tparam SpMatType The sparse matrix type
 * @tparam ALPHAT Type of the ALPHA scaling factor
 * @tparam BETAT Type of the BETA scaling factor
 */
template <typename SpMatType, typename ALPHAT, typename BETAT>
inline constexpr bool spmbv_uses_generic_csr_v =
    spmbv_uses_generic_csr<SpMatType, ALPHAT, BETAT>::value;

/**
 * @brief Type trait to determine if sparse matrix-vector multiplication should
 * use generic COO implementation.
 *
 * This trait checks if the sparse matrix is in COO (Coordinate) format, the
 * scalar types are convertible. When these
 * conditions are met, the generic COO implementation will be selected.
 *
 * @tparam SpMatType The sparse matrix type
 * @tparam ALPHAT Type of the ALPHA scaling factor
 * @tparam BETAT Type of the BETA scaling factor
 */
template <typename SpMatType, typename ALPHAT, typename BETAT>
struct spmbv_uses_generic_coo {
  inline static constexpr bool value =
      are_alpha_beta_convertible_v<SpMatType, ALPHAT, BETAT> and
      sparsexx::detail::is_coo_matrix_v<SpMatType>;
};

/**
 * @brief Convenience variable template for spmbv_uses_generic_coo trait.
 *
 * @tparam SpMatType The sparse matrix type
 * @tparam ALPHAT Type of the ALPHA scaling factor
 * @tparam BETAT Type of the BETA scaling factor
 */
template <typename SpMatType, typename ALPHAT, typename BETAT>
inline constexpr bool spmbv_uses_generic_coo_v =
    spmbv_uses_generic_coo<SpMatType, ALPHAT, BETAT>::value;

}  // namespace sparsexx::spblas::detail
