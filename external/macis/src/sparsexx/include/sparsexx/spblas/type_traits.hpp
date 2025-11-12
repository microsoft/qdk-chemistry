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

#if SPARSEXX_ENABLE_MKL
#include <sparsexx/wrappers/mkl_sparse_matrix.hpp>
#endif /* SPARSEXX_ENABLE_MKL */

namespace sparsexx::spblas::detail {

template <typename SpMatType, typename ALPHAT, typename BETAT>
struct are_alpha_beta_convertible {
  inline static constexpr bool value =
      std::is_convertible_v<ALPHAT, typename SpMatType::value_type> and
      std::is_convertible_v<BETAT, typename SpMatType::value_type>;
};

template <typename SpMatType, typename ALPHAT, typename BETAT>
inline constexpr bool are_alpha_beta_convertible_v =
    are_alpha_beta_convertible<SpMatType, ALPHAT, BETAT>::value;

#if SPARSEXX_ENABLE_MKL

template <typename SpMatType, typename ALPHAT, typename BETAT>
struct spmbv_uses_mkl {
  inline static constexpr bool value =
      are_alpha_beta_convertible_v<SpMatType, ALPHAT, BETAT> and
      sparsexx::detail::mkl::is_mkl_sparse_matrix_v<SpMatType>;
};

#else

template <typename SpMatType, typename ALPHAT, typename BETAT>
struct spmbv_uses_mkl : public std::false_type {};

#endif /* SPARSEXX_ENABLE_MKL */

template <typename SpMatType, typename ALPHAT, typename BETAT>
inline constexpr bool spmbv_uses_mkl_v =
    spmbv_uses_mkl<SpMatType, ALPHAT, BETAT>::value;

template <typename SpMatType, typename ALPHAT, typename BETAT>
struct spmbv_uses_generic_csr {
  inline static constexpr bool value =
      are_alpha_beta_convertible_v<SpMatType, ALPHAT, BETAT> and
      sparsexx::detail::is_csr_matrix_v<SpMatType> and
      not spmbv_uses_mkl_v<SpMatType, ALPHAT, BETAT>;
};

template <typename SpMatType, typename ALPHAT, typename BETAT>
inline constexpr bool spmbv_uses_generic_csr_v =
    spmbv_uses_generic_csr<SpMatType, ALPHAT, BETAT>::value;

template <typename SpMatType, typename ALPHAT, typename BETAT>
struct spmbv_uses_generic_coo {
  inline static constexpr bool value =
      are_alpha_beta_convertible_v<SpMatType, ALPHAT, BETAT> and
      sparsexx::detail::is_coo_matrix_v<SpMatType> and
      not spmbv_uses_mkl_v<SpMatType, ALPHAT, BETAT>;
};

template <typename SpMatType, typename ALPHAT, typename BETAT>
inline constexpr bool spmbv_uses_generic_coo_v =
    spmbv_uses_generic_coo<SpMatType, ALPHAT, BETAT>::value;

}  // namespace sparsexx::spblas::detail
