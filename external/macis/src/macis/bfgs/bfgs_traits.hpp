/*
 * MACIS Copyright (c) 2023, The Regents of the University of California,
 * through Lawrence Berkeley National Laboratory (subject to receipt of
 * any required approvals from the U.S. Dept. of Energy). All rights reserved.
 * Portions Copyright (c) Microsoft Corporation.
 *
 * See LICENSE.txt for details
 */

#pragma once
#include <type_traits>

namespace bfgs::detail {

/**
 * @brief Primary type traits template for extracting types from BFGS function
 * objects
 *
 * This struct provides a standardized interface for extracting argument and
 * return types from function objects used in BFGS optimization. Function
 * objects are expected to define `argument_type` and `return_type` member
 * typedefs.
 *
 * @tparam Functor Function object type that provides vector operations for BFGS
 *
 * @note Functor must define:
 *       - `argument_type`: Type of vectors/arguments operated on
 *       - `return_type`: Type returned by function evaluations
 */
template <typename Functor>
struct bfgs_traits {
  /// Type of arguments/vectors that the functor operates on
  using arg_type = typename Functor::argument_type;

  /// Type returned by functor evaluations (typically scalar for optimization)
  using ret_type = typename Functor::return_type;
};

/**
 * @brief Convenience alias template for extracting argument type from BFGS
 * functors
 *
 * This alias template provides a shorthand for accessing the argument type
 * (typically vector type) that a BFGS functor operates on. It eliminates the
 * need to write the full `typename bfgs_traits<Functor>::arg_type` syntax.
 *
 * @tparam Functor Function object type that provides vector operations
 *
 * @example
 * ```cpp
 * // Instead of: typename bfgs_traits<MyFunctor>::arg_type
 * using vector_type = arg_type_t<MyFunctor>;
 * ```
 */
template <typename Functor>
using arg_type_t = typename bfgs_traits<Functor>::arg_type;

/**
 * @brief Convenience alias template for extracting return type from BFGS
 * functors
 *
 * This alias template provides a shorthand for accessing the return type
 * (typically scalar type) that a BFGS functor returns. It eliminates the
 * need to write the full `typename bfgs_traits<Functor>::ret_type` syntax.
 *
 * @tparam Functor Function object type that provides function evaluations
 *
 * @example
 * ```cpp
 * // Instead of: typename bfgs_traits<MyFunctor>::ret_type
 * using scalar_type = ret_type_t<MyFunctor>;
 * ```
 */
template <typename Functor>
using ret_type_t = typename bfgs_traits<Functor>::ret_type;

}  // namespace bfgs::detail
