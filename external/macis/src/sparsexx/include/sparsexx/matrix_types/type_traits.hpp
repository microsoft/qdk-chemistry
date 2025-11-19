/*
 * MACIS Copyright (c) 2023, The Regents of the University of California,
 * through Lawrence Berkeley National Laboratory (subject to receipt of
 * any required approvals from the U.S. Dept. of Energy). All rights reserved.
 *
 * See LICENSE.txt for details
 */

#pragma once

#include <sparsexx/matrix_types/coo_matrix.hpp>
#include <sparsexx/matrix_types/csr_matrix.hpp>
#include <type_traits>

namespace sparsexx::detail {

/**
 * @brief Type trait to detect CSR matrix types.
 *
 * This struct provides compile-time detection of CSR matrix types using SFINAE.
 * The default case evaluates to false_type.
 *
 * @tparam SpMatType The type to test
 * @tparam void SFINAE parameter for template specialization
 */
template <typename SpMatType, typename = void>
struct is_csr_matrix : public std::false_type {};

/**
 * @brief Type trait to detect CSC matrix types.
 *
 * This struct provides compile-time detection of CSC matrix types using SFINAE.
 * The default case evaluates to false_type.
 *
 * @tparam SpMatType The type to test
 * @tparam void SFINAE parameter for template specialization
 */
template <typename SpMatType, typename = void>
struct is_csc_matrix : public std::false_type {};

/**
 * @brief Type trait to detect COO matrix types.
 *
 * This struct provides compile-time detection of COO matrix types using SFINAE.
 * The default case evaluates to false_type.
 *
 * @tparam SpMatType The type to test
 * @tparam void SFINAE parameter for template specialization
 */
template <typename SpMatType, typename = void>
struct is_coo_matrix : public std::false_type {};

/**
 * @brief Specialization of is_csr_matrix for actual CSR matrix types.
 *
 * This specialization uses std::is_base_of to detect types that inherit from
 * csr_matrix with compatible template parameters. It evaluates to true_type
 * for CSR matrix types.
 *
 * @tparam SpMatType The CSR matrix type being detected
 */
template <typename SpMatType>
struct is_csr_matrix<SpMatType,
                     std::enable_if_t<std::is_base_of_v<
                         csr_matrix<typename SpMatType::value_type,
                                    typename SpMatType::index_type,
                                    typename SpMatType::allocator_type>,
                         SpMatType> > > : public std::true_type {};

/**
 * @brief Specialization of is_csc_matrix for actual CSC matrix types.
 *
 * This specialization uses std::is_base_of to detect types that inherit from
 * csc_matrix with compatible template parameters. It evaluates to true_type
 * for CSC matrix types.
 *
 * @tparam SpMatType The CSC matrix type being detected
 */
template <typename SpMatType>
struct is_csc_matrix<SpMatType,
                     std::enable_if_t<std::is_base_of_v<
                         csc_matrix<typename SpMatType::value_type,
                                    typename SpMatType::index_type,
                                    typename SpMatType::allocator_type>,
                         SpMatType> > > : public std::true_type {};

/**
 * @brief Specialization of is_coo_matrix for actual COO matrix types.
 *
 * This specialization uses std::is_base_of to detect types that inherit from
 * coo_matrix with compatible template parameters. It evaluates to true_type
 * for COO matrix types.
 *
 * @tparam SpMatType The COO matrix type being detected
 */
template <typename SpMatType>
struct is_coo_matrix<SpMatType,
                     std::enable_if_t<std::is_base_of_v<
                         coo_matrix<typename SpMatType::value_type,
                                    typename SpMatType::index_type,
                                    typename SpMatType::allocator_type>,
                         SpMatType> > > : public std::true_type {};

/**
 * @brief Convenience variable template for is_csr_matrix.
 * @tparam SpMatType The type to test
 * @return true if SpMatType is a CSR matrix, false otherwise
 */
template <typename SpMatType>
inline constexpr bool is_csr_matrix_v = is_csr_matrix<SpMatType>::value;

/**
 * @brief Convenience variable template for is_csc_matrix.
 * @tparam SpMatType The type to test
 * @return true if SpMatType is a CSC matrix, false otherwise
 */
template <typename SpMatType>
inline constexpr bool is_csc_matrix_v = is_csc_matrix<SpMatType>::value;

/**
 * @brief Convenience variable template for is_coo_matrix.
 * @tparam SpMatType The type to test
 * @return true if SpMatType is a COO matrix, false otherwise
 */
template <typename SpMatType>
inline constexpr bool is_coo_matrix_v = is_coo_matrix<SpMatType>::value;

/**
 * @brief SFINAE enabler for CSR matrix types.
 *
 * This struct provides a type alias that exists only if the template parameter
 * is a CSR matrix type. Used for template specialization and function
 * overloading.
 *
 * @tparam SpMatType The type to test
 * @tparam U The type to return if SpMatType is a CSR matrix (default: void)
 */
template <typename SpMatType, typename U = void>
struct enable_if_csr_matrix {
  using type = std::enable_if_t<is_csr_matrix_v<SpMatType>, U>;
};

/**
 * @brief SFINAE enabler for CSC matrix types.
 *
 * This struct provides a type alias that exists only if the template parameter
 * is a CSC matrix type. Used for template specialization and function
 * overloading.
 *
 * @tparam SpMatType The type to test
 * @tparam U The type to return if SpMatType is a CSC matrix (default: void)
 */
template <typename SpMatType, typename U = void>
struct enable_if_csc_matrix {
  using type = std::enable_if_t<is_csc_matrix_v<SpMatType>, U>;
};

/**
 * @brief SFINAE enabler for COO matrix types.
 *
 * This struct provides a type alias that exists only if the template parameter
 * is a COO matrix type. Used for template specialization and function
 * overloading.
 *
 * @tparam SpMatType The type to test
 * @tparam U The type to return if SpMatType is a COO matrix (default: void)
 */
template <typename SpMatType, typename U = void>
struct enable_if_coo_matrix {
  using type = std::enable_if_t<is_coo_matrix_v<SpMatType>, U>;
};

/**
 * @brief Convenience type alias for enable_if_csr_matrix.
 * @tparam SpMatType The type to test
 * @tparam U The type to return if SpMatType is a CSR matrix (default: void)
 */
template <typename SpMatType, typename U = void>
using enable_if_csr_matrix_t =
    typename enable_if_csr_matrix<SpMatType, U>::type;

/**
 * @brief Convenience type alias for enable_if_csc_matrix.
 * @tparam SpMatType The type to test
 * @tparam U The type to return if SpMatType is a CSC matrix (default: void)
 */
template <typename SpMatType, typename U = void>
using enable_if_csc_matrix_t =
    typename enable_if_csc_matrix<SpMatType, U>::type;

/**
 * @brief Convenience type alias for enable_if_coo_matrix.
 * @tparam SpMatType The type to test
 * @tparam U The type to return if SpMatType is a COO matrix (default: void)
 */
template <typename SpMatType, typename U = void>
using enable_if_coo_matrix_t =
    typename enable_if_coo_matrix<SpMatType, U>::type;

/**
 * @brief Extracts the value type from a sparse matrix type.
 * @tparam SpMatType The sparse matrix type
 */
template <typename SpMatType>
using value_type_t = typename SpMatType::value_type;

/**
 * @brief Extracts the size type from a sparse matrix type.
 * @tparam SpMatType The sparse matrix type
 */
template <typename SpMatType>
using size_type_t = typename SpMatType::size_type;

/**
 * @brief Extracts the index type from a sparse matrix type.
 * @tparam SpMatType The sparse matrix type
 */
template <typename SpMatType>
using index_type_t = typename SpMatType::index_type;

/**
 * @brief Extracts the allocator type from a sparse matrix type.
 * @tparam SpMatType The sparse matrix type
 */
template <typename SpMatType>
using allocator_type_t = typename SpMatType::allocator_type;

/**
 * @brief Identity type trait that returns the input type unchanged.
 *
 * This is a utility trait that can be used to prevent template argument
 * deduction in certain contexts.
 *
 * @tparam T The type to return
 */
template <typename T>
struct type_identity {
  using type = T;
};

/**
 * @brief Convenience type alias for type_identity.
 * @tparam T The type to return
 */
template <typename T>
using type_identity_t = typename type_identity<T>::type;

}  // namespace sparsexx::detail
