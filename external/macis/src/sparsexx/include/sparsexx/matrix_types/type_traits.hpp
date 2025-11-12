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

template <typename SpMatType, typename = void>
struct is_csr_matrix : public std::false_type {};
template <typename SpMatType, typename = void>
struct is_csc_matrix : public std::false_type {};
template <typename SpMatType, typename = void>
struct is_coo_matrix : public std::false_type {};

template <typename SpMatType>
struct is_csr_matrix<SpMatType,
                     std::enable_if_t<std::is_base_of_v<
                         csr_matrix<typename SpMatType::value_type,
                                    typename SpMatType::index_type,
                                    typename SpMatType::allocator_type>,
                         SpMatType> > > : public std::true_type {};

template <typename SpMatType>
struct is_csc_matrix<SpMatType,
                     std::enable_if_t<std::is_base_of_v<
                         csc_matrix<typename SpMatType::value_type,
                                    typename SpMatType::index_type,
                                    typename SpMatType::allocator_type>,
                         SpMatType> > > : public std::true_type {};

template <typename SpMatType>
struct is_coo_matrix<SpMatType,
                     std::enable_if_t<std::is_base_of_v<
                         coo_matrix<typename SpMatType::value_type,
                                    typename SpMatType::index_type,
                                    typename SpMatType::allocator_type>,
                         SpMatType> > > : public std::true_type {};

template <typename SpMatType>
inline constexpr bool is_csr_matrix_v = is_csr_matrix<SpMatType>::value;
template <typename SpMatType>
inline constexpr bool is_csc_matrix_v = is_csc_matrix<SpMatType>::value;
template <typename SpMatType>
inline constexpr bool is_coo_matrix_v = is_coo_matrix<SpMatType>::value;

template <typename SpMatType, typename U = void>
struct enable_if_csr_matrix {
  using type = std::enable_if_t<is_csr_matrix_v<SpMatType>, U>;
};
template <typename SpMatType, typename U = void>
struct enable_if_csc_matrix {
  using type = std::enable_if_t<is_csc_matrix_v<SpMatType>, U>;
};
template <typename SpMatType, typename U = void>
struct enable_if_coo_matrix {
  using type = std::enable_if_t<is_coo_matrix_v<SpMatType>, U>;
};

template <typename SpMatType, typename U = void>
using enable_if_csr_matrix_t =
    typename enable_if_csr_matrix<SpMatType, U>::type;
template <typename SpMatType, typename U = void>
using enable_if_csc_matrix_t =
    typename enable_if_csc_matrix<SpMatType, U>::type;
template <typename SpMatType, typename U = void>
using enable_if_coo_matrix_t =
    typename enable_if_coo_matrix<SpMatType, U>::type;

template <typename SpMatType>
using value_type_t = typename SpMatType::value_type;

template <typename SpMatType>
using size_type_t = typename SpMatType::size_type;

template <typename SpMatType>
using index_type_t = typename SpMatType::index_type;

template <typename SpMatType>
using allocator_type_t = typename SpMatType::allocator_type;

template <typename T>
struct type_identity {
  using type = T;
};

template <typename T>
using type_identity_t = typename type_identity<T>::type;

}  // namespace sparsexx::detail
