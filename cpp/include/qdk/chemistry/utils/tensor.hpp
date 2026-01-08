// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include <memory>
#include <qdk/chemistry/utils/tensor_span.hpp>
#include <vector>

// Include mdarray from Kokkos implementation
// C++23: Will be <mdarray> when standardized (P1684)
#include <experimental/__p1684_bits/mdarray.hpp>

namespace qdk::chemistry {

/**
 * @brief Owning multidimensional array with column-major (Fortran) layout.
 *
 * Uses mdarray from the Kokkos implementation with std::vector as the
 * underlying container. Provides implicit conversion to tensor_span for
 * non-owning views.
 *
 * @tparam T Element type
 * @tparam Rank Number of dimensions
 */
template <typename T, size_t Rank>
using tensor =
    detail::mdspan_ns::mdarray<T, detail::mdspan_ns::dextents<size_t, Rank>,
                               detail::mdspan_ns::layout_left, std::vector<T>>;

/**
 * @brief Four-dimensional owning tensor with column-major layout.
 * @tparam T Element type
 *
 * Implicit conversion to rank4_span<T> and rank4_span<const T> is provided
 * by mdarray's conversion operators.
 *
 * @code
 * auto eri = make_rank4_tensor<double>(norb);  // Create norb^4 tensor
 * eri(0, 1, 2, 3) = 1.5;                       // Access element
 * rank4_span<double> view = eri;               // Implicit conversion to span
 * @endcode
 */
template <typename T>
using rank4_tensor = tensor<T, 4>;

/**
 * @brief Create an empty rank-4 tensor with uniform dimensions.
 *
 * Allocates storage for n^4 elements initialized to zero.
 *
 * @tparam T Element type
 * @param n Extent along each dimension (total elements = n^4)
 * @return An owning rank-4 tensor with shape [n, n, n, n]
 *
 * @code
 * auto eri = make_rank4_tensor<double>(10);  // 10^4 = 10000 elements
 * @endcode
 */
template <typename T>
inline rank4_tensor<T> make_rank4_tensor(size_t n) {
  return rank4_tensor<T>(n, n, n, n);
}

/**
 * @brief Create a rank-4 tensor with explicit dimensions.
 *
 * Allocates storage for n0*n1*n2*n3 elements initialized to zero.
 *
 * @tparam T Element type
 * @param n0 Extent along first dimension
 * @param n1 Extent along second dimension
 * @param n2 Extent along third dimension
 * @param n3 Extent along fourth dimension
 * @return An owning rank-4 tensor
 */
template <typename T>
inline rank4_tensor<T> make_rank4_tensor(size_t n0, size_t n1, size_t n2,
                                         size_t n3) {
  return rank4_tensor<T>(n0, n1, n2, n3);
}

/**
 * @brief Create a rank-4 tensor by moving container data with uniform
 * dimensions.
 *
 * Takes ownership of the provided vector's storage. The vector must contain
 * exactly n^4 elements in column-major order.
 *
 * @tparam T Element type
 * @param data Vector containing flattened tensor data (moved from)
 * @param n Extent along each dimension
 * @return An owning rank-4 tensor with shape [n, n, n, n]
 *
 * @pre data.size() >= n^4
 *
 * @code
 * std::vector<double> storage(10000);
 * // ... fill storage ...
 * auto eri = make_rank4_tensor(std::move(storage), 10);  // Zero-copy
 * @endcode
 */
template <typename T>
inline rank4_tensor<T> make_rank4_tensor(std::vector<T>&& data, size_t n) {
  return rank4_tensor<T>(std::move(data), n, n, n, n);
}

/**
 * @brief Create a rank-4 tensor by moving container data with explicit
 * dimensions.
 *
 * @tparam T Element type
 * @param data Vector containing flattened tensor data (moved from)
 * @param n0 Extent along first dimension
 * @param n1 Extent along second dimension
 * @param n2 Extent along third dimension
 * @param n3 Extent along fourth dimension
 * @return An owning rank-4 tensor
 *
 * @pre data.size() >= n0*n1*n2*n3
 */
template <typename T>
inline rank4_tensor<T> make_rank4_tensor(std::vector<T>&& data, size_t n0,
                                         size_t n1, size_t n2, size_t n3) {
  return rank4_tensor<T>(std::move(data), n0, n1, n2, n3);
}

/**
 * @brief Create a rank-4 tensor by copying from a span.
 *
 * Copies the span's data into a new owning tensor. Use this when you need
 * to take ownership of externally-managed data.
 *
 * @tparam T Element type (const-qualified types will be decayed)
 * @param span A rank-4 span to copy from
 * @return An owning rank-4 tensor with the same dimensions and data
 *
 * @code
 * auto view = hamiltonian.get_two_body_integrals();
 * auto owned = make_rank4_tensor(std::get<0>(view));  // Copy from span
 * @endcode
 */
template <typename T>
inline rank4_tensor<std::remove_const_t<T>> make_rank4_tensor(
    const rank4_span<T>& span) {
  using value_type = std::remove_const_t<T>;
  std::vector<value_type> data(
      span.data_handle(), span.data_handle() + span.extent(0) * span.extent(1) *
                                                   span.extent(2) *
                                                   span.extent(3));
  return rank4_tensor<value_type>(std::move(data), span.extent(0),
                                  span.extent(1), span.extent(2),
                                  span.extent(3));
}

/**
 * @brief Get an mdspan view of a rank-4 tensor.
 *
 * Explicit conversion to span. Note that mdarray also provides implicit
 * conversion operators, so this is only needed when explicit conversion
 * is preferred.
 *
 * @tparam T Element type
 * @param tensor The tensor to view
 * @return A non-owning span over the tensor's data
 */
template <typename T>
inline rank4_span<T> as_span(rank4_tensor<T>& tensor) {
  return tensor.to_mdspan();
}

/**
 * @brief Get a const mdspan view of a rank-4 tensor.
 *
 * @tparam T Element type
 * @param tensor The tensor to view
 * @return A non-owning const span over the tensor's data
 */
template <typename T>
inline rank4_span<const T> as_span(const rank4_tensor<T>& tensor) {
  return tensor.to_mdspan();
}

}  // namespace qdk::chemistry
