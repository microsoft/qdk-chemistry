// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#if __cplusplus >= 202302L
#include <mdspan>
namespace qdk::chemistry::detail {
namespace mdspan_ns = std;
}
#else
#include <experimental/mdspan>
namespace qdk::chemistry::detail {
namespace mdspan_ns = std::experimental;
}
#endif

#include <cstddef>

namespace qdk::chemistry {

/**
 * @brief Column-major (Fortran-order) multidimensional span.
 *
 * Uses layout_left for column-major storage order, which is compatible with
 * Fortran-ordered data and Eigen's default storage order.
 *
 * @tparam T Element type
 * @tparam Rank Number of dimensions
 */
template <typename T, size_t Rank>
using tensor_span =
    detail::mdspan_ns::mdspan<T, detail::mdspan_ns::dextents<size_t, Rank>,
                              detail::mdspan_ns::layout_left>;

/**
 * @brief Four-dimensional tensor span with column-major layout.
 * @tparam T Element type
 */
template <typename T>
using rank4_span = tensor_span<T, 4>;

/**
 * @brief Create a rank-4 span from flattened storage with uniform dimensions.
 *
 * Constructs a 4D view over linearly-stored data with shape [n, n, n, n].
 * The data must be in column-major order matching the index pattern
 * data[i + n*j + n*n*k + n*n*n*l] = tensor(i,j,k,l).
 *
 * @tparam T Element type
 * @param data Pointer to the first element of the flattened storage
 * @param n Extent along each dimension (total elements = n^4)
 * @return A rank-4 span viewing the data
 *
 * @pre data must point to at least n^4 valid elements
 * @pre data must remain valid for the lifetime of the returned span
 *
 * @code
 * std::vector<double> storage(16); // 2^4 elements
 * auto span = make_rank4_span(storage.data(), 2);
 * span(0, 1, 0, 1) = 3.14; // Access element
 * @endcode
 */
template <typename T>
inline rank4_span<T> make_rank4_span(T* data, size_t n) {
  return rank4_span<T>(data, n, n, n, n);
}

/**
 * @brief Create a rank-4 span from flattened storage with explicit dimensions.
 *
 * Constructs a 4D view over linearly-stored data with shape [n0, n1, n2, n3].
 *
 * @tparam T Element type
 * @param data Pointer to the first element of the flattened storage
 * @param n0 Extent along first dimension
 * @param n1 Extent along second dimension
 * @param n2 Extent along third dimension
 * @param n3 Extent along fourth dimension
 * @return A rank-4 span viewing the data
 */
template <typename T>
inline rank4_span<T> make_rank4_span(T* data, size_t n0, size_t n1, size_t n2,
                                     size_t n3) {
  return rank4_span<T>(data, n0, n1, n2, n3);
}
}  // namespace qdk::chemistry
