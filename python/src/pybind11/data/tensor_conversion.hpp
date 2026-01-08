// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once

/**
 * @file tensor_conversion.hpp
 * @brief Utilities for converting between mdspan tensors and NumPy arrays.
 *
 * This header provides functions to convert qdk::chemistry::rank4_span to numpy
 * arrays, with both zero-copy (view) and copying semantics.
 */

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <qdk/chemistry/utils/tensor_span.hpp>

namespace py = pybind11;

namespace qdk::chemistry::python::utils {

/**
 * @brief Convert a rank-4 span to a NumPy array (copy).
 *
 * Creates an owned NumPy array by copying the data from the span.
 * The returned array is independent of the source span and can outlive it.
 *
 * @tparam T Element type (typically const double)
 * @param span The rank-4 span to convert
 * @return A 4D NumPy array with shape (n0, n1, n2, n3) in Fortran order
 *
 * @code
 * auto span = hamiltonian.get_two_body_span_aaaa();
 * py::array_t<double> arr = span_to_numpy_copy(span);
 * @endcode
 */
template <typename T>
py::array_t<double> span_to_numpy_copy(
    const qdk::chemistry::rank4_span<T>& span) {
  // Get dimensions
  const auto n0 = span.extent(0);
  const auto n1 = span.extent(1);
  const auto n2 = span.extent(2);
  const auto n3 = span.extent(3);

  // Create array with Fortran (column-major) order to match layout_left
  // pybind11 uses row-major by default, so we specify strides explicitly
  std::vector<ssize_t> shape = {
      static_cast<ssize_t>(n0), static_cast<ssize_t>(n1),
      static_cast<ssize_t>(n2), static_cast<ssize_t>(n3)};

  // Column-major strides: stride[0] = sizeof(T), stride[1] = n0 * sizeof(T),
  // etc.
  std::vector<ssize_t> strides = {
      static_cast<ssize_t>(sizeof(double)),
      static_cast<ssize_t>(n0 * sizeof(double)),
      static_cast<ssize_t>(n0 * n1 * sizeof(double)),
      static_cast<ssize_t>(n0 * n1 * n2 * sizeof(double))};

  // Create array and copy data
  py::array_t<double> result(shape, strides);
  auto ptr = result.mutable_data();
  const size_t total_size = n0 * n1 * n2 * n3;
  std::copy(span.data_handle(), span.data_handle() + total_size, ptr);

  return result;
}

/**
 * @brief Convert a rank-4 span to a NumPy array view (zero-copy).
 *
 * Creates a NumPy array that views the same memory as the span.
 * The returned array does NOT own the data - the parent object (e.g.,
 * Hamiltonian) must remain alive while the array is in use.
 *
 * @tparam T Element type (typically const double)
 * @tparam Parent The type of the parent object that owns the data
 * @param span The rank-4 span to view
 * @param parent Shared pointer to the parent object that owns the span's data
 * @return A 4D NumPy array view with shape (n0, n1, n2, n3) in Fortran order
 *
 * @warning The returned array is only valid while the parent object is alive.
 *          Use span_to_numpy_copy() if you need an independent array.
 *
 * @code
 * auto span = hamiltonian.get_two_body_span_aaaa();
 * py::array_t<double> view = span_to_numpy_view(span, hamiltonian_ptr);
 * @endcode
 */
template <typename T, typename Parent>
py::array_t<double, py::array::f_style> span_to_numpy_view(
    const qdk::chemistry::rank4_span<T>& span, std::shared_ptr<Parent> parent) {
  // Get dimensions
  const auto n0 = span.extent(0);
  const auto n1 = span.extent(1);
  const auto n2 = span.extent(2);
  const auto n3 = span.extent(3);

  std::vector<ssize_t> shape = {
      static_cast<ssize_t>(n0), static_cast<ssize_t>(n1),
      static_cast<ssize_t>(n2), static_cast<ssize_t>(n3)};

  // Column-major strides for layout_left
  std::vector<ssize_t> strides = {
      static_cast<ssize_t>(sizeof(double)),
      static_cast<ssize_t>(n0 * sizeof(double)),
      static_cast<ssize_t>(n0 * n1 * sizeof(double)),
      static_cast<ssize_t>(n0 * n1 * n2 * sizeof(double))};

  // Create a capsule that prevents the parent from being destroyed
  // The capsule destructor releases the shared_ptr reference
  py::capsule prevent_gc(new std::shared_ptr<Parent>(parent), [](void* ptr) {
    delete static_cast<std::shared_ptr<Parent>*>(ptr);
  });

  // Create array view (does not copy data)
  return py::array_t<double, py::array::f_style>(
      shape, strides,
      const_cast<double*>(span.data_handle()),  // data pointer
      prevent_gc                                // prevents parent destruction
  );
}

}  // namespace qdk::chemistry::python::utils
