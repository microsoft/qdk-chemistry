// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once

#include <nanobind/eigen/dense.h>
#include <nanobind/eigen/sparse.h>

namespace nanobind::detail {

// Preserve the array-like inputs and singleton dimensions accepted by the
// Python API, while using nanobind's ndarray conversion and ownership rules.
template <typename Matrix>
struct array_like_eigen_caster {
  using Scalar = typename Matrix::Scalar;
  static constexpr int Rows = Matrix::RowsAtCompileTime;
  using Array = array_for_eigen_t<Matrix, const Scalar>;
  NB_TYPE_CASTER(Matrix, make_caster<Array>::Name)

  bool from_python(handle source, uint32_t flags,
                   cleanup_list* cleanup) noexcept {
    try {
      make_caster<Array> array;
      if (!array.from_python(source, flags, cleanup)) {
        if (!(flags & cast_flags::convert) || source.is_none()) return false;
        object converted = module_::import_("numpy").attr("asarray")(source);
        const auto dimensions = cast<size_t>(converted.attr("ndim"));
        if constexpr (Matrix::IsVectorAtCompileTime) {
          if (dimensions == 2) {
            auto shape = converted.attr("shape");
            if (cast<size_t>(shape[0]) == 1 || cast<size_t>(shape[1]) == 1)
              converted = converted.attr("reshape")(-1);
          }
        } else if (dimensions == 1) {
          converted = converted.attr("reshape")(-1, 1);
        }
        if (!array.from_python(converted, flags, cleanup)) return false;
      }
      const auto& input = array.value;
      const Eigen::Index rows =
          Matrix::IsVectorAtCompileTime && Rows == 1 ? 1 : input.shape(0);
      const Eigen::Index cols = Matrix::IsVectorAtCompileTime
                                    ? (Rows == 1 ? input.shape(0) : 1)
                                    : input.shape(1);
      value = Eigen::Map<const Matrix>(input.data(), rows, cols);
      return true;
    } catch (...) {
      PyErr_Clear();
      return false;
    }
  }

  template <typename T>
  static handle from_cpp(T&& source, rv_policy policy,
                         cleanup_list* cleanup) noexcept {
    try {
      using Source = std::remove_reference_t<T>;
      if constexpr (std::is_pointer_v<Source>) {
        if (!source) return none().release();
      }
      const Matrix& matrix = [&]() -> const Matrix& {
        if constexpr (std::is_pointer_v<Source>)
          return *source;
        else
          return source;
      }();
      policy = infer_policy<T>(policy);
      object owner;
      const Matrix* output_matrix = &matrix;
      if (policy == rv_policy::reference_internal) {
        if (!cleanup || !cleanup->self()) return handle();
        owner = borrow(cleanup->self());
        policy = rv_policy::reference;
      } else if (policy == rv_policy::take_ownership) {
        owner = capsule(const_cast<Matrix*>(&matrix), [](void* ptr) noexcept {
          delete static_cast<Matrix*>(ptr);
        });
        policy = rv_policy::reference;
      } else if (policy == rv_policy::move) {
        if constexpr (Matrix::SizeAtCompileTime == Eigen::Dynamic) {
          output_matrix = [&]() {
            if constexpr (std::is_pointer_v<Source>)
              return new Matrix(std::move(*source));
            else
              return new Matrix(std::forward<T>(source));
          }();
          owner = capsule(output_matrix, [](void* ptr) noexcept {
            delete static_cast<Matrix*>(ptr);
          });
          policy = rv_policy::reference;
        } else {
          policy = rv_policy::copy;
        }
      }
      using OutputScalar =
          std::conditional_t<std::is_const_v<std::remove_pointer_t<Source>>,
                             const Scalar, Scalar>;
      using Output = array_for_eigen_t<Matrix, OutputScalar>;
      const size_t shape[] = {static_cast<size_t>(output_matrix->rows()),
                              static_cast<size_t>(output_matrix->cols())};
      const size_t vector_shape[] = {
          static_cast<size_t>(output_matrix->size())};
      Output output(const_cast<Scalar*>(output_matrix->data()), ndim_v<Matrix>,
                    Matrix::IsVectorAtCompileTime ? vector_shape : shape,
                    owner);
      return make_caster<Output>::from_cpp(output, policy, cleanup);
    } catch (const std::exception& error) {
      PyErr_SetString(PyExc_RuntimeError, error.what());
      return handle();
    }
  }
};

template <>
struct type_caster<Eigen::MatrixXd> : array_like_eigen_caster<Eigen::MatrixXd> {
};
template <>
struct type_caster<Eigen::MatrixXcd>
    : array_like_eigen_caster<Eigen::MatrixXcd> {};
template <>
struct type_caster<Eigen::VectorXd> : array_like_eigen_caster<Eigen::VectorXd> {
};
template <>
struct type_caster<Eigen::RowVectorXd>
    : array_like_eigen_caster<Eigen::RowVectorXd> {};
template <>
struct type_caster<Eigen::VectorXcd>
    : array_like_eigen_caster<Eigen::VectorXcd> {};
template <>
struct type_caster<Eigen::VectorXi> : array_like_eigen_caster<Eigen::VectorXi> {
};
template <>
struct type_caster<Eigen::Vector3d> : array_like_eigen_caster<Eigen::Vector3d> {
};

}  // namespace nanobind::detail
