// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <Eigen/Core>
#include <cstddef>
#include <qdk/chemistry/data/two_body_symmetry.hpp>
#include <string>

namespace py = pybind11;

namespace {

namespace detail = qdk::chemistry::data::detail;

using Integrals =
    py::array_t<double, py::array::c_style | py::array::forcecast>;

/// Number of orbitals n of (n, n, n, n) integrals.
std::size_t orbital_count(const Integrals& integrals) {
  if (integrals.ndim() != 4 || integrals.shape(1) != integrals.shape(0) ||
      integrals.shape(2) != integrals.shape(0) ||
      integrals.shape(3) != integrals.shape(0)) {
    throw py::value_error(
        "The two-body integrals must have shape (n, n, n, n).");
  }
  return static_cast<std::size_t>(integrals.shape(0));
}

/// A copy of `values` with the shape of `integrals`.
Integrals shaped_like(const Integrals& integrals, const double* values) {
  const py::ssize_t n = integrals.shape(0);
  return Integrals({n, n, n, n}, values);
}

/// `integrals` symmetrized by `symmetrize`, for a consumer that takes them only
/// if they have `symmetry` to isApprox precision, as measured by `compare`.
template <class Compare, class Symmetrize>
Integrals required_symmetrized(const Integrals& integrals,
                               const std::string& consumer,
                               const std::string& symmetry, Compare compare,
                               Symmetrize symmetrize) {
  const std::size_t n = orbital_count(integrals);
  detail::require_finite_two_body(integrals.data(), n * n * n * n, consumer);
  if (const auto comparison = compare(integrals.data(), n);
      !comparison.approx()) {
    throw py::value_error(
        detail::not_symmetric_error(consumer, symmetry, comparison).what());
  }
  Integrals symmetrized = shaped_like(integrals, integrals.data());
  symmetrize(symmetrized.mutable_data(), n);
  return symmetrized;
}

}  // namespace

void bind_two_body_symmetry(py::module& data) {
  data.def(
      "_fourfold_symmetrized",
      [](const Integrals& integrals, const std::string& consumer) {
        Eigen::VectorXd storage;
        const double* symmetrized = detail::fourfold_symmetrized(
            integrals.data(), orbital_count(integrals), storage, consumer);
        return shaped_like(integrals, symmetrized);
      },
      R"(
Average of restricted or same-spin integrals over ``(pq|rs)``, ``(rs|pq)``,
``(qp|sr)``, ``(sr|qp)``, with a warning if they are not Hermitian.
)",
      py::arg("integrals"), py::arg("consumer"));

  data.def(
      "_reversal_symmetrized",
      [](const Integrals& integrals, const std::string& consumer) {
        Eigen::VectorXd storage;
        const double* symmetrized = detail::reversal_symmetrized(
            integrals.data(), orbital_count(integrals), storage, consumer);
        return shaped_like(integrals, symmetrized);
      },
      R"(
Average of alpha-beta integrals over ``(pq|rs)``, ``(qp|sr)``, with a warning
if they are not Hermitian.
)",
      py::arg("integrals"), py::arg("consumer"));

  data.def(
      "_eightfold_symmetrized",
      [](const Integrals& integrals, const std::string& consumer) {
        return required_symmetrized(integrals, consumer, "8-fold",
                                    detail::eightfold_comparison,
                                    detail::eightfold_symmetrize);
      },
      R"(
Average of integrals over their 8-fold permutation classes. Raises
``ValueError`` unless they equal it to ``isApprox`` precision.
)",
      py::arg("integrals"), py::arg("consumer"));

  data.def(
      "_pair_symmetrized",
      [](const Integrals& integrals, const std::string& consumer) {
        return required_symmetrized(integrals, consumer, "pair",
                                    detail::pair_comparison,
                                    detail::pair_symmetrize);
      },
      R"(
Average of alpha-beta integrals over ``(pq|rs)``, ``(qp|rs)``, ``(pq|sr)``,
``(qp|sr)``. Raises ``ValueError`` unless they equal it to ``isApprox``
precision.
)",
      py::arg("integrals"), py::arg("consumer"));

  data.def(
      "_is_approx",
      [](const Eigen::Ref<const Eigen::VectorXd>& x,
         const Eigen::Ref<const Eigen::VectorXd>& y) {
        if (x.size() != y.size()) {
          throw py::value_error("The vectors must have the same size.");
        }
        return x.isApprox(y);
      },
      R"(
Eigen's ``x.isApprox(y)``: ``||x - y|| <= p min(||x||, ||y||)`` with Eigen's
default precision ``p``.
)",
      py::arg("x"), py::arg("y"));
}
