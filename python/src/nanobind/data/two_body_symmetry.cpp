// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include <Eigen/Core>
#include <cstddef>
#include <qdk/chemistry/data/two_body_symmetry.hpp>
#include <string>
#include <vector>

#include "eigen.hpp"
#include "stl.hpp"

namespace nb = nanobind;

namespace {

namespace detail = qdk::chemistry::data::detail;

using Integrals = nb::ndarray<nb::numpy, const double, nb::c_contig>;
using Symmetrized = nb::ndarray<nb::numpy, double, nb::ndim<4>>;

/// Number of orbitals n of (n, n, n, n) integrals.
std::size_t orbital_count(const Integrals& integrals) {
  if (integrals.ndim() != 4 || integrals.shape(1) != integrals.shape(0) ||
      integrals.shape(2) != integrals.shape(0) ||
      integrals.shape(3) != integrals.shape(0)) {
    throw nb::value_error(
        "The two-body integrals must have shape (n, n, n, n).");
  }
  return integrals.shape(0);
}

/// An (n, n, n, n) array that owns a copy of `values`.
Symmetrized shaped_copy(const double* values, std::size_t n) {
  auto* owned = new std::vector<double>(values, values + n * n * n * n);
  nb::capsule owner(owned, [](void* p) noexcept {
    delete static_cast<std::vector<double>*>(p);
  });
  return Symmetrized(owned->data(), {n, n, n, n}, owner);
}

/// `integrals` symmetrized by `symmetrize`, for a consumer that takes them only
/// if they have `symmetry`, i.e. equal the result to isApprox precision.
template <class Symmetrize>
Symmetrized required_symmetrized(const Integrals& integrals,
                                 const std::string& consumer,
                                 const std::string& symmetry,
                                 Symmetrize symmetrize) {
  const std::size_t n = orbital_count(integrals);
  const std::size_t count = n * n * n * n;
  detail::require_finite_two_body(integrals.data(), count, consumer);
  Symmetrized symmetrized = shaped_copy(integrals.data(), n);
  symmetrize(symmetrized.data(), n);
  const double* input = integrals.data();
  const double* output = symmetrized.data();
  detail::TwoBodyComparison comparison;
  for (std::size_t i = 0; i < count; ++i) comparison.add(input[i], output[i]);
  if (!comparison.approx()) {
    throw nb::value_error(
        detail::not_symmetric_error(consumer, symmetry, comparison).what());
  }
  return symmetrized;
}

}  // namespace

void bind_two_body_symmetry(nb::module_& data) {
  data.def(
      "_fourfold_symmetrized",
      [](const Integrals& integrals, const std::string& consumer) {
        Eigen::VectorXd storage;
        const std::size_t n = orbital_count(integrals);
        return shaped_copy(detail::fourfold_symmetrized(integrals.data(), n,
                                                        storage, consumer),
                           n);
      },
      R"(
Average of restricted or same-spin integrals over ``(pq|rs)``, ``(rs|pq)``,
``(qp|sr)``, ``(sr|qp)``, with a warning if they are not Hermitian.
)",
      nb::arg("integrals"), nb::arg("consumer"));

  data.def(
      "_reversal_symmetrized",
      [](const Integrals& integrals, const std::string& consumer) {
        Eigen::VectorXd storage;
        const std::size_t n = orbital_count(integrals);
        return shaped_copy(detail::reversal_symmetrized(integrals.data(), n,
                                                        storage, consumer),
                           n);
      },
      R"(
Average of alpha-beta integrals over ``(pq|rs)``, ``(qp|sr)``, with a warning
if they are not Hermitian.
)",
      nb::arg("integrals"), nb::arg("consumer"));

  data.def(
      "_required_eightfold_symmetrized",
      [](const Integrals& integrals, const std::string& consumer) {
        return required_symmetrized(integrals, consumer, "8-fold",
                                    detail::eightfold_symmetrize);
      },
      R"(
Average of integrals over their 8-fold permutation classes. Raises
``ValueError`` unless they equal it to ``isApprox`` precision.
)",
      nb::arg("integrals"), nb::arg("consumer"));

  data.def(
      "_required_pair_symmetrized",
      [](const Integrals& integrals, const std::string& consumer) {
        return required_symmetrized(integrals, consumer,
                                    "(pq|rs) = (qp|rs) = (pq|sr)",
                                    detail::pair_symmetrize);
      },
      R"(
Average of alpha-beta integrals over ``(pq|rs)``, ``(qp|rs)``, ``(pq|sr)``,
``(qp|sr)``. Raises ``ValueError`` unless they equal it to ``isApprox``
precision.
)",
      nb::arg("integrals"), nb::arg("consumer"));

  data.def(
      "_is_approx",
      [](const Eigen::VectorXd& x, const Eigen::VectorXd& y) {
        if (x.size() != y.size()) {
          throw nb::value_error("The vectors must have the same size.");
        }
        return x.isApprox(y);
      },
      R"(
Eigen's ``x.isApprox(y)``: ``||x - y|| <= p min(||x||, ||y||)`` with Eigen's
default precision ``p``.
)",
      nb::arg("x"), nb::arg("y"));
}
