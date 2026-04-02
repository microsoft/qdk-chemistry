// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <qdk/chemistry/utils/orbital_entropies.hpp>

namespace py = pybind11;

void bind_orbital_entropies(pybind11::module& m) {
  using namespace qdk::chemistry::utils::orbital_entropies;
  using namespace qdk::chemistry::data;

  auto oe = m.def_submodule("orbital_entropies", R"(
Orbital entropy and mutual information utilities.

Provides factory functions for entropy measures (von Neumann, Rényi, min,
max) and functions to compute single-orbital entropies, two-orbital
entropies, and mutual information from orbital reduced density matrices
or directly from a :class:`~qdk_chemistry.Wavefunction`.

All entropy functions accept a custom entropy measure as an optional
argument. If omitted, von Neumann entropy is used by default.

Example::

    from qdk_chemistry.utils.orbital_entropies import (
        build_single_orbital_entropies,
        build_mutual_information,
        von_neumann_entropy,
        renyi_entropy,
    )

    # Von Neumann (default)
    s1 = build_single_orbital_entropies(wfn)
    mi = build_mutual_information(wfn)

    # Rényi with alpha=2
    s1_renyi = build_single_orbital_entropies(wfn, renyi_entropy(2.0))

    # Custom Python callable
    s1_custom = build_single_orbital_entropies(
        wfn, lambda eigs: sum(e**2 for e in eigs)
    )
)");

  // ---- Entropy measure factories ----

  oe.def("von_neumann_entropy", &von_neumann_entropy, R"(
Create a von Neumann entropy measure.

S = -sum_k lambda_k * ln(lambda_k)

Returns:
    callable: Entropy function taking a list of eigenvalues.
)");

  oe.def("renyi_entropy", &renyi_entropy, R"(
Create a Rényi entropy measure of order alpha.

S_alpha = 1/(1 - alpha) * ln(sum_k lambda_k^alpha)

Args:
    alpha (float): Rényi order (must not be 1).

Returns:
    callable: Entropy function taking a list of eigenvalues.
)",
         py::arg("alpha"));

  oe.def("min_entropy", &min_entropy, R"(
Create a min-entropy measure (Rényi entropy of order infinity).

S_min = -ln(max_k lambda_k)

Returns:
    callable: Entropy function taking a list of eigenvalues.
)");

  oe.def("max_entropy", &max_entropy, R"(
Create a max-entropy (Hartley) measure (Rényi entropy of order 0).

S_max = ln(|{k : lambda_k > 0}|)

Returns:
    callable: Entropy function taking a list of eigenvalues.
)");

  // ---- Wavefunction-based API ----

  oe.def(
      "build_single_orbital_entropies",
      [](const Wavefunction& wfn, const EntropyFunction& func) {
        return build_single_orbital_entropies(wfn, func);
      },
      R"(
Compute single-orbital entropies from a wavefunction.

Requires that the wavefunction has 1-orbital RDMs available
(``calculate_one_orbital_rdm=True`` when running the solver).

Args:
    wavefunction (Wavefunction): Wavefunction with orbital RDM data.
    entropy_func (callable, optional): Entropy measure. Defaults to
        von Neumann entropy.

Returns:
    numpy.ndarray: Vector of single-orbital entropies (shape ``(norb,)``).
)",
      py::arg("wavefunction"),
      py::arg("entropy_func") = von_neumann_entropy());

  oe.def(
      "build_two_orbital_entropies",
      [](const Wavefunction& wfn, const EntropyFunction& func) {
        return build_two_orbital_entropies(wfn, func);
      },
      R"(
Compute two-orbital entropies from a wavefunction.

Requires that the wavefunction has both 1- and 2-orbital RDMs available.

Args:
    wavefunction (Wavefunction): Wavefunction with orbital RDM data.
    entropy_func (callable, optional): Entropy measure. Defaults to
        von Neumann entropy.

Returns:
    numpy.ndarray: Symmetric matrix of two-orbital entropies
        (shape ``(norb, norb)``).
)",
      py::arg("wavefunction"),
      py::arg("entropy_func") = von_neumann_entropy());

  oe.def(
      "build_mutual_information",
      py::overload_cast<const Wavefunction&, const EntropyFunction&>(
          &build_mutual_information),
      R"(
Compute mutual information from a wavefunction.

I(i,j) = S_1(i) + S_1(j) - S_2(i,j)

Requires that the wavefunction has both 1- and 2-orbital RDMs available.

Args:
    wavefunction (Wavefunction): Wavefunction with orbital RDM data.
    entropy_func (callable, optional): Entropy measure. Defaults to
        von Neumann entropy.

Returns:
    numpy.ndarray: Symmetric mutual information matrix
        (shape ``(norb, norb)``).
)",
      py::arg("wavefunction"),
      py::arg("entropy_func") = von_neumann_entropy());

  // ---- Low-level eigenvalue-based API ----

  oe.def(
      "build_single_orbital_entropies",
      [](const Eigen::MatrixXd& eigenvalues, const EntropyFunction& func) {
        return build_single_orbital_entropies(eigenvalues, func);
      },
      R"(
Compute single-orbital entropies from an eigenvalue matrix.

Each row contains the eigenvalue spectrum for that orbital.

Args:
    eigenvalues (numpy.ndarray): Matrix of shape ``(norb, n_eig)``.
    entropy_func (callable, optional): Entropy measure. Defaults to
        von Neumann entropy.

Returns:
    numpy.ndarray: Vector of entropies (shape ``(norb,)``).
)",
      py::arg("eigenvalues"),
      py::arg("entropy_func") = von_neumann_entropy());

  oe.def(
      "build_two_orbital_entropies",
      [](const Eigen::VectorXd& two_ordm, Eigen::Index norb,
         const EntropyFunction& func) {
        return build_two_orbital_entropies(two_ordm, norb, func);
      },
      R"(
Compute two-orbital entropies from the 2-orbital RDM tensor.

The tensor has shape (norb, norb, 16, 16) stored as a flat column-major
vector matching the MACIS rank4_span layout.

Args:
    two_ordm (numpy.ndarray): Flat vector of size norb*norb*16*16.
    norb (int): Number of orbitals.
    entropy_func (callable, optional): Entropy measure. Defaults to
        von Neumann entropy.

Returns:
    numpy.ndarray: Symmetric entropy matrix (shape ``(norb, norb)``).
)",
      py::arg("two_ordm"), py::arg("norb"),
      py::arg("entropy_func") = von_neumann_entropy());

  oe.def(
      "build_mutual_information",
      py::overload_cast<const Eigen::VectorXd&, const Eigen::MatrixXd&>(
          &build_mutual_information),
      R"(
Compute mutual information from single- and two-orbital entropies.

I(i,j) = S_1(i) + S_1(j) - S_2(i,j)

Args:
    s1 (numpy.ndarray): Single-orbital entropies (shape ``(norb,)``).
    s2 (numpy.ndarray): Two-orbital entropies (shape ``(norb, norb)``).

Returns:
    numpy.ndarray: Symmetric mutual information matrix
        (shape ``(norb, norb)``).
)",
      py::arg("s1"), py::arg("s2"));
}
