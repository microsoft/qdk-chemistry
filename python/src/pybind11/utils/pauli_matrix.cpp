// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <pybind11/complex.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <algorithm>
#include <bit>
#include <complex>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace py = pybind11;

namespace {

/**
 * @brief Bitmask decomposition of a Pauli label string.
 *
 * Uses Little-Endian convention: label[0] corresponds to qubit n-1 (MSB).
 */
struct PauliMasks {
  int64_t x_mask;  ///< Bitmask with 1s where X or Y appears
  int64_t z_mask;  ///< Bitmask with 1s where Z or Y appears
  int y_count;     ///< Number of Y operators in the label
};

/**
 * @brief Decompose a Pauli label string into X/Z bitmasks and Y count.
 *
 * Each character in the label sets the corresponding bit in the X and/or Z
 * mask: X sets x_mask, Z sets z_mask, Y sets both.
 *
 * @param label Pauli string of length n (characters in {I, X, Y, Z})
 * @param n Number of qubits (must equal label.size())
 * @return PauliMasks containing x_mask, z_mask, and y_count
 */
PauliMasks pauli_string_to_masks(const std::string& label, int n) {
  int64_t x_mask = 0, z_mask = 0;
  int y_count = 0;
  for (int i = 0; i < n; ++i) {
    int64_t bit = int64_t{1} << (n - 1 - i);
    switch (label[i]) {
      case 'X':
        x_mask |= bit;
        break;
      case 'Z':
        z_mask |= bit;
        break;
      case 'Y':
        x_mask |= bit;
        z_mask |= bit;
        ++y_count;
        break;
      default:
        break;  // 'I'
    }
  }
  return {x_mask, z_mask, y_count};
}

/**
 * @brief Compute the phase factor (i)^y_count.
 *
 * Returns the complex value of i raised to the power y_count,
 * cycling through {1, i, -1, -i}.
 *
 * @param y_count Number of Y operators
 * @return (i)^y_count as a complex double
 */
std::complex<double> y_phase(int y_count) {
  switch (y_count & 3) {
    case 0:
      return {1.0, 0.0};
    case 1:
      return {0.0, 1.0};
    case 2:
      return {-1.0, 0.0};
    case 3:
      return {0.0, -1.0};
  }
  return {1.0, 0.0};  // unreachable
}

}  // namespace

/**
 * @brief Decompose a Pauli label into bitmasks and phase.
 *
 * Python-facing wrapper that returns the X bitmask, Z bitmask, and
 * the cumulative Y phase factor as a tuple.
 *
 * @param label Pauli string (characters in {I, X, Y, Z})
 * @return Tuple of (x_mask, z_mask, y_phase)
 */
static py::tuple py_pauli_string_to_masks(const std::string& label) {
  const int n = static_cast<int>(label.size());
  auto m = pauli_string_to_masks(label, n);
  return py::make_tuple(m.x_mask, m.z_mask, y_phase(m.y_count));
}

/**
 * @brief Compute the expectation value <psi|P|psi> for a single Pauli string.
 *
 * Uses the bitmask approach to evaluate the expectation value without
 * materialising the full Pauli matrix. Only the real part is returned
 * because every Pauli string is Hermitian.
 *
 * @param pauli_str Pauli label of length n (characters in {I, X, Y, Z}),
 *                  using Little-Endian convention (label[0] = qubit n-1)
 * @param psi_arr Complex state vector of length 2^n
 * @return Real-valued expectation value
 */
static double pauli_expectation(const std::string& pauli_str,
                                py::array_t<std::complex<double>> psi_arr) {
  auto psi = psi_arr.unchecked<1>();
  const int n = static_cast<int>(pauli_str.size());
  const int64_t dim = int64_t{1} << n;
  auto m = pauli_string_to_masks(pauli_str, n);
  const auto phase = y_phase(m.y_count);

  double result = 0.0;
#pragma omp parallel for schedule(static) reduction(+ : result)
  for (int64_t r = 0; r < dim; ++r) {
    int64_t c = r ^ m.x_mask;
    int parity = std::popcount(static_cast<uint64_t>(c & m.z_mask));
    double sign = 1.0 - 2.0 * (parity & 1);
    auto val = std::conj(psi(r)) * phase * sign * psi(c);
    result += val.real();
  }
  return result;
}

/**
 * @brief Build a dense Hamiltonian matrix from Pauli strings and coefficients.
 *
 * Computes H = sum_t coeff[t] * P_t, where each P_t is a Pauli string
 * in Little-Endian convention (label[0] = qubit n-1).
 *
 * @param pauli_strings Vector of Pauli label strings, each of length n
 * @param coefficients_arr Complex coefficient for each Pauli term
 * @return Dense complex matrix of dimension 2^n x 2^n
 */
static Eigen::MatrixXcd pauli_to_dense_matrix(
    const std::vector<std::string>& pauli_strings,
    py::array_t<std::complex<double>> coefficients_arr) {
  auto coeffs = coefficients_arr.unchecked<1>();
  const int T = static_cast<int>(pauli_strings.size());
  const int n = static_cast<int>(pauli_strings[0].size());
  const int64_t dim = int64_t{1} << n;

  Eigen::MatrixXcd H = Eigen::MatrixXcd::Zero(dim, dim);

  // Pre-compute masks for all terms
  std::vector<int64_t> x_masks(T), z_masks(T);
  std::vector<std::complex<double>> scaled(T);
  for (int t = 0; t < T; ++t) {
    auto m = pauli_string_to_masks(pauli_strings[t], n);
    x_masks[t] = m.x_mask;
    z_masks[t] = m.z_mask;
    scaled[t] = coeffs(t) * y_phase(m.y_count);
  }

#pragma omp parallel for schedule(static)
  for (int64_t r = 0; r < dim; ++r) {
    for (int t = 0; t < T; ++t) {
      int64_t c = r ^ x_masks[t];
      int parity = std::popcount(static_cast<uint64_t>(c & z_masks[t]));
      double sign = 1 - 2 * (parity & 1);
      H(r, c) += scaled[t] * sign;
    }
  }

  return H;
}

/**
 * @brief Build a sparse CSR Hamiltonian matrix from Pauli strings and
 *        coefficients.
 *
 * Computes H = sum_t coeff[t] * P_t and returns the result as a
 * scipy.sparse.csr_matrix constructed directly in C++.
 *
 * @param pauli_strings Vector of Pauli label strings, each of length n
 * @param coefficients_arr Complex coefficient for each Pauli term
 * @return scipy.sparse.csr_matrix of dimension 2^n x 2^n
 *
 * @throws std::runtime_error If the matrix dimensions exceed int32 index
 *         limits
 */
static py::object pauli_to_sparse_matrix(
    const std::vector<std::string>& pauli_strings,
    py::array_t<std::complex<double>> coefficients_arr) {
  auto coeffs = coefficients_arr.unchecked<1>();
  const int T = static_cast<int>(pauli_strings.size());
  const int n = static_cast<int>(pauli_strings[0].size());
  const int64_t dim = int64_t{1} << n;
  using StorageIndex = int32_t;

  // Pre-compute masks for all terms
  std::vector<int64_t> x_masks(T), z_masks(T);
  std::vector<std::complex<double>> scaled(T);
  for (int t = 0; t < T; ++t) {
    auto m = pauli_string_to_masks(pauli_strings[t], n);
    x_masks[t] = m.x_mask;
    z_masks[t] = m.z_mask;
    scaled[t] = coeffs(t) * y_phase(m.y_count);
  }

  // Deduplicate x_masks to find K = nnz per row (same for every row).
  std::vector<int64_t> unique_xm = x_masks;
  std::sort(unique_xm.begin(), unique_xm.end());
  unique_xm.erase(std::unique(unique_xm.begin(), unique_xm.end()),
                  unique_xm.end());
  const int64_t K = static_cast<int64_t>(unique_xm.size());
  const int64_t total_nnz = K * dim;

  // Guard against overflow when dimensions/nnz exceed int32 limits.
  if (dim > static_cast<int64_t>(std::numeric_limits<StorageIndex>::max()) ||
      total_nnz >
          static_cast<int64_t>(std::numeric_limits<StorageIndex>::max())) {
    throw std::runtime_error(
        "Matrix too large for int32 CSR indices in pauli_to_sparse_matrix.");
  }

  // Allocate Python-owned CSR arrays once and fill in-place.
  py::array_t<StorageIndex> indptr(dim + 1);
  py::array_t<StorageIndex> indices(total_nnz);
  py::array_t<std::complex<double>> data(total_nnz);

  auto indptr_m = indptr.mutable_unchecked<1>();
  auto indices_m = indices.mutable_unchecked<1>();
  auto data_m = data.mutable_unchecked<1>();

  // Uniform row pointers: indptr[r] = r * K
  for (int64_t r = 0; r <= dim; ++r)
    indptr_m(r) = static_cast<StorageIndex>(r * K);

  // Group terms by x_mask once. Then each column can accumulate values for
  // every unique x_mask directly, avoiding per-column sort/merge work.
  std::vector<int64_t> term_x_index(T);
  for (int t = 0; t < T; ++t) {
    auto it = std::lower_bound(unique_xm.begin(), unique_xm.end(), x_masks[t]);
    term_x_index[t] = static_cast<int64_t>(it - unique_xm.begin());
  }

  std::vector<std::vector<int>> terms_by_x(static_cast<size_t>(K));
  for (int t = 0; t < T; ++t)
    terms_by_x[static_cast<size_t>(term_x_index[t])].push_back(t);

  // Single parallel pass over rows. Column indices are not sorted within a row;
  // scipy CSR supports this and can sort lazily when needed.
#pragma omp parallel for schedule(static)
  for (int64_t r = 0; r < dim; ++r) {
    int64_t base = r * K;
    for (int64_t k = 0; k < K; ++k) {
      const int64_t c = r ^ unique_xm[static_cast<size_t>(k)];
      std::complex<double> acc{0.0, 0.0};
      const auto& group = terms_by_x[static_cast<size_t>(k)];
      for (int t : group) {
        int parity = std::popcount(static_cast<uint64_t>(c & z_masks[t]));
        double sign = 1.0 - 2.0 * (parity & 1);
        acc += scaled[t] * sign;
      }
      indices_m(base + k) = static_cast<StorageIndex>(c);
      data_m(base + k) = acc;
    }
  }

  py::object scipy_sparse = py::module_::import("scipy.sparse");
  py::tuple triplet = py::make_tuple(data, indices, indptr);
  py::tuple shape = py::make_tuple(dim, dim);
  return scipy_sparse.attr("csr_matrix")(triplet, shape);
}

void bind_pauli_matrix(py::module_& m) {
  m.def("pauli_to_dense_matrix", &pauli_to_dense_matrix,
        py::arg("pauli_strings"), py::arg("coefficients"),
        R"(Build a dense Hamiltonian matrix from Pauli strings and coefficients.

Args:
    pauli_strings: List of Pauli label strings (characters in {I, X, Y, Z}),
        all of the same length n.
    coefficients: Complex array of coefficients, one per Pauli term.

Returns:
    numpy.ndarray: Dense complex matrix of shape (2**n, 2**n).
)");

  m.def(
      "pauli_to_sparse_matrix", &pauli_to_sparse_matrix,
      py::arg("pauli_strings"), py::arg("coefficients"),
      R"(Build a sparse CSR Hamiltonian matrix from Pauli strings and coefficients.
Args:
    pauli_strings: List of Pauli label strings (characters in {I, X, Y, Z}),
        all of the same length n.
    coefficients: Complex array of coefficients, one per Pauli term.

Returns:
    csr_matrix: Sparse complex matrix of shape (2**n, 2**n).

Raises:
    RuntimeError: If the matrix dimensions exceed int32 index limits.
)");

  m.def("pauli_string_to_masks", &py_pauli_string_to_masks,
        py::arg("pauli_str"),
        R"(Decompose a Pauli label string into bitmasks and phase factor.

X sets a bit in x_mask, Z sets a bit in z_mask, Y sets both.
The cumulative phase factor (i)^(number of Y operators) is also returned.

Args:
    pauli_str: Pauli string (characters in {I, X, Y, Z}), using Little-Endian
        label convention (label[0] = qubit n-1).

Returns:
    (x_mask, z_mask, y_phase) where x_mask and z_mask are integers
        and y_phase is a complex number.
)");

  m.def("pauli_expectation", &pauli_expectation, py::arg("pauli_str"),
        py::arg("psi"),
        R"(Compute the expectation value <psi|P|psi> for a single Pauli string.

Args:
    pauli_str: Pauli label of length n (characters in {I, X, Y, Z}),
        using Little-Endian convention (label[0] = qubit n-1).
    psi: Complex state vector of length 2**n.

Returns:
    Expectation value.
)");
}
