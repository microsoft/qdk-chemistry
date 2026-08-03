// --------------------------------------------------------------------------------------------
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.
// --------------------------------------------------------------------------------------------

#include <algorithm>
#include <cctype>
#include <cmath>
#include <complex>
#include <fstream>
#include <map>
#include <mutex>
#include <nlohmann/json.hpp>
#include <optional>
#include <qdk/chemistry/data/boson_mapping.hpp>
#include <qdk/chemistry/data/bosonic_modes.hpp>
#include <qdk/chemistry/utils/hash_context.hpp>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

namespace qdk::chemistry::data {

namespace detail {

/// Local Pauli terms on a single mode's register, keyed by a base-4 packed
/// word: local qubit k contributes op_k * 4^k with op in {0=I,1=X,2=Y,3=Z}.
using LocalPauliMap = std::unordered_map<std::uint64_t, std::complex<double>>;

/// Dense d x d complex matrix in row-major order.
using LocalMatrix = std::vector<std::complex<double>>;

namespace {

bool is_power_of_two(std::size_t value) {
  return value != 0 && (value & (value - 1)) == 0;
}

std::size_t log2_exact(std::size_t value) {
  std::size_t bits = 0;
  while ((std::size_t{1} << bits) < value) {
    ++bits;
  }
  return bits;
}

/// Accumulate the exact Pauli expansion of coeff * |row><col| into @p out.
///
/// |0><0| = (I+Z)/2, |1><1| = (I-Z)/2, |0><1| = (X+iY)/2, |1><0| = (X-iY)/2.
void accumulate_outer(LocalPauliMap& out, std::uint64_t row, std::uint64_t col,
                      std::complex<double> coeff, std::size_t num_qubits) {
  std::vector<std::pair<std::uint64_t, std::complex<double>>> current{
      {0, coeff}};
  std::vector<std::pair<std::uint64_t, std::complex<double>>> next;
  std::uint64_t stride = 1;
  for (std::size_t k = 0; k < num_qubits; ++k) {
    const bool row_bit = ((row >> k) & 1U) != 0;
    const bool col_bit = ((col >> k) & 1U) != 0;

    // (op_a, factor_a) and (op_b, factor_b) for this qubit.
    std::uint64_t op_a = 0;
    std::complex<double> factor_a{0.5, 0.0};
    std::uint64_t op_b = 3;
    std::complex<double> factor_b{0.5, 0.0};
    if (!row_bit && !col_bit) {
      op_a = 0;  // I
      op_b = 3;  // Z
      factor_b = {0.5, 0.0};
    } else if (row_bit && col_bit) {
      op_a = 0;  // I
      op_b = 3;  // Z
      factor_b = {-0.5, 0.0};
    } else if (!row_bit && col_bit) {
      op_a = 1;  // X
      op_b = 2;  // Y
      factor_b = {0.0, 0.5};
    } else {
      op_a = 1;  // X
      op_b = 2;  // Y
      factor_b = {0.0, -0.5};
    }

    next.clear();
    next.reserve(current.size() * 2);
    for (const auto& [word, value] : current) {
      next.emplace_back(word + op_a * stride, value * factor_a);
      next.emplace_back(word + op_b * stride, value * factor_b);
    }
    current.swap(next);
    stride *= 4;
  }
  for (const auto& [word, value] : current) {
    out[word] += value;
  }
}

/// Exact Pauli decomposition of a dense local matrix.
LocalPauliMap decompose_local(const LocalMatrix& matrix, std::size_t dimension,
                              std::size_t num_qubits,
                              const std::vector<std::uint64_t>& codewords) {
  LocalPauliMap out;
  for (std::size_t row = 0; row < dimension; ++row) {
    for (std::size_t col = 0; col < dimension; ++col) {
      const std::complex<double> value = matrix[row * dimension + col];
      if (std::abs(value) < 1e-15) {
        continue;
      }
      accumulate_outer(out, codewords[row], codewords[col], value, num_qubits);
    }
  }
  return out;
}

/// Exact Pauli decomposition of a diagonal operator via the fast
/// Walsh-Hadamard transform.  The register basis index x carries the value
/// f(level(x)); the transform yields the coefficient of each Z-product.
LocalPauliMap decompose_diagonal(const std::vector<double>& values,
                                 std::size_t num_qubits,
                                 const std::vector<std::uint64_t>& codewords) {
  const std::size_t size = std::size_t{1} << num_qubits;
  std::vector<double> spectrum(size, 0.0);
  for (std::size_t level = 0; level < values.size(); ++level) {
    spectrum[codewords[level]] = values[level];
  }
  for (std::size_t len = 1; len < size; len <<= 1) {
    for (std::size_t i = 0; i < size; i += (len << 1)) {
      for (std::size_t j = i; j < i + len; ++j) {
        const double a = spectrum[j];
        const double b = spectrum[j + len];
        spectrum[j] = a + b;
        spectrum[j + len] = a - b;
      }
    }
  }
  const double scale = 1.0 / static_cast<double>(size);
  LocalPauliMap out;
  for (std::size_t mask = 0; mask < size; ++mask) {
    const double coefficient = spectrum[mask] * scale;
    if (coefficient == 0.0) {
      continue;
    }
    std::uint64_t word = 0;
    std::uint64_t stride = 1;
    for (std::size_t k = 0; k < num_qubits; ++k) {
      if (((mask >> k) & 1U) != 0) {
        word += 3 * stride;  // Z on local qubit k
      }
      stride *= 4;
    }
    out[word] += std::complex<double>(coefficient, 0.0);
  }
  return out;
}

/// Dense matrix of an ordered product of ladder operators on one mode.
/// @p sequence contains 'D' for a creation and 'A' for an annihilation
/// operator, leftmost factor first.
LocalMatrix ladder_sequence_matrix(const std::string& sequence,
                                   std::size_t dimension) {
  LocalMatrix result(dimension * dimension, {0.0, 0.0});
  for (std::size_t i = 0; i < dimension; ++i) {
    result[i * dimension + i] = {1.0, 0.0};
  }
  for (const char factor : sequence) {
    LocalMatrix single(dimension * dimension, {0.0, 0.0});
    for (std::size_t n = 1; n < dimension; ++n) {
      const double amplitude = std::sqrt(static_cast<double>(n));
      if (factor == 'D') {
        single[n * dimension + (n - 1)] = {amplitude, 0.0};
      } else {
        single[(n - 1) * dimension + n] = {amplitude, 0.0};
      }
    }
    LocalMatrix product(dimension * dimension, {0.0, 0.0});
    for (std::size_t i = 0; i < dimension; ++i) {
      for (std::size_t k = 0; k < dimension; ++k) {
        const std::complex<double> left = result[i * dimension + k];
        if (left == std::complex<double>(0.0, 0.0)) {
          continue;
        }
        for (std::size_t j = 0; j < dimension; ++j) {
          product[i * dimension + j] += left * single[k * dimension + j];
        }
      }
    }
    result.swap(product);
  }
  return result;
}

bool is_diagonal(const LocalMatrix& matrix, std::size_t dimension) {
  for (std::size_t row = 0; row < dimension; ++row) {
    for (std::size_t col = 0; col < dimension; ++col) {
      if (row == col) {
        continue;
      }
      if (std::abs(matrix[row * dimension + col]) > 1e-15) {
        return false;
      }
    }
  }
  return true;
}

/// Cached local decomposition of a ladder-operator sequence.  The result
/// depends only on (encoding, dimension, sequence), never on the mode.
const LocalPauliMap& cached_sequence_terms(
    BosonEncoding encoding, std::size_t dimension, std::size_t num_qubits,
    const std::vector<std::uint64_t>& codewords, const std::string& sequence) {
  using Key = std::tuple<std::uint8_t, std::size_t, std::string>;
  static std::mutex mutex;
  static std::map<Key, LocalPauliMap> cache;

  const Key key{static_cast<std::uint8_t>(encoding), dimension, sequence};
  const std::lock_guard<std::mutex> guard(mutex);
  auto it = cache.find(key);
  if (it != cache.end()) {
    return it->second;
  }

  const LocalMatrix matrix = ladder_sequence_matrix(sequence, dimension);
  LocalPauliMap terms;
  if (is_diagonal(matrix, dimension)) {
    std::vector<double> values(dimension, 0.0);
    for (std::size_t n = 0; n < dimension; ++n) {
      values[n] = matrix[n * dimension + n].real();
    }
    terms = decompose_diagonal(values, num_qubits, codewords);
  } else {
    terms = decompose_local(matrix, dimension, num_qubits, codewords);
  }
  return cache.emplace(key, std::move(terms)).first->second;
}

/// Sorted, pruned local terms as (packed word, coefficient) pairs.
std::vector<std::pair<std::uint64_t, std::complex<double>>> sorted_local(
    const LocalPauliMap& terms, double threshold) {
  std::vector<std::pair<std::uint64_t, std::complex<double>>> out;
  out.reserve(terms.size());
  for (const auto& [word, coefficient] : terms) {
    if (std::abs(coefficient) >= threshold) {
      out.emplace_back(word, coefficient);
    }
  }
  std::sort(out.begin(), out.end(), [](const auto& lhs, const auto& rhs) {
    return lhs.first < rhs.first;
  });
  return out;
}

}  // namespace
}  // namespace detail

std::string to_string(BosonEncoding encoding) {
  switch (encoding) {
    case BosonEncoding::StandardBinary:
      return "standard-binary";
    case BosonEncoding::GrayCode:
      return "gray-code";
  }
  throw std::invalid_argument("Unknown BosonEncoding value");
}

BosonEncoding boson_encoding_from_string(const std::string& name) {
  std::string key;
  key.reserve(name.size());
  for (const char c : name) {
    key.push_back(static_cast<char>(
        std::tolower(static_cast<unsigned char>(c == '_' ? '-' : c))));
  }
  if (key == "standard-binary" || key == "sb" || key == "binary" ||
      key == "standardbinary") {
    return BosonEncoding::StandardBinary;
  }
  if (key == "gray-code" || key == "gray" || key == "gc" || key == "graycode") {
    return BosonEncoding::GrayCode;
  }
  throw std::invalid_argument("Unknown boson encoding '" + name +
                              "'; expected 'standard-binary' or 'gray-code'");
}

void BosonMapping::validate_mode_dimension(std::size_t mode,
                                           std::size_t mode_dimension) {
  if (mode_dimension < 2) {
    throw std::invalid_argument(
        "BosonMapping: mode " + std::to_string(mode) + " has local dimension " +
        std::to_string(mode_dimension) +
        ", but a mode dimension must be at least 2 (a mode with fewer than "
        "two levels carries no bosonic degree of freedom).");
  }
  if (!detail::is_power_of_two(mode_dimension)) {
    throw std::invalid_argument(
        "BosonMapping: mode " + std::to_string(mode) +
        " has local dimension d=" + std::to_string(mode_dimension) +
        ", which is not a power of two. A power-of-two dimension makes the "
        "qubit register exactly the code space, so there is no unphysical "
        "subspace and leakage is identically zero. Padding is free: d=3 and "
        "d=4 both cost 32 hopping terms. Pad the basis explicitly, e.g. "
        "BosonicModes::padded_to_power_of_two(num_modes, " +
        std::to_string(mode_dimension) +
        ") or basis.with_padded_dimensions()"
        ", which gives d=" +
        std::to_string(BosonicModes::padded_dimension(mode_dimension)) +
        " for this mode. The basis owns the cutoff and is never padded "
        "implicitly.");
  }
}

void BosonMapping::check_mode(std::size_t mode) const {
  if (mode >= mode_dimensions_.size()) {
    throw std::out_of_range("BosonMapping: mode index " + std::to_string(mode) +
                            " is out of range for " +
                            std::to_string(mode_dimensions_.size()) + " modes");
  }
}

BosonMapping::BosonMapping(std::vector<std::size_t> mode_dimensions,
                           BosonEncoding encoding)
    : mode_dimensions_(std::move(mode_dimensions)),
      num_qubits_(0),
      encoding_(encoding),
      name_(to_string(encoding)) {
  if (mode_dimensions_.empty()) {
    throw std::invalid_argument("BosonMapping: num_modes must be at least 1");
  }
  const std::size_t num_modes = mode_dimensions_.size();
  for (std::size_t i = 0; i < num_modes; ++i) {
    validate_mode_dimension(i, mode_dimensions_[i]);
  }

  // Widths first, then offsets: mode 0 owns the most significant block, so the
  // block of mode i starts above every block with a larger index.
  mode_qubit_counts_.resize(num_modes);
  for (std::size_t i = 0; i < num_modes; ++i) {
    mode_qubit_counts_[i] = detail::log2_exact(mode_dimensions_[i]);
    num_qubits_ += mode_qubit_counts_[i];
  }
  mode_qubit_offsets_.assign(num_modes, 0);
  std::size_t offset = 0;
  for (std::size_t i = num_modes; i-- > 0;) {
    mode_qubit_offsets_[i] = offset;
    offset += mode_qubit_counts_[i];
  }

  codewords_.resize(num_modes);
  levels_.resize(num_modes);
  for (std::size_t i = 0; i < num_modes; ++i) {
    const std::size_t dimension = mode_dimensions_[i];
    codewords_[i].resize(dimension);
    levels_[i].assign(dimension, 0);
    for (std::size_t n = 0; n < dimension; ++n) {
      const auto value = static_cast<std::uint64_t>(n);
      const std::uint64_t code = (encoding_ == BosonEncoding::GrayCode)
                                     ? (value ^ (value >> 1))
                                     : value;
      codewords_[i][n] = code;
      levels_[i][static_cast<std::size_t>(code)] = n;
    }
  }
}

BosonMapping BosonMapping::standard_binary(std::size_t num_modes,
                                           std::size_t mode_dimension) {
  return for_encoding(num_modes, mode_dimension, BosonEncoding::StandardBinary);
}

BosonMapping BosonMapping::gray_code(std::size_t num_modes,
                                     std::size_t mode_dimension) {
  return for_encoding(num_modes, mode_dimension, BosonEncoding::GrayCode);
}

BosonMapping BosonMapping::for_encoding(std::size_t num_modes,
                                        std::size_t mode_dimension,
                                        BosonEncoding encoding) {
  if (num_modes == 0) {
    throw std::invalid_argument("BosonMapping: num_modes must be at least 1");
  }
  return BosonMapping(std::vector<std::size_t>(num_modes, mode_dimension),
                      encoding);
}

BosonMapping BosonMapping::for_basis(const BosonicModes& modes,
                                     BosonEncoding encoding) {
  // Read the cutoff per mode; the mapping never owns or duplicates it, and
  // never assumes the dimensions are homogeneous.
  std::vector<std::size_t> dimensions;
  dimensions.reserve(modes.num_modes());
  for (std::size_t i = 0; i < modes.num_modes(); ++i) {
    dimensions.push_back(modes.mode_dimension(i));
  }
  return BosonMapping(std::move(dimensions), encoding);
}

std::size_t BosonMapping::mode_dimension(std::size_t mode) const {
  check_mode(mode);
  return mode_dimensions_[mode];
}

std::optional<std::size_t> BosonMapping::uniform_dimension() const {
  if (mode_dimensions_.empty()) {
    return std::nullopt;
  }
  const std::size_t first = mode_dimensions_.front();
  for (const std::size_t dimension : mode_dimensions_) {
    if (dimension != first) {
      return std::nullopt;
    }
  }
  return first;
}

std::size_t BosonMapping::max_occupation(std::size_t mode) const {
  return mode_dimension(mode) - 1;
}

std::size_t BosonMapping::qubits_per_mode(std::size_t mode) const {
  check_mode(mode);
  return mode_qubit_counts_[mode];
}

std::uint64_t BosonMapping::codeword(std::size_t mode,
                                     std::size_t level) const {
  check_mode(mode);
  if (level >= mode_dimensions_[mode]) {
    throw std::out_of_range(
        "BosonMapping: occupation level " + std::to_string(level) +
        " is out of range for mode " + std::to_string(mode) +
        " with dimension " + std::to_string(mode_dimensions_[mode]));
  }
  return codewords_[mode][level];
}

std::size_t BosonMapping::level(std::size_t mode,
                                std::uint64_t codeword) const {
  check_mode(mode);
  if (codeword >= mode_dimensions_[mode]) {
    throw std::out_of_range(
        "BosonMapping: codeword " + std::to_string(codeword) +
        " is out of range for mode " + std::to_string(mode) +
        " with dimension " + std::to_string(mode_dimensions_[mode]));
  }
  return levels_[mode][static_cast<std::size_t>(codeword)];
}

const std::vector<std::uint64_t>& BosonMapping::codeword_table(
    std::size_t mode) const {
  check_mode(mode);
  return codewords_[mode];
}

std::vector<double> BosonMapping::isometry(std::size_t mode) const {
  check_mode(mode);
  const std::size_t dimension = mode_dimensions_[mode];
  const std::size_t rows = std::size_t{1} << mode_qubit_counts_[mode];
  if (rows * dimension > (std::size_t{1} << 22)) {
    throw std::overflow_error(
        "BosonMapping::isometry: the dense isometry would have " +
        std::to_string(rows * dimension) +
        " entries; use codeword_table(mode) instead");
  }
  std::vector<double> matrix(rows * dimension, 0.0);
  for (std::size_t n = 0; n < dimension; ++n) {
    matrix[static_cast<std::size_t>(codewords_[mode][n]) * dimension + n] = 1.0;
  }
  return matrix;
}

std::uint64_t BosonMapping::global_qubit(std::size_t mode,
                                         std::size_t local) const {
  return static_cast<std::uint64_t>(mode_qubit_offsets_[mode] + local);
}

std::vector<std::uint64_t> BosonMapping::mode_qubits(std::size_t mode) const {
  check_mode(mode);
  std::vector<std::uint64_t> qubits(mode_qubit_counts_[mode]);
  for (std::size_t k = 0; k < mode_qubit_counts_[mode]; ++k) {
    qubits[k] = global_qubit(mode, k);
  }
  return qubits;
}

void BosonMapping::validate_basis(const BosonicModes& modes) const {
  if (modes.num_modes() != mode_dimensions_.size()) {
    throw std::invalid_argument("BosonMapping: the basis has " +
                                std::to_string(modes.num_modes()) +
                                " modes but the mapping was built for " +
                                std::to_string(mode_dimensions_.size()));
  }
  for (std::size_t i = 0; i < mode_dimensions_.size(); ++i) {
    if (modes.mode_dimension(i) != mode_dimensions_[i]) {
      throw std::invalid_argument(
          "BosonMapping: mode " + std::to_string(i) +
          " of the basis has dimension " +
          std::to_string(modes.mode_dimension(i)) +
          " but the mapping was built for dimension " +
          std::to_string(mode_dimensions_[i]) +
          ". The occupation cutoff lives on the basis; rebuild the mapping "
          "with BosonMapping::for_basis(...) so the two agree.");
    }
  }
}

BosonPauliTerms BosonMapping::ladder_product(
    const std::vector<std::pair<std::size_t, bool>>& factors,
    double threshold) const {
  // Bosonic operators on different modes commute, so the factors can be
  // grouped by mode as long as their relative order within a mode is kept.
  std::map<std::size_t, std::string> sequences;
  for (const auto& [mode, is_creation] : factors) {
    check_mode(mode);
    sequences[mode].push_back(is_creation ? 'D' : 'A');
  }

  // Walk modes in decreasing index order so that global qubit indices are
  // produced in increasing order and each word is sorted by construction.
  BosonPauliTerms result;
  result.emplace_back(std::complex<double>(1.0, 0.0), SparsePauliWord{});
  for (auto it = sequences.rbegin(); it != sequences.rend(); ++it) {
    const std::size_t mode = it->first;
    const std::size_t mode_qubits_count = mode_qubit_counts_[mode];
    const auto& local = detail::cached_sequence_terms(
        encoding_, mode_dimensions_[mode], mode_qubits_count, codewords_[mode],
        it->second);
    const auto local_terms = detail::sorted_local(local, threshold);

    BosonPauliTerms next;
    next.reserve(result.size() * local_terms.size());
    for (const auto& [coefficient, word] : result) {
      for (const auto& [packed, local_coefficient] : local_terms) {
        SparsePauliWord extended = word;
        std::uint64_t remaining = packed;
        for (std::size_t k = 0; k < mode_qubits_count; ++k) {
          const auto op = static_cast<std::uint8_t>(remaining & 3U);
          remaining >>= 2;
          if (op != 0) {
            extended.emplace_back(global_qubit(mode, k), op);
          }
        }
        next.emplace_back(coefficient * local_coefficient, std::move(extended));
      }
    }
    result.swap(next);
  }

  BosonPauliTerms pruned;
  pruned.reserve(result.size());
  for (auto& term : result) {
    if (std::abs(term.first) >= threshold) {
      pruned.push_back(std::move(term));
    }
  }
  std::sort(pruned.begin(), pruned.end(), [](const auto& lhs, const auto& rhs) {
    return lhs.second < rhs.second;
  });
  return pruned;
}

BosonPauliTerms BosonMapping::diagonal(const std::vector<double>& values,
                                       std::size_t mode,
                                       double threshold) const {
  check_mode(mode);
  if (values.size() != mode_dimensions_[mode]) {
    throw std::invalid_argument("BosonMapping::diagonal: expected " +
                                std::to_string(mode_dimensions_[mode]) +
                                " values (one per occupation level of mode " +
                                std::to_string(mode) + ") but got " +
                                std::to_string(values.size()));
  }
  const std::size_t mode_qubits_count = mode_qubit_counts_[mode];
  const auto local =
      detail::decompose_diagonal(values, mode_qubits_count, codewords_[mode]);
  const auto local_terms = detail::sorted_local(local, threshold);

  BosonPauliTerms result;
  result.reserve(local_terms.size());
  for (const auto& [packed, coefficient] : local_terms) {
    SparsePauliWord word;
    std::uint64_t remaining = packed;
    for (std::size_t k = 0; k < mode_qubits_count; ++k) {
      const auto op = static_cast<std::uint8_t>(remaining & 3U);
      remaining >>= 2;
      if (op != 0) {
        word.emplace_back(global_qubit(mode, k), op);
      }
    }
    result.emplace_back(coefficient, std::move(word));
  }
  std::sort(result.begin(), result.end(), [](const auto& lhs, const auto& rhs) {
    return lhs.second < rhs.second;
  });
  return result;
}

BosonPauliTerms BosonMapping::number(std::size_t mode, double threshold) const {
  const std::size_t dimension = mode_dimension(mode);
  std::vector<double> values(dimension);
  for (std::size_t n = 0; n < dimension; ++n) {
    values[n] = static_cast<double>(n);
  }
  return diagonal(values, mode, threshold);
}

BosonPauliTerms BosonMapping::number_squared(std::size_t mode,
                                             double threshold) const {
  const std::size_t dimension = mode_dimension(mode);
  std::vector<double> values(dimension);
  for (std::size_t n = 0; n < dimension; ++n) {
    values[n] = static_cast<double>(n) * static_cast<double>(n);
  }
  return diagonal(values, mode, threshold);
}

BosonPauliTerms BosonMapping::number_times_number_minus_one(
    std::size_t mode, double threshold) const {
  const std::size_t dimension = mode_dimension(mode);
  std::vector<double> values(dimension);
  for (std::size_t n = 0; n < dimension; ++n) {
    values[n] = static_cast<double>(n) * (static_cast<double>(n) - 1.0);
  }
  return diagonal(values, mode, threshold);
}

BosonPauliTerms BosonMapping::annihilation(std::size_t mode,
                                           double threshold) const {
  return ladder_product({{mode, false}}, threshold);
}

BosonPauliTerms BosonMapping::creation(std::size_t mode,
                                       double threshold) const {
  return ladder_product({{mode, true}}, threshold);
}

std::string BosonMapping::get_summary() const {
  std::ostringstream ss;
  ss << "BosonMapping '" << name_ << "'";
  ss << "\n  Modes: " << num_modes();
  const auto uniform = uniform_dimension();
  if (uniform.has_value()) {
    ss << "\n  Local dimension d: " << *uniform
       << " (n_max = " << (*uniform - 1) << ")";
    ss << "\n  Qubits per mode: " << mode_qubit_counts_.front();
  } else {
    ss << "\n  Local dimensions d: [";
    for (std::size_t i = 0; i < mode_dimensions_.size(); ++i) {
      ss << (i == 0 ? "" : ", ") << mode_dimensions_[i];
    }
    ss << "] (per mode)";
    ss << "\n  Qubits per mode: [";
    for (std::size_t i = 0; i < mode_qubit_counts_.size(); ++i) {
      ss << (i == 0 ? "" : ", ") << mode_qubit_counts_[i];
    }
    ss << "]";
  }
  ss << "\n  Qubits: " << num_qubits();
  return ss.str();
}

nlohmann::json BosonMapping::to_json() const {
  // The cutoff is written per mode so that a heterogeneous mapping round-trips
  // without a schema change; num_modes is redundant but kept for readability.
  return nlohmann::json{{"version", SERIALIZATION_VERSION},
                        {"num_modes", mode_dimensions_.size()},
                        {"mode_dimensions", mode_dimensions_},
                        {"encoding", to_string(encoding_)}};
}

BosonMapping BosonMapping::from_json(const nlohmann::json& data) {
  const auto encoding =
      boson_encoding_from_string(data.at("encoding").get<std::string>());

  // The schema is an array of one dimension per mode, always. num_modes is
  // written alongside it for readability and is cross-checked when present.
  if (!data.contains("mode_dimensions") ||
      !data["mode_dimensions"].is_array()) {
    throw std::invalid_argument(
        "BosonMapping: JSON field mode_dimensions must be an array holding "
        "one local Fock-space dimension per mode");
  }
  auto dimensions = data["mode_dimensions"].get<std::vector<std::size_t>>();
  if (data.contains("num_modes")) {
    const auto num_modes = data.at("num_modes").get<std::size_t>();
    if (num_modes != dimensions.size()) {
      throw std::invalid_argument(
          "BosonMapping: num_modes is " + std::to_string(num_modes) + " but " +
          std::to_string(dimensions.size()) + " mode dimensions were given");
    }
  }
  return BosonMapping(std::move(dimensions), encoding);
}

void BosonMapping::to_file(const std::string& filename,
                           const std::string& type) const {
  if (type == "json") {
    to_json_file(filename);
  } else if (type == "hdf5" || type == "h5") {
    to_hdf5_file(filename);
  } else {
    throw std::invalid_argument("Unsupported format type: " + type);
  }
}

void BosonMapping::to_json_file(const std::string& filename) const {
  std::ofstream file(filename);
  if (!file) {
    throw std::runtime_error("Unable to open file for writing: " + filename);
  }
  file << to_json().dump(2);
}

BosonMapping BosonMapping::from_json_file(const std::string& filename) {
  std::ifstream file(filename);
  if (!file) {
    throw std::runtime_error("Unable to open file for reading: " + filename);
  }
  nlohmann::json data;
  file >> data;
  return from_json(data);
}

void BosonMapping::to_hdf5(H5::Group& group) const {
  const std::string json = to_json().dump();
  group.createAttribute("json", H5::StrType(0, H5T_VARIABLE), H5::DataSpace())
      .write(H5::StrType(0, H5T_VARIABLE), json);
}

BosonMapping BosonMapping::from_hdf5(H5::Group& group) {
  std::string json;
  group.openAttribute("json").read(H5::StrType(0, H5T_VARIABLE), json);
  return from_json(nlohmann::json::parse(json));
}

void BosonMapping::to_hdf5_file(const std::string& filename) const {
  H5::H5File file(filename, H5F_ACC_TRUNC);
  H5::Group root = file.openGroup("/");
  to_hdf5(root);
}

BosonMapping BosonMapping::from_hdf5_file(const std::string& filename) {
  H5::H5File file(filename, H5F_ACC_RDONLY);
  H5::Group root = file.openGroup("/");
  return from_hdf5(root);
}

BosonMapping BosonMapping::from_file(const std::string& filename,
                                     const std::string& type) {
  if (type == "json") {
    return from_json_file(filename);
  }
  if (type == "hdf5" || type == "h5") {
    return from_hdf5_file(filename);
  }
  throw std::invalid_argument("Unsupported format type: " + type);
}

void BosonMapping::hash_update(qdk::chemistry::utils::HashContext& ctx) const {
  qdk::chemistry::utils::hash_value(ctx, std::string("boson_mapping"));
  qdk::chemistry::utils::hash_value(
      ctx, static_cast<std::uint64_t>(mode_dimensions_.size()));
  for (const std::size_t dimension : mode_dimensions_) {
    qdk::chemistry::utils::hash_value(ctx,
                                      static_cast<std::uint64_t>(dimension));
  }
  qdk::chemistry::utils::hash_value(ctx, to_string(encoding_));
}

}  // namespace qdk::chemistry::data
