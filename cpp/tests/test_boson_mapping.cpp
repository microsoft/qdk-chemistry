// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <gtest/gtest.h>
#include <spdlog/sinks/ringbuffer_sink.h>

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <qdk/chemistry/data/boson_mapping.hpp>
#include <qdk/chemistry/data/bosonic_modes.hpp>
#include <qdk/chemistry/data/hamiltonian.hpp>
#include <qdk/chemistry/data/lattice_graph.hpp>
#include <qdk/chemistry/data/pauli_operator.hpp>
#include <qdk/chemistry/utils/logger.hpp>
#include <qdk/chemistry/utils/model_hamiltonians.hpp>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "ut_common.hpp"

using namespace qdk::chemistry::data;
namespace mh = qdk::chemistry::utils::model_hamiltonians;

namespace {

/// Reference tolerance for every exact-algebra comparison in this file.
constexpr double kTol = 1e-10;

/// Pauli terms rendered as a label -> coefficient map, using the
/// QubitOperator convention (qubit 0 is the rightmost character).
std::map<std::string, std::complex<double>> as_labels(
    const BosonMapResult& result, std::size_t num_qubits) {
  std::map<std::string, std::complex<double>> out;
  for (std::size_t i = 0; i < result.words.size(); ++i) {
    out[sparse_pauli_word_to_label(result.words[i],
                                   static_cast<std::uint64_t>(num_qubits))] =
        result.coefficients[i];
  }
  return out;
}

std::map<std::string, std::complex<double>> as_labels(
    const BosonPauliTerms& terms, std::size_t num_qubits) {
  std::map<std::string, std::complex<double>> out;
  for (const auto& [coefficient, word] : terms) {
    out[sparse_pauli_word_to_label(
        word, static_cast<std::uint64_t>(num_qubits))] = coefficient;
  }
  return out;
}

/// Dense matrix of a Pauli sum on @p num_qubits qubits.
///
/// Convention: qubit k is bit k of the basis index; |0>/|1> are the +1/-1
/// eigenvectors of Z_k.  X flips the bit, Y|0> = i|1>, Y|1> = -i|0>.
Eigen::MatrixXcd to_matrix(const std::vector<SparsePauliWord>& words,
                           const std::vector<std::complex<double>>& coeffs,
                           std::size_t num_qubits) {
  const Eigen::Index dim = Eigen::Index{1} << num_qubits;
  Eigen::MatrixXcd matrix = Eigen::MatrixXcd::Zero(dim, dim);
  for (std::size_t term = 0; term < words.size(); ++term) {
    for (Eigen::Index col = 0; col < dim; ++col) {
      auto row = static_cast<std::uint64_t>(col);
      std::complex<double> amplitude = coeffs[term];
      for (const auto& [qubit, op] : words[term]) {
        const std::uint64_t bit = (row >> qubit) & 1U;
        switch (op) {
          case 1:  // X
            row ^= (std::uint64_t{1} << qubit);
            break;
          case 2:  // Y
            row ^= (std::uint64_t{1} << qubit);
            amplitude *= (bit == 0) ? std::complex<double>(0.0, 1.0)
                                    : std::complex<double>(0.0, -1.0);
            break;
          case 3:  // Z
            if (bit != 0) {
              amplitude = -amplitude;
            }
            break;
          default:
            break;
        }
      }
      matrix(static_cast<Eigen::Index>(row), col) += amplitude;
    }
  }
  return matrix;
}

/// All occupation configurations (n_0, ..., n_{L-1}) in row-major order,
/// matching the global index sum_i n_i d^{L-1-i}.
std::vector<std::vector<std::size_t>> occupation_configs(std::size_t modes,
                                                         std::size_t dim) {
  std::vector<std::vector<std::size_t>> configs;
  std::vector<std::size_t> current(modes, 0);
  while (true) {
    configs.push_back(current);
    std::size_t position = modes;
    while (position > 0) {
      --position;
      if (++current[position] < dim) {
        break;
      }
      current[position] = 0;
      if (position == 0) {
        return configs;
      }
    }
  }
}

/// Independent reference builder #1: the Bose-Hubbard Hamiltonian assembled
/// directly in the truncated occupation basis.
Eigen::MatrixXd occupation_basis_hamiltonian(std::size_t modes, std::size_t dim,
                                             double t, double u, double mu) {
  const auto configs = occupation_configs(modes, dim);
  std::map<std::vector<std::size_t>, Eigen::Index> index;
  for (Eigen::Index i = 0; i < static_cast<Eigen::Index>(configs.size()); ++i) {
    index[configs[static_cast<std::size_t>(i)]] = i;
  }
  Eigen::MatrixXd matrix =
      Eigen::MatrixXd::Zero(static_cast<Eigen::Index>(configs.size()),
                            static_cast<Eigen::Index>(configs.size()));
  for (const auto& config : configs) {
    const Eigen::Index col = index.at(config);
    double diagonal = 0.0;
    for (std::size_t i = 0; i < modes; ++i) {
      const auto n = static_cast<double>(config[i]);
      diagonal += 0.5 * u * n * (n - 1.0) - mu * n;
    }
    matrix(col, col) += diagonal;
    for (std::size_t i = 0; i + 1 < modes; ++i) {
      const std::size_t j = i + 1;
      // b_i^dag b_j and b_j^dag b_i, both with amplitude -t.
      const std::vector<std::pair<std::size_t, std::size_t>> moves{{j, i},
                                                                   {i, j}};
      for (const auto& [from, to] : moves) {
        if (config[from] == 0 || config[to] + 1 > dim - 1) {
          continue;
        }
        auto target = config;
        const double amplitude = std::sqrt(static_cast<double>(config[from])) *
                                 std::sqrt(static_cast<double>(config[to] + 1));
        target[from] -= 1;
        target[to] += 1;
        matrix(index.at(target), col) += -t * amplitude;
      }
    }
  }
  return matrix;
}

Eigen::MatrixXd kron(const Eigen::MatrixXd& a, const Eigen::MatrixXd& b) {
  Eigen::MatrixXd out =
      Eigen::MatrixXd::Zero(a.rows() * b.rows(), a.cols() * b.cols());
  for (Eigen::Index i = 0; i < a.rows(); ++i) {
    for (Eigen::Index j = 0; j < a.cols(); ++j) {
      out.block(i * b.rows(), j * b.cols(), b.rows(), b.cols()) = a(i, j) * b;
    }
  }
  return out;
}

/// Independent reference builder #2: Kronecker products of single-mode
/// operators.  Mode 0 is the leftmost (most significant) factor.
Eigen::MatrixXd kronecker_hamiltonian(std::size_t modes, std::size_t dim,
                                      double t, double u, double mu) {
  const auto d = static_cast<Eigen::Index>(dim);
  Eigen::MatrixXd identity = Eigen::MatrixXd::Identity(d, d);
  Eigen::MatrixXd annihilate = Eigen::MatrixXd::Zero(d, d);
  Eigen::MatrixXd number = Eigen::MatrixXd::Zero(d, d);
  for (Eigen::Index n = 1; n < d; ++n) {
    annihilate(n - 1, n) = std::sqrt(static_cast<double>(n));
  }
  for (Eigen::Index n = 0; n < d; ++n) {
    number(n, n) = static_cast<double>(n);
  }
  const Eigen::MatrixXd create = annihilate.transpose();
  const Eigen::MatrixXd on_site =
      0.5 * u * number * (number - Eigen::MatrixXd::Identity(d, d)) -
      mu * number;

  auto embed = [&](const std::map<std::size_t, Eigen::MatrixXd>& factors) {
    Eigen::MatrixXd out = Eigen::MatrixXd::Ones(1, 1);
    for (std::size_t i = 0; i < modes; ++i) {
      auto it = factors.find(i);
      out = kron(out, it == factors.end() ? identity : it->second);
    }
    return out;
  };

  Eigen::MatrixXd total = Eigen::MatrixXd::Zero(1, 1);
  {
    Eigen::Index full = 1;
    for (std::size_t i = 0; i < modes; ++i) {
      full *= d;
    }
    total = Eigen::MatrixXd::Zero(full, full);
  }
  for (std::size_t i = 0; i < modes; ++i) {
    total += embed({{i, on_site}});
  }
  for (std::size_t i = 0; i + 1 < modes; ++i) {
    total += -t * embed({{i, create}, {i + 1, annihilate}});
    total += -t * embed({{i + 1, create}, {i, annihilate}});
  }
  return total;
}

/// Global register index of an occupation configuration under a mapping.
std::size_t register_index(const BosonMapping& mapping,
                           const std::vector<std::size_t>& config) {
  std::size_t index = 0;
  const std::size_t modes = mapping.num_modes();
  // Written as if the dimensions already varied: mode i's block starts above
  // every block with a larger index.
  std::size_t shift = 0;
  for (std::size_t i = modes; i-- > 0;) {
    index |= static_cast<std::size_t>(mapping.codeword(i, config[i])) << shift;
    shift += mapping.qubits_per_mode(i);
  }
  return index;
}

qdk::chemistry::data::LatticeGraph chain(std::size_t sites) {
  return qdk::chemistry::data::LatticeGraph::chain(
      static_cast<std::uint64_t>(sites));
}

}  // namespace

// ---------------------------------------------------------------------------
// Encoding / isometry
// ---------------------------------------------------------------------------

TEST(BosonMapping, StandardBinaryAndGrayCodewords) {
  const auto sb = BosonMapping::standard_binary(1, 8);
  EXPECT_EQ(sb.qubits_per_mode(0), 3u);
  EXPECT_EQ(sb.num_qubits(), 3u);
  EXPECT_EQ(sb.encoding(), BosonEncoding::StandardBinary);
  EXPECT_EQ(sb.name(), "standard-binary");
  for (std::size_t n = 0; n < 8; ++n) {
    EXPECT_EQ(sb.codeword(0, n), n);
    EXPECT_EQ(sb.level(0, static_cast<std::uint64_t>(n)), n);
  }

  const auto gc = BosonMapping::gray_code(1, 8);
  const std::vector<std::uint64_t> expected{0, 1, 3, 2, 6, 7, 5, 4};
  for (std::size_t n = 0; n < 8; ++n) {
    EXPECT_EQ(gc.codeword(0, n), expected[n]) << "level " << n;
    EXPECT_EQ(gc.level(0, expected[n]), n);
  }
  // Gray property: adjacent levels differ in exactly one bit.
  for (std::size_t n = 1; n < 8; ++n) {
    const std::uint64_t diff = gc.codeword(0, n) ^ gc.codeword(0, n - 1);
    EXPECT_EQ(diff & (diff - 1), 0u);
    EXPECT_NE(diff, 0u);
  }
}

TEST(BosonMapping, IsometryIsAPermutationAtPowerOfTwoCutoff) {
  for (const auto encoding :
       {BosonEncoding::StandardBinary, BosonEncoding::GrayCode}) {
    const auto mapping = BosonMapping::for_encoding(1, 8, encoding);
    const auto v = mapping.isometry(0);
    ASSERT_EQ(v.size(), 8u * 8u);
    // Exactly one 1 per row and per column: the code space is the whole
    // register, so there is no unphysical subspace and leakage is zero.
    for (std::size_t row = 0; row < 8; ++row) {
      double row_sum = 0.0;
      double col_sum = 0.0;
      for (std::size_t col = 0; col < 8; ++col) {
        row_sum += v[row * 8 + col];
        col_sum += v[col * 8 + row];
      }
      EXPECT_DOUBLE_EQ(row_sum, 1.0);
      EXPECT_DOUBLE_EQ(col_sum, 1.0);
    }
  }
}

TEST(BosonMapping, ModeQubitLayoutPutsModeZeroInTheMostSignificantBlock) {
  const auto mapping = BosonMapping::standard_binary(3, 4);
  EXPECT_EQ(mapping.num_qubits(), 6u);
  EXPECT_EQ(mapping.mode_qubits(0), (std::vector<std::uint64_t>{4, 5}));
  EXPECT_EQ(mapping.mode_qubits(1), (std::vector<std::uint64_t>{2, 3}));
  EXPECT_EQ(mapping.mode_qubits(2), (std::vector<std::uint64_t>{0, 1}));
  EXPECT_THROW(mapping.mode_qubits(3), std::out_of_range);
}

TEST(BosonMapping, HeterogeneousCutoffsLayOutBlocksByPerModeWidth) {
  // Phase 1 has no public heterogeneous constructor, but the mapping reads the
  // cutoff per mode from the basis, so a heterogeneous basis (which only
  // deserialization can produce today) must lay out correctly.  This pins the
  // generalized offset rule sum_{j>i} nq(j) that replaces (L-1-i)*nq.
  BosonicModes uniform(3, 4);
  auto json = uniform.to_json();
  json["mode_dimensions"] = std::vector<std::size_t>{4u, 8u, 2u};
  auto modes = BosonicModes::from_json(json);
  ASSERT_NE(modes, nullptr);

  const auto mapping = BosonMapping::for_basis(*modes);
  EXPECT_EQ(mapping.mode_dimensions(), (std::vector<std::size_t>{4u, 8u, 2u}));
  EXPECT_FALSE(mapping.uniform_dimension().has_value());
  EXPECT_EQ(mapping.qubits_per_mode(0), 2u);
  EXPECT_EQ(mapping.qubits_per_mode(1), 3u);
  EXPECT_EQ(mapping.qubits_per_mode(2), 1u);
  EXPECT_EQ(mapping.num_qubits(), 6u);
  // Mode 0 stays the most significant block: offsets are 4, 1 and 0.
  EXPECT_EQ(mapping.mode_qubits(0), (std::vector<std::uint64_t>{4, 5}));
  EXPECT_EQ(mapping.mode_qubits(1), (std::vector<std::uint64_t>{1, 2, 3}));
  EXPECT_EQ(mapping.mode_qubits(2), (std::vector<std::uint64_t>{0}));
  EXPECT_NO_THROW(mapping.validate_basis(*modes));

  // Each mode's operator primitives use that mode's own dimension.
  EXPECT_EQ(mapping.number(1).size(), 4u);
  EXPECT_EQ(mapping.number(2).size(), 2u);
  EXPECT_EQ(mapping.annihilation(1).size(), 3u * 8u);
  EXPECT_EQ(mapping.annihilation(2).size(), 1u * 2u);
  EXPECT_TRUE(mapping.number_times_number_minus_one(2).empty());
  EXPECT_EQ(mapping.codeword_table(1).size(), 8u);
  EXPECT_EQ(mapping.isometry(2).size(), 4u);
  EXPECT_THROW(mapping.diagonal(std::vector<double>(4, 1.0), 1),
               std::invalid_argument);

  // A non-power-of-two mode is rejected by index, not silently padded.
  json["mode_dimensions"] = std::vector<std::size_t>{4u, 3u, 2u};
  auto ragged = BosonicModes::from_json(json);
  try {
    BosonMapping::for_basis(*ragged);
    FAIL() << "expected std::invalid_argument";
  } catch (const std::invalid_argument& e) {
    const std::string message = e.what();
    EXPECT_NE(message.find("mode 1"), std::string::npos) << message;
    EXPECT_NE(message.find("d=3"), std::string::npos) << message;
    EXPECT_NE(message.find("d=4"), std::string::npos) << message;
  }
}

TEST(BosonMapping, RejectsNonPowerOfTwoCutoffWithAnActionableMessage) {
  try {
    BosonMapping::standard_binary(2, 3);
    FAIL() << "expected std::invalid_argument";
  } catch (const std::invalid_argument& e) {
    const std::string message = e.what();
    // The message must be actionable: which mode, what it found, the next
    // power of two, and that padding is free.
    EXPECT_NE(message.find("mode 0"), std::string::npos) << message;
    EXPECT_NE(message.find("d=3"), std::string::npos) << message;
    EXPECT_NE(message.find("power of two"), std::string::npos) << message;
    EXPECT_NE(message.find("d=4"), std::string::npos) << message;
    EXPECT_NE(message.find("32 hopping terms"), std::string::npos) << message;
    EXPECT_NE(message.find("padded_to_power_of_two"), std::string::npos)
        << message;
  }
  EXPECT_THROW(BosonMapping::standard_binary(2, 6), std::invalid_argument);
  EXPECT_THROW(BosonMapping::standard_binary(0, 4), std::invalid_argument);
  EXPECT_THROW(BosonMapping::standard_binary(2, 1), std::invalid_argument);
}

TEST(BosonMapping, ForBasisReadsTheCutoffFromTheBasis) {
  BosonicModes modes(3, 4);
  const auto mapping = BosonMapping::for_basis(modes);
  EXPECT_EQ(mapping.num_modes(), 3u);
  EXPECT_EQ(mapping.mode_dimension(0), 4u);
  EXPECT_EQ(mapping.mode_dimensions(), (std::vector<std::size_t>{4u, 4u, 4u}));
  ASSERT_TRUE(mapping.uniform_dimension().has_value());
  EXPECT_EQ(*mapping.uniform_dimension(), 4u);
  EXPECT_NO_THROW(mapping.validate_basis(modes));

  // A mismatched cutoff is a hard error, never silently-wrong physics.
  BosonicModes wrong_dimension(3, 8);
  EXPECT_THROW(mapping.validate_basis(wrong_dimension), std::invalid_argument);
  BosonicModes wrong_count(2, 4);
  EXPECT_THROW(mapping.validate_basis(wrong_count), std::invalid_argument);

  BosonicModes not_mappable(2, 3);
  EXPECT_THROW(BosonMapping::for_basis(not_mappable), std::invalid_argument);
}

// ---------------------------------------------------------------------------
// Single-mode operator primitives (report section 1.1)
// ---------------------------------------------------------------------------

TEST(BosonMapping, NumberOperatorMatchesTheClosedForm) {
  // n = (3/2) I - (1/2) Z_0 - Z_1 for d = 4 standard binary.
  const auto mapping = BosonMapping::standard_binary(1, 4);
  const auto terms = as_labels(mapping.number(0), 2);
  ASSERT_EQ(terms.size(), 3u);
  EXPECT_NEAR(terms.at("II").real(), 1.5, kTol);
  EXPECT_NEAR(terms.at("IZ").real(), -0.5, kTol);
  EXPECT_NEAR(terms.at("ZI").real(), -1.0, kTol);

  // Every coefficient of a diagonal operator is real.
  for (const auto& [label, coefficient] : terms) {
    EXPECT_NEAR(coefficient.imag(), 0.0, kTol) << label;
  }

  // Weight-1 with nq + 1 terms at any power-of-two d.
  for (const std::size_t d : {2u, 4u, 8u, 16u, 32u}) {
    const auto wide = BosonMapping::standard_binary(1, d);
    const auto image = wide.number(0);
    EXPECT_EQ(image.size(), wide.qubits_per_mode(0) + 1);
    for (const auto& [coefficient, word] : image) {
      EXPECT_LE(word.size(), 1u);
    }
  }
}

TEST(BosonMapping, NumberTimesNumberMinusOneMatchesTheClosedForm) {
  // d = 4: n(n-1) = 2 I - Z_0 - 2 Z_1 + Z_0 Z_1
  const auto four = BosonMapping::standard_binary(1, 4);
  const auto terms4 = as_labels(four.number_times_number_minus_one(0), 2);
  ASSERT_EQ(terms4.size(), 4u);
  EXPECT_NEAR(terms4.at("II").real(), 2.0, kTol);
  EXPECT_NEAR(terms4.at("IZ").real(), -1.0, kTol);
  EXPECT_NEAR(terms4.at("ZI").real(), -2.0, kTol);
  EXPECT_NEAR(terms4.at("ZZ").real(), 1.0, kTol);

  // d = 8: 14 I - 3 Z_0 - 6 Z_1 - 12 Z_2 + Z_0 Z_1 + 2 Z_0 Z_2 + 4 Z_1 Z_2
  const auto eight = BosonMapping::standard_binary(1, 8);
  const auto terms8 = as_labels(eight.number_times_number_minus_one(0), 3);
  ASSERT_EQ(terms8.size(), 7u);
  EXPECT_NEAR(terms8.at("III").real(), 14.0, kTol);
  EXPECT_NEAR(terms8.at("IIZ").real(), -3.0, kTol);
  EXPECT_NEAR(terms8.at("IZI").real(), -6.0, kTol);
  EXPECT_NEAR(terms8.at("ZII").real(), -12.0, kTol);
  EXPECT_NEAR(terms8.at("IZZ").real(), 1.0, kTol);
  EXPECT_NEAR(terms8.at("ZIZ").real(), 2.0, kTol);
  EXPECT_NEAR(terms8.at("ZZI").real(), 4.0, kTol);

  // Hard-core bosons: n(n-1) vanishes identically at d = 2.
  const auto two = BosonMapping::standard_binary(1, 2);
  EXPECT_TRUE(two.number_times_number_minus_one(0).empty());

  // 1 + nq(nq+1)/2 terms, weight <= 2, at every power-of-two d.
  for (const std::size_t d : {4u, 8u, 16u, 32u}) {
    const auto wide = BosonMapping::standard_binary(1, d);
    const auto image = wide.number_times_number_minus_one(0);
    const std::size_t nq = wide.qubits_per_mode(0);
    EXPECT_EQ(image.size(), 1u + nq * (nq + 1) / 2);
    for (const auto& [coefficient, word] : image) {
      EXPECT_LE(word.size(), 2u);
    }
  }
}

TEST(BosonMapping, AnnihilationOperatorMatchesTheClosedForm) {
  // d = 4 standard binary: b expands into exactly 8 Pauli strings.
  const auto mapping = BosonMapping::standard_binary(1, 4);
  const auto terms = as_labels(mapping.annihilation(0), 2);
  ASSERT_EQ(terms.size(), 8u);
  const double a = (1.0 + std::sqrt(3.0)) / 4.0;
  const double b = (1.0 - std::sqrt(3.0)) / 4.0;
  const double c = std::sqrt(2.0) / 4.0;
  EXPECT_NEAR(terms.at("IX").real(), a, kTol);
  EXPECT_NEAR(terms.at("IY").imag(), a, kTol);
  EXPECT_NEAR(terms.at("ZX").real(), b, kTol);
  EXPECT_NEAR(terms.at("ZY").imag(), b, kTol);
  EXPECT_NEAR(terms.at("XX").real(), c, kTol);
  EXPECT_NEAR(terms.at("XY").imag(), -c, kTol);
  EXPECT_NEAR(terms.at("YX").imag(), c, kTol);
  EXPECT_NEAR(terms.at("YY").real(), c, kTol);

  // Term count is nq * 2^nq for every power-of-two d and both encodings.
  for (const auto encoding :
       {BosonEncoding::StandardBinary, BosonEncoding::GrayCode}) {
    for (const std::size_t d : {2u, 4u, 8u, 16u}) {
      const auto wide = BosonMapping::for_encoding(1, d, encoding);
      const std::size_t nq = wide.qubits_per_mode(0);
      EXPECT_EQ(wide.annihilation(0).size(), nq * (std::size_t{1} << nq))
          << "d=" << d;
    }
  }
}

TEST(BosonMapping, CreationIsTheAdjointOfAnnihilation) {
  for (const auto encoding :
       {BosonEncoding::StandardBinary, BosonEncoding::GrayCode}) {
    const auto mapping = BosonMapping::for_encoding(1, 8, encoding);
    const auto annihilate = mapping.annihilation(0);
    const auto create = mapping.creation(0);
    ASSERT_EQ(annihilate.size(), create.size());
    std::map<std::string, std::complex<double>> adjoint;
    for (const auto& [coefficient, word] : annihilate) {
      adjoint[sparse_pauli_word_to_label(word, 3)] = std::conj(coefficient);
    }
    for (const auto& [coefficient, word] : create) {
      const auto label = sparse_pauli_word_to_label(word, 3);
      ASSERT_TRUE(adjoint.count(label)) << label;
      EXPECT_NEAR(std::abs(coefficient - adjoint.at(label)), 0.0, kTol)
          << label;
    }
  }
}

TEST(BosonMapping, DiagonalHandlesArbitraryFunctionsIncludingPenalties) {
  const auto mapping = BosonMapping::standard_binary(1, 4);

  // n^2 = n(n-1) + n.
  const auto squared = as_labels(mapping.number_squared(0), 2);
  const auto shifted = as_labels(mapping.number_times_number_minus_one(0), 2);
  const auto linear = as_labels(mapping.number(0), 2);
  for (const auto& [label, coefficient] : squared) {
    const auto lhs = coefficient;
    std::complex<double> rhs{0.0, 0.0};
    if (shifted.count(label)) rhs += shifted.at(label);
    if (linear.count(label)) rhs += linear.at(label);
    EXPECT_NEAR(std::abs(lhs - rhs), 0.0, kTol) << label;
  }

  // A projector onto the top level is a legitimate occupation penalty.
  const auto penalty = as_labels(mapping.diagonal({0.0, 0.0, 0.0, 1.0}, 0), 2);
  ASSERT_EQ(penalty.size(), 4u);
  EXPECT_NEAR(penalty.at("II").real(), 0.25, kTol);
  EXPECT_NEAR(penalty.at("IZ").real(), -0.25, kTol);
  EXPECT_NEAR(penalty.at("ZI").real(), -0.25, kTol);
  EXPECT_NEAR(penalty.at("ZZ").real(), 0.25, kTol);

  EXPECT_THROW(mapping.diagonal({0.0, 1.0}, 0), std::invalid_argument);
  EXPECT_THROW(mapping.diagonal({0.0, 1.0, 2.0, 3.0}, 1), std::out_of_range);
}

TEST(BosonMapping, GrayCodeNumberOperatorIsExactButHigherWeight) {
  // Gray code trades the weight-1 number operator for single-bit-flip
  // adjacency; the decomposition must still be exact.
  const auto mapping = BosonMapping::gray_code(1, 4);
  const auto terms = as_labels(mapping.number(0), 2);
  const Eigen::MatrixXcd matrix = [&] {
    std::vector<SparsePauliWord> words;
    std::vector<std::complex<double>> coefficients;
    for (const auto& [coefficient, word] : mapping.number(0)) {
      words.push_back(word);
      coefficients.push_back(coefficient);
    }
    return to_matrix(words, coefficients, 2);
  }();
  for (std::size_t n = 0; n < 4; ++n) {
    const auto index = static_cast<Eigen::Index>(mapping.codeword(0, n));
    EXPECT_NEAR(matrix(index, index).real(), static_cast<double>(n), kTol);
  }
  EXPECT_GT(terms.size(), 0u);
}

// ---------------------------------------------------------------------------
// Verified fixtures from the encoding report (section 5.6)
// ---------------------------------------------------------------------------

BosonMapResult map_bose_hubbard(std::size_t sites, std::size_t dim, double t,
                                double u, double mu, BosonEncoding encoding) {
  const auto hamiltonian =
      mh::create_bose_hubbard_hamiltonian(chain(sites), t, u, mu, dim);
  const auto* modes =
      dynamic_cast<const BosonicModes*>(hamiltonian.get_orbitals().get());
  EXPECT_NE(modes, nullptr);
  const auto mapping = BosonMapping::for_basis(*modes, encoding);
  return boson_map_hamiltonian(mapping, hamiltonian, 1e-12, 1e-14);
}

TEST(BosonMapEngine, Fixture1TwoHardCoreModes) {
  // L = 2, d = 2, t = 1, U = 4, mu = 0 -> -0.5 (XX + YY) on 2 qubits.
  const auto result =
      map_bose_hubbard(2, 2, 1.0, 4.0, 0.0, BosonEncoding::StandardBinary);
  const auto terms = as_labels(result, 2);
  ASSERT_EQ(terms.size(), 2u);
  EXPECT_NEAR(terms.at("XX").real(), -0.5, kTol);
  EXPECT_NEAR(terms.at("YY").real(), -0.5, kTol);
  for (const auto& [label, coefficient] : terms) {
    EXPECT_NEAR(coefficient.imag(), 0.0, kTol) << label;
  }
}

TEST(BosonMapEngine, Fixture4ThreeHardCoreModesPinsTheRegisterLayout) {
  // L = 3, d = 2, t = 1, U = 8, mu = 0.  The exact strings — not just the
  // spectrum — distinguish the mode-to-qubit layout and the endianness.
  const auto result =
      map_bose_hubbard(3, 2, 1.0, 8.0, 0.0, BosonEncoding::StandardBinary);
  const auto terms = as_labels(result, 3);
  ASSERT_EQ(terms.size(), 4u);
  EXPECT_NEAR(terms.at("IXX").real(), -0.5, kTol);
  EXPECT_NEAR(terms.at("IYY").real(), -0.5, kTol);
  EXPECT_NEAR(terms.at("XXI").real(), -0.5, kTol);
  EXPECT_NEAR(terms.at("YYI").real(), -0.5, kTol);
}

TEST(BosonMapEngine, Fixture2TwoModesFourLevels) {
  // L = 2, d = 4, t = 1, U = 4, mu = 0 -> 39 terms on 4 qubits.
  const auto result =
      map_bose_hubbard(2, 4, 1.0, 4.0, 0.0, BosonEncoding::StandardBinary);
  const auto terms = as_labels(result, 4);
  const std::vector<std::pair<std::string, double>> reference{
      {"IIII", 8.0},
      {"IIIZ", -2.0},
      {"IIZI", -4.0},
      {"IIZZ", 2.0},
      {"IXIX", -0.9330127018922193},
      {"IXXX", -0.48296291314453416},
      {"IXYY", -0.48296291314453416},
      {"IXZX", 0.24999999999999994},
      {"IYIY", -0.9330127018922193},
      {"IYXY", 0.48296291314453416},
      {"IYYX", -0.48296291314453416},
      {"IYZY", 0.24999999999999994},
      {"IZII", -2.0},
      {"XXIX", -0.48296291314453416},
      {"XXXX", -0.25000000000000006},
      {"XXYY", -0.25000000000000006},
      {"XXZX", 0.12940952255126037},
      {"XYIY", 0.48296291314453416},
      {"XYXY", -0.25000000000000006},
      {"XYYX", 0.25000000000000006},
      {"XYZY", -0.12940952255126037},
      {"YXIY", -0.48296291314453416},
      {"YXXY", 0.25000000000000006},
      {"YXYX", -0.25000000000000006},
      {"YXZY", 0.12940952255126037},
      {"YYIX", -0.48296291314453416},
      {"YYXX", -0.25000000000000006},
      {"YYYY", -0.25000000000000006},
      {"YYZX", 0.12940952255126037},
      {"ZIII", -4.0},
      {"ZXIX", 0.24999999999999994},
      {"ZXXX", 0.12940952255126037},
      {"ZXYY", 0.12940952255126037},
      {"ZXZX", -0.06698729810778066},
      {"ZYIY", 0.24999999999999994},
      {"ZYXY", -0.12940952255126037},
      {"ZYYX", 0.12940952255126037},
      {"ZYZY", -0.06698729810778066},
      {"ZZII", 2.0}};
  ASSERT_EQ(terms.size(), reference.size());
  for (const auto& [label, value] : reference) {
    ASSERT_TRUE(terms.count(label)) << "missing " << label;
    EXPECT_NEAR(terms.at(label).real(), value, kTol) << label;
    EXPECT_NEAR(terms.at(label).imag(), 0.0, kTol) << label;
  }
}

// ---------------------------------------------------------------------------
// Three-way builder agreement and exact diagonalization
// ---------------------------------------------------------------------------

TEST(BosonMapEngine, ThreeWayBuilderAgreementAndZeroLeakage) {
  struct Case {
    std::size_t sites;
    std::size_t dim;
    double t;
    double u;
    double mu;
  };
  const std::vector<Case> cases{{2, 2, 1.0, 4.0, 0.0},
                                {2, 4, 1.0, 4.0, 0.0},
                                {3, 2, 1.0, 8.0, 0.0},
                                {3, 4, 0.7, 3.3, 0.9},
                                {2, 8, 0.5, 2.0, -1.0}};
  for (const auto& c : cases) {
    // Builder 1: occupation basis.  Builder 2: Kronecker products.
    const Eigen::MatrixXd occupation =
        occupation_basis_hamiltonian(c.sites, c.dim, c.t, c.u, c.mu);
    const Eigen::MatrixXd kronecker =
        kronecker_hamiltonian(c.sites, c.dim, c.t, c.u, c.mu);
    ASSERT_EQ(occupation.rows(), kronecker.rows());
    EXPECT_LT((occupation - kronecker).cwiseAbs().maxCoeff(), kTol)
        << "L=" << c.sites << " d=" << c.dim;

    for (const auto encoding :
         {BosonEncoding::StandardBinary, BosonEncoding::GrayCode}) {
      // Builder 3: the library's chemist-notation contraction, encoded.
      const auto result =
          map_bose_hubbard(c.sites, c.dim, c.t, c.u, c.mu, encoding);
      const auto mapping = BosonMapping::for_encoding(c.sites, c.dim, encoding);
      const Eigen::MatrixXcd encoded =
          to_matrix(result.words, result.coefficients, mapping.num_qubits());
      ASSERT_EQ(encoded.rows(), occupation.rows());

      const auto configs = occupation_configs(c.sites, c.dim);
      double max_error = 0.0;
      for (std::size_t row = 0; row < configs.size(); ++row) {
        for (std::size_t col = 0; col < configs.size(); ++col) {
          const auto encoded_value = encoded(
              static_cast<Eigen::Index>(register_index(mapping, configs[row])),
              static_cast<Eigen::Index>(register_index(mapping, configs[col])));
          const double reference = occupation(static_cast<Eigen::Index>(row),
                                              static_cast<Eigen::Index>(col));
          max_error = std::max(
              max_error,
              std::abs(encoded_value - std::complex<double>(reference, 0.0)));
        }
      }
      EXPECT_LT(max_error, kTol) << "L=" << c.sites << " d=" << c.dim
                                 << " encoding=" << to_string(encoding);

      // Leakage: the codeword map is a bijection onto the whole register at a
      // power-of-two cutoff, so every register state is physical and the
      // comparison above already covered the entire matrix.
      EXPECT_EQ(static_cast<std::size_t>(encoded.rows()), configs.size());
    }
  }
}

TEST(BosonMapEngine, ExactGroundAndExcitedEnergies) {
  // Single particle on a two-site chain: E = -t, +t independent of U.
  {
    const Eigen::MatrixXd h = occupation_basis_hamiltonian(2, 4, 1.0, 4.0, 0.0);
    // Sector N = 1 is spanned by |1,0> and |0,1>.
    const auto configs = occupation_configs(2, 4);
    std::vector<Eigen::Index> sector;
    for (Eigen::Index i = 0; i < static_cast<Eigen::Index>(configs.size());
         ++i) {
      const auto& config = configs[static_cast<std::size_t>(i)];
      if (config[0] + config[1] == 1) sector.push_back(i);
    }
    ASSERT_EQ(sector.size(), 2u);
    Eigen::MatrixXd block(2, 2);
    for (int a = 0; a < 2; ++a) {
      for (int b = 0; b < 2; ++b) {
        block(a, b) = h(sector[static_cast<std::size_t>(a)],
                        sector[static_cast<std::size_t>(b)]);
      }
    }
    const Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(block);
    EXPECT_NEAR(solver.eigenvalues()(0), -1.0, kTol);
    EXPECT_NEAR(solver.eigenvalues()(1), 1.0, kTol);
  }

  // Two particles on a two-site chain with d >= 3: the N = 2 sector has
  // eigenvalues U and (U -/+ sqrt(U^2 + 16 t^2)) / 2.
  {
    const double t = 1.0;
    const double u = 4.0;
    const Eigen::MatrixXd h = occupation_basis_hamiltonian(2, 4, t, u, 0.0);
    const auto configs = occupation_configs(2, 4);
    std::vector<Eigen::Index> sector;
    for (Eigen::Index i = 0; i < static_cast<Eigen::Index>(configs.size());
         ++i) {
      const auto& config = configs[static_cast<std::size_t>(i)];
      if (config[0] + config[1] == 2) sector.push_back(i);
    }
    ASSERT_EQ(sector.size(), 3u);
    Eigen::MatrixXd block(3, 3);
    for (int a = 0; a < 3; ++a) {
      for (int b = 0; b < 3; ++b) {
        block(a, b) = h(sector[static_cast<std::size_t>(a)],
                        sector[static_cast<std::size_t>(b)]);
      }
    }
    const Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(block);
    const double root = std::sqrt(u * u + 16.0 * t * t);
    EXPECT_NEAR(solver.eigenvalues()(0), (u - root) / 2.0, kTol);
    EXPECT_NEAR(solver.eigenvalues()(0), 2.0 - 2.0 * std::sqrt(2.0), kTol);
    EXPECT_NEAR(solver.eigenvalues()(1), u, kTol);
    EXPECT_NEAR(solver.eigenvalues()(2), (u + root) / 2.0, kTol);
  }
}

TEST(BosonMapEngine, EncodedSpectrumMatchesTheReferenceFixture) {
  // Fixture 2 full 16-dimensional spectrum (report section 5.4).
  const auto result =
      map_bose_hubbard(2, 4, 1.0, 4.0, 0.0, BosonEncoding::StandardBinary);
  const Eigen::MatrixXcd encoded =
      to_matrix(result.words, result.coefficients, 4);
  const Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> solver(encoded);
  const std::vector<double> reference{-1.0,
                                      -0.828427124746,
                                      0.0,
                                      1.0,
                                      1.708497377871,
                                      4.0,
                                      4.828427124746,
                                      5.535898384862,
                                      6.0,
                                      12.0,
                                      12.291502622129,
                                      12.464101615138,
                                      13.0,
                                      14.0,
                                      19.0,
                                      24.0};
  ASSERT_EQ(static_cast<std::size_t>(solver.eigenvalues().size()),
            reference.size());
  for (std::size_t i = 0; i < reference.size(); ++i) {
    EXPECT_NEAR(solver.eigenvalues()(static_cast<Eigen::Index>(i)),
                reference[i], 1e-9)
        << "eigenvalue " << i;
  }
  // The encoded operator is Hermitian.
  EXPECT_LT((encoded - encoded.adjoint()).cwiseAbs().maxCoeff(), kTol);
}

TEST(BosonMapEngine, GrayCodeIsIsospectralWithStandardBinary) {
  const auto sb =
      map_bose_hubbard(3, 4, 0.7, 3.3, 0.9, BosonEncoding::StandardBinary);
  const auto gc =
      map_bose_hubbard(3, 4, 0.7, 3.3, 0.9, BosonEncoding::GrayCode);
  const Eigen::MatrixXcd sb_matrix = to_matrix(sb.words, sb.coefficients, 6);
  const Eigen::MatrixXcd gc_matrix = to_matrix(gc.words, gc.coefficients, 6);
  const Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> sb_solver(sb_matrix);
  const Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> gc_solver(gc_matrix);
  EXPECT_LT(
      (sb_solver.eigenvalues() - gc_solver.eigenvalues()).cwiseAbs().maxCoeff(),
      1e-9);
  // Same Pauli-term count: Gray code buys circuit depth, not term count.
  EXPECT_EQ(sb.words.size(), gc.words.size());
}

// ---------------------------------------------------------------------------
// Data-class behaviour
// ---------------------------------------------------------------------------

TEST(BosonMapping, JsonRoundTrip) {
  const auto mapping = BosonMapping::gray_code(3, 8);
  const auto json = mapping.to_json();
  const auto loaded = BosonMapping::from_json(json);
  EXPECT_EQ(loaded.num_modes(), 3u);
  EXPECT_EQ(loaded.mode_dimension(0), 8u);
  EXPECT_EQ(json.at("mode_dimensions").get<std::vector<std::size_t>>(),
            (std::vector<std::size_t>{8u, 8u, 8u}));
  EXPECT_EQ(loaded.encoding(), BosonEncoding::GrayCode);
  EXPECT_EQ(loaded.content_hash(), mapping.content_hash());
  EXPECT_NE(loaded.content_hash(),
            BosonMapping::standard_binary(3, 8).content_hash());
  EXPECT_EQ(mapping.get_data_type_name(), "boson_mapping");
  EXPECT_NE(mapping.get_summary().find("gray-code"), std::string::npos);
}

TEST(BosonMapping, EncodingNameParsing) {
  EXPECT_EQ(boson_encoding_from_string("standard-binary"),
            BosonEncoding::StandardBinary);
  EXPECT_EQ(boson_encoding_from_string("standard_binary"),
            BosonEncoding::StandardBinary);
  EXPECT_EQ(boson_encoding_from_string("SB"), BosonEncoding::StandardBinary);
  EXPECT_EQ(boson_encoding_from_string("Gray-Code"), BosonEncoding::GrayCode);
  EXPECT_EQ(boson_encoding_from_string("gray"), BosonEncoding::GrayCode);
  EXPECT_THROW(boson_encoding_from_string("unary"), std::invalid_argument);
  EXPECT_EQ(to_string(BosonEncoding::StandardBinary), "standard-binary");
  EXPECT_EQ(to_string(BosonEncoding::GrayCode), "gray-code");
}

TEST(BoseHubbardBuilder, StoresChemistNotationIntegralsOnABosonicBasis) {
  const double t = 1.5;
  const double u = 3.0;
  const double mu = 0.75;
  const auto hamiltonian =
      mh::create_bose_hubbard_hamiltonian(chain(3), t, u, mu, 4);

  const auto orbitals = hamiltonian.get_orbitals();
  ASSERT_NE(orbitals, nullptr);
  const auto* modes = dynamic_cast<const BosonicModes*>(orbitals.get());
  ASSERT_NE(modes, nullptr);
  EXPECT_EQ(modes->num_modes(), 3u);
  EXPECT_EQ(modes->mode_dimension(0), 4u);

  // h_ii = -mu, h_ij = -t on bonds, (ii|ii) = U.
  for (unsigned i = 0; i < 3; ++i) {
    EXPECT_NEAR(hamiltonian.get_one_body_element(i, i), -mu, kTol);
    EXPECT_NEAR(hamiltonian.get_two_body_element(i, i, i, i), u, kTol);
  }
  EXPECT_NEAR(hamiltonian.get_one_body_element(0, 1), -t, kTol);
  EXPECT_NEAR(hamiltonian.get_one_body_element(1, 0), -t, kTol);
  EXPECT_NEAR(hamiltonian.get_one_body_element(1, 2), -t, kTol);
  EXPECT_NEAR(hamiltonian.get_one_body_element(0, 2), 0.0, kTol);

  // A clone must keep the bosonic basis: dropping it would silently change
  // the physics of any downstream mapping.
  const auto copy = hamiltonian;
  const auto* copied_modes =
      dynamic_cast<const BosonicModes*>(copy.get_orbitals().get());
  ASSERT_NE(copied_modes, nullptr);
  EXPECT_EQ(copied_modes->mode_dimension(2), 4u);
}

TEST(BoseHubbardBuilder, MismatchedCutoffIsAHardError) {
  const auto hamiltonian =
      mh::create_bose_hubbard_hamiltonian(chain(2), 1.0, 4.0, 0.0, 4);
  const auto wrong = BosonMapping::standard_binary(2, 8);
  EXPECT_THROW(boson_map_hamiltonian(wrong, hamiltonian, 1e-12, 1e-14),
               std::invalid_argument);
  const auto wrong_modes = BosonMapping::standard_binary(3, 4);
  EXPECT_THROW(boson_map_hamiltonian(wrong_modes, hamiltonian, 1e-12, 1e-14),
               std::invalid_argument);
}

// ---------------------------------------------------------------------------
// Custom codeword tables (from_codeword_table)
// ---------------------------------------------------------------------------

namespace {

/// Codeword table of a named encoding, read back off a mapping built by the
/// named factory -- so a test can feed exactly that table to
/// from_codeword_table and compare the two.
std::vector<std::vector<std::uint64_t>> table_of(const BosonMapping& mapping) {
  std::vector<std::vector<std::uint64_t>> table;
  table.reserve(mapping.num_modes());
  for (std::size_t i = 0; i < mapping.num_modes(); ++i) {
    table.push_back(mapping.codeword_table(i));
  }
  return table;
}

/// Every ladder primitive of every mode, as label -> coefficient maps, so two
/// mappings can be compared operator by operator rather than field by field.
std::vector<std::map<std::string, std::complex<double>>> all_primitives(
    const BosonMapping& mapping) {
  std::vector<std::map<std::string, std::complex<double>>> out;
  const std::size_t nq = mapping.num_qubits();
  for (std::size_t i = 0; i < mapping.num_modes(); ++i) {
    out.push_back(as_labels(mapping.annihilation(i), nq));
    out.push_back(as_labels(mapping.creation(i), nq));
    out.push_back(as_labels(mapping.number(i), nq));
    out.push_back(as_labels(mapping.number_squared(i), nq));
    out.push_back(as_labels(mapping.number_times_number_minus_one(i), nq));
  }
  for (std::size_t i = 0; i + 1 < mapping.num_modes(); ++i) {
    out.push_back(
        as_labels(mapping.ladder_product({{i, true}, {i + 1, false}}), nq));
  }
  return out;
}

void expect_same_operators(const BosonMapping& lhs, const BosonMapping& rhs) {
  const auto left = all_primitives(lhs);
  const auto right = all_primitives(rhs);
  ASSERT_EQ(left.size(), right.size());
  for (std::size_t k = 0; k < left.size(); ++k) {
    ASSERT_EQ(left[k].size(), right[k].size()) << "operator " << k;
    for (const auto& [label, coefficient] : left[k]) {
      ASSERT_TRUE(right[k].count(label) == 1)
          << "operator " << k << " missing " << label;
      EXPECT_NEAR(coefficient.real(), right[k].at(label).real(), kTol) << label;
      EXPECT_NEAR(coefficient.imag(), right[k].at(label).imag(), kTol) << label;
    }
  }
}

}  // namespace

TEST(BosonMappingCustomTable, ReproducesTheNamedEncodingsExactly) {
  // The oracle claim: an encoding *is* its codeword table, so feeding the
  // standard-binary (resp. Gray) table back in must give the very same
  // operators the named factory produces -- every primitive, every mode.
  for (const std::size_t dimension : {2u, 4u, 8u}) {
    const auto sb = BosonMapping::standard_binary(2, dimension);
    const auto sb_custom = BosonMapping::from_codeword_table(table_of(sb));
    EXPECT_EQ(sb_custom.num_modes(), sb.num_modes());
    EXPECT_EQ(sb_custom.num_qubits(), sb.num_qubits());
    EXPECT_EQ(sb_custom.mode_dimensions(), sb.mode_dimensions());
    EXPECT_EQ(sb_custom.qubits_per_mode(0), sb.qubits_per_mode(0));
    EXPECT_EQ(sb_custom.mode_qubits(1), sb.mode_qubits(1));
    expect_same_operators(sb, sb_custom);

    const auto gray = BosonMapping::gray_code(2, dimension);
    const auto gray_custom = BosonMapping::from_codeword_table(table_of(gray));
    expect_same_operators(gray, gray_custom);
  }
}

TEST(BosonMappingCustomTable, ReportsCustomEvenWhenTheTableIsANamedOne) {
  // The tag records how the mapping was built; no table recognition happens,
  // so encoding() never has to guess and can never be subtly wrong.
  const auto mapping = BosonMapping::from_codeword_table(
      table_of(BosonMapping::standard_binary(1, 4)));
  EXPECT_EQ(mapping.encoding(), BosonEncoding::Custom);
  EXPECT_EQ(mapping.name(), "custom");
  EXPECT_EQ(to_string(BosonEncoding::Custom), "custom");
  EXPECT_EQ(boson_encoding_from_string("custom"), BosonEncoding::Custom);
  EXPECT_EQ(boson_encoding_from_string("CUSTOM"), BosonEncoding::Custom);

  const auto named =
      BosonMapping::from_codeword_table({{0, 1, 3, 2}}, "reflected");
  EXPECT_EQ(named.name(), "reflected");
  EXPECT_EQ(named.encoding(), BosonEncoding::Custom);
  EXPECT_NE(named.get_summary().find("reflected"), std::string::npos);
}

TEST(BosonMappingCustomTable, CustomEncodingCannotSelectAnEncoding) {
  // "custom" names no particular table, so it cannot stand in for one.
  EXPECT_THROW(BosonMapping::for_encoding(2, 4, BosonEncoding::Custom),
               std::invalid_argument);
}

TEST(BosonMappingCustomTable, HeterogeneousTablesLayOutBlocksPerMode) {
  // Mode 0 (d = 8, 3 qubits) is the most significant block, mode 1 (d = 2, 1
  // qubit) the least. The dimensions and widths come from the table alone.
  const auto mapping = BosonMapping::from_codeword_table(
      {{7, 6, 4, 5, 1, 0, 2, 3}, {1, 0}}, "reflected-gray-and-flip");
  EXPECT_EQ(mapping.num_modes(), 2u);
  EXPECT_EQ(mapping.mode_dimension(0), 8u);
  EXPECT_EQ(mapping.mode_dimension(1), 2u);
  EXPECT_EQ(mapping.qubits_per_mode(0), 3u);
  EXPECT_EQ(mapping.qubits_per_mode(1), 1u);
  EXPECT_EQ(mapping.num_qubits(), 4u);
  EXPECT_EQ(mapping.mode_qubits(0), (std::vector<std::uint64_t>{1, 2, 3}));
  EXPECT_EQ(mapping.mode_qubits(1), (std::vector<std::uint64_t>{0}));
  EXPECT_EQ(mapping.max_occupation(0), 7u);
  EXPECT_FALSE(mapping.uniform_dimension().has_value());

  // codeword and level are exact inverses of one another on both modes.
  for (std::size_t mode = 0; mode < mapping.num_modes(); ++mode) {
    for (std::size_t n = 0; n < mapping.mode_dimension(mode); ++n) {
      EXPECT_EQ(mapping.level(mode, mapping.codeword(mode, n)), n);
    }
  }
}

TEST(BosonMappingCustomTable, DistinctTablesGiveDistinctOperators) {
  // Regression guard for the local-decomposition cache. It used to be keyed on
  // (encoding, dimension, sequence); with a single Custom tag two different
  // tables of the same dimension would then collide and silently return each
  // other's Pauli terms. The table itself is the key.
  const auto flip = BosonMapping::from_codeword_table({{1, 0, 2, 3}}, "flip01");
  const auto swap = BosonMapping::from_codeword_table({{0, 1, 3, 2}}, "swap23");
  const auto flip_terms = as_labels(flip.annihilation(0), 2);
  const auto swap_terms = as_labels(swap.annihilation(0), 2);
  EXPECT_NE(flip_terms, swap_terms);

  // ... and each still agrees with the operator its own table defines: b maps
  // |n> to sqrt(n) |n-1>, i.e. <cw(n-1)| b |cw(n)> = sqrt(n).
  for (const auto* mapping : {&flip, &swap}) {
    const auto terms = mapping->annihilation(0);
    std::vector<SparsePauliWord> words;
    std::vector<std::complex<double>> coefficients;
    for (const auto& term : terms) {
      coefficients.push_back(term.first);
      words.push_back(term.second);
    }
    const auto matrix = to_matrix(words, coefficients, 2);
    for (std::size_t n = 0; n < 4; ++n) {
      for (std::size_t m = 0; m < 4; ++m) {
        const auto row = static_cast<Eigen::Index>(mapping->codeword(0, m));
        const auto col = static_cast<Eigen::Index>(mapping->codeword(0, n));
        const double expected =
            (m + 1 == n) ? std::sqrt(static_cast<double>(n)) : 0.0;
        EXPECT_NEAR(matrix(row, col).real(), expected, kTol)
            << mapping->name() << " <" << m << "|b|" << n << ">";
        EXPECT_NEAR(matrix(row, col).imag(), 0.0, kTol);
      }
    }
  }
}

TEST(BosonMappingCustomTable, RejectsInvalidTablesWithActionableMessages) {
  // Empty.
  EXPECT_THROW(BosonMapping::from_codeword_table({}), std::invalid_argument);

  // Fewer than two levels.
  EXPECT_THROW(BosonMapping::from_codeword_table({{0}}), std::invalid_argument);

  // Not a power of two.
  try {
    BosonMapping::from_codeword_table({{0, 1, 2, 3}, {0, 1, 2}});
    FAIL() << "expected a non-power-of-two table to be rejected";
  } catch (const std::invalid_argument& error) {
    const std::string message = error.what();
    EXPECT_NE(message.find("mode 1"), std::string::npos) << message;
    EXPECT_NE(message.find("d=3"), std::string::npos) << message;
    EXPECT_NE(message.find("4 codewords"), std::string::npos) << message;
  }

  // A codeword that does not fit in the mode's qubits.
  try {
    BosonMapping::from_codeword_table({{0, 1, 2, 9}});
    FAIL() << "expected an out-of-register codeword to be rejected";
  } catch (const std::invalid_argument& error) {
    const std::string message = error.what();
    EXPECT_NE(message.find("mode 0"), std::string::npos) << message;
    EXPECT_NE(message.find("level 3"), std::string::npos) << message;
    EXPECT_NE(message.find("codeword 9"), std::string::npos) << message;
  }

  // A repeated codeword, i.e. a non-injective map.
  try {
    BosonMapping::from_codeword_table({{0, 1, 2, 3}, {2, 1, 3, 1}});
    FAIL() << "expected a repeated codeword to be rejected";
  } catch (const std::invalid_argument& error) {
    const std::string message = error.what();
    EXPECT_NE(message.find("mode 1"), std::string::npos) << message;
    EXPECT_NE(message.find("levels 1 and 3"), std::string::npos) << message;
    EXPECT_NE(message.find("injective"), std::string::npos) << message;
  }
}

TEST(BosonMappingCustomTable, SerializationRoundTripsTheTable) {
  const auto mapping =
      BosonMapping::from_codeword_table({{2, 0, 3, 1}, {1, 0}}, "my-encoding");
  const auto json = mapping.to_json();

  // The table -- not the tag -- is what goes on the wire for a custom mapping.
  EXPECT_EQ(json.at("encoding").get<std::string>(), "custom");
  EXPECT_EQ(json.at("name").get<std::string>(), "my-encoding");
  EXPECT_EQ(json.at("codewords").get<std::vector<std::vector<std::uint64_t>>>(),
            (std::vector<std::vector<std::uint64_t>>{{2, 0, 3, 1}, {1, 0}}));
  EXPECT_EQ(json.at("mode_dimensions").get<std::vector<std::size_t>>(),
            (std::vector<std::size_t>{4u, 2u}));
  EXPECT_EQ(json.at("version").get<std::string>(), "0.1.0");

  const auto loaded = BosonMapping::from_json(json);
  EXPECT_EQ(loaded.encoding(), BosonEncoding::Custom);
  EXPECT_EQ(loaded.name(), "my-encoding");
  EXPECT_EQ(loaded.codeword_table(0), (std::vector<std::uint64_t>{2, 0, 3, 1}));
  EXPECT_EQ(loaded.codeword_table(1), (std::vector<std::uint64_t>{1, 0}));
  EXPECT_EQ(loaded.content_hash(), mapping.content_hash());
  expect_same_operators(mapping, loaded);

  // Two custom mappings that differ only in their table must not collide.
  const auto other =
      BosonMapping::from_codeword_table({{0, 1, 3, 2}, {1, 0}}, "my-encoding");
  EXPECT_NE(other.content_hash(), mapping.content_hash());
  // ... nor two that differ only in their label.
  const auto relabelled =
      BosonMapping::from_codeword_table({{2, 0, 3, 1}, {1, 0}}, "other-label");
  EXPECT_NE(relabelled.content_hash(), mapping.content_hash());
}

TEST(BosonMappingCustomTable, NamedEncodingPayloadsAndHashesAreUnchanged) {
  // The new fields are written only for custom mappings, so a document for a
  // named encoding is exactly what it was before custom tables existed.
  for (const auto encoding :
       {BosonEncoding::StandardBinary, BosonEncoding::GrayCode}) {
    const auto json = BosonMapping::for_encoding(2, 4, encoding).to_json();
    EXPECT_FALSE(json.contains("codewords"));
    EXPECT_FALSE(json.contains("name"));
    EXPECT_EQ(json.size(), 4u);
  }

  // And an old-style document -- no codewords field at all -- still loads.
  const nlohmann::json legacy{{"version", "0.1.0"},
                              {"num_modes", 2},
                              {"mode_dimensions", {4, 4}},
                              {"encoding", "gray-code"}};
  const auto loaded = BosonMapping::from_json(legacy);
  EXPECT_EQ(loaded.encoding(), BosonEncoding::GrayCode);
  expect_same_operators(loaded, BosonMapping::gray_code(2, 4));
}

TEST(BosonMappingCustomTable, RejectsSelfContradictoryDocuments) {
  // "custom" without a table has nothing to rebuild the mapping from.
  const nlohmann::json no_table{{"version", "0.1.0"},
                                {"num_modes", 1},
                                {"mode_dimensions", {4}},
                                {"encoding", "custom"}};
  EXPECT_THROW(BosonMapping::from_json(no_table), std::invalid_argument);

  // A named encoding shipped with a table that is not that encoding's table:
  // encoding() would then describe an operator set the mapping does not have.
  nlohmann::json inconsistent{{"version", "0.1.0"},
                              {"num_modes", 1},
                              {"mode_dimensions", {4}},
                              {"encoding", "standard-binary"}};
  inconsistent["codewords"] =
      std::vector<std::vector<std::uint64_t>>{{0, 1, 3, 2}};
  EXPECT_THROW(BosonMapping::from_json(inconsistent), std::invalid_argument);

  // The declared dimensions must agree with the table they accompany.
  nlohmann::json bad_dimensions{{"version", "0.1.0"},
                                {"num_modes", 1},
                                {"mode_dimensions", {8}},
                                {"encoding", "custom"}};
  bad_dimensions["codewords"] =
      std::vector<std::vector<std::uint64_t>>{{0, 1, 3, 2}};
  EXPECT_THROW(BosonMapping::from_json(bad_dimensions), std::invalid_argument);
}

TEST(BosonMappingCustomTable, ValidatesABasisJustLikeANamedEncoding) {
  const auto mapping =
      BosonMapping::from_codeword_table({{1, 0, 3, 2}, {1, 0, 3, 2}}, "flip");
  const BosonicModes matching(2, 4);
  EXPECT_NO_THROW(mapping.validate_basis(matching));
  const BosonicModes wrong_cutoff(2, 8);
  EXPECT_THROW(mapping.validate_basis(wrong_cutoff), std::invalid_argument);
}

TEST(BosonMappingCustomTable, MapsAHamiltonianIdenticallyToTheNamedEncoding) {
  // End to end through the mapping engine, not just the primitives.
  const auto hamiltonian =
      mh::create_bose_hubbard_hamiltonian(chain(2), 0.7, 3.3, 0.9, 4);
  const auto named = BosonMapping::standard_binary(2, 4);
  const auto custom = BosonMapping::from_codeword_table(table_of(named));
  const auto from_named =
      as_labels(boson_map_hamiltonian(named, hamiltonian, 1e-12, 1e-14), 4);
  const auto from_custom =
      as_labels(boson_map_hamiltonian(custom, hamiltonian, 1e-12, 1e-14), 4);
  ASSERT_EQ(from_named.size(), from_custom.size());
  for (const auto& [label, coefficient] : from_named) {
    ASSERT_TRUE(from_custom.count(label) == 1) << label;
    EXPECT_NEAR(coefficient.real(), from_custom.at(label).real(), kTol)
        << label;
    EXPECT_NEAR(coefficient.imag(), from_custom.at(label).imag(), kTol)
        << label;
  }
}

// ---------------------------------------------------------------------------
// Hard-core bosons (d = 2)
// ---------------------------------------------------------------------------

TEST(HardCoreBosons, OnSiteInteractionVanishesSoUIsInert) {
  // n(n-1) = 0 for n in {0, 1}, so U cannot appear in the mapped operator.
  const auto free_terms = as_labels(
      map_bose_hubbard(3, 2, 1.0, 0.0, 0.0, BosonEncoding::StandardBinary), 3);
  const auto huge_terms = as_labels(
      map_bose_hubbard(3, 2, 1.0, 400.0, 0.0, BosonEncoding::StandardBinary),
      3);
  ASSERT_EQ(free_terms.size(), huge_terms.size());
  for (const auto& [label, coefficient] : free_terms) {
    ASSERT_TRUE(huge_terms.count(label) == 1) << label;
    EXPECT_EQ(coefficient, huge_terms.at(label)) << label;
  }

  // The primitive itself is empty, which is why.
  const auto mapping = BosonMapping::standard_binary(1, 2);
  EXPECT_TRUE(mapping.number_times_number_minus_one(0).empty());
  // ... and it is not empty as soon as there are three levels to interact.
  EXPECT_FALSE(BosonMapping::standard_binary(1, 4)
                   .number_times_number_minus_one(0)
                   .empty());
}

TEST(HardCoreBosons, HardCoreBasisIsTwoLevelAndNeedsNoPadding) {
  const auto modes = BosonicModes::hard_core(4);
  ASSERT_NE(modes, nullptr);
  EXPECT_EQ(modes->num_modes(), 4u);
  for (std::size_t i = 0; i < modes->num_modes(); ++i) {
    EXPECT_EQ(modes->mode_dimension(i), 2u) << "mode " << i;
    EXPECT_EQ(modes->max_occupation(i), 1u) << "mode " << i;
  }
  EXPECT_EQ(modes->uniform_dimension(), std::optional<std::size_t>{2u});
  EXPECT_TRUE(modes->has_power_of_two_dimensions());
  EXPECT_EQ(modes->fock_space_dimension(), 16u);

  // Power of two already, so padding is a no-op and the basis maps directly.
  const auto padded = modes->with_padded_dimensions();
  EXPECT_EQ(padded->mode_dimensions(), modes->mode_dimensions());
  const auto mapping = BosonMapping::for_basis(*modes);
  EXPECT_EQ(mapping.num_qubits(), 4u);
  EXPECT_EQ(mapping.qubits_per_mode(0), 1u);
  EXPECT_NO_THROW(mapping.validate_basis(*modes));
}

TEST(HardCoreBosons, AnnihilationIsExactlySigmaMinus) {
  // b = |0><1| = (X + iY)/2 at d = 2.
  const auto mapping = BosonMapping::for_basis(*BosonicModes::hard_core(1));
  const auto terms = as_labels(mapping.annihilation(0), 1);
  ASSERT_EQ(terms.size(), 2u);
  EXPECT_NEAR(terms.at("X").real(), 0.5, kTol);
  EXPECT_NEAR(terms.at("X").imag(), 0.0, kTol);
  EXPECT_NEAR(terms.at("Y").real(), 0.0, kTol);
  EXPECT_NEAR(terms.at("Y").imag(), 0.5, kTol);
}

TEST(HardCoreBosons, ReproducesResearchFixtureOne) {
  // Fixture 1: L = 2, d = 2, t = 1, U = 4, mu = 0 built on a hard_core basis.
  const auto modes = BosonicModes::hard_core(2);
  const auto hamiltonian = mh::create_bose_hubbard_hamiltonian(
      chain(2), 1.0, 4.0, 0.0, modes->mode_dimension(0));
  const auto mapping = BosonMapping::for_basis(*modes);
  const auto terms =
      as_labels(boson_map_hamiltonian(mapping, hamiltonian, 1e-12, 1e-14), 2);
  ASSERT_EQ(terms.size(), 2u);
  EXPECT_NEAR(terms.at("XX").real(), -0.5, kTol);
  EXPECT_NEAR(terms.at("YY").real(), -0.5, kTol);
}

namespace {

/// Log lines emitted while @p action runs, captured off the global logger.
std::vector<std::string> captured_log(const std::function<void()>& action) {
  namespace utils = qdk::chemistry::utils;
  auto logger = utils::Logger::get();
  auto sink = std::make_shared<spdlog::sinks::ringbuffer_sink_mt>(16);
  const auto previous_level = utils::Logger::get_global_level();
  logger->sinks().push_back(sink);
  utils::Logger::set_global_level(utils::LogLevel::warn);
  action();
  utils::Logger::set_global_level(previous_level);
  logger->sinks().pop_back();
  return sink->last_formatted();
}

bool mentions_hard_core(const std::vector<std::string>& lines) {
  return std::any_of(lines.begin(), lines.end(), [](const std::string& line) {
    return line.find("hard-core limit") != std::string::npos;
  });
}

}  // namespace

TEST(HardCoreBosons, InertUIsReportedButChangesNothing) {
  // The Hamiltonian is built exactly as asked -- the physics of the hard-core
  // limit is right -- but the caller is told that U cannot be felt.
  const auto with_u = captured_log(
      [] { mh::create_bose_hubbard_hamiltonian(chain(2), 1.0, 4.0, 0.0, 2); });
  EXPECT_TRUE(mentions_hard_core(with_u))
      << "expected a hard-core warning, got " << with_u.size() << " line(s)";

  // No warning when there is nothing to warn about.
  const auto zero_u = captured_log(
      [] { mh::create_bose_hubbard_hamiltonian(chain(2), 1.0, 0.0, 0.0, 2); });
  EXPECT_FALSE(mentions_hard_core(zero_u));

  const auto four_levels = captured_log(
      [] { mh::create_bose_hubbard_hamiltonian(chain(2), 1.0, 4.0, 0.0, 4); });
  EXPECT_FALSE(mentions_hard_core(four_levels));

  // The warning must not alter the stored integrals: (ii|ii) = U verbatim.
  const auto hamiltonian =
      mh::create_bose_hubbard_hamiltonian(chain(2), 1.0, 4.0, 0.0, 2);
  EXPECT_NEAR(hamiltonian.get_two_body_element(0, 0, 0, 0), 4.0, kTol);
  EXPECT_NEAR(hamiltonian.get_one_body_element(0, 1), -1.0, kTol);
}
