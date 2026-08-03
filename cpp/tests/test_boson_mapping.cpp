// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <gtest/gtest.h>

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <complex>
#include <map>
#include <memory>
#include <qdk/chemistry/data/boson_mapping.hpp>
#include <qdk/chemistry/data/bosonic_modes.hpp>
#include <qdk/chemistry/data/hamiltonian.hpp>
#include <qdk/chemistry/data/lattice_graph.hpp>
#include <qdk/chemistry/data/pauli_operator.hpp>
#include <qdk/chemistry/utils/model_hamiltonians.hpp>
#include <stdexcept>
#include <string>
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
