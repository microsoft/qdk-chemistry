// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <algorithm>
#include <cmath>
#include <complex>
#include <qdk/chemistry/data/boson_mapping.hpp>
#include <qdk/chemistry/data/bosonic_modes.hpp>
#include <qdk/chemistry/data/hamiltonian.hpp>
#include <qdk/chemistry/data/hamiltonian_containers/sparse.hpp>
#include <qdk/chemistry/utils/logger.hpp>
#include <stdexcept>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

namespace qdk::chemistry::data {

namespace {

using TermAccumulator =
    std::unordered_map<SparsePauliWord, std::complex<double>,
                       SparsePauliWordHash>;

void accumulate(TermAccumulator& sink, const BosonPauliTerms& terms,
                double scale) {
  for (const auto& [coefficient, word] : terms) {
    sink[word] += coefficient * scale;
  }
}

BosonMapResult finalize(TermAccumulator& sink, double threshold) {
  std::vector<std::pair<SparsePauliWord, std::complex<double>>> ordered;
  ordered.reserve(sink.size());
  for (auto& entry : sink) {
    if (std::abs(entry.second) >= threshold) {
      ordered.emplace_back(entry.first, entry.second);
    }
  }
  std::sort(
      ordered.begin(), ordered.end(),
      [](const auto& lhs, const auto& rhs) { return lhs.first < rhs.first; });

  BosonMapResult result;
  result.words.reserve(ordered.size());
  result.coefficients.reserve(ordered.size());
  for (auto& [word, coefficient] : ordered) {
    result.words.push_back(std::move(word));
    result.coefficients.push_back(coefficient);
  }
  return result;
}

}  // namespace

BosonMapResult boson_map_hamiltonian(const BosonMapping& mapping,
                                     double core_energy, const double* one_body,
                                     const int* two_body_indices,
                                     const double* two_body_values,
                                     std::size_t num_entries,
                                     std::size_t num_modes, double threshold,
                                     double integral_threshold) {
  QDK_LOG_TRACE_ENTERING();
  if (num_modes != mapping.num_modes()) {
    throw std::invalid_argument("boson_map_hamiltonian: the Hamiltonian has " +
                                std::to_string(num_modes) +
                                " modes but the mapping was built for " +
                                std::to_string(mapping.num_modes()));
  }
  if (one_body == nullptr) {
    throw std::invalid_argument(
        "boson_map_hamiltonian: one_body must not be null");
  }
  if (num_entries > 0 &&
      (two_body_indices == nullptr || two_body_values == nullptr)) {
    throw std::invalid_argument(
        "boson_map_hamiltonian: two-body buffers must not be null when "
        "num_entries > 0");
  }

  TermAccumulator sink;
  if (std::abs(core_energy) >= integral_threshold) {
    sink[SparsePauliWord{}] += std::complex<double>(core_energy, 0.0);
  }

  // One-body: sum_pq h_pq b_p^dag b_q.
  for (std::size_t p = 0; p < num_modes; ++p) {
    for (std::size_t q = 0; q < num_modes; ++q) {
      const double value = one_body[p * num_modes + q];
      if (std::abs(value) < integral_threshold) {
        continue;
      }
      accumulate(sink, mapping.ladder_product({{p, true}, {q, false}}), value);
    }
  }

  // Two-body: (1/2) sum_pqrs (pq|rs) b_p^dag b_r^dag b_s b_q.
  for (std::size_t entry = 0; entry < num_entries; ++entry) {
    const double value = two_body_values[entry];
    if (std::abs(value) < integral_threshold) {
      continue;
    }
    const int* idx = two_body_indices + 4 * entry;
    for (int component = 0; component < 4; ++component) {
      if (idx[component] < 0 ||
          static_cast<std::size_t>(idx[component]) >= num_modes) {
        throw std::invalid_argument("boson_map_hamiltonian: two-body index " +
                                    std::to_string(idx[component]) +
                                    " is outside [0, " +
                                    std::to_string(num_modes) + ")");
      }
    }
    const auto p = static_cast<std::size_t>(idx[0]);
    const auto q = static_cast<std::size_t>(idx[1]);
    const auto r = static_cast<std::size_t>(idx[2]);
    const auto s = static_cast<std::size_t>(idx[3]);
    accumulate(
        sink,
        mapping.ladder_product({{p, true}, {r, true}, {s, false}, {q, false}}),
        0.5 * value);
  }

  return finalize(sink, threshold);
}

BosonMapResult boson_map_hamiltonian(const BosonMapping& mapping,
                                     const Hamiltonian& hamiltonian,
                                     double threshold,
                                     double integral_threshold) {
  QDK_LOG_TRACE_ENTERING();
  const auto orbitals = hamiltonian.get_orbitals();
  if (!orbitals) {
    throw std::invalid_argument(
        "boson_map_hamiltonian: the Hamiltonian has no orbital basis");
  }
  const std::size_t num_modes = orbitals->num_modes();
  if (num_modes != mapping.num_modes()) {
    throw std::invalid_argument("boson_map_hamiltonian: the Hamiltonian has " +
                                std::to_string(num_modes) +
                                " modes but the mapping was built for " +
                                std::to_string(mapping.num_modes()));
  }
  // The occupation cutoff lives on the basis: if the Hamiltonian carries a
  // bosonic basis, its cutoff must agree with the mapping's.
  if (const auto* modes = dynamic_cast<const BosonicModes*>(orbitals.get())) {
    mapping.validate_basis(*modes);
  }

  // One-body integrals, flattened row-major.
  std::vector<double> one_body(num_modes * num_modes, 0.0);
  std::vector<int> two_body_indices;
  std::vector<double> two_body_values;

  if (hamiltonian.has_container_type<SparseHamiltonianContainer>()) {
    const auto& container =
        hamiltonian.get_container<SparseHamiltonianContainer>();
    const auto& sparse_one_body = container.sparse_one_body_integrals();
    for (int k = 0; k < sparse_one_body.outerSize(); ++k) {
      for (Eigen::SparseMatrix<double>::InnerIterator it(sparse_one_body, k);
           it; ++it) {
        one_body[static_cast<std::size_t>(it.row()) * num_modes +
                 static_cast<std::size_t>(it.col())] = it.value();
      }
    }
    if (container.has_two_body_integrals()) {
      const auto two_body = container.sparse_two_body_integrals();
      two_body_indices.reserve(4 * two_body.size());
      two_body_values.reserve(two_body.size());
      for (const auto& [index, value] : two_body) {
        two_body_indices.push_back(std::get<0>(index));
        two_body_indices.push_back(std::get<1>(index));
        two_body_indices.push_back(std::get<2>(index));
        two_body_indices.push_back(std::get<3>(index));
        two_body_values.push_back(value);
      }
    }
  } else {
    const auto& [h_alpha, h_beta] = hamiltonian.get_one_body_integrals();
    (void)h_beta;
    for (std::size_t p = 0; p < num_modes; ++p) {
      for (std::size_t q = 0; q < num_modes; ++q) {
        one_body[p * num_modes + q] =
            h_alpha(static_cast<Eigen::Index>(p), static_cast<Eigen::Index>(q));
      }
    }
    if (hamiltonian.has_two_body_integrals()) {
      const auto& [aaaa, aabb, bbbb] = hamiltonian.get_two_body_integrals();
      (void)aabb;
      (void)bbbb;
      const std::size_t n2 = num_modes * num_modes;
      const std::size_t n3 = n2 * num_modes;
      for (std::size_t p = 0; p < num_modes; ++p) {
        for (std::size_t q = 0; q < num_modes; ++q) {
          for (std::size_t r = 0; r < num_modes; ++r) {
            for (std::size_t s = 0; s < num_modes; ++s) {
              const double value = aaaa(static_cast<Eigen::Index>(
                  p * n3 + q * n2 + r * num_modes + s));
              if (std::abs(value) < integral_threshold) {
                continue;
              }
              two_body_indices.push_back(static_cast<int>(p));
              two_body_indices.push_back(static_cast<int>(q));
              two_body_indices.push_back(static_cast<int>(r));
              two_body_indices.push_back(static_cast<int>(s));
              two_body_values.push_back(value);
            }
          }
        }
      }
    }
  }

  return boson_map_hamiltonian(mapping, 0.0, one_body.data(),
                               two_body_indices.data(), two_body_values.data(),
                               two_body_values.size(), num_modes, threshold,
                               integral_threshold);
}

}  // namespace qdk::chemistry::data
