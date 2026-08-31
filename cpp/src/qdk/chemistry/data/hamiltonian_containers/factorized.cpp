// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <Eigen/Dense>
#include <cmath>
#include <cstddef>
#include <memory>
#include <qdk/chemistry/data/hamiltonian_containers/factorized.hpp>
#include <qdk/chemistry/data/orbitals.hpp>
#include <qdk/chemistry/utils/logger.hpp>
#include <stdexcept>
#include <string>

#include "../hdf5_serialization.hpp"
#include "../json_serialization.hpp"

namespace qdk::chemistry::data {

namespace {

/// Cross-check the rank/basis/copy recorded in a serialized payload
/// against the buffers. Shared by the JSON and HDF5 readers.
void validate_stored_shape(std::size_t num_ranks, std::size_t num_bases,
                           std::size_t num_copies, Eigen::Index u_size,
                           Eigen::Index w_size, const Eigen::MatrixXd& wb) {
  const auto mismatch = [](const std::string& what, std::size_t expected,
                           std::size_t found) {
    throw std::invalid_argument("FactorizedHamiltonianContainer: serialized " +
                                what + " is " + std::to_string(found) +
                                " but the stored buffers imply " +
                                std::to_string(expected) + ".");
  };

  if (static_cast<std::size_t>(wb.rows()) != num_ranks) {
    mismatch("num_ranks", static_cast<std::size_t>(wb.rows()), num_ranks);
  }
  if (static_cast<std::size_t>(wb.cols()) != num_copies) {
    mismatch("num_copies", static_cast<std::size_t>(wb.cols()), num_copies);
  }
  if (num_ranks == 0 || num_bases == 0) {
    throw std::invalid_argument(
        "FactorizedHamiltonianContainer: serialized num_ranks and num_bases "
        "must both be non-zero.");
  }
  const std::size_t expected_w = num_ranks * num_bases * num_copies;
  if (static_cast<std::size_t>(w_size) != expected_w) {
    mismatch("w_matrices length", expected_w, static_cast<std::size_t>(w_size));
  }
  if (static_cast<std::size_t>(u_size) % (num_ranks * num_bases) != 0) {
    throw std::invalid_argument(
        "FactorizedHamiltonianContainer: serialized u_matrices length " +
        std::to_string(u_size) + " is not a multiple of num_ranks*num_bases.");
  }
}

}  // namespace

FactorizedHamiltonianContainer::FactorizedHamiltonianContainer(
    double core_energy, const Eigen::VectorXd& u_matrices,
    const Eigen::VectorXd& w_matrices, const Eigen::MatrixXd& wb_matrix,
    const Eigen::MatrixXd& one_body_integrals,
    const Eigen::MatrixXd& inactive_fock_matrix,
    std::shared_ptr<Orbitals> orbitals, const Eigen::VectorXd& signs,
    double energy_gap, HamiltonianType type)
    : HamiltonianContainer(one_body_integrals, orbitals, core_energy,
                           inactive_fock_matrix, type),
      _u(u_matrices),
      _w(w_matrices),
      _wb(wb_matrix),
      _signs(signs.size() > 0 ? signs
                              : Eigen::VectorXd::Ones(wb_matrix.rows())),
      _energy_gap(energy_gap) {
  QDK_LOG_TRACE_ENTERING();

  validate_integral_dimensions();
  validate_restrictedness_consistency();
  validate_active_space_dimensions();

  if (!is_valid()) {
    throw std::invalid_argument(
        "Tried to generate invalid factorized Hamiltonian object.");
  }
}

// === HamiltonianContainer overrides ===

std::unique_ptr<HamiltonianContainer> FactorizedHamiltonianContainer::clone()
    const {
  QDK_LOG_TRACE_ENTERING();
  auto [h1_alpha, h1_beta] = get_one_body_integrals();
  Eigen::MatrixXd fock_alpha = Eigen::MatrixXd::Zero(0, 0);
  if (has_inactive_fock_matrix()) {
    auto [fa, fb] = get_inactive_fock_matrix();
    fock_alpha = fa;
  }
  return std::make_unique<FactorizedHamiltonianContainer>(
      _core_energy, _u, _w, _wb, h1_alpha, fock_alpha, _orbitals, _signs,
      _energy_gap, _type);
}

std::string FactorizedHamiltonianContainer::get_container_type() const {
  QDK_LOG_TRACE_ENTERING();
  return "factorized";
}

std::tuple<const Eigen::VectorXd&, const Eigen::VectorXd&,
           const Eigen::VectorXd&>
FactorizedHamiltonianContainer::get_two_body_integrals() const {
  QDK_LOG_TRACE_ENTERING();
  if (!has_two_body_integrals()) {
    throw std::runtime_error(
        "Factorized Hamiltonian two-body integrals not set");
  }
  if (!_cached_two_body) {
    _build_two_body_cache();
  }
  // Restricted: all three channels share the same data
  return std::make_tuple(std::cref(*_cached_two_body),
                         std::cref(*_cached_two_body),
                         std::cref(*_cached_two_body));
}

double FactorizedHamiltonianContainer::get_two_body_element(
    unsigned i, unsigned j, unsigned k, unsigned l, SpinChannel channel) const {
  QDK_LOG_TRACE_ENTERING();
  if (!has_two_body_integrals()) {
    throw std::runtime_error(
        "Factorized Hamiltonian two-body integrals not set");
  }
  size_t norb = get_num_orbitals();
  if (i >= norb || j >= norb || k >= norb || l >= norb) {
    throw std::out_of_range("Orbital index out of range");
  }
  if (!_cached_two_body) {
    _build_two_body_cache();
  }
  size_t idx = i * norb * norb * norb + j * norb * norb + k * norb + l;
  return (*_cached_two_body)(idx);
}

bool FactorizedHamiltonianContainer::has_two_body_integrals() const {
  QDK_LOG_TRACE_ENTERING();
  return _u.size() > 0 && _w.size() > 0;
}

bool FactorizedHamiltonianContainer::is_restricted() const {
  QDK_LOG_TRACE_ENTERING();
  return true;  // Factorized container is always restricted (spin-free)
}

bool FactorizedHamiltonianContainer::is_valid() const {
  QDK_LOG_TRACE_ENTERING();
  // Check if essential data is present
  if (!has_one_body_integrals()) return false;
  if (!has_two_body_integrals()) return false;
  if (!has_orbitals()) return false;

  // Check dimension consistency
  try {
    validate_integral_dimensions();
  } catch (const std::exception&) {
    return false;
  }

  return true;
}

// === Two-body reconstruction ===

Eigen::VectorXd FactorizedHamiltonianContainer::reconstruct_two_body_integrals()
    const {
  //   h2_{pqrs} = Σ_{r,c} s_r M^{rc}_{pq} M^{rc}_{rs},
  //   M^{rc}_{pq} = Σ_b W^r_{bc} U^r_{bp} U^r_{bq},
  size_t norb = get_num_orbitals();
  size_t R = get_num_ranks();
  size_t B = get_num_bases();
  size_t C = get_num_copies();
  size_t norb2 = norb * norb;

  Eigen::VectorXd h2 = Eigen::VectorXd::Zero(norb2 * norb2);
  Eigen::Map<
      Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
      h2_matrix(h2.data(), norb2, norb2);

  for (size_t r = 0; r < R; ++r) {
    Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,
                                   Eigen::RowMajor>>
        Ur(_u.data() + r * B * norb, B, norb);  // U^r : [B x N]
    Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,
                                   Eigen::RowMajor>>
        Wr(_w.data() + r * B * C, B, C);  // W^r : [B x C]

    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
        pair_products(B, norb2);
    for (size_t b = 0; b < B; ++b) {
      Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,
                               Eigen::RowMajor>>
          basis_outer_product(pair_products.row(b).data(), norb, norb);
      basis_outer_product.noalias() = Ur.row(b).transpose() * Ur.row(b);
    }

    Eigen::MatrixXd M = Wr.transpose() * pair_products;
    h2_matrix.noalias() += _signs(r) * (M.transpose() * M);
  }

  return h2;
}

void FactorizedHamiltonianContainer::_build_two_body_cache() const {
  QDK_LOG_TRACE_ENTERING();
  _cached_two_body =
      std::make_shared<Eigen::VectorXd>(reconstruct_two_body_integrals());
}

// === Factorized-specific accessors ===

const Eigen::VectorXd& FactorizedHamiltonianContainer::get_u_matrices() const {
  return _u;
}

const Eigen::VectorXd& FactorizedHamiltonianContainer::get_w_matrices() const {
  return _w;
}

const Eigen::MatrixXd& FactorizedHamiltonianContainer::get_wb_matrix() const {
  return _wb;
}

const Eigen::VectorXd& FactorizedHamiltonianContainer::get_signs() const {
  return _signs;
}

size_t FactorizedHamiltonianContainer::get_num_orbitals() const {
  return _orbitals->get_active_space_indices().first.size();
}

size_t FactorizedHamiltonianContainer::get_num_ranks() const {
  return static_cast<size_t>(_wb.rows());
}

size_t FactorizedHamiltonianContainer::get_num_bases() const {
  // Inferred from U [R*B*N]; guard against an empty factorization.
  size_t denom = get_num_ranks() * get_num_orbitals();
  return denom == 0 ? 0 : static_cast<size_t>(_u.size()) / denom;
}

size_t FactorizedHamiltonianContainer::get_num_copies() const {
  return static_cast<size_t>(_wb.cols());
}

double FactorizedHamiltonianContainer::get_energy_gap() const {
  return _energy_gap;
}

double FactorizedHamiltonianContainer::get_lambda() const {
  // norm (Eq. 34):
  // Λ = Σ|eig(h1_prime)| + 1/4 Σ_{rc} (|WB^{rc}| + Σ_b |W^{rc}_b|)^2.
  //
  // The per-rank signs are absent on purpose: every two-body contribution
  // enters through an absolute value and |s_r| = 1, so a negated fragment has
  // exactly the same 1-norm. They do still reach this function indirectly,
  // through get_h1_prime().
  Eigen::MatrixXd h1p = get_h1_prime();
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(h1p);
  if (solver.info() != Eigen::Success) {
    throw std::runtime_error(
        "FactorizedHamiltonianContainer::get_lambda: failed to diagonalize the "
        "adjusted one-body matrix.");
  }
  double one_body_norm = solver.eigenvalues().array().abs().sum();

  size_t R = get_num_ranks();
  size_t B = get_num_bases();
  size_t C = get_num_copies();

  auto W = [&](size_t r, size_t b, size_t c) -> double {
    return _w(r * B * C + b * C + c);
  };

  double two_body_norm = 0.0;
  for (size_t r = 0; r < R; ++r) {
    for (size_t c = 0; c < C; ++c) {
      double sum_abs_w = 0.0;
      for (size_t b = 0; b < B; ++b) {
        sum_abs_w += std::abs(W(r, b, c));
      }
      double term = std::abs(_wb(r, c)) + sum_abs_w;
      two_body_norm += term * term;
    }
  }

  return one_body_norm + 0.25 * two_body_norm;
}

double FactorizedHamiltonianContainer::get_lambda_eff() const {
  // Effective SOS-walk normalization (Eq. 12): λ_eff = √(E_gap·(2Λ - E_gap)).
  double lambda = get_lambda();
  if (_energy_gap <= 0.0) {
    throw std::runtime_error("E_gap must be positive for a valid SOS walk");
  }
  if (_energy_gap >= 2.0 * lambda) {
    throw std::runtime_error(
        "E_gap must be less than 2*Lambda for a valid SOS walk");
  }
  return std::sqrt(_energy_gap * (2.0 * lambda - _energy_gap));
}

Eigen::MatrixXd FactorizedHamiltonianContainer::get_h1_prime() const {
  // Adjusted one-body matrix h^(1)' (Eq. 37). Writing the rank-r copy-c leaf as
  //   M^{rc}_{pq} = Σ_{b∈[B]} w_b^{rc} u^r_{b,p} u^r_{b,q},
  // the three accumulated corrections are
  //   h1'_{pq} = h1_{pq} - ½ Σ_{rc} s_r (M^{rc} M^{rc})_{pq}  (a) normal-order
  //                      + Σ_{rc} s_r tr(M^{rc}) M^{rc}_{pq}  (b)
  //                      - Σ_{rc} s_r wB^{rc} M^{rc}_{pq}     (c)
  //
  // All three carry the rank's sign s_r: they are the one-body residue of the
  // single fragment operator s_r (Σ_b w_b n_b - wB)², so negating the fragment
  // negates the square, the cross term and the trace correction together.
  //
  // Term (a) has no counterpart in Eq. 37 as printed, because the paper writes
  // the two-body term as a plain product of E operators while this container
  // stores h2 = (pq|rs) normal-ordered, i.e.
  //   H = E_core + Σ h1_{pq} E_pq + ½ Σ h2_{pqrs} (E_pq E_rs - δ_qr E_ps).
  // Unpacking that -½ δ_qr E_ps piece leaves the one-body remainder
  // -½ Σ_s h2_{pssq} = -½ Σ_{rc} s_r (M^{rc} M^{rc})_{pq}. Dropping it would
  // make h1' -- and hence Λ -- silently wrong. See also the docstring in
  // factorized.hpp, which lists all three terms.
  size_t norb = get_num_orbitals();
  size_t R = get_num_ranks();
  size_t B = get_num_bases();
  size_t C = get_num_copies();

  auto [h1_alpha, h1_beta] = get_one_body_integrals();
  Eigen::MatrixXd h1 = 0.5 * (h1_alpha + h1_beta);

  // h2[p,q,r,s] = Σ_rank Σ_c s_rank * M_rc[p,q] * M_rc[r,s]
  // where M_rc[p,q] = Σ_b W[rank,b,c] * U[rank,b,p] * U[rank,b,q]
  //                  = (U_r^T * diag(w_c) * U_r)[p,q]
  for (size_t r = 0; r < R; ++r) {
    // Map U_r as [B x norb] row-major: _u[r*B*norb + b*norb + p]
    Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,
                                   Eigen::RowMajor>>
        Ur(_u.data() + r * B * norb, B, norb);

    const double sign = _signs(r);

    for (size_t c = 0; c < C; ++c) {
      // Build scaled = diag(w[r,:,c]) * U_r,  shape [B x norb]
      Eigen::MatrixXd scaled(B, norb);
      for (size_t b = 0; b < B; ++b) {
        double w_rbc = _w(r * B * C + b * C + c);
        scaled.row(b) = w_rbc * Ur.row(b);
      }

      // M_rc = U_r^T * scaled = U_r^T * diag(w_c) * U_r,  shape [N x N]
      Eigen::MatrixXd Mrc = Ur.transpose() * scaled;

      // h1 -= 0.5 * s_r * M_rc^2  (contraction of einsum "prrq")
      h1.noalias() -= 0.5 * sign * (Mrc * Mrc);

      // h1 += s_r * tr(M_rc) * M_rc  (contraction of einsum "pqrr")
      h1 += sign * Mrc.trace() * Mrc;

      // h1 -= s_r * wb[r,c] * M_rc  (wb term)
      h1 -= sign * _wb(r, c) * Mrc;
    }
  }

  return h1;
}

// === Serialization ===

nlohmann::json FactorizedHamiltonianContainer::to_json() const {
  QDK_LOG_TRACE_ENTERING();
  nlohmann::json j;
  j["version"] = SERIALIZATION_VERSION;
  j["container_type"] = "factorized";

  auto [h1_alpha, h1_beta] = get_one_body_integrals();
  j["one_body_integrals"] = matrix_to_json(h1_alpha);
  j["u_matrices"] = vector_to_json(_u);
  j["w_matrices"] = vector_to_json(_w);
  j["wb_matrix"] = matrix_to_json(_wb);
  j["signs"] = vector_to_json(_signs);
  j["num_ranks"] = get_num_ranks();
  j["num_bases"] = get_num_bases();
  j["num_copies"] = get_num_copies();
  j["core_energy"] = _core_energy;
  j["energy_gap"] = _energy_gap;
  j["type"] =
      (_type == HamiltonianType::Hermitian) ? "Hermitian" : "NonHermitian";
  j["is_restricted"] = is_restricted();

  // Orbitals are guaranteed non-null by the base container constructor, so
  // this is unconditional and from_json() can read it back the same way.
  j["orbitals"] = _orbitals->to_json();

  if (has_inactive_fock_matrix()) {
    auto [fock_a, fock_b] = get_inactive_fock_matrix();
    j["inactive_fock_matrix"] = matrix_to_json(fock_a);
  }

  return j;
}

std::unique_ptr<FactorizedHamiltonianContainer>
FactorizedHamiltonianContainer::from_json(const nlohmann::json& j) {
  QDK_LOG_TRACE_ENTERING();
  validate_serialization_version(SERIALIZATION_VERSION, j.at("version"));

  auto h1 = json_to_matrix(j.at("one_body_integrals"));
  auto u = json_to_vector(j.at("u_matrices"));
  auto w = json_to_vector(j.at("w_matrices"));
  auto wb = json_to_matrix(j.at("wb_matrix"));
  // Optional: an absent sign vector means an all-positive factorization.
  Eigen::VectorXd signs;
  if (j.contains("signs")) {
    signs = json_to_vector(j.at("signs"));
  }
  double core_energy = j.at("core_energy");
  double energy_gap = j.at("energy_gap");

  // A JSON payload written before "type" was serialized carries no such key.
  // Those all predate the non-Hermitian support, so Hermitian is the correct
  // reading rather than merely a convenient default.
  HamiltonianType type = HamiltonianType::Hermitian;
  if (j.contains("type") && j.at("type").get<std::string>() == "NonHermitian") {
    type = HamiltonianType::NonHermitian;
  }

  validate_stored_shape(j.at("num_ranks").get<std::size_t>(),
                        j.at("num_bases").get<std::size_t>(),
                        j.at("num_copies").get<std::size_t>(), u.size(),
                        w.size(), wb);

  auto orbitals = Orbitals::from_json(j.at("orbitals"));

  Eigen::MatrixXd fock = Eigen::MatrixXd::Zero(0, 0);
  if (j.contains("inactive_fock_matrix")) {
    fock = json_to_matrix(j.at("inactive_fock_matrix"));
  }

  return std::make_unique<FactorizedHamiltonianContainer>(
      core_energy, u, w, wb, h1, fock, orbitals, signs, energy_gap, type);
}

void FactorizedHamiltonianContainer::to_hdf5(H5::Group& group) const {
  QDK_LOG_TRACE_ENTERING();
  H5::StrType string_type(H5::PredType::C_S1, H5T_VARIABLE);

  H5::Attribute version_attr =
      group.createAttribute("version", string_type, H5::DataSpace(H5S_SCALAR));
  std::string v(SERIALIZATION_VERSION);
  version_attr.write(string_type, v);

  H5::Attribute ct_attr = group.createAttribute("container_type", string_type,
                                                H5::DataSpace(H5S_SCALAR));
  std::string ct("factorized");
  ct_attr.write(string_type, ct);

  // Scalars live in a "metadata" subgroup, matching the other Hamiltonian
  // containers (cholesky.cpp, canonical_four_center.cpp, sparse.cpp) and the
  // layout the Python readers expect.
  H5::Group metadata_group = group.createGroup("metadata");
  metadata_group
      .createAttribute("core_energy", H5::PredType::NATIVE_DOUBLE,
                       H5::DataSpace(H5S_SCALAR))
      .write(H5::PredType::NATIVE_DOUBLE, &_core_energy);
  metadata_group
      .createAttribute("energy_gap", H5::PredType::NATIVE_DOUBLE,
                       H5::DataSpace(H5S_SCALAR))
      .write(H5::PredType::NATIVE_DOUBLE, &_energy_gap);

  hsize_t r_val = get_num_ranks(), b_val = get_num_bases(),
          c_val = get_num_copies();
  metadata_group
      .createAttribute("num_ranks", H5::PredType::NATIVE_HSIZE,
                       H5::DataSpace(H5S_SCALAR))
      .write(H5::PredType::NATIVE_HSIZE, &r_val);
  metadata_group
      .createAttribute("num_bases", H5::PredType::NATIVE_HSIZE,
                       H5::DataSpace(H5S_SCALAR))
      .write(H5::PredType::NATIVE_HSIZE, &b_val);
  metadata_group
      .createAttribute("num_copies", H5::PredType::NATIVE_HSIZE,
                       H5::DataSpace(H5S_SCALAR))
      .write(H5::PredType::NATIVE_HSIZE, &c_val);

  std::string type_str =
      (_type == HamiltonianType::Hermitian) ? "Hermitian" : "NonHermitian";
  H5::StrType type_string_type(H5::PredType::C_S1, type_str.length() + 1);
  metadata_group
      .createAttribute("type", type_string_type, H5::DataSpace(H5S_SCALAR))
      .write(type_string_type, type_str.c_str());

  hbool_t is_restricted_flag = is_restricted() ? 1 : 0;
  metadata_group
      .createAttribute("is_restricted", H5::PredType::NATIVE_HBOOL,
                       H5::DataSpace(H5S_SCALAR))
      .write(H5::PredType::NATIVE_HBOOL, &is_restricted_flag);

  auto [h1_alpha, h1_beta] = get_one_body_integrals();
  save_matrix_to_group(group, "one_body_integrals", h1_alpha);
  save_vector_to_group(group, "u_matrices", _u);
  save_vector_to_group(group, "w_matrices", _w);
  save_matrix_to_group(group, "wb_matrix", _wb);
  save_vector_to_group(group, "signs", _signs);

  if (has_inactive_fock_matrix()) {
    auto [fock_a, fock_b] = get_inactive_fock_matrix();
    save_matrix_to_group(group, "inactive_fock_matrix", fock_a);
  }

  // Orbitals are guaranteed non-null by the base container constructor.
  H5::Group orb_group = group.createGroup("orbitals");
  _orbitals->to_hdf5(orb_group);
}

std::unique_ptr<FactorizedHamiltonianContainer>
FactorizedHamiltonianContainer::from_hdf5(H5::Group& group) {
  QDK_LOG_TRACE_ENTERING();

  H5::StrType string_type(H5::PredType::C_S1, H5T_VARIABLE);
  H5::Attribute version_attr = group.openAttribute("version");
  std::string version;
  version_attr.read(string_type, version);
  validate_serialization_version(SERIALIZATION_VERSION, version);

  double core_energy, energy_gap;
  H5::Group metadata_group = group.openGroup("metadata");
  metadata_group.openAttribute("core_energy")
      .read(H5::PredType::NATIVE_DOUBLE, &core_energy);
  metadata_group.openAttribute("energy_gap")
      .read(H5::PredType::NATIVE_DOUBLE, &energy_gap);

  hsize_t num_ranks, num_bases, num_copies;
  metadata_group.openAttribute("num_ranks")
      .read(H5::PredType::NATIVE_HSIZE, &num_ranks);
  metadata_group.openAttribute("num_bases")
      .read(H5::PredType::NATIVE_HSIZE, &num_bases);
  metadata_group.openAttribute("num_copies")
      .read(H5::PredType::NATIVE_HSIZE, &num_copies);

  // An HDF5 payload written before "type" was serialized carries no such
  // attribute. Those all predate the non-Hermitian support, so Hermitian is the
  // correct reading rather than merely a convenient default.
  HamiltonianType type = HamiltonianType::Hermitian;
  if (metadata_group.attrExists("type")) {
    H5::Attribute type_attr = metadata_group.openAttribute("type");
    std::string type_str;
    type_attr.read(type_attr.getStrType(), type_str);
    if (type_str == "NonHermitian") {
      type = HamiltonianType::NonHermitian;
    }
  }

  auto h1 = load_matrix_from_group(group, "one_body_integrals");
  auto u = load_vector_from_group(group, "u_matrices");
  auto w = load_vector_from_group(group, "w_matrices");
  auto wb = load_matrix_from_group(group, "wb_matrix");
  // Optional: an absent sign vector means an all-positive factorization.
  Eigen::VectorXd signs;
  if (group.nameExists("signs")) {
    signs = load_vector_from_group(group, "signs");
  }

  validate_stored_shape(num_ranks, num_bases, num_copies, u.size(), w.size(),
                        wb);

  H5::Group orb_group = group.openGroup("orbitals");
  std::shared_ptr<Orbitals> orbitals = Orbitals::from_hdf5(orb_group);

  Eigen::MatrixXd fock = Eigen::MatrixXd::Zero(0, 0);
  if (group.nameExists("inactive_fock_matrix")) {
    fock = load_matrix_from_group(group, "inactive_fock_matrix");
  }

  return std::make_unique<FactorizedHamiltonianContainer>(
      core_energy, u, w, wb, h1, fock, orbitals, signs, energy_gap, type);
}

// === Validation ===

void FactorizedHamiltonianContainer::validate_integral_dimensions() const {
  QDK_LOG_TRACE_ENTERING();
  HamiltonianContainer::validate_integral_dimensions();

  size_t norb = get_num_orbitals();
  size_t R = get_num_ranks();
  size_t C = get_num_copies();

  if (R == 0) {
    throw std::invalid_argument(
        "WB matrix must have at least one rank (row), got 0.");
  }
  if (norb == 0) {
    throw std::invalid_argument("Number of active orbitals must be positive.");
  }
  if (static_cast<size_t>(_u.size()) % (R * norb) != 0) {
    throw std::invalid_argument(
        "U matrices size " + std::to_string(_u.size()) +
        " is not a multiple of R*N = " + std::to_string(R * norb) +
        " (R=" + std::to_string(R) + ", N=" + std::to_string(norb) + ").");
  }
  size_t B = static_cast<size_t>(_u.size()) / (R * norb);
  size_t expected_w = R * B * C;
  if (static_cast<size_t>(_w.size()) != expected_w) {
    throw std::invalid_argument(
        "W matrices size mismatch: expected R*B*C = " +
        std::to_string(expected_w) + " (R=" + std::to_string(R) +
        ", B=" + std::to_string(B) + ", C=" + std::to_string(C) + "), got " +
        std::to_string(_w.size()) + ".");
  }
  if (static_cast<size_t>(_signs.size()) != R) {
    throw std::invalid_argument(
        "Sign vector size mismatch: expected one sign per rank R = " +
        std::to_string(R) + ", got " + std::to_string(_signs.size()) + ".");
  }
  for (Eigen::Index r = 0; r < _signs.size(); ++r) {
    if (_signs(r) != 1.0 && _signs(r) != -1.0) {
      throw std::invalid_argument("Sign for rank " + std::to_string(r) +
                                  " must be exactly +1.0 or -1.0, got " +
                                  std::to_string(_signs(r)) + ".");
    }
  }
}

// === Hashing ===

void FactorizedHamiltonianContainer::hash_update(
    qdk::chemistry::utils::HashContext& ctx) const {
  HamiltonianContainer::hash_update(ctx);
  hash_value(ctx, get_container_type());
  hash_value(ctx, static_cast<int64_t>(get_num_ranks()));
  hash_value(ctx, static_cast<int64_t>(get_num_bases()));
  hash_value(ctx, static_cast<int64_t>(get_num_copies()));
  hash_value(ctx, _energy_gap);
  hash_value(ctx, _u);
  hash_value(ctx, _w);
  hash_value(ctx, _wb);
  hash_value(ctx, _signs);
}

}  // namespace qdk::chemistry::data
