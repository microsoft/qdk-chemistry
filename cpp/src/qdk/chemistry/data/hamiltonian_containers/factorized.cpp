// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <Eigen/Dense>
#include <cmath>
#include <memory>
#include <qdk/chemistry/data/hamiltonian_containers/factorized.hpp>
#include <qdk/chemistry/data/orbitals.hpp>
#include <qdk/chemistry/utils/logger.hpp>
#include <stdexcept>

#include "../hdf5_serialization.hpp"
#include "../json_serialization.hpp"

namespace qdk::chemistry::data {

FactorizedHamiltonianContainer::FactorizedHamiltonianContainer(
    const Eigen::MatrixXd& one_body_integrals,
    const Eigen::VectorXd& u_matrices, const Eigen::VectorXd& w_matrices,
    const Eigen::MatrixXd& wb_matrix, size_t num_ranks, size_t num_bases,
    size_t num_copies, std::shared_ptr<Orbitals> orbitals, double core_energy,
    const Eigen::MatrixXd& inactive_fock_matrix, double bliss_core_shift,
    double energy_gap, HamiltonianType type)
    : HamiltonianContainer(one_body_integrals, orbitals, core_energy,
                           inactive_fock_matrix, type),
      _u(u_matrices),
      _w(w_matrices),
      _wb(wb_matrix),
      _num_ranks(num_ranks),
      _num_bases(num_bases),
      _num_copies(num_copies),
      _bliss_core_shift(bliss_core_shift),
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
  // Reconstruct from stored data. The base-class SBT one-body
  // is immutable/shared, but for clone we go through the dense ctor.
  auto [h1_alpha, h1_beta] = get_one_body_integrals();
  auto [fock_alpha, fock_beta] = get_inactive_fock_matrix();
  return std::make_unique<FactorizedHamiltonianContainer>(
      h1_alpha, _u, _w, _wb, _num_ranks, _num_bases, _num_copies, _orbitals,
      _core_energy, fock_alpha, _bliss_core_shift, _energy_gap, _type);
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
    throw std::runtime_error("Factorized two-body data is not set");
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
    throw std::runtime_error("Two-body integrals are not set");
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
  if (!has_one_body_integrals()) return false;
  if (!has_orbitals()) return false;
  size_t norb = get_num_orbitals();
  if (_u.size() != static_cast<Eigen::Index>(_num_ranks * _num_bases * norb))
    return false;
  if (_w.size() !=
      static_cast<Eigen::Index>(_num_ranks * _num_bases * _num_copies))
    return false;
  if (_wb.rows() != static_cast<Eigen::Index>(_num_ranks) ||
      _wb.cols() != static_cast<Eigen::Index>(_num_copies))
    return false;
  return true;
}

// === Two-body reconstruction ===

Eigen::VectorXd FactorizedHamiltonianContainer::reconstruct_two_body_integrals()
    const {
  size_t norb = get_num_orbitals();
  size_t R = _num_ranks;
  size_t B = _num_bases;
  size_t C = _num_copies;
  size_t norb4 = norb * norb * norb * norb;

  Eigen::VectorXd h2 = Eigen::VectorXd::Zero(norb4);

  // Map flat arrays to 3D views:
  // U[r][b][p] stored as flat [R*B*N], row-major in (r,b,p)
  // W[r][b][c] stored as flat [R*B*C], row-major in (r,b,c)
  auto U = [&](size_t r, size_t b, size_t p) -> double {
    return _u(r * B * norb + b * norb + p);
  };
  auto W = [&](size_t r, size_t b, size_t c) -> double {
    return _w(r * B * C + b * C + c);
  };

  // h2_{pqrs} = Σ_r Σ_c (Σ_b U^r_{bp} U^r_{bq} W^r_{bc})
  //                      (Σ_{b'} U^r_{b'r} U^r_{b's} W^r_{b'c})
  for (size_t r = 0; r < R; ++r) {
    // Build intermediate: M^r_c[p][q] = Σ_b U^r_{bp} U^r_{bq} W^r_{bc}
    // Shape: [C][norb][norb]
    Eigen::MatrixXd M = Eigen::MatrixXd::Zero(C, norb * norb);
    for (size_t b = 0; b < B; ++b) {
      for (size_t c = 0; c < C; ++c) {
        double w_rbc = W(r, b, c);
        for (size_t p = 0; p < norb; ++p) {
          double u_rbp = U(r, b, p);
          for (size_t q = 0; q < norb; ++q) {
            M(c, p * norb + q) += u_rbp * U(r, b, q) * w_rbc;
          }
        }
      }
    }

    // h2_{pqrs} += Σ_c M^r_c[p][q] * M^r_c[r][s]
    // = M^T * M where M is [C x norb^2]
    // Result is [norb^2 x norb^2], stored in h2
    for (size_t c = 0; c < C; ++c) {
      Eigen::Map<const Eigen::MatrixXd> Mc(M.row(c).data(), norb, norb);
      // Flatten: h2_{pq,rs} += M_c[pq] * M_c[rs]
      Eigen::Map<const Eigen::VectorXd> Mc_flat(M.row(c).data(), norb * norb);
      // Outer product: h2 += Mc_flat * Mc_flat^T (as flat norb^4)
      for (size_t pq = 0; pq < norb * norb; ++pq) {
        for (size_t rs = 0; rs < norb * norb; ++rs) {
          h2(pq * norb * norb + rs) += Mc_flat(pq) * Mc_flat(rs);
        }
      }
    }
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

size_t FactorizedHamiltonianContainer::get_num_orbitals() const {
  return _orbitals->get_active_space_indices().first.size();
}

size_t FactorizedHamiltonianContainer::get_num_ranks() const { return _num_ranks; }

size_t FactorizedHamiltonianContainer::get_num_bases() const { return _num_bases; }

size_t FactorizedHamiltonianContainer::get_num_copies() const {
  return _num_copies;
}

double FactorizedHamiltonianContainer::get_bliss_core_shift() const {
  return _bliss_core_shift;
}

double FactorizedHamiltonianContainer::get_energy_gap() const {
  return _energy_gap;
}

double FactorizedHamiltonianContainer::get_lambda() const {
  Eigen::MatrixXd h1m = get_h1_majorana();
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(h1m);
  double one_body_norm = solver.eigenvalues().array().abs().sum();

  size_t R = _num_ranks;
  size_t B = _num_bases;
  size_t C = _num_copies;

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
  double lambda = get_lambda();
  double eg = _energy_gap;
  if (eg <= 0.0) {
    throw std::runtime_error(
        "E_gap must be positive for a valid SOS walk");
  }
  if (eg >= 2.0 * lambda) {
    throw std::runtime_error(
        "E_gap must be less than 2*Lambda for a valid SOS walk");
  }
  return std::sqrt(eg * (2.0 * lambda - eg));
}

Eigen::MatrixXd FactorizedHamiltonianContainer::get_h1_majorana() const {
  size_t norb = get_num_orbitals();
  size_t R = _num_ranks;
  size_t B = _num_bases;
  size_t C = _num_copies;

  // Start from one-body integrals
  auto [h1_alpha, h1_beta] = get_one_body_integrals();
  // Spin-free h1 = 0.5*(h1_alpha + h1_beta); for restricted they're equal
  Eigen::MatrixXd h1 = 0.5 * (h1_alpha + h1_beta);

  // Reconstruct h2 for the contraction terms
  if (!_cached_two_body) {
    _build_two_body_cache();
  }
  // Map h2 as 4D: h2[p][q][r][s]
  const double* h2 = _cached_two_body->data();
  auto H2 = [&](size_t p, size_t q, size_t r, size_t s) -> double {
    return h2[p * norb * norb * norb + q * norb * norb + r * norb + s];
  };

  // h1 -= 0.5 * einsum("prrq -> pq", h2)
  for (size_t p = 0; p < norb; ++p) {
    for (size_t q = 0; q < norb; ++q) {
      double sum = 0.0;
      for (size_t r = 0; r < norb; ++r) {
        sum += H2(p, r, r, q);
      }
      h1(p, q) -= 0.5 * sum;
    }
  }

  // Majorana: h1 += einsum("pqrr -> pq", h2)
  for (size_t p = 0; p < norb; ++p) {
    for (size_t q = 0; q < norb; ++q) {
      double sum = 0.0;
      for (size_t r = 0; r < norb; ++r) {
        sum += H2(p, q, r, r);
      }
      h1(p, q) += sum;
    }
  }

  // h1 -= einsum("rc, rbc, rbp, rbq -> pq", wb, w, u, u)
  auto U = [&](size_t r, size_t b, size_t p) -> double {
    return _u(r * B * norb + b * norb + p);
  };
  auto W = [&](size_t r, size_t b, size_t c) -> double {
    return _w(r * B * C + b * C + c);
  };

  for (size_t r = 0; r < R; ++r) {
    for (size_t c = 0; c < C; ++c) {
      for (size_t b = 0; b < B; ++b) {
        double coeff = _wb(r, c) * W(r, b, c);
        for (size_t p = 0; p < norb; ++p) {
          double cu = coeff * U(r, b, p);
          for (size_t q = 0; q < norb; ++q) {
            h1(p, q) -= cu * U(r, b, q);
          }
        }
      }
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
  j["num_ranks"] = _num_ranks;
  j["num_bases"] = _num_bases;
  j["num_copies"] = _num_copies;
  j["core_energy"] = _core_energy;
  j["bliss_core_shift"] = _bliss_core_shift;
  j["energy_gap"] = _energy_gap;

  if (has_orbitals()) {
    j["orbitals"] = _orbitals->to_json();
  }

  if (has_inactive_fock_matrix()) {
    auto [fock_a, fock_b] = get_inactive_fock_matrix();
    j["inactive_fock_matrix"] = matrix_to_json(fock_a);
  }

  return j;
}

std::unique_ptr<FactorizedHamiltonianContainer>
FactorizedHamiltonianContainer::from_json(const nlohmann::json& j) {
  QDK_LOG_TRACE_ENTERING();
  validate_serialization_version(SERIALIZATION_VERSION, j["version"]);

  auto h1 = json_to_matrix(j["one_body_integrals"]);
  auto u = json_to_vector(j["u_matrices"]);
  auto w = json_to_vector(j["w_matrices"]);
  auto wb = json_to_matrix(j["wb_matrix"]);
  size_t R = j["num_ranks"];
  size_t B = j["num_bases"];
  size_t C = j["num_copies"];
  double core_energy = j["core_energy"];
  double bliss_core_shift = j.value("bliss_core_shift", 0.0);
  double energy_gap = j.value("energy_gap", 0.0);

  auto orbitals = Orbitals::from_json(j["orbitals"]);

  Eigen::MatrixXd fock = Eigen::MatrixXd::Zero(0, 0);
  if (j.contains("inactive_fock_matrix")) {
    fock = json_to_matrix(j["inactive_fock_matrix"]);
  }

  return std::make_unique<FactorizedHamiltonianContainer>(
      h1, u, w, wb, R, B, C, orbitals, core_energy, fock, bliss_core_shift,
      energy_gap);
}

void FactorizedHamiltonianContainer::to_hdf5(H5::Group& group) const {
  QDK_LOG_TRACE_ENTERING();
  H5::StrType string_type(H5::PredType::C_S1, H5T_VARIABLE);

  // Version
  H5::Attribute version_attr = group.createAttribute(
      "version", string_type, H5::DataSpace(H5S_SCALAR));
  std::string v(SERIALIZATION_VERSION);
  version_attr.write(string_type, v);

  // Container type
  H5::Attribute ct_attr = group.createAttribute("container_type", string_type,
                                                H5::DataSpace(H5S_SCALAR));
  std::string ct("factorized");
  ct_attr.write(string_type, ct);

  // Scalar attributes
  group.createAttribute("core_energy", H5::PredType::NATIVE_DOUBLE,
                        H5::DataSpace(H5S_SCALAR))
      .write(H5::PredType::NATIVE_DOUBLE, &_core_energy);
  group.createAttribute("bliss_core_shift", H5::PredType::NATIVE_DOUBLE,
                        H5::DataSpace(H5S_SCALAR))
      .write(H5::PredType::NATIVE_DOUBLE, &_bliss_core_shift);
  group.createAttribute("energy_gap", H5::PredType::NATIVE_DOUBLE,
                        H5::DataSpace(H5S_SCALAR))
      .write(H5::PredType::NATIVE_DOUBLE, &_energy_gap);

  hsize_t r_val = _num_ranks, b_val = _num_bases, c_val = _num_copies;
  group.createAttribute("num_ranks", H5::PredType::NATIVE_HSIZE,
                        H5::DataSpace(H5S_SCALAR))
      .write(H5::PredType::NATIVE_HSIZE, &r_val);
  group.createAttribute("num_bases", H5::PredType::NATIVE_HSIZE,
                        H5::DataSpace(H5S_SCALAR))
      .write(H5::PredType::NATIVE_HSIZE, &b_val);
  group.createAttribute("num_copies", H5::PredType::NATIVE_HSIZE,
                        H5::DataSpace(H5S_SCALAR))
      .write(H5::PredType::NATIVE_HSIZE, &c_val);

  // Datasets
  auto write_vector = [&](const std::string& name,
                          const Eigen::VectorXd& vec) {
    hsize_t dims = vec.size();
    H5::DataSpace space(1, &dims);
    auto ds = group.createDataSet(name, H5::PredType::NATIVE_DOUBLE, space);
    ds.write(vec.data(), H5::PredType::NATIVE_DOUBLE);
  };

  auto write_matrix = [&](const std::string& name,
                          const Eigen::MatrixXd& mat) {
    hsize_t dims[2] = {static_cast<hsize_t>(mat.rows()),
                       static_cast<hsize_t>(mat.cols())};
    H5::DataSpace space(2, dims);
    auto ds = group.createDataSet(name, H5::PredType::NATIVE_DOUBLE, space);
    // Eigen is column-major; write as-is and record that
    ds.write(mat.data(), H5::PredType::NATIVE_DOUBLE);
  };

  auto [h1_alpha, h1_beta] = get_one_body_integrals();
  write_matrix("one_body_integrals", h1_alpha);
  write_vector("u_matrices", _u);
  write_vector("w_matrices", _w);
  write_matrix("wb_matrix", _wb);

  if (has_inactive_fock_matrix()) {
    auto [fock_a, fock_b] = get_inactive_fock_matrix();
    write_matrix("inactive_fock_matrix", fock_a);
  }

  if (has_orbitals()) {
    H5::Group orb_group = group.createGroup("orbitals");
    _orbitals->to_hdf5(orb_group);
  }
}

std::unique_ptr<FactorizedHamiltonianContainer>
FactorizedHamiltonianContainer::from_hdf5(H5::Group& group) {
  QDK_LOG_TRACE_ENTERING();

  H5::StrType string_type(H5::PredType::C_S1, H5T_VARIABLE);
  H5::Attribute version_attr = group.openAttribute("version");
  std::string version;
  version_attr.read(string_type, version);
  validate_serialization_version(SERIALIZATION_VERSION, version);

  double core_energy, bliss_core_shift, energy_gap;
  group.openAttribute("core_energy")
      .read(H5::PredType::NATIVE_DOUBLE, &core_energy);
  group.openAttribute("bliss_core_shift")
      .read(H5::PredType::NATIVE_DOUBLE, &bliss_core_shift);
  group.openAttribute("energy_gap")
      .read(H5::PredType::NATIVE_DOUBLE, &energy_gap);

  hsize_t r_val, b_val, c_val;
  group.openAttribute("num_ranks")
      .read(H5::PredType::NATIVE_HSIZE, &r_val);
  group.openAttribute("num_bases")
      .read(H5::PredType::NATIVE_HSIZE, &b_val);
  group.openAttribute("num_copies")
      .read(H5::PredType::NATIVE_HSIZE, &c_val);

  auto read_vector = [&](const std::string& name) -> Eigen::VectorXd {
    auto ds = group.openDataSet(name);
    auto space = ds.getSpace();
    hsize_t dims;
    space.getSimpleExtentDims(&dims);
    Eigen::VectorXd vec(dims);
    ds.read(vec.data(), H5::PredType::NATIVE_DOUBLE);
    return vec;
  };

  auto read_matrix = [&](const std::string& name) -> Eigen::MatrixXd {
    auto ds = group.openDataSet(name);
    auto space = ds.getSpace();
    hsize_t dims[2];
    space.getSimpleExtentDims(dims);
    Eigen::MatrixXd mat(dims[0], dims[1]);
    ds.read(mat.data(), H5::PredType::NATIVE_DOUBLE);
    return mat;
  };

  auto h1 = read_matrix("one_body_integrals");
  auto u = read_vector("u_matrices");
  auto w = read_vector("w_matrices");
  auto wb = read_matrix("wb_matrix");

  std::shared_ptr<Orbitals> orbitals;
  if (group.nameExists("orbitals")) {
    H5::Group orb_group = group.openGroup("orbitals");
    orbitals = Orbitals::from_hdf5(orb_group);
  }

  Eigen::MatrixXd fock = Eigen::MatrixXd::Zero(0, 0);
  if (group.nameExists("inactive_fock_matrix")) {
    fock = read_matrix("inactive_fock_matrix");
  }

  return std::make_unique<FactorizedHamiltonianContainer>(
      h1, u, w, wb, r_val, b_val, c_val, orbitals, core_energy, fock,
      bliss_core_shift, energy_gap);
}

// === Validation ===

void FactorizedHamiltonianContainer::validate_integral_dimensions() const {
  QDK_LOG_TRACE_ENTERING();
  HamiltonianContainer::validate_integral_dimensions();

  size_t norb = get_num_orbitals();
  size_t expected_u = _num_ranks * _num_bases * norb;
  size_t expected_w = _num_ranks * _num_bases * _num_copies;

  if (static_cast<size_t>(_u.size()) != expected_u) {
    throw std::invalid_argument(
        "U matrices size mismatch: expected " + std::to_string(expected_u) +
        ", got " + std::to_string(_u.size()));
  }
  if (static_cast<size_t>(_w.size()) != expected_w) {
    throw std::invalid_argument(
        "W matrices size mismatch: expected " + std::to_string(expected_w) +
        ", got " + std::to_string(_w.size()));
  }
  if (_wb.rows() != static_cast<Eigen::Index>(_num_ranks) ||
      _wb.cols() != static_cast<Eigen::Index>(_num_copies)) {
    throw std::invalid_argument(
        "WB matrix shape mismatch: expected (" + std::to_string(_num_ranks) +
        ", " + std::to_string(_num_copies) + "), got (" +
        std::to_string(_wb.rows()) + ", " + std::to_string(_wb.cols()) + ")");
  }
}

// === Hashing ===

void FactorizedHamiltonianContainer::hash_update(
    qdk::chemistry::utils::HashContext& ctx) const {
  HamiltonianContainer::hash_update(ctx);
  hash_value(ctx, get_container_type());
  hash_value(ctx, static_cast<int64_t>(_num_ranks));
  hash_value(ctx, static_cast<int64_t>(_num_bases));
  hash_value(ctx, static_cast<int64_t>(_num_copies));
  hash_value(ctx, _bliss_core_shift);
  hash_value(ctx, _energy_gap);
  hash_eigen(ctx, _u);
  hash_eigen(ctx, _w);
  hash_eigen(ctx, _wb);
}

}  // namespace qdk::chemistry::data
