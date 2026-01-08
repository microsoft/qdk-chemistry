/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE.txt in the project root for
 * license information.
 */

#include "../../algorithms/microsoft/mp2.hpp"

#include <algorithm>
#include <macis/hamiltonian_generator/double_loop.hpp>
#include <macis/sd_operations.hpp>
#include <macis/util/rdms.hpp>
#include <optional>
#include <qdk/chemistry/data/wavefunction_containers/mp2.hpp>
#include <qdk/chemistry/utils/logger.hpp>
#include <stdexcept>
#include <variant>

#include "../json_serialization.hpp"

namespace qdk::chemistry::data {

MP2Container::MP2Container(std::shared_ptr<Hamiltonian> hamiltonian,
                           std::shared_ptr<Wavefunction> wavefunction,
                           const std::string& partitioning)
    // Mp2 is always not self dual
    : WavefunctionContainer(WavefunctionType::NotSelfDual),
      _wavefunction(wavefunction),
      _hamiltonian(hamiltonian) {
  QDK_LOG_TRACE_ENTERING();
  if (!hamiltonian) {
    throw std::invalid_argument("Hamiltonian cannot be null");
  }

  if (wavefunction->get_total_determinants().empty()) {
    throw std::invalid_argument("Reference determinants cannot be empty");
  }

  if (partitioning != "mp") {
    throw std::runtime_error(
        "Only Moeller-Plesset Hamiltonian partitioning is implemented");
  }
}

std::unique_ptr<WavefunctionContainer> MP2Container::clone() const {
  QDK_LOG_TRACE_ENTERING();
  return std::make_unique<MP2Container>(_hamiltonian, _wavefunction);
}

std::shared_ptr<Orbitals> MP2Container::get_orbitals() const {
  QDK_LOG_TRACE_ENTERING();
  return _hamiltonian->get_orbitals();
}

std::shared_ptr<Hamiltonian> MP2Container::get_hamiltonian() const {
  QDK_LOG_TRACE_ENTERING();
  return _hamiltonian;
}

std::shared_ptr<Wavefunction> MP2Container::get_wavefunction() const {
  QDK_LOG_TRACE_ENTERING();
  return _wavefunction;
}

const MP2Container::VectorVariant& MP2Container::get_coefficients() const {
  QDK_LOG_TRACE_ENTERING();
  if (!_coefficients_cache) {
    _generate_ci_expansion();
  }
  return *_coefficients_cache;
}

MP2Container::ScalarVariant MP2Container::get_coefficient(
    const Configuration& det) const {
  QDK_LOG_TRACE_ENTERING();
  // Ensure expansion is generated
  if (!_coefficients_cache) {
    _generate_ci_expansion();
  }

  // Find the determinant
  auto it = std::find(_determinant_vector_cache->begin(),
                      _determinant_vector_cache->end(), det);
  if (it == _determinant_vector_cache->end()) {
    throw std::runtime_error("Determinant not found in MP2 expansion");
  }

  size_t idx = std::distance(_determinant_vector_cache->begin(), it);

  return std::visit(
      [idx](const auto& vec) -> ScalarVariant {
        return ScalarVariant(vec[idx]);
      },
      *_coefficients_cache);
}

const MP2Container::DeterminantVector& MP2Container::get_active_determinants()
    const {
  QDK_LOG_TRACE_ENTERING();
  if (!_determinant_vector_cache) {
    _generate_ci_expansion();
  }
  return *_determinant_vector_cache;
}

void MP2Container::_compute_t1_amplitudes() const {
  QDK_LOG_TRACE_ENTERING();
  // Check if T1 amplitudes are already computed
  if (_t1_amplitudes_aa && _t1_amplitudes_bb) {
    return;
  }

  // Get electron counts
  auto [n_alpha, n_beta] = get_active_num_electrons();

  // Get active space size
  int active_space_size = get_orbitals()->get_num_molecular_orbitals();
  if (get_orbitals()->has_active_space()) {
    const auto& [active_space_ind_alpha, active_space_ind_beta] =
        get_orbitals()->get_active_space_indices();
    active_space_size = active_space_ind_alpha.size();
  }

  const size_t n_vir_alpha = active_space_size - n_alpha;
  const size_t n_vir_beta = active_space_size - n_beta;

  // For MP2, T1 amplitudes are always zero
  Eigen::VectorXd t1_aa_vec = Eigen::VectorXd::Zero(n_alpha * n_vir_alpha);
  Eigen::VectorXd t1_bb_vec = Eigen::VectorXd::Zero(n_beta * n_vir_beta);

  _t1_amplitudes_aa = std::make_shared<VectorVariant>(t1_aa_vec);
  _t1_amplitudes_bb = std::make_shared<VectorVariant>(t1_bb_vec);
}

void MP2Container::_compute_t2_amplitudes() const {
  QDK_LOG_TRACE_ENTERING();
  // Check if amplitudes are already computed
  if (_t2_amplitudes_abab && _t2_amplitudes_aaaa && _t2_amplitudes_bbbb) {
    return;
  }

  // Get electron counts
  auto [n_alpha, n_beta] = get_active_num_electrons();

  // Check if we need unrestricted calculation
  bool use_unrestricted =
      _hamiltonian->is_unrestricted() || (n_alpha != n_beta);

  if (!get_orbitals()->has_energies()) {
    throw std::runtime_error(
        "Orbital energies are required for MP2 amplitude calculation");
  }

  // Get active space size
  int active_space_size = get_orbitals()->get_num_molecular_orbitals();
  if (get_orbitals()->has_active_space()) {
    const auto& [active_space_ind_alpha, active_space_ind_beta] =
        get_orbitals()->get_active_space_indices();
    active_space_size = active_space_ind_alpha.size();
  }

  if (use_unrestricted) {
    // Unrestricted MP2
    const auto& [eps_alpha, eps_beta] = get_orbitals()->get_energies();
    const size_t n_vir_alpha = active_space_size - n_alpha;
    const size_t n_vir_beta = active_space_size - n_beta;

    const auto& [moeri_aaaa, moeri_aabb, moeri_bbbb] =
        _hamiltonian->get_two_body_integrals();

    // Initialize T2 amplitudes storage
    size_t t2_aa_size = n_alpha * n_alpha * n_vir_alpha * n_vir_alpha;
    size_t t2_ab_size = n_alpha * n_beta * n_vir_alpha * n_vir_beta;
    size_t t2_bb_size = n_beta * n_beta * n_vir_beta * n_vir_beta;

    Eigen::VectorXd t2_aa(t2_aa_size);
    Eigen::VectorXd t2_ab(t2_ab_size);
    Eigen::VectorXd t2_bb(t2_bb_size);
    t2_aa.setZero();
    t2_ab.setZero();
    t2_bb.setZero();

    // Pre-compute strides for tensor indexing
    const size_t stride_k = active_space_size;
    const size_t stride_j = active_space_size * active_space_size;
    const size_t stride_i =
        active_space_size * active_space_size * active_space_size;

    // Alpha-Alpha contribution
    algorithms::microsoft::MP2Calculator::compute_same_spin_t2(
        eps_alpha, moeri_aaaa, n_alpha, n_vir_alpha, stride_i, stride_j,
        stride_k, t2_aa);

    // Alpha-Beta contribution
    algorithms::microsoft::MP2Calculator::compute_opposite_spin_t2(
        eps_alpha, eps_beta, moeri_aabb, n_alpha, n_beta, n_vir_alpha,
        n_vir_beta, stride_i, stride_j, stride_k, t2_ab);

    // Beta-Beta contribution
    algorithms::microsoft::MP2Calculator::compute_same_spin_t2(
        eps_beta, moeri_bbbb, n_beta, n_vir_beta, stride_i, stride_j, stride_k,
        t2_bb);

    _t2_amplitudes_abab = std::make_shared<VectorVariant>(t2_ab);
    _t2_amplitudes_aaaa = std::make_shared<VectorVariant>(t2_aa);
    _t2_amplitudes_bbbb = std::make_shared<VectorVariant>(t2_bb);
  } else {
    // Restricted MP2
    if (n_alpha != n_beta) {
      throw std::runtime_error(
          "Restricted MP2 requires equal alpha and beta electrons");
    }

    const auto& [eps_alpha, eps_beta] = get_orbitals()->get_energies();
    const size_t n_occ = n_alpha;
    const size_t n_vir = active_space_size - n_occ;

    const auto& [moeri_aaaa, moeri_aabb, moeri_bbbb] =
        _hamiltonian->get_two_body_integrals();
    // For restricted case, all components are the same; use aaaa
    const auto& moeri = moeri_aaaa;

    size_t t2_size = n_occ * n_occ * n_vir * n_vir;
    Eigen::VectorXd t2_amplitudes(t2_size);
    t2_amplitudes.setZero();

    // Pre-compute strides for tensor indexing
    const size_t stride_k = active_space_size;
    const size_t stride_j = active_space_size * active_space_size;
    const size_t stride_i =
        active_space_size * active_space_size * active_space_size;

    // Compute T2 amplitudes
    algorithms::microsoft::MP2Calculator::compute_restricted_t2(
        eps_alpha, moeri, n_occ, n_vir, stride_i, stride_j, stride_k,
        t2_amplitudes);

    _t2_amplitudes_abab = std::make_shared<VectorVariant>(t2_amplitudes);
    _t2_amplitudes_aaaa = std::make_shared<VectorVariant>(t2_amplitudes);
    _t2_amplitudes_bbbb = std::make_shared<VectorVariant>(t2_amplitudes);
  }
}

std::pair<const MP2Container::VectorVariant&,
          const MP2Container::VectorVariant&>
MP2Container::get_t1_amplitudes() const {
  QDK_LOG_TRACE_ENTERING();
  if (!_t1_amplitudes_aa) {
    _compute_t1_amplitudes();
  }

  return std::make_pair(std::cref(*_t1_amplitudes_aa),
                        std::cref(*_t1_amplitudes_bb));
}

std::tuple<const MP2Container::VectorVariant&,
           const MP2Container::VectorVariant&,
           const MP2Container::VectorVariant&>
MP2Container::get_t2_amplitudes() const {
  QDK_LOG_TRACE_ENTERING();
  if (!_t2_amplitudes_abab) {
    _compute_t2_amplitudes();
  }

  return std::make_tuple(std::cref(*_t2_amplitudes_abab),
                         std::cref(*_t2_amplitudes_aaaa),
                         std::cref(*_t2_amplitudes_bbbb));
}

bool MP2Container::has_t1_amplitudes() const {
  QDK_LOG_TRACE_ENTERING();
  return _t1_amplitudes_aa != nullptr;
}

bool MP2Container::has_t2_amplitudes() const {
  QDK_LOG_TRACE_ENTERING();
  return _t2_amplitudes_abab != nullptr;
}

size_t MP2Container::size() const {
  QDK_LOG_TRACE_ENTERING();
  if (!_determinant_vector_cache) {
    _generate_ci_expansion();
  }
  return _determinant_vector_cache->size();
}

MP2Container::ScalarVariant MP2Container::overlap(
    const WavefunctionContainer& other) const {
  QDK_LOG_TRACE_ENTERING();
  throw std::runtime_error(
      "overlap() is not implemented for MP2 wavefunctions.");
}

double MP2Container::norm() const {
  QDK_LOG_TRACE_ENTERING();
  throw std::runtime_error("norm() is not implemented for MP2 wavefunctions.");
}

bool MP2Container::contains_determinant(const Configuration& det) const {
  QDK_LOG_TRACE_ENTERING();
  throw std::runtime_error(
      "contains_determinant() is not implemented for MP2 wavefunctions.");
}

bool MP2Container::contains_reference(const Configuration& det) const {
  QDK_LOG_TRACE_ENTERING();
  const auto& references = _wavefunction->get_total_determinants();
  return std::find(references.begin(), references.end(), det) !=
         references.end();
}

void MP2Container::clear_caches() const {
  QDK_LOG_TRACE_ENTERING();
  _t1_amplitudes_aa = nullptr;
  _t1_amplitudes_bb = nullptr;
  _t2_amplitudes_abab = nullptr;
  _t2_amplitudes_aaaa = nullptr;
  _t2_amplitudes_bbbb = nullptr;

  _determinant_vector_cache = nullptr;
  _coefficients_cache = nullptr;
}

nlohmann::json MP2Container::to_json() const {
  QDK_LOG_TRACE_ENTERING();
  nlohmann::json j;
  j["type"] = "mp2";
  j["version"] = SERIALIZATION_VERSION;

  // Serialize orbitals
  j["orbitals"] = get_orbitals()->to_json();

  // Serialize Hamiltonian
  j["hamiltonian"] = _hamiltonian->to_json();

  // Serialize wavefunction
  j["wavefunction"] = _wavefunction->to_json();

  // Note: We don't serialize amplitudes in JSON - they are computed on
  // demand when Hamiltonian is available

  return j;
}

std::unique_ptr<MP2Container> MP2Container::from_json(const nlohmann::json& j) {
  QDK_LOG_TRACE_ENTERING();
  // Deserialize orbitals
  auto orbitals = Orbitals::from_json(j.at("orbitals"));

  // Deserialize Hamiltonian (can be null)
  std::shared_ptr<Hamiltonian> hamiltonian = nullptr;
  if (!j.at("hamiltonian").is_null()) {
    hamiltonian = Hamiltonian::from_json(j.at("hamiltonian"));
  }

  // Deserialize Wavefunction (can be null)
  std::shared_ptr<Wavefunction> wavefunction = nullptr;
  if (!j.at("wavefunction").is_null()) {
    wavefunction = Wavefunction::from_json(j.at("wavefunction"));
  }

  return std::make_unique<MP2Container>(hamiltonian, wavefunction);
}

void MP2Container::to_hdf5(H5::Group& group) const {
  QDK_LOG_TRACE_ENTERING();
  try {
    H5::StrType string_type(H5::PredType::C_S1, H5T_VARIABLE);

    // version
    H5::Attribute version_attr = group.createAttribute(
        "version", string_type, H5::DataSpace(H5S_SCALAR));
    std::string version_str = SERIALIZATION_VERSION;
    version_attr.write(string_type, version_str);
    version_attr.close();

    // container type
    std::string container_type = get_container_type();
    H5::Attribute container_type_attr = group.createAttribute(
        "container_type", string_type, H5::DataSpace(H5S_SCALAR));
    container_type_attr.write(string_type, container_type);

    // complex flag
    bool is_complex_flag = this->is_complex();
    H5::Attribute is_complex_attr = group.createAttribute(
        "is_complex", H5::PredType::NATIVE_HBOOL, H5::DataSpace(H5S_SCALAR));
    hbool_t is_complex_hbool = is_complex_flag ? 1 : 0;
    is_complex_attr.write(H5::PredType::NATIVE_HBOOL, &is_complex_hbool);

    // Store wfn if available
    if (_wavefunction) {
      H5::Group wavefunction_group = group.createGroup("wavefunction");
      _wavefunction->to_hdf5(wavefunction_group);
    }

    // Store Hamiltonian if available
    if (_hamiltonian) {
      H5::Group hamiltonian_group = group.createGroup("hamiltonian");
      _hamiltonian->to_hdf5(hamiltonian_group);
    }

    // Note: We don't serialize amplitudes - they are computed on demand
    // when Hamiltonian is available

  } catch (const H5::Exception& e) {
    throw std::runtime_error("HDF5 error: " + std::string(e.getCDetailMsg()));
  }
}

std::unique_ptr<MP2Container> MP2Container::from_hdf5(H5::Group& group) {
  QDK_LOG_TRACE_ENTERING();
  try {
    H5::StrType string_type(H5::PredType::C_S1, H5T_VARIABLE);

    // version
    H5::Attribute version_attr = group.openAttribute("version");
    std::string version_str;
    version_attr.read(string_type, version_str);
    validate_serialization_version(SERIALIZATION_VERSION, version_str);

    // Load Hamiltonian (if available)
    std::shared_ptr<Hamiltonian> hamiltonian = nullptr;
    if (group.nameExists("hamiltonian")) {
      H5::Group hamiltonian_group = group.openGroup("hamiltonian");
      hamiltonian = Hamiltonian::from_hdf5(hamiltonian_group);
    }

    // Load wavefunction (if available)
    std::shared_ptr<Wavefunction> wavefunction = nullptr;
    if (group.nameExists("wavefunction")) {
      H5::Group wavefunction_group = group.openGroup("wavefunction");
      wavefunction = Wavefunction::from_hdf5(wavefunction_group);
    }

    return std::make_unique<MP2Container>(hamiltonian, wavefunction);
  } catch (const H5::Exception& e) {
    throw std::runtime_error("HDF5 error: " + std::string(e.getCDetailMsg()));
  }
}

std::pair<size_t, size_t> MP2Container::get_total_num_electrons() const {
  QDK_LOG_TRACE_ENTERING();
  // Get from first reference determinant
  const auto& references = _wavefunction->get_total_determinants();
  if (references.size() == 0) {
    throw std::runtime_error("No reference determinants available");
  }
  const auto& first_ref = references[0];
  auto [n_alpha, n_beta] = first_ref.get_n_electrons();
  return std::make_pair(n_alpha, n_beta);
}

std::pair<size_t, size_t> MP2Container::get_active_num_electrons() const {
  QDK_LOG_TRACE_ENTERING();
  return get_total_num_electrons();
}

std::pair<Eigen::VectorXd, Eigen::VectorXd>
MP2Container::get_total_orbital_occupations() const {
  QDK_LOG_TRACE_ENTERING();
  throw std::runtime_error(
      "get_total_orbital_occupations() is not implemented for MP2 "
      "wavefunctions.");
}

std::pair<Eigen::VectorXd, Eigen::VectorXd>
MP2Container::get_active_orbital_occupations() const {
  QDK_LOG_TRACE_ENTERING();
  throw std::runtime_error(
      "get_active_orbital_occupations() is not implemented for MP2 "
      "wavefunctions.");
}

std::string MP2Container::get_container_type() const {
  QDK_LOG_TRACE_ENTERING();
  return "mp2";
}

bool MP2Container::is_complex() const {
  QDK_LOG_TRACE_ENTERING();
  if (_t1_amplitudes_aa) {
    if (std::holds_alternative<Eigen::VectorXcd>(*_t1_amplitudes_aa)) {
      return true;
    }
  }
  if (_t2_amplitudes_abab) {
    if (std::holds_alternative<Eigen::VectorXcd>(*_t2_amplitudes_abab)) {
      return true;
    }
  }
  return false;
}

Configuration MP2Container::_apply_excitations(
    const Configuration& ref,
    const std::vector<std::pair<size_t, size_t>>& alpha_excitations,
    const std::vector<std::pair<size_t, size_t>>& beta_excitations) {
  // Convert reference to string, apply excitations, convert back
  std::string config_str = ref.to_string();

  // Apply alpha excitations
  for (const auto& [from_idx, to_idx] : alpha_excitations) {
    if (from_idx >= config_str.size() || to_idx >= config_str.size()) {
      throw std::out_of_range("Excitation index out of range");
    }

    char& from_char = config_str[from_idx];
    char& to_char = config_str[to_idx];

    // Remove alpha from source
    if (from_char == '2') {
      from_char = 'd';  // Doubly -> beta only
    } else if (from_char == 'u') {
      from_char = '0';  // Alpha -> unoccupied
    } else {
      throw std::runtime_error("Invalid alpha excitation: source has no alpha");
    }

    // Add alpha to target
    if (to_char == '0') {
      to_char = 'u';  // Unoccupied -> alpha
    } else if (to_char == 'd') {
      to_char = '2';  // Beta -> doubly
    } else {
      throw std::runtime_error(
          "Invalid alpha excitation: target already has alpha");
    }
  }

  // Apply beta excitations
  for (const auto& [from_idx, to_idx] : beta_excitations) {
    if (from_idx >= config_str.size() || to_idx >= config_str.size()) {
      throw std::out_of_range("Excitation index out of range");
    }

    char& from_char = config_str[from_idx];
    char& to_char = config_str[to_idx];

    // Remove beta from source
    if (from_char == '2') {
      from_char = 'u';  // Doubly -> alpha only
    } else if (from_char == 'd') {
      from_char = '0';  // Beta -> unoccupied
    } else {
      throw std::runtime_error("Invalid beta excitation: source has no beta");
    }

    // Add beta to target
    if (to_char == '0') {
      to_char = 'd';  // Unoccupied -> beta
    } else if (to_char == 'u') {
      to_char = '2';  // Alpha -> doubly
    } else {
      throw std::runtime_error(
          "Invalid beta excitation: target already has beta");
    }
  }

  return Configuration(config_str);
}

void MP2Container::_generate_ci_expansion() const {
  // MP2 is a perturbation theory method, not an exponential ansatz.
  // The first-order wavefunction correction is:
  //   |Ψ^(1)⟩ = Σ_{ijab} t_{ij}^{ab} |Φ_{ij}^{ab}⟩
  // The complete MP2 wavefunction is |Ψ⟩ = |Φ₀⟩ + |Ψ^(1)⟩, i.e., the CI
  // expansion includes the reference determinant and all double excitations.

  // Get T2 amplitudes (T1 = 0 for MP2)
  auto [t2_abab, t2_aaaa, t2_bbbb] = get_t2_amplitudes();

  // Get reference determinant
  const auto& references = _wavefunction->get_total_determinants();
  if (references.empty()) {
    throw std::runtime_error("No reference determinants for CI expansion");
  }
  const Configuration& ref = references[0];

  // Get orbital information
  auto [n_alpha, n_beta] = get_active_num_electrons();
  size_t n_orbitals = get_orbitals()->get_num_molecular_orbitals();
  size_t n_virt_alpha = n_orbitals - n_alpha;
  size_t n_virt_beta = n_orbitals - n_beta;

  // Determine if complex
  bool is_complex_wfn = std::holds_alternative<Eigen::VectorXcd>(t2_abab) ||
                        std::holds_alternative<Eigen::VectorXcd>(t2_aaaa) ||
                        std::holds_alternative<Eigen::VectorXcd>(t2_bbbb);

  if (is_complex_wfn) {
    // Complex case
    std::vector<std::complex<double>> coefficients;
    DeterminantVector determinants;

    // Reference determinant (coefficient = 1)
    determinants.push_back(ref);
    coefficients.push_back(std::complex<double>(1.0, 0.0));

    // Helper to get T2 element (complex)
    auto get_t2_abab_c = [&](size_t i, size_t j, size_t a,
                             size_t b) -> std::complex<double> {
      size_t idx = i * n_beta * n_virt_alpha * n_virt_beta +
                   j * n_virt_alpha * n_virt_beta + a * n_virt_beta + b;
      if (std::holds_alternative<Eigen::VectorXcd>(t2_abab)) {
        const auto& vec = std::get<Eigen::VectorXcd>(t2_abab);
        return idx < static_cast<size_t>(vec.size())
                   ? vec[idx]
                   : std::complex<double>(0.0, 0.0);
      } else {
        const auto& vec = std::get<Eigen::VectorXd>(t2_abab);
        return idx < static_cast<size_t>(vec.size())
                   ? std::complex<double>(vec[idx], 0.0)
                   : std::complex<double>(0.0, 0.0);
      }
    };

    auto get_t2_aaaa_c = [&](size_t i, size_t j, size_t a,
                             size_t b) -> std::complex<double> {
      if (i >= j || a >= b) return std::complex<double>(0.0, 0.0);
      // Full rectangular storage: nocc * nocc * nvir * nvir
      size_t idx = i * n_alpha * n_virt_alpha * n_virt_alpha +
                   j * n_virt_alpha * n_virt_alpha + a * n_virt_alpha + b;
      if (std::holds_alternative<Eigen::VectorXcd>(t2_aaaa)) {
        const auto& vec = std::get<Eigen::VectorXcd>(t2_aaaa);
        return idx < static_cast<size_t>(vec.size())
                   ? vec[idx]
                   : std::complex<double>(0.0, 0.0);
      } else {
        const auto& vec = std::get<Eigen::VectorXd>(t2_aaaa);
        return idx < static_cast<size_t>(vec.size())
                   ? std::complex<double>(vec[idx], 0.0)
                   : std::complex<double>(0.0, 0.0);
      }
    };

    auto get_t2_bbbb_c = [&](size_t i, size_t j, size_t a,
                             size_t b) -> std::complex<double> {
      if (i >= j || a >= b) return std::complex<double>(0.0, 0.0);
      // Full rectangular storage: nocc * nocc * nvir * nvir
      size_t idx = i * n_beta * n_virt_beta * n_virt_beta +
                   j * n_virt_beta * n_virt_beta + a * n_virt_beta + b;
      if (std::holds_alternative<Eigen::VectorXcd>(t2_bbbb)) {
        const auto& vec = std::get<Eigen::VectorXcd>(t2_bbbb);
        return idx < static_cast<size_t>(vec.size())
                   ? vec[idx]
                   : std::complex<double>(0.0, 0.0);
      } else {
        const auto& vec = std::get<Eigen::VectorXd>(t2_bbbb);
        return idx < static_cast<size_t>(vec.size())
                   ? std::complex<double>(vec[idx], 0.0)
                   : std::complex<double>(0.0, 0.0);
      }
    };

    // Doubles from T2 (first-order wavefunction correction)
    // Alpha-beta doubles
    for (size_t i = 0; i < n_alpha; ++i) {
      for (size_t j = 0; j < n_beta; ++j) {
        for (size_t a = 0; a < n_virt_alpha; ++a) {
          for (size_t b = 0; b < n_virt_beta; ++b) {
            auto t_ijab = get_t2_abab_c(i, j, a, b);
            if (std::abs(t_ijab) > std::numeric_limits<double>::epsilon()) {
              auto det = _apply_excitations(ref, {{i, n_alpha + a}},
                                            {{j, n_beta + b}});
              determinants.push_back(det);
              coefficients.push_back(t_ijab);
            }
          }
        }
      }
    }

    // Alpha-alpha doubles
    for (size_t i = 0; i < n_alpha; ++i) {
      for (size_t j = i + 1; j < n_alpha; ++j) {
        for (size_t a = 0; a < n_virt_alpha; ++a) {
          for (size_t b = a + 1; b < n_virt_alpha; ++b) {
            auto t_ijab = get_t2_aaaa_c(i, j, a, b);
            if (std::abs(t_ijab) > std::numeric_limits<double>::epsilon()) {
              auto det = _apply_excitations(
                  ref, {{i, n_alpha + a}, {j, n_alpha + b}}, {});
              determinants.push_back(det);
              coefficients.push_back(t_ijab);
            }
          }
        }
      }
    }

    // Beta-beta doubles
    for (size_t i = 0; i < n_beta; ++i) {
      for (size_t j = i + 1; j < n_beta; ++j) {
        for (size_t a = 0; a < n_virt_beta; ++a) {
          for (size_t b = a + 1; b < n_virt_beta; ++b) {
            auto t_ijab = get_t2_bbbb_c(i, j, a, b);
            if (std::abs(t_ijab) > std::numeric_limits<double>::epsilon()) {
              auto det = _apply_excitations(ref, {},
                                            {{i, n_beta + a}, {j, n_beta + b}});
              determinants.push_back(det);
              coefficients.push_back(t_ijab);
            }
          }
        }
      }
    }

    // Consolidate duplicates (shouldn't be any, but for safety)
    // Convert to VectorVariant for consolidate_determinants
    Eigen::VectorXcd coef_vec(coefficients.size());
    for (size_t i = 0; i < coefficients.size(); ++i) {
      coef_vec[i] = coefficients[i];
    }
    VectorVariant coef_variant(std::move(coef_vec));
    detail::consolidate_determinants(determinants, coef_variant);

    // Normalize the wavefunction: |Ψ⟩ = |Φ₀⟩ + Σ t_{ijab} |Φ_{ij}^{ab}⟩
    // Norm² = 1 + Σ |t_{ijab}|², so we divide by sqrt(norm²)
    auto& final_coefs = std::get<Eigen::VectorXcd>(coef_variant);
    double norm_sq = 0.0;
    for (Eigen::Index i = 0; i < final_coefs.size(); ++i) {
      norm_sq += std::norm(final_coefs[i]);  // |c|² for complex
    }
    double norm = std::sqrt(norm_sq);
    for (Eigen::Index i = 0; i < final_coefs.size(); ++i) {
      final_coefs[i] /= norm;
    }

    // Store results
    _determinant_vector_cache =
        std::make_unique<DeterminantVector>(std::move(determinants));
    _coefficients_cache =
        std::make_unique<VectorVariant>(std::move(coef_variant));

  } else {
    // Real case
    std::vector<double> coefficients;
    DeterminantVector determinants;

    // Reference determinant (coefficient = 1)
    determinants.push_back(ref);
    coefficients.push_back(1.0);

    // Helper to get T2 element (real)
    auto get_t2_abab_r = [&](size_t i, size_t j, size_t a, size_t b) -> double {
      size_t idx = i * n_beta * n_virt_alpha * n_virt_beta +
                   j * n_virt_alpha * n_virt_beta + a * n_virt_beta + b;
      const auto& vec = std::get<Eigen::VectorXd>(t2_abab);
      return idx < static_cast<size_t>(vec.size()) ? vec[idx] : 0.0;
    };

    auto get_t2_aaaa_r = [&](size_t i, size_t j, size_t a, size_t b) -> double {
      if (i >= j || a >= b) return 0.0;
      // Full rectangular storage: nocc * nocc * nvir * nvir
      size_t idx = i * n_alpha * n_virt_alpha * n_virt_alpha +
                   j * n_virt_alpha * n_virt_alpha + a * n_virt_alpha + b;
      const auto& vec = std::get<Eigen::VectorXd>(t2_aaaa);
      return idx < static_cast<size_t>(vec.size()) ? vec[idx] : 0.0;
    };

    auto get_t2_bbbb_r = [&](size_t i, size_t j, size_t a, size_t b) -> double {
      if (i >= j || a >= b) return 0.0;
      // Full rectangular storage: nocc * nocc * nvir * nvir
      size_t idx = i * n_beta * n_virt_beta * n_virt_beta +
                   j * n_virt_beta * n_virt_beta + a * n_virt_beta + b;
      const auto& vec = std::get<Eigen::VectorXd>(t2_bbbb);
      return idx < static_cast<size_t>(vec.size()) ? vec[idx] : 0.0;
    };

    // Doubles from T2 (first-order wavefunction correction)
    // Alpha-beta doubles
    for (size_t i = 0; i < n_alpha; ++i) {
      for (size_t j = 0; j < n_beta; ++j) {
        for (size_t a = 0; a < n_virt_alpha; ++a) {
          for (size_t b = 0; b < n_virt_beta; ++b) {
            double t_ijab = get_t2_abab_r(i, j, a, b);
            if (std::abs(t_ijab) > std::numeric_limits<double>::epsilon()) {
              auto det = _apply_excitations(ref, {{i, n_alpha + a}},
                                            {{j, n_beta + b}});
              determinants.push_back(det);
              coefficients.push_back(t_ijab);
            }
          }
        }
      }
    }

    // Alpha-alpha doubles
    for (size_t i = 0; i < n_alpha; ++i) {
      for (size_t j = i + 1; j < n_alpha; ++j) {
        for (size_t a = 0; a < n_virt_alpha; ++a) {
          for (size_t b = a + 1; b < n_virt_alpha; ++b) {
            double t_ijab = get_t2_aaaa_r(i, j, a, b);
            if (std::abs(t_ijab) > std::numeric_limits<double>::epsilon()) {
              auto det = _apply_excitations(
                  ref, {{i, n_alpha + a}, {j, n_alpha + b}}, {});
              determinants.push_back(det);
              coefficients.push_back(t_ijab);
            }
          }
        }
      }
    }

    // Beta-beta doubles
    for (size_t i = 0; i < n_beta; ++i) {
      for (size_t j = i + 1; j < n_beta; ++j) {
        for (size_t a = 0; a < n_virt_beta; ++a) {
          for (size_t b = a + 1; b < n_virt_beta; ++b) {
            double t_ijab = get_t2_bbbb_r(i, j, a, b);
            if (std::abs(t_ijab) > std::numeric_limits<double>::epsilon()) {
              auto det = _apply_excitations(ref, {},
                                            {{i, n_beta + a}, {j, n_beta + b}});
              determinants.push_back(det);
              coefficients.push_back(t_ijab);
            }
          }
        }
      }
    }

    // Consolidate duplicates (shouldn't be any, but for safety)
    // Convert to VectorVariant for consolidate_determinants
    Eigen::VectorXd coef_vec(coefficients.size());
    for (size_t i = 0; i < coefficients.size(); ++i) {
      coef_vec[i] = coefficients[i];
    }
    VectorVariant coef_variant(std::move(coef_vec));
    detail::consolidate_determinants(determinants, coef_variant);

    // Normalize the wavefunction: |Ψ⟩ = |Φ₀⟩ + Σ t_{ijab} |Φ_{ij}^{ab}⟩
    // Norm² = 1 + Σ |t_{ijab}|², so we divide by sqrt(norm²)
    auto& final_coefs = std::get<Eigen::VectorXd>(coef_variant);
    double norm_sq = 0.0;
    for (Eigen::Index i = 0; i < final_coefs.size(); ++i) {
      norm_sq += final_coefs[i] * final_coefs[i];
    }
    double norm = std::sqrt(norm_sq);
    for (Eigen::Index i = 0; i < final_coefs.size(); ++i) {
      final_coefs[i] /= norm;
    }

    // Store results
    _determinant_vector_cache =
        std::make_unique<DeterminantVector>(std::move(determinants));
    _coefficients_cache =
        std::make_unique<VectorVariant>(std::move(coef_variant));
  }
}

// =============================================================================
// Lazy RDM computation from CI expansion
// =============================================================================

namespace {
// Helper: Dispatch RDM computation using MACIS HamiltonianGenerator
template <size_t N>
void mp2_compute_rdms_with_ham_gen(
    const std::vector<Configuration>& determinants,
    const std::vector<double>& coeffs, size_t norb, const Eigen::MatrixXd& T,
    const Eigen::VectorXd& V, std::vector<double>& one_rdm_aa,
    std::vector<double>& one_rdm_bb, std::vector<double>& two_rdm_aaaa,
    std::vector<double>& two_rdm_bbbb, std::vector<double>& two_rdm_aabb) {
  using wfn_t = macis::wfn_t<N>;
  using generator_t = macis::DoubleLoopHamiltonianGenerator<wfn_t>;

  const size_t ndets = determinants.size();

  // Convert QDK Configurations to MACIS wfn_t format
  std::vector<wfn_t> macis_dets;
  macis_dets.reserve(ndets);
  for (const auto& config : determinants) {
    auto bitset = config.to_bitset<N>();
    macis_dets.push_back(wfn_t(bitset));
  }

  // Create Hamiltonian generator with one-body and two-body integrals
  generator_t ham_gen(
      macis::matrix_span<double>(const_cast<double*>(T.data()), norb, norb),
      macis::rank4_span<double>(const_cast<double*>(V.data()), norb, norb, norb,
                                norb));

  // Create spans for RDM storage
  macis::matrix_span<double> ordm_aa(one_rdm_aa.data(), norb, norb);
  macis::matrix_span<double> ordm_bb(one_rdm_bb.data(), norb, norb);
  macis::rank4_span<double> trdm_aaaa(two_rdm_aaaa.data(), norb, norb, norb,
                                      norb);
  macis::rank4_span<double> trdm_bbbb(two_rdm_bbbb.data(), norb, norb, norb,
                                      norb);
  macis::rank4_span<double> trdm_aabb(two_rdm_aabb.data(), norb, norb, norb,
                                      norb);

  // Make a non-const copy of coefficients for MACIS API
  std::vector<double> coeffs_copy = coeffs;

  // Use ham_gen.form_rdms_spin_dep to compute RDMs - same as macis_base.hpp
  ham_gen.form_rdms_spin_dep(macis_dets.begin(), macis_dets.end(),
                             macis_dets.begin(), macis_dets.end(),
                             coeffs_copy.data(), ordm_aa, ordm_bb, trdm_aaaa,
                             trdm_bbbb, trdm_aabb);
}
}  // namespace

void MP2Container::_generate_rdms_from_ci_expansion() const {
  // Ensure CI expansion is available
  if (!_determinant_vector_cache || !_coefficients_cache) {
    _generate_ci_expansion();
  }

  // Only support real coefficients for now
  // MP2 should always be real, but check anyway for safety
  if (is_complex()) {
    throw std::runtime_error(
        "Lazy RDM computation from complex MP2 amplitudes is not yet "
        "supported. MP2 wavefunctions should always be real-valued.");
  }

  const auto& determinants = *_determinant_vector_cache;
  const auto& coeffs_variant = *_coefficients_cache;
  const auto& coeffs_eigen = std::get<Eigen::VectorXd>(coeffs_variant);

  // Convert to std::vector<double> for ham_gen
  std::vector<double> coeffs(coeffs_eigen.data(),
                             coeffs_eigen.data() + coeffs_eigen.size());

  size_t norb = get_orbitals()->get_num_molecular_orbitals();
  size_t norb2 = norb * norb;
  size_t norb4 = norb2 * norb2;

  // Get integrals from Hamiltonian
  const auto& [T_a, T_b] = _hamiltonian->get_one_body_integrals();
  const auto& [V_aaaa, V_aabb, V_bbbb] = _hamiltonian->get_two_body_integrals();

  // Allocate RDM storage
  std::vector<double> one_rdm_aa(norb2, 0.0);
  std::vector<double> one_rdm_bb(norb2, 0.0);
  std::vector<double> two_rdm_aaaa(norb4, 0.0);
  std::vector<double> two_rdm_bbbb(norb4, 0.0);
  std::vector<double> two_rdm_aabb(norb4, 0.0);

  // Dispatch based on number of orbitals
  if (norb <= 32) {
    mp2_compute_rdms_with_ham_gen<64>(determinants, coeffs, norb, T_a, V_aaaa,
                                      one_rdm_aa, one_rdm_bb, two_rdm_aaaa,
                                      two_rdm_bbbb, two_rdm_aabb);
  } else if (norb <= 64) {
    mp2_compute_rdms_with_ham_gen<128>(determinants, coeffs, norb, T_a, V_aaaa,
                                       one_rdm_aa, one_rdm_bb, two_rdm_aaaa,
                                       two_rdm_bbbb, two_rdm_aabb);
  } else if (norb <= 128) {
    mp2_compute_rdms_with_ham_gen<256>(determinants, coeffs, norb, T_a, V_aaaa,
                                       one_rdm_aa, one_rdm_bb, two_rdm_aaaa,
                                       two_rdm_bbbb, two_rdm_aabb);
  } else {
    throw std::runtime_error(
        "Number of orbitals exceeds maximum supported (128) for RDM "
        "computation");
  }

  // Store in base class RDM member variables
  // Scale 2-RDMs by 2.0 to match convention (MACIS uses 0.5 prefactor)
  Eigen::MatrixXd one_aa_mat =
      Eigen::Map<Eigen::MatrixXd>(one_rdm_aa.data(), norb, norb);
  Eigen::MatrixXd one_bb_mat =
      Eigen::Map<Eigen::MatrixXd>(one_rdm_bb.data(), norb, norb);
  Eigen::VectorXd two_aaaa_vec =
      Eigen::Map<Eigen::VectorXd>(two_rdm_aaaa.data(), norb4) * 2.0;
  Eigen::VectorXd two_bbbb_vec =
      Eigen::Map<Eigen::VectorXd>(two_rdm_bbbb.data(), norb4) * 2.0;
  Eigen::VectorXd two_aabb_vec =
      Eigen::Map<Eigen::VectorXd>(two_rdm_aabb.data(), norb4) * 2.0;

  _one_rdm_spin_dependent_aa =
      std::make_shared<MatrixVariant>(std::move(one_aa_mat));
  _one_rdm_spin_dependent_bb =
      std::make_shared<MatrixVariant>(std::move(one_bb_mat));
  _two_rdm_spin_dependent_aaaa =
      std::make_shared<VectorVariant>(std::move(two_aaaa_vec));
  _two_rdm_spin_dependent_bbbb =
      std::make_shared<VectorVariant>(std::move(two_bbbb_vec));
  _two_rdm_spin_dependent_aabb =
      std::make_shared<VectorVariant>(std::move(two_aabb_vec));
}

bool MP2Container::has_one_rdm_spin_dependent() const {
  // RDMs available if explicitly set OR if we have amplitudes to compute them
  if (_one_rdm_spin_dependent_aa != nullptr &&
      _one_rdm_spin_dependent_bb != nullptr) {
    return true;
  }
  // Can compute from amplitudes if available (always have Hamiltonian)
  return _hamiltonian != nullptr;
}

bool MP2Container::has_one_rdm_spin_traced() const {
  if (_one_rdm_spin_traced != nullptr) {
    return true;
  }
  return has_one_rdm_spin_dependent();
}

bool MP2Container::has_two_rdm_spin_dependent() const {
  if (_two_rdm_spin_dependent_aabb != nullptr &&
      _two_rdm_spin_dependent_aaaa != nullptr &&
      _two_rdm_spin_dependent_bbbb != nullptr) {
    return true;
  }
  // Can compute from amplitudes if available (always have Hamiltonian)
  return _hamiltonian != nullptr;
}

bool MP2Container::has_two_rdm_spin_traced() const {
  if (_two_rdm_spin_traced != nullptr) {
    return true;
  }
  return has_two_rdm_spin_dependent();
}

std::tuple<const MP2Container::MatrixVariant&,
           const MP2Container::MatrixVariant&>
MP2Container::get_active_one_rdm_spin_dependent() const {
  // If not already computed, generate from CI expansion
  if (_one_rdm_spin_dependent_aa == nullptr ||
      _one_rdm_spin_dependent_bb == nullptr) {
    _generate_rdms_from_ci_expansion();
  }
  return std::make_tuple(std::cref(*_one_rdm_spin_dependent_aa),
                         std::cref(*_one_rdm_spin_dependent_bb));
}

std::tuple<const MP2Container::VectorVariant&,
           const MP2Container::VectorVariant&,
           const MP2Container::VectorVariant&>
MP2Container::get_active_two_rdm_spin_dependent() const {
  // If not already computed, generate from CI expansion
  if (_two_rdm_spin_dependent_aabb == nullptr ||
      _two_rdm_spin_dependent_aaaa == nullptr ||
      _two_rdm_spin_dependent_bbbb == nullptr) {
    _generate_rdms_from_ci_expansion();
  }
  return std::make_tuple(std::cref(*_two_rdm_spin_dependent_aabb),
                         std::cref(*_two_rdm_spin_dependent_aaaa),
                         std::cref(*_two_rdm_spin_dependent_bbbb));
}

const MP2Container::MatrixVariant&
MP2Container::get_active_one_rdm_spin_traced() const {
  // If spin-traced already available, return it
  if (_one_rdm_spin_traced != nullptr) {
    return *_one_rdm_spin_traced;
  }

  // Ensure spin-dependent RDMs are computed (this triggers lazy eval)
  get_active_one_rdm_spin_dependent();

  // Now compute spin-traced from spin-dependent
  _one_rdm_spin_traced = detail::add_matrix_variants(
      *_one_rdm_spin_dependent_aa, *_one_rdm_spin_dependent_bb);
  return *_one_rdm_spin_traced;
}

const MP2Container::VectorVariant&
MP2Container::get_active_two_rdm_spin_traced() const {
  // If spin-traced already available, return it
  if (_two_rdm_spin_traced != nullptr) {
    return *_two_rdm_spin_traced;
  }

  // Ensure spin-dependent RDMs are computed (this triggers lazy eval)
  get_active_two_rdm_spin_dependent();

  // Compute spin-traced from spin-dependent components
  // spin-traced = aaaa + bbbb + aabb + bbaa
  auto two_rdm_ss_part = detail::add_vector_variants(
      *_two_rdm_spin_dependent_aaaa, *_two_rdm_spin_dependent_bbbb);
  auto two_rdm_spin_bbaa = detail::transpose_ijkl_klij_vector_variant(
      *_two_rdm_spin_dependent_aabb,
      get_orbitals()->get_num_molecular_orbitals());
  auto two_rdm_os_part = detail::add_vector_variants(
      *_two_rdm_spin_dependent_aabb, *two_rdm_spin_bbaa);
  _two_rdm_spin_traced =
      detail::add_vector_variants(*two_rdm_os_part, *two_rdm_ss_part);
  return *_two_rdm_spin_traced;
}

}  // namespace qdk::chemistry::data
