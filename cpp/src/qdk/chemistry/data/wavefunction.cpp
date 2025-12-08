// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <H5Cpp.h>

#include <Eigen/Dense>
#include <fstream>
#include <memory>
#include <qdk/chemistry/data/wavefunction.hpp>
#include <qdk/chemistry/data/wavefunction_containers/cas.hpp>
#include <qdk/chemistry/data/wavefunction_containers/sci.hpp>
#include <qdk/chemistry/data/wavefunction_containers/sd.hpp>
#include <sstream>
#include <tuple>
#include <variant>

#include "hdf5_error_handling.hpp"
#include "json_serialization.hpp"

namespace qdk::chemistry::data {
namespace detail {
std::shared_ptr<ContainerTypes::MatrixVariant> add_matrix_variants(
    const ContainerTypes::MatrixVariant& mat1,
    const ContainerTypes::MatrixVariant& mat2) {
  return std::visit(
      [](const auto& m1,
         const auto& m2) -> std::shared_ptr<ContainerTypes::MatrixVariant> {
        if constexpr (std::is_same_v<std::decay_t<decltype(m1)>,
                                     std::decay_t<decltype(m2)>>) {
          using MatType = std::decay_t<decltype(m1)>;
          return std::make_shared<ContainerTypes::MatrixVariant>(
              std::in_place_type<MatType>, m1 + m2);
        } else {
          throw std::runtime_error(
              "Cannot add ContainerTypes::MatrixVariants of different types");
        }
      },
      mat1, mat2);
}

std::shared_ptr<ContainerTypes::VectorVariant> add_vector_variants(
    const ContainerTypes::VectorVariant& vec1,
    const ContainerTypes::VectorVariant& vec2) {
  return std::visit(
      [](const auto& v1,
         const auto& v2) -> std::shared_ptr<ContainerTypes::VectorVariant> {
        if constexpr (std::is_same_v<std::decay_t<decltype(v1)>,
                                     std::decay_t<decltype(v2)>>) {
          using VecType = std::decay_t<decltype(v1)>;
          return std::make_shared<ContainerTypes::VectorVariant>(
              std::in_place_type<VecType>, v1 + v2);
        } else {
          throw std::runtime_error(
              "Cannot add ContainerTypes::VectorVariants of different types");
        }
      },
      vec1, vec2);
}

bool is_matrix_variant_complex(const ContainerTypes::MatrixVariant& variant) {
  return std::holds_alternative<Eigen::MatrixXcd>(variant);
}

bool is_vector_variant_complex(const ContainerTypes::VectorVariant& variant) {
  return std::holds_alternative<Eigen::VectorXcd>(variant);
}

std::shared_ptr<ContainerTypes::VectorVariant>
transpose_ijkl_klij_vector_variant(const ContainerTypes::VectorVariant& variant,
                                   int norbs) {
  return std::visit(
      [norbs](
          const auto& vec) -> std::shared_ptr<ContainerTypes::VectorVariant> {
        using VecType = std::decay_t<decltype(vec)>;
        VecType output(vec.size());
        output.setZero();

        for (int i = 0; i < norbs; ++i)
          for (int j = 0; j < norbs; ++j)
            for (int k = 0; k < norbs; ++k)
              for (int l = 0; l < norbs; ++l) {
                int new_index = k * norbs * norbs * norbs + l * norbs * norbs +
                                i * norbs + j;
                int index = i * norbs * norbs * norbs + j * norbs * norbs +
                            k * norbs + l;
                output(new_index) = vec(index);
              }
        return make_shared<ContainerTypes::VectorVariant>(
            std::in_place_type<VecType>, output);
      },
      variant);
}

}  // namespace detail

// WavefunctionContainer constructor
WavefunctionContainer::WavefunctionContainer(WavefunctionType type)
    : WavefunctionContainer(std::nullopt,  // one_rdm_spin_traced
                            std::nullopt,  // one_rdm_aa
                            std::nullopt,  // one_rdm_bb
                            std::nullopt,  // two_rdm_spin_traced
                            std::nullopt,  // two_rdm_aabb
                            std::nullopt,  // two_rdm_aaaa
                            std::nullopt,  // two_rdm_bbbb
                            type) {}

// \cond DOXYGEN_SUPPRESS (suppress warnings for declaration with "using")
WavefunctionContainer::WavefunctionContainer(
    const std::optional<ContainerTypes::MatrixVariant>& one_rdm_spin_traced,
    const std::optional<ContainerTypes::VectorVariant>& two_rdm_spin_traced,
    WavefunctionType type)
    : WavefunctionContainer(one_rdm_spin_traced,
                            std::nullopt,  // one_rdm_aa
                            std::nullopt,  // one_rdm_bb
                            two_rdm_spin_traced,
                            std::nullopt,  // two_rdm_aabb
                            std::nullopt,  // two_rdm_aaaa
                            std::nullopt,  // two_rdm_bbbb
                            type) {}
// \endcond

// \cond DOXYGEN_SUPPRESS (suppress warnings for declaration with "using")
WavefunctionContainer::WavefunctionContainer(
    const std::optional<ContainerTypes::MatrixVariant>& one_rdm_spin_traced,
    const std::optional<ContainerTypes::MatrixVariant>& one_rdm_aa,
    const std::optional<ContainerTypes::MatrixVariant>& one_rdm_bb,
    const std::optional<ContainerTypes::VectorVariant>& two_rdm_spin_traced,
    const std::optional<ContainerTypes::VectorVariant>& two_rdm_aabb,
    const std::optional<ContainerTypes::VectorVariant>& two_rdm_aaaa,
    const std::optional<ContainerTypes::VectorVariant>& two_rdm_bbbb,
    WavefunctionType type)
    : _type(type) {
  if (one_rdm_spin_traced.has_value()) {
    _one_rdm_spin_traced = std::make_shared<ContainerTypes::MatrixVariant>(
        one_rdm_spin_traced.value());
  } else {
    _one_rdm_spin_traced = nullptr;
  }
  if (one_rdm_aa.has_value()) {
    _one_rdm_spin_dependent_aa =
        std::make_shared<ContainerTypes::MatrixVariant>(one_rdm_aa.value());
  } else {
    _one_rdm_spin_dependent_aa = nullptr;
  }
  if (one_rdm_bb.has_value()) {
    _one_rdm_spin_dependent_bb =
        std::make_shared<ContainerTypes::MatrixVariant>(one_rdm_bb.value());
  } else {
    _one_rdm_spin_dependent_bb = nullptr;
  }
  if (two_rdm_spin_traced.has_value()) {
    _two_rdm_spin_traced = std::make_shared<ContainerTypes::VectorVariant>(
        two_rdm_spin_traced.value());
  } else {
    _two_rdm_spin_traced = nullptr;
  }
  if (two_rdm_aabb.has_value()) {
    _two_rdm_spin_dependent_aabb =
        std::make_shared<ContainerTypes::VectorVariant>(two_rdm_aabb.value());
  } else {
    _two_rdm_spin_dependent_aabb = nullptr;
  }
  if (two_rdm_aaaa.has_value()) {
    _two_rdm_spin_dependent_aaaa =
        std::make_shared<ContainerTypes::VectorVariant>(two_rdm_aaaa.value());
  } else {
    _two_rdm_spin_dependent_aaaa = nullptr;
  }
  if (two_rdm_bbbb.has_value()) {
    _two_rdm_spin_dependent_bbbb =
        std::make_shared<ContainerTypes::VectorVariant>(two_rdm_bbbb.value());
  } else {
    _two_rdm_spin_dependent_bbbb = nullptr;
  }
}
// \endcond

// RDM handling
std::tuple<const ContainerTypes::MatrixVariant&,
           const ContainerTypes::MatrixVariant&>
WavefunctionContainer::get_active_one_rdm_spin_dependent() const {
  if (!has_one_rdm_spin_dependent()) {
    throw std::runtime_error("Spin-dependent one-body RDM not set");
  }
  bool is_singlet =
      get_active_num_electrons().first == get_active_num_electrons().second;
  if (_one_rdm_spin_dependent_aa != nullptr &&
      _one_rdm_spin_dependent_bb != nullptr) {
    return std::make_tuple(std::cref(*_one_rdm_spin_dependent_aa),
                           std::cref(*_one_rdm_spin_dependent_bb));
  }

  // For restricted singlets, if only one spin component is available, use it
  // for both
  if ((get_orbitals()->is_restricted() && is_singlet) &&
      _one_rdm_spin_dependent_aa != nullptr) {
    return std::make_tuple(std::cref(*_one_rdm_spin_dependent_aa),
                           std::cref(*_one_rdm_spin_dependent_aa));
  }
  if ((get_orbitals()->is_restricted() && is_singlet) &&
      _one_rdm_spin_dependent_bb != nullptr) {
    return std::make_tuple(std::cref(*_one_rdm_spin_dependent_bb),
                           std::cref(*_one_rdm_spin_dependent_bb));
  }

  // If restricted singlet and only spin-traced RDM is available, derive spin
  // components
  if (get_orbitals()->is_restricted() && is_singlet &&
      _one_rdm_spin_traced != nullptr) {
    // Lazy evaluation - only compute if necessary
    if (_one_rdm_spin_dependent_aa == nullptr) {
      _one_rdm_spin_dependent_aa =
          detail::multiply_matrix_variant(*_one_rdm_spin_traced, 0.5);
    }
    return std::make_tuple(std::cref(*_one_rdm_spin_dependent_aa),
                           std::cref(*_one_rdm_spin_dependent_aa));
  }

  // Should not reach this exception
  throw std::runtime_error("No one-body RDMs are set");
}

const ContainerTypes::MatrixVariant&
WavefunctionContainer::get_active_one_rdm_spin_traced() const {
  if (!has_one_rdm_spin_traced()) {
    throw std::runtime_error("Spin-traced one-body RDM not set");
  }
  bool is_singlet =
      get_active_num_electrons().first == get_active_num_electrons().second;
  if (_one_rdm_spin_traced != nullptr) {
    return *_one_rdm_spin_traced;
  }
  if (_one_rdm_spin_dependent_aa != nullptr &&
      _one_rdm_spin_dependent_bb != nullptr) {
    // evaluate only if necessary
    _one_rdm_spin_traced = detail::add_matrix_variants(
        *_one_rdm_spin_dependent_aa, *_one_rdm_spin_dependent_bb);
    return *_one_rdm_spin_traced;
  }
  // If restricted singlets, we can use the 0.5 * spin-traced RDM.
  if (get_orbitals()->is_restricted() && is_singlet &&
      _one_rdm_spin_dependent_aa != nullptr) {
    _one_rdm_spin_traced =
        detail::multiply_matrix_variant(*_one_rdm_spin_dependent_aa, 2.0);
    return *_one_rdm_spin_traced;
  }
  if (get_orbitals()->is_restricted() && is_singlet &&
      _one_rdm_spin_dependent_bb != nullptr) {
    _one_rdm_spin_traced =
        detail::multiply_matrix_variant(*_one_rdm_spin_dependent_bb, 2.0);
    return *_one_rdm_spin_traced;
  }
  // Should not reach this exception.
  throw std::runtime_error("No spin-traced one-body RDMs are set");
}

std::tuple<const ContainerTypes::VectorVariant&,
           const ContainerTypes::VectorVariant&,
           const ContainerTypes::VectorVariant&>
WavefunctionContainer::get_active_two_rdm_spin_dependent() const {
  if (!has_two_rdm_spin_dependent()) {
    throw std::runtime_error("Spin-dependent two-body RDM not set");
  }
  bool is_singlet =
      get_active_num_electrons().first == get_active_num_electrons().second;
  if (_two_rdm_spin_dependent_aabb != nullptr &&
      _two_rdm_spin_dependent_aaaa != nullptr &&
      _two_rdm_spin_dependent_bbbb != nullptr) {
    return std::make_tuple(std::cref(*_two_rdm_spin_dependent_aabb),
                           std::cref(*_two_rdm_spin_dependent_aaaa),
                           std::cref(*_two_rdm_spin_dependent_bbbb));
  }
  if (get_orbitals()->is_restricted() && is_singlet &&
      _two_rdm_spin_dependent_aabb != nullptr &&
      _two_rdm_spin_dependent_aaaa != nullptr) {
    return std::make_tuple(std::cref(*_two_rdm_spin_dependent_aabb),
                           std::cref(*_two_rdm_spin_dependent_aaaa),
                           std::cref(*_two_rdm_spin_dependent_aaaa));
  }
  // Should not reach this exception
  throw std::runtime_error("No spin-dependent two-body RDMs are set");
}

const ContainerTypes::VectorVariant&
WavefunctionContainer::get_active_two_rdm_spin_traced() const {
  if (!has_two_rdm_spin_traced()) {
    throw std::runtime_error("Spin-traced two-body RDM not set");
  }
  bool is_singlet =
      get_active_num_electrons().first == get_active_num_electrons().second;
  if (_two_rdm_spin_traced != nullptr) {
    return *_two_rdm_spin_traced;
  }
  if (_two_rdm_spin_dependent_aabb != nullptr &&
      _two_rdm_spin_dependent_aaaa != nullptr &&
      _two_rdm_spin_dependent_bbbb != nullptr) {
    auto two_rdm_ss_part = detail::add_vector_variants(
        *(_two_rdm_spin_dependent_aaaa), *(_two_rdm_spin_dependent_bbbb));
    auto two_rdm_spin_bbaa = detail::transpose_ijkl_klij_vector_variant(
        *_two_rdm_spin_dependent_aabb,
        get_orbitals()->get_active_space_indices().first.size());

    auto two_rdm_os_part = detail::add_vector_variants(
        *(_two_rdm_spin_dependent_aabb), *(two_rdm_spin_bbaa));

    _two_rdm_spin_traced =
        detail::add_vector_variants(*two_rdm_os_part, *two_rdm_ss_part);

    return *_two_rdm_spin_traced;
  }
  if (get_orbitals()->is_restricted() && is_singlet &&
      _two_rdm_spin_dependent_aabb != nullptr &&
      _two_rdm_spin_dependent_aaaa != nullptr) {
    // For restricted singlet case: aabb + bbaa + aaaa + bbbb = 2*aabb + 2*aaaa
    auto double_aabb =
        detail::multiply_vector_variant(*_two_rdm_spin_dependent_aabb, 2.0);
    auto double_aaaa =
        detail::multiply_vector_variant(*_two_rdm_spin_dependent_aaaa, 2.0);
    _two_rdm_spin_traced =
        detail::add_vector_variants(*double_aabb, *double_aaaa);
    return *_two_rdm_spin_traced;
  }
  // Should not reach this exception
  throw std::runtime_error("No spin-traced two-body RDMs are set");
}

bool WavefunctionContainer::has_one_rdm_spin_dependent() const {
  bool is_singlet =
      get_active_num_electrons().first == get_active_num_electrons().second;
  return (_one_rdm_spin_dependent_aa != nullptr &&
          _one_rdm_spin_dependent_bb != nullptr) ||
         ((get_orbitals()->is_restricted() && is_singlet) &&
          (_one_rdm_spin_dependent_aa != nullptr ||
           _one_rdm_spin_dependent_bb != nullptr)) ||
         (get_orbitals()->is_restricted() && _one_rdm_spin_traced != nullptr);
}

bool WavefunctionContainer::has_one_rdm_spin_traced() const {
  return _one_rdm_spin_traced != nullptr || has_one_rdm_spin_dependent();
}

bool WavefunctionContainer::has_two_rdm_spin_dependent() const {
  bool is_singlet =
      get_active_num_electrons().first == get_active_num_electrons().second;
  return (_two_rdm_spin_dependent_aabb != nullptr &&
          _two_rdm_spin_dependent_aaaa != nullptr &&
          _two_rdm_spin_dependent_bbbb != nullptr) ||
         ((get_orbitals()->is_restricted() && is_singlet) &&
          _two_rdm_spin_dependent_aabb != nullptr &&
          (_two_rdm_spin_dependent_aaaa != nullptr ||
           _two_rdm_spin_dependent_bbbb != nullptr));
}

bool WavefunctionContainer::has_two_rdm_spin_traced() const {
  return _two_rdm_spin_traced != nullptr || has_two_rdm_spin_dependent();
}

// entropies
Eigen::VectorXd WavefunctionContainer::get_single_orbital_entropies() const {
  if (!has_one_rdm_spin_dependent()) {
    throw std::runtime_error(
        "Spin-dependent one-body RDMs must be set to evaluate "
        "single-orbital entropies");
  }
  if (_two_rdm_spin_dependent_aabb == nullptr) {
    throw std::runtime_error(
        "alpha-alpha-beta-beta block of spin-dependent two-body RDMs must be "
        "set to evaluate single-orbital entropies");
  }

  const auto& one_rdm_aa_var = std::get<0>(get_active_one_rdm_spin_dependent());
  const auto& one_rdm_bb_var = std::get<1>(get_active_one_rdm_spin_dependent());
  const auto& two_rdm_ab_var = std::get<0>(get_active_two_rdm_spin_dependent());

  Eigen::MatrixXd one_rdm_aa;
  Eigen::MatrixXd one_rdm_bb;
  Eigen::VectorXd two_rdm_ab;

  if (detail::is_matrix_variant_complex(one_rdm_aa_var) ||
      detail::is_vector_variant_complex(two_rdm_ab_var) ||
      detail::is_matrix_variant_complex(one_rdm_bb_var)) {
    throw std::runtime_error("Complex entropy calculation not yet implemented");
  } else {
    one_rdm_aa = std::get<Eigen::MatrixXd>(one_rdm_aa_var);
    one_rdm_bb = std::get<Eigen::MatrixXd>(one_rdm_bb_var);
    two_rdm_ab = std::get<Eigen::VectorXd>(two_rdm_ab_var);
  }

  int norbs = one_rdm_aa.rows();

  // Lambda function to get the two-body RDM element
  auto get_active_two_rdm_element = [&two_rdm_ab, norbs](int i, int j, int k,
                                                         int l) {
    if (i >= norbs || j >= norbs || k >= norbs || l >= norbs) {
      throw std::out_of_range("Index out of bounds for two-body RDM");
    }
    int norbs2 = norbs * norbs;
    return two_rdm_ab(i * norbs * norbs2 + j * norbs2 + k * norbs + l);
  };

  // Source: https://doi.org/10.1002/qua.24832
  // s1_i  = - \sum_alpha \omega_i,alpha * ln(omega_i,alpha)
  Eigen::VectorXd s1_entropies = Eigen::VectorXd::Zero(norbs);
  for (std::size_t i = 0; i < norbs; ++i) {
    // omega_1 = 1 - \gamma_{ii} - \gamma_{\bar{i}\bar{i}} +
    // \Gamma_{i\bar{i}i\bar{i}}
    auto ordm1 = 1 - one_rdm_aa(i, i) - one_rdm_bb(i, i) +
                 get_active_two_rdm_element(i, i, i, i);
    if (ordm1 > 0) {
      s1_entropies(i) -= ordm1 * std::log(ordm1);
    }
    // omega_2 = \gamma_{ii} - \Gamma_{i\bar{i}i\bar{i}}
    auto ordm2 = one_rdm_aa(i, i) - get_active_two_rdm_element(i, i, i, i);
    if (ordm2 > 0) {
      s1_entropies(i) -= ordm2 * std::log(ordm2);
    }
    // omega_3 = \gamma_{\bar{i}\bar{i}} - \Gamma_{i\bar{i}i\bar{i}}
    auto ordm3 = one_rdm_bb(i, i) - get_active_two_rdm_element(i, i, i, i);
    if (ordm3 > 0) {
      s1_entropies(i) -= ordm3 * std::log(ordm3);
    }
    // omega_4 = \Gamma_{i\bar{i}i\bar{i}}
    auto ordm4 = get_active_two_rdm_element(i, i, i, i);
    if (ordm4 > 0) {
      s1_entropies(i) -= ordm4 * std::log(ordm4);
    }
  }
  return s1_entropies;
}

bool WavefunctionContainer::has_single_orbital_entropies() const {
  return has_one_rdm_spin_dependent() &&
         _two_rdm_spin_dependent_aabb != nullptr;
}

void WavefunctionContainer::_clear_rdms() const {
  _one_rdm_spin_traced.reset();
  _one_rdm_spin_dependent_aa.reset();
  _one_rdm_spin_dependent_bb.reset();
  _two_rdm_spin_traced.reset();
  _two_rdm_spin_dependent_aabb.reset();
  _two_rdm_spin_dependent_aaaa.reset();
  _two_rdm_spin_dependent_bbbb.reset();
}

std::unique_ptr<WavefunctionContainer> WavefunctionContainer::from_json(
    const nlohmann::json& j) {
  if (!j.contains("container_type")) {
    throw std::runtime_error("JSON missing required 'container_type' field");
  }

  std::string container_type = j["container_type"];

  // Forward to appropriate container implementation
  if (container_type == "cas") {
    return CasWavefunctionContainer::from_json(j);
  } else if (container_type == "sci") {
    return SciWavefunctionContainer::from_json(j);
  } else if (container_type == "sd") {
    return SlaterDeterminantContainer::from_json(j);
  } else {
    throw std::runtime_error("Unknown container type: " + container_type);
  }
}

std::unique_ptr<WavefunctionContainer> WavefunctionContainer::from_hdf5(
    H5::Group& group) {
  try {
    // Read container type identifier
    if (!group.attrExists("container_type")) {
      throw std::runtime_error(
          "HDF5 group missing required 'container_type' attribute");
    }

    H5::StrType string_type(H5::PredType::C_S1, H5T_VARIABLE);
    H5::Attribute type_attr = group.openAttribute("container_type");
    std::string container_type;
    type_attr.read(string_type, container_type);

    // Forward to appropriate container implementation
    if (container_type == "cas") {
      return CasWavefunctionContainer::from_hdf5(group);
    } else if (container_type == "sci") {
      return SciWavefunctionContainer::from_hdf5(group);
    } else if (container_type == "sd") {
      return SlaterDeterminantContainer::from_hdf5(group);
    } else {
      throw std::runtime_error("Unknown container type: " + container_type);
    }

  } catch (const H5::Exception& e) {
    throw std::runtime_error("HDF5 error: " + std::string(e.getCDetailMsg()));
  }
}

WavefunctionType WavefunctionContainer::get_type() const { return _type; }

// Wavefunction implementations
Wavefunction::Wavefunction(std::unique_ptr<WavefunctionContainer> container)
    : _container(std::move(container)) {}

// Copy constructor
Wavefunction::Wavefunction(const Wavefunction& other)
    : _container(other._container->clone()) {}

// Copy assignment operator
Wavefunction& Wavefunction::operator=(const Wavefunction& other) {
  if (this != &other) {
    _container = other._container->clone();
  }
  return *this;
}

std::shared_ptr<Orbitals> Wavefunction::get_orbitals() const {
  return _container->get_orbitals();
}

std::string Wavefunction::get_container_type() const {
  return _container->get_container_type();
}

std::pair<size_t, size_t> Wavefunction::get_total_num_electrons() const {
  return _container->get_total_num_electrons();
}

std::pair<size_t, size_t> Wavefunction::get_active_num_electrons() const {
  return _container->get_active_num_electrons();
}

std::pair<Eigen::VectorXd, Eigen::VectorXd>
Wavefunction::get_total_orbital_occupations() const {
  return _container->get_total_orbital_occupations();
}

std::pair<Eigen::VectorXd, Eigen::VectorXd>
Wavefunction::get_active_orbital_occupations() const {
  return _container->get_active_orbital_occupations();
}

Wavefunction::ScalarVariant Wavefunction::get_coefficient(
    const Configuration& det) const {
  return _container->get_coefficient(det);
}

const Wavefunction::VectorVariant& Wavefunction::get_coefficients() const {
  return _container->get_coefficients();
}

const Wavefunction::DeterminantVector& Wavefunction::get_active_determinants()
    const {
  return _container->get_active_determinants();
}

Wavefunction::DeterminantVector Wavefunction::get_total_determinants() const {
  const auto& active_dets = get_active_determinants();
  DeterminantVector total_dets;
  total_dets.reserve(active_dets.size());

  for (const auto& active_det : active_dets) {
    total_dets.push_back(get_total_determinant(active_det));
  }

  return total_dets;
}

Configuration Wavefunction::get_active_determinant(
    const Configuration& total_determinant) const {
  auto orbitals = get_orbitals();

  if (!orbitals->has_active_space()) {
    // If no active space is defined, return the total determinant as-is
    return total_determinant;
  }

  auto [alpha_active, beta_active] = orbitals->get_active_space_indices();
  const auto& active_indices = alpha_active;

  if (active_indices.empty()) {
    // Empty active space - return empty configuration
    return Configuration("");
  }

  const std::string total_str = total_determinant.to_string();
  std::string active_str;
  active_str.reserve(active_indices.size());

  // Extract only the active space orbitals
  for (size_t idx : active_indices) {
    if (idx < total_str.length()) {
      active_str += total_str[idx];
    } else {
      active_str += '0';  // Pad with unoccupied if needed
    }
  }

  return Configuration(active_str);
}

Configuration Wavefunction::get_total_determinant(
    const Configuration& active_determinant) const {
  auto orbitals = get_orbitals();

  if (!orbitals->has_active_space()) {
    // If no active space is defined, return the active determinant as-is
    return active_determinant;
  }

  auto [alpha_inactive, beta_inactive] = orbitals->get_inactive_space_indices();
  auto [alpha_active, beta_active] = orbitals->get_active_space_indices();
  auto [alpha_virtual, beta_virtual] = orbitals->get_virtual_space_indices();

  const auto& inactive_indices = alpha_inactive;
  const auto& active_indices = alpha_active;
  const auto& virtual_indices = alpha_virtual;

  size_t num_molecular_orbitals = orbitals->get_num_molecular_orbitals();
  std::string total_str(num_molecular_orbitals, '0');

  // Fill inactive orbitals as doubly occupied
  for (size_t idx : inactive_indices) {
    if (idx < num_molecular_orbitals) {
      total_str[idx] = '2';
    }
  }

  // Fill active orbitals from the active determinant
  const std::string active_str = active_determinant.to_string();
  for (size_t i = 0; i < active_indices.size() && i < active_str.length();
       ++i) {
    size_t idx = active_indices[i];
    if (idx < num_molecular_orbitals) {
      total_str[idx] = active_str[i];
    }
  }

  return Configuration(total_str);
}

size_t Wavefunction::size() const { return _container->size(); }

double Wavefunction::norm() const { return _container->norm(); }

Wavefunction::ScalarVariant Wavefunction::overlap(
    const Wavefunction& other) const {
  return _container->overlap(*other._container);
}

std::tuple<const Wavefunction::MatrixVariant&,
           const Wavefunction::MatrixVariant&>
Wavefunction::get_active_one_rdm_spin_dependent() const {
  return _container->get_active_one_rdm_spin_dependent();
}

std::tuple<const Wavefunction::VectorVariant&,
           const Wavefunction::VectorVariant&,
           const Wavefunction::VectorVariant&>
Wavefunction::get_active_two_rdm_spin_dependent() const {
  return _container->get_active_two_rdm_spin_dependent();
}

const Wavefunction::MatrixVariant&
Wavefunction::get_active_one_rdm_spin_traced() const {
  return _container->get_active_one_rdm_spin_traced();
}

const Wavefunction::VectorVariant&
Wavefunction::get_active_two_rdm_spin_traced() const {
  return _container->get_active_two_rdm_spin_traced();
}

bool Wavefunction::has_single_orbital_entropies() const {
  return _container->has_single_orbital_entropies();
}

Eigen::VectorXd Wavefunction::get_single_orbital_entropies() const {
  return _container->get_single_orbital_entropies();
}

bool Wavefunction::has_one_rdm_spin_dependent() const {
  return _container->has_one_rdm_spin_dependent();
}

bool Wavefunction::has_one_rdm_spin_traced() const {
  return _container->has_one_rdm_spin_traced();
}

bool Wavefunction::has_two_rdm_spin_dependent() const {
  return _container->has_two_rdm_spin_dependent();
}

bool Wavefunction::has_two_rdm_spin_traced() const {
  return _container->has_two_rdm_spin_traced();
}

// Cache management
void Wavefunction::_clear_caches() const { _container->clear_caches(); }

// Type access
WavefunctionType Wavefunction::get_type() const {
  return _container->get_type();
}

bool Wavefunction::is_complex() const { return _container->is_complex(); }

nlohmann::json Wavefunction::to_json() const {
  nlohmann::json j;

  // Store version first
  j["version"] = SERIALIZATION_VERSION;

  // Add container type identifier
  j["container_type"] = _container->get_container_type();

  // Delegate to container serialization (orbitals are included within the
  // container)
  j["container"] = _container->to_json();

  return j;
}

std::shared_ptr<Wavefunction> Wavefunction::from_json(const nlohmann::json& j) {
  try {
    // Validate version first
    if (!j.contains("version")) {
      throw std::runtime_error("Invalid JSON: missing version field");
    }
    validate_serialization_version(SERIALIZATION_VERSION, j["version"]);

    // Load container using factory method (orbitals are loaded internally by
    // the container)
    if (!j.contains("container")) {
      throw std::runtime_error("JSON missing required 'container' field");
    }
    auto container = WavefunctionContainer::from_json(j["container"]);

    return std::make_shared<Wavefunction>(std::move(container));

  } catch (const std::exception& e) {
    throw std::runtime_error("Failed to parse Wavefunction from JSON: " +
                             std::string(e.what()));
  }
}

void Wavefunction::to_json_file(const std::string& filename) const {
  _to_json_file(filename);
}

std::shared_ptr<Wavefunction> Wavefunction::from_json_file(
    const std::string& filename) {
  return _from_json_file(filename);
}

void Wavefunction::to_hdf5(H5::Group& group) const {
  try {
    H5::StrType string_type(H5::PredType::C_S1, H5T_VARIABLE);

    // Add version attribute
    H5::Attribute version_attr = group.createAttribute(
        "version", string_type, H5::DataSpace(H5S_SCALAR));
    std::string version_str(SERIALIZATION_VERSION);
    version_attr.write(string_type, version_str);
    version_attr.close();

    // Write container type identifier
    std::string container_type = _container->get_container_type();
    H5::Attribute type_attr = group.createAttribute(
        "container_type", string_type, H5::DataSpace(H5S_SCALAR));
    type_attr.write(string_type, container_type);

    // Delegate to container serialization (orbitals are included within the
    // container)
    H5::Group container_group = group.createGroup("container");
    _container->to_hdf5(container_group);

  } catch (const H5::Exception& e) {
    throw std::runtime_error("HDF5 error: " + std::string(e.getCDetailMsg()));
  }
}

std::shared_ptr<Wavefunction> Wavefunction::from_hdf5(H5::Group& group) {
  try {
    // Check version first
    H5::StrType string_type(H5::PredType::C_S1, H5T_VARIABLE);
    H5::Attribute version_attr = group.openAttribute("version");
    std::string version;
    version_attr.read(string_type, version);
    validate_serialization_version(SERIALIZATION_VERSION, version);

    // Load container using factory method (orbitals are loaded internally by
    // the container)
    if (!group.nameExists("container")) {
      throw std::runtime_error(
          "HDF5 group missing required 'container' subgroup");
    }
    H5::Group container_group = group.openGroup("container");
    auto container = WavefunctionContainer::from_hdf5(container_group);

    return std::make_shared<Wavefunction>(std::move(container));

  } catch (const H5::Exception& e) {
    throw std::runtime_error("HDF5 error: " + std::string(e.getCDetailMsg()));
  }
}

void Wavefunction::to_hdf5_file(const std::string& filename) const {
  _to_hdf5_file(filename);
}

std::shared_ptr<Wavefunction> Wavefunction::from_hdf5_file(
    const std::string& filename) {
  return _from_hdf5_file(filename);
}

void Wavefunction::to_file(const std::string& filename,
                           const std::string& type) const {
  if (type == "json") {
    to_json_file(filename);
  } else if (type == "hdf5") {
    to_hdf5_file(filename);
  } else {
    throw std::invalid_argument("Unsupported file type: " + type +
                                ". Supported types are: json, hdf5");
  }
}

std::shared_ptr<Wavefunction> Wavefunction::from_file(
    const std::string& filename, const std::string& type) {
  if (type == "json") {
    return from_json_file(filename);
  } else if (type == "hdf5") {
    return from_hdf5_file(filename);
  } else {
    throw std::invalid_argument("Unsupported file type: " + type +
                                ". Supported types are: json, hdf5");
  }
}

void Wavefunction::_to_json_file(const std::string& filename) const {
  std::ofstream file(filename);
  if (!file.is_open()) {
    throw std::runtime_error("Cannot open file for writing: " + filename);
  }
  nlohmann::json j = to_json();
  file << j.dump(2);  // Pretty print with 2-space indentation
  file.close();
}

void Wavefunction::_to_hdf5_file(const std::string& filename) const {
  try {
    H5::H5File file(filename, H5F_ACC_TRUNC);
    H5::Group wavefunction_group = file.createGroup("/wavefunction");
    to_hdf5(wavefunction_group);
  } catch (const H5::Exception& e) {
    throw std::runtime_error("HDF5 error: " + std::string(e.getCDetailMsg()));
  }
}

std::shared_ptr<Wavefunction> Wavefunction::_from_json_file(
    const std::string& filename) {
  std::ifstream file(filename);
  if (!file.is_open()) {
    throw std::runtime_error(
        "Unable to open Wavefunction JSON file '" + filename +
        "'. Please check that the file exists and you have read permissions.");
  }
  nlohmann::json j;
  file >> j;
  return from_json(j);
}

std::shared_ptr<Wavefunction> Wavefunction::_from_hdf5_file(
    const std::string& filename) {
  // Disable HDF5 automatic error printing to stderr unless verbose mode
  if (hdf5_errors_should_be_suppressed()) {
    H5::Exception::dontPrint();
  }

  H5::H5File file;
  try {
    file.openFile(filename, H5F_ACC_RDONLY);
  } catch (const H5::Exception& e) {
    throw std::runtime_error("Unable to open Wavefunction HDF5 file '" +
                             filename + "'. " +
                             "Please check that the file exists, is a valid "
                             "HDF5 file, and you have read permissions.");
  }

  try {
    H5::Group wavefunction_group = file.openGroup("/wavefunction");
    return from_hdf5(wavefunction_group);
  } catch (const H5::Exception& e) {
    throw std::runtime_error(
        "Unable to read Wavefunction data from HDF5 file '" + filename + "'. " +
        "HDF5 error: " + std::string(e.getCDetailMsg()));
  }
}

std::string Wavefunction::get_summary() const {
  std::ostringstream oss;
  oss << "Wavefunction Summary:\n";
  oss << "  Container type: " << _container->get_container_type() << "\n";
  oss << "  Number of determinants: " << size() << "\n";
  oss << "  Wavefunction type: "
      << (get_type() == WavefunctionType::SelfDual ? "SelfDual" : "NotSelfDual")
      << "\n";
  oss << "  Complex: " << (is_complex() ? "yes" : "no") << "\n";
  oss << "  Norm: " << norm() << "\n";

  auto [n_alpha_total, n_beta_total] = get_total_num_electrons();
  auto [n_alpha_active, n_beta_active] = get_active_num_electrons();

  oss << "  Total electrons (α,β): (" << n_alpha_total << "," << n_beta_total
      << ")\n";
  oss << "  Active electrons (α,β): (" << n_alpha_active << "," << n_beta_active
      << ")\n";

  // RDM availability
  oss << "  1-RDM available: " << (has_one_rdm_spin_dependent() ? "yes" : "no")
      << "\n";
  oss << "  2-RDM available: " << (has_two_rdm_spin_dependent() ? "yes" : "no")
      << "\n";

  if (auto orbitals = get_orbitals()) {
    oss << "  Orbitals: " << orbitals->get_num_molecular_orbitals() << " MOs, "
        << (orbitals->is_restricted() ? "restricted" : "unrestricted");
  } else {
    oss << "  Orbitals: none";
  }

  return oss.str();
}
}  // namespace qdk::chemistry::data
