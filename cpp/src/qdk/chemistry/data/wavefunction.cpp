// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <H5Cpp.h>

#include <Eigen/Dense>
#include <algorithm>
#include <fstream>
#include <memory>
#include <numeric>
#include <qdk/chemistry/data/wavefunction.hpp>
#include <qdk/chemistry/data/wavefunction_containers/cas.hpp>
#include <qdk/chemistry/data/wavefunction_containers/cc.hpp>
#include <qdk/chemistry/data/wavefunction_containers/mp2.hpp>
#include <qdk/chemistry/data/wavefunction_containers/sci.hpp>
#include <qdk/chemistry/data/wavefunction_containers/sd.hpp>
#include <qdk/chemistry/utils/logger.hpp>
#include <qdk/chemistry/utils/string_utils.hpp>
#include <sstream>
#include <tuple>
#include <variant>

#include "filename_utils.hpp"
#include "hdf5_error_handling.hpp"
#include "hdf5_serialization.hpp"
#include "json_serialization.hpp"

namespace qdk::chemistry::data {
namespace detail {
std::shared_ptr<ContainerTypes::MatrixVariant> add_matrix_variants(
    const ContainerTypes::MatrixVariant& mat1,
    const ContainerTypes::MatrixVariant& mat2) {
  QDK_LOG_TRACE_ENTERING();
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
  QDK_LOG_TRACE_ENTERING();
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
  QDK_LOG_TRACE_ENTERING();
  return std::holds_alternative<Eigen::MatrixXcd>(variant);
}

bool is_vector_variant_complex(const ContainerTypes::VectorVariant& variant) {
  QDK_LOG_TRACE_ENTERING();
  return std::holds_alternative<Eigen::VectorXcd>(variant);
}

std::shared_ptr<ContainerTypes::VectorVariant>
transpose_ijkl_klij_vector_variant(const ContainerTypes::VectorVariant& variant,
                                   int norbs) {
  QDK_LOG_TRACE_ENTERING();
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
                            type) {
  QDK_LOG_TRACE_ENTERING();
}

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
                            type) {
  QDK_LOG_TRACE_ENTERING();
}
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
  QDK_LOG_TRACE_ENTERING();
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
  QDK_LOG_TRACE_ENTERING();
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
  QDK_LOG_TRACE_ENTERING();
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
  QDK_LOG_TRACE_ENTERING();
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
  QDK_LOG_TRACE_ENTERING();
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
  QDK_LOG_TRACE_ENTERING();
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
  QDK_LOG_TRACE_ENTERING();
  return _one_rdm_spin_traced != nullptr || has_one_rdm_spin_dependent();
}

bool WavefunctionContainer::has_two_rdm_spin_dependent() const {
  QDK_LOG_TRACE_ENTERING();
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
  QDK_LOG_TRACE_ENTERING();
  return _two_rdm_spin_traced != nullptr || has_two_rdm_spin_dependent();
}

// entropies
Eigen::VectorXd WavefunctionContainer::get_single_orbital_entropies() const {
  QDK_LOG_TRACE_ENTERING();
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

  // Source: Boguslawski & Tecmer (2015). doi:10.1002/qua.24832
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
  QDK_LOG_TRACE_ENTERING();
  return has_one_rdm_spin_dependent() &&
         _two_rdm_spin_dependent_aabb != nullptr;
}

void WavefunctionContainer::_clear_rdms() const {
  QDK_LOG_TRACE_ENTERING();
  _one_rdm_spin_traced.reset();
  _one_rdm_spin_dependent_aa.reset();
  _one_rdm_spin_dependent_bb.reset();
  _two_rdm_spin_traced.reset();
  _two_rdm_spin_dependent_aabb.reset();
  _two_rdm_spin_dependent_aaaa.reset();
  _two_rdm_spin_dependent_bbbb.reset();
}

void WavefunctionContainer::_serialize_rdms_to_hdf5(H5::Group& group) const {
  QDK_LOG_TRACE_ENTERING();

  // If rdms are available, store them
  if (has_one_rdm_spin_dependent() || has_two_rdm_spin_dependent()) {
    H5::Group rdm_group = group.createGroup("rdms");

    if (has_one_rdm_spin_dependent()) {
      // For restricted systems, only store one spin component
      if (get_orbitals()->is_restricted()) {
        std::string storage_name = "one_rdm_aa";
        H5::Attribute one_rdm_aa_complex_attr = rdm_group.createAttribute(
            "is_one_rdm_aa_complex", H5::PredType::NATIVE_HBOOL,
            H5::DataSpace(H5S_SCALAR));

        if (_one_rdm_spin_dependent_aa != nullptr) {
          bool is_one_rdm_complex =
              detail::is_matrix_variant_complex(*_one_rdm_spin_dependent_aa);
          save_matrix_variant_to_group(is_one_rdm_complex,
                                       _one_rdm_spin_dependent_aa, rdm_group,
                                       storage_name);
          hbool_t is_one_rdm_aa_complex_flag = is_one_rdm_complex ? 1 : 0;
          one_rdm_aa_complex_attr.write(H5::PredType::NATIVE_HBOOL,
                                        &is_one_rdm_aa_complex_flag);
        } else if (_one_rdm_spin_dependent_bb != nullptr) {
          bool is_one_rdm_complex =
              detail::is_matrix_variant_complex(*_one_rdm_spin_dependent_bb);
          save_matrix_variant_to_group(is_one_rdm_complex,
                                       _one_rdm_spin_dependent_bb, rdm_group,
                                       storage_name);
          hbool_t is_one_rdm_aa_complex_flag = is_one_rdm_complex ? 1 : 0;
          one_rdm_aa_complex_attr.write(H5::PredType::NATIVE_HBOOL,
                                        &is_one_rdm_aa_complex_flag);
        } else if (_one_rdm_spin_traced != nullptr &&
                   get_orbitals()->is_restricted()) {
          auto derived_one_rdm_spin_dependent_aa =
              detail::multiply_matrix_variant(*_one_rdm_spin_traced, 0.5);
          bool is_one_rdm_complex = detail::is_matrix_variant_complex(
              *derived_one_rdm_spin_dependent_aa);
          save_matrix_variant_to_group(is_one_rdm_complex,
                                       derived_one_rdm_spin_dependent_aa,
                                       rdm_group, storage_name);
          hbool_t is_one_rdm_aa_complex_flag = is_one_rdm_complex ? 1 : 0;
          one_rdm_aa_complex_attr.write(H5::PredType::NATIVE_HBOOL,
                                        &is_one_rdm_aa_complex_flag);
        } else {
          throw std::runtime_error(
              "Spin-dependent one-body RDMs are supposedly available, but "
              "could not be retrieved in the expected format");
        }
      } else {
        // Unrestricted - store both spin components
        std::string storage_name_aa = "one_rdm_aa";
        H5::Attribute one_rdm_aa_complex_attr = rdm_group.createAttribute(
            "is_one_rdm_aa_complex", H5::PredType::NATIVE_HBOOL,
            H5::DataSpace(H5S_SCALAR));
        H5::Attribute one_rdm_bb_complex_attr = rdm_group.createAttribute(
            "is_one_rdm_bb_complex", H5::PredType::NATIVE_HBOOL,
            H5::DataSpace(H5S_SCALAR));

        if (_one_rdm_spin_dependent_aa != nullptr) {
          bool is_aa_rdm_complex =
              detail::is_matrix_variant_complex(*_one_rdm_spin_dependent_aa);
          save_matrix_variant_to_group(is_aa_rdm_complex,
                                       _one_rdm_spin_dependent_aa, rdm_group,
                                       storage_name_aa);
          hbool_t is_one_rdm_aa_complex_flag = is_aa_rdm_complex ? 1 : 0;
          one_rdm_aa_complex_attr.write(H5::PredType::NATIVE_HBOOL,
                                        &is_one_rdm_aa_complex_flag);
        } else {
          throw std::runtime_error(
              "Expected _one_rdm_spin_dependent_aa to point to something "
              "other than nullptr");
        }

        if (_one_rdm_spin_dependent_bb != nullptr) {
          std::string storage_name_bb = "one_rdm_bb";
          bool is_bb_rdm_complex =
              detail::is_matrix_variant_complex(*_one_rdm_spin_dependent_bb);
          save_matrix_variant_to_group(is_bb_rdm_complex,
                                       _one_rdm_spin_dependent_bb, rdm_group,
                                       storage_name_bb);
          hbool_t is_one_rdm_bb_complex_flag = is_bb_rdm_complex ? 1 : 0;
          one_rdm_bb_complex_attr.write(H5::PredType::NATIVE_HBOOL,
                                        &is_one_rdm_bb_complex_flag);
        } else {
          throw std::runtime_error(
              "Expected _one_rdm_spin_dependent_bb to point to something "
              "other than nullptr");
        }
      }
    }

    if (has_two_rdm_spin_dependent()) {
      std::string storage_name_aabb = "two_rdm_aabb";
      std::string storage_name_aaaa = "two_rdm_aaaa";
      H5::Attribute two_rdm_aabb_complex_attr = rdm_group.createAttribute(
          "is_two_rdm_aabb_complex", H5::PredType::NATIVE_HBOOL,
          H5::DataSpace(H5S_SCALAR));
      H5::Attribute two_rdm_aaaa_complex_attr = rdm_group.createAttribute(
          "is_two_rdm_aaaa_complex", H5::PredType::NATIVE_HBOOL,
          H5::DataSpace(H5S_SCALAR));

      if (_two_rdm_spin_dependent_aabb != nullptr) {
        bool is_aabb_rdm_complex =
            detail::is_vector_variant_complex(*_two_rdm_spin_dependent_aabb);
        save_vector_variant_to_group(is_aabb_rdm_complex,
                                     _two_rdm_spin_dependent_aabb, rdm_group,
                                     storage_name_aabb);
        hbool_t is_two_rdm_aabb_complex_flag = is_aabb_rdm_complex ? 1 : 0;
        two_rdm_aabb_complex_attr.write(H5::PredType::NATIVE_HBOOL,
                                        &is_two_rdm_aabb_complex_flag);
      } else {
        throw std::runtime_error(
            "Expected _two_rdm_spin_dependent_aabb to point to something "
            "other than nullptr");
      }

      if (_two_rdm_spin_dependent_aaaa != nullptr) {
        bool is_aaaa_rdm_complex =
            detail::is_vector_variant_complex(*_two_rdm_spin_dependent_aaaa);
        save_vector_variant_to_group(is_aaaa_rdm_complex,
                                     _two_rdm_spin_dependent_aaaa, rdm_group,
                                     storage_name_aaaa);
        hbool_t is_two_rdm_aaaa_complex_flag = is_aaaa_rdm_complex ? 1 : 0;
        two_rdm_aaaa_complex_attr.write(H5::PredType::NATIVE_HBOOL,
                                        &is_two_rdm_aaaa_complex_flag);
      } else {
        throw std::runtime_error(
            "Expected _two_rdm_spin_dependent_aaaa to point to something "
            "other than nullptr");
      }

      if (get_orbitals()->is_unrestricted()) {
        if (_two_rdm_spin_dependent_bbbb != nullptr) {
          std::string storage_name_bbbb = "two_rdm_bbbb";
          H5::Attribute two_rdm_bbbb_complex_attr = rdm_group.createAttribute(
              "is_two_rdm_bbbb_complex", H5::PredType::NATIVE_HBOOL,
              H5::DataSpace(H5S_SCALAR));
          bool is_bbbb_rdm_complex =
              detail::is_vector_variant_complex(*_two_rdm_spin_dependent_bbbb);
          save_vector_variant_to_group(is_bbbb_rdm_complex,
                                       _two_rdm_spin_dependent_bbbb, rdm_group,
                                       storage_name_bbbb);
          hbool_t is_two_rdm_bbbb_complex_flag = is_bbbb_rdm_complex ? 1 : 0;
          two_rdm_bbbb_complex_attr.write(H5::PredType::NATIVE_HBOOL,
                                          &is_two_rdm_bbbb_complex_flag);
        } else {
          throw std::runtime_error(
              "Expected _two_rdm_spin_dependent_bbbb to point to something "
              "other than nullptr");
        }
      }
    }
  }
}

std::tuple<std::optional<ContainerTypes::MatrixVariant>,
           std::optional<ContainerTypes::MatrixVariant>,
           std::optional<ContainerTypes::VectorVariant>,
           std::optional<ContainerTypes::VectorVariant>,
           std::optional<ContainerTypes::VectorVariant>,
           std::optional<ContainerTypes::MatrixVariant>,
           std::optional<ContainerTypes::VectorVariant>>
WavefunctionContainer::_deserialize_rdms_from_hdf5_restricted(
    H5::Group& rdm_group, const std::shared_ptr<Orbitals>& orbitals) {
  QDK_LOG_TRACE_ENTERING();

  std::optional<MatrixVariant> one_rdm_aa;
  std::optional<MatrixVariant> one_rdm_bb;
  std::optional<VectorVariant> two_rdm_aabb;
  std::optional<VectorVariant> two_rdm_aaaa;
  std::optional<VectorVariant> two_rdm_bbbb;
  std::optional<MatrixVariant> one_rdm_spin_traced;
  std::optional<VectorVariant> two_rdm_spin_traced;

  // Load one-RDMs if available
  if (rdm_group.nameExists("one_rdm_aa")) {
    bool is_one_rdm_aa_complex = false;
    if (rdm_group.attrExists("is_one_rdm_aa_complex")) {
      H5::Attribute complex_attr =
          rdm_group.openAttribute("is_one_rdm_aa_complex");
      hbool_t is_complex_flag;
      complex_attr.read(H5::PredType::NATIVE_HBOOL, &is_complex_flag);
      is_one_rdm_aa_complex = (is_complex_flag != 0);
    }
    one_rdm_aa = load_matrix_variant_from_group(rdm_group, "one_rdm_aa",
                                                is_one_rdm_aa_complex);
    one_rdm_bb = one_rdm_aa;  // For restricted, both spins are the same

    // Compute spin-traced RDM
    auto spin_traced_result = detail::multiply_matrix_variant(*one_rdm_aa, 2.0);
    if (spin_traced_result != nullptr) {
      one_rdm_spin_traced = *spin_traced_result;
    }
  }

  // Load two-RDMs if available
  if (rdm_group.nameExists("two_rdm_aabb") &&
      rdm_group.nameExists("two_rdm_aaaa")) {
    bool is_two_rdm_aabb_complex = false;
    if (rdm_group.attrExists("is_two_rdm_aabb_complex")) {
      H5::Attribute complex_attr =
          rdm_group.openAttribute("is_two_rdm_aabb_complex");
      hbool_t is_complex_flag;
      complex_attr.read(H5::PredType::NATIVE_HBOOL, &is_complex_flag);
      is_two_rdm_aabb_complex = (is_complex_flag != 0);
    }
    bool is_two_rdm_aaaa_complex = false;
    if (rdm_group.attrExists("is_two_rdm_aaaa_complex")) {
      H5::Attribute complex_attr =
          rdm_group.openAttribute("is_two_rdm_aaaa_complex");
      hbool_t is_complex_flag;
      complex_attr.read(H5::PredType::NATIVE_HBOOL, &is_complex_flag);
      is_two_rdm_aaaa_complex = (is_complex_flag != 0);
    }
    two_rdm_aabb = load_vector_variant_from_group(rdm_group, "two_rdm_aabb",
                                                  is_two_rdm_aabb_complex);
    two_rdm_aaaa = load_vector_variant_from_group(rdm_group, "two_rdm_aaaa",
                                                  is_two_rdm_aaaa_complex);
    two_rdm_bbbb = two_rdm_aaaa;  // For restricted, bbbb is the same as aaaa

    // Compute spin-traced two-RDM
    auto two_rdm_ss_result =
        detail::multiply_vector_variant(*two_rdm_aaaa, 2.0);
    auto two_rdm_bbaa_result = detail::transpose_ijkl_klij_vector_variant(
        *two_rdm_aabb, orbitals->get_active_space_indices().first.size());

    if (two_rdm_ss_result != nullptr && two_rdm_bbaa_result != nullptr) {
      auto two_rdm_os_result =
          detail::add_vector_variants(*two_rdm_aabb, *two_rdm_bbaa_result);
      if (two_rdm_os_result != nullptr) {
        auto final_result =
            detail::add_vector_variants(*two_rdm_ss_result, *two_rdm_os_result);
        if (final_result != nullptr) {
          two_rdm_spin_traced = *final_result;
        }
      }
    }
  }

  return std::make_tuple(one_rdm_aa, one_rdm_bb, two_rdm_aabb, two_rdm_aaaa,
                         two_rdm_bbbb, one_rdm_spin_traced,
                         two_rdm_spin_traced);
}

std::tuple<std::optional<ContainerTypes::MatrixVariant>,
           std::optional<ContainerTypes::MatrixVariant>,
           std::optional<ContainerTypes::VectorVariant>,
           std::optional<ContainerTypes::VectorVariant>,
           std::optional<ContainerTypes::VectorVariant>,
           std::optional<ContainerTypes::MatrixVariant>,
           std::optional<ContainerTypes::VectorVariant>>
WavefunctionContainer::_deserialize_rdms_from_hdf5_unrestricted(
    H5::Group& rdm_group, const std::shared_ptr<Orbitals>& orbitals) {
  QDK_LOG_TRACE_ENTERING();

  std::optional<MatrixVariant> one_rdm_aa;
  std::optional<MatrixVariant> one_rdm_bb;
  std::optional<VectorVariant> two_rdm_aabb;
  std::optional<VectorVariant> two_rdm_aaaa;
  std::optional<VectorVariant> two_rdm_bbbb;
  std::optional<MatrixVariant> one_rdm_spin_traced;
  std::optional<VectorVariant> two_rdm_spin_traced;

  // Load one-RDMs (both aa and bb for unrestricted)
  if (rdm_group.nameExists("one_rdm_aa")) {
    bool is_one_rdm_aa_complex = false;
    if (rdm_group.attrExists("is_one_rdm_aa_complex")) {
      H5::Attribute complex_attr =
          rdm_group.openAttribute("is_one_rdm_aa_complex");
      hbool_t is_complex_flag;
      complex_attr.read(H5::PredType::NATIVE_HBOOL, &is_complex_flag);
      is_one_rdm_aa_complex = (is_complex_flag != 0);
    }
    one_rdm_aa = load_matrix_variant_from_group(rdm_group, "one_rdm_aa",
                                                is_one_rdm_aa_complex);
  }

  if (rdm_group.nameExists("one_rdm_bb")) {
    bool is_one_rdm_bb_complex = false;
    if (rdm_group.attrExists("is_one_rdm_bb_complex")) {
      H5::Attribute complex_attr =
          rdm_group.openAttribute("is_one_rdm_bb_complex");
      hbool_t is_complex_flag;
      complex_attr.read(H5::PredType::NATIVE_HBOOL, &is_complex_flag);
      is_one_rdm_bb_complex = (is_complex_flag != 0);
    }
    one_rdm_bb = load_matrix_variant_from_group(rdm_group, "one_rdm_bb",
                                                is_one_rdm_bb_complex);

    // Compute spin-traced one-RDM
    if (one_rdm_aa.has_value()) {
      auto spin_traced_result =
          detail::add_matrix_variants(*one_rdm_aa, *one_rdm_bb);
      if (spin_traced_result != nullptr) {
        one_rdm_spin_traced = *spin_traced_result;
      }
    }
  }

  // Load two-RDMs
  if (rdm_group.nameExists("two_rdm_aabb") &&
      rdm_group.nameExists("two_rdm_aaaa")) {
    bool is_two_rdm_aabb_complex = false;
    if (rdm_group.attrExists("is_two_rdm_aabb_complex")) {
      H5::Attribute complex_attr =
          rdm_group.openAttribute("is_two_rdm_aabb_complex");
      hbool_t is_complex_flag;
      complex_attr.read(H5::PredType::NATIVE_HBOOL, &is_complex_flag);
      is_two_rdm_aabb_complex = (is_complex_flag != 0);
    }
    bool is_two_rdm_aaaa_complex = false;
    if (rdm_group.attrExists("is_two_rdm_aaaa_complex")) {
      H5::Attribute complex_attr =
          rdm_group.openAttribute("is_two_rdm_aaaa_complex");
      hbool_t is_complex_flag;
      complex_attr.read(H5::PredType::NATIVE_HBOOL, &is_complex_flag);
      is_two_rdm_aaaa_complex = (is_complex_flag != 0);
    }
    two_rdm_aabb = load_vector_variant_from_group(rdm_group, "two_rdm_aabb",
                                                  is_two_rdm_aabb_complex);
    two_rdm_aaaa = load_vector_variant_from_group(rdm_group, "two_rdm_aaaa",
                                                  is_two_rdm_aaaa_complex);
  }

  if (rdm_group.nameExists("two_rdm_bbbb")) {
    bool is_two_rdm_bbbb_complex = false;
    if (rdm_group.attrExists("is_two_rdm_bbbb_complex")) {
      H5::Attribute complex_attr =
          rdm_group.openAttribute("is_two_rdm_bbbb_complex");
      hbool_t is_complex_flag;
      complex_attr.read(H5::PredType::NATIVE_HBOOL, &is_complex_flag);
      is_two_rdm_bbbb_complex = (is_complex_flag != 0);
    }
    two_rdm_bbbb = load_vector_variant_from_group(rdm_group, "two_rdm_bbbb",
                                                  is_two_rdm_bbbb_complex);

    // Compute spin-traced two-RDM
    if (two_rdm_aabb.has_value() && two_rdm_aaaa.has_value()) {
      auto two_rdm_ss_result =
          detail::add_vector_variants(*two_rdm_aaaa, *two_rdm_bbbb);
      auto two_rdm_bbaa_result = detail::transpose_ijkl_klij_vector_variant(
          *two_rdm_aabb, orbitals->get_active_space_indices().first.size());

      if (two_rdm_ss_result != nullptr && two_rdm_bbaa_result != nullptr) {
        auto two_rdm_os_result =
            detail::add_vector_variants(*two_rdm_aabb, *two_rdm_bbaa_result);
        if (two_rdm_os_result != nullptr) {
          auto final_result = detail::add_vector_variants(*two_rdm_ss_result,
                                                          *two_rdm_os_result);
          if (final_result != nullptr) {
            two_rdm_spin_traced = *final_result;
          }
        }
      }
    }
  }

  return std::make_tuple(one_rdm_aa, one_rdm_bb, two_rdm_aabb, two_rdm_aaaa,
                         two_rdm_bbbb, one_rdm_spin_traced,
                         two_rdm_spin_traced);
}

void WavefunctionContainer::_serialize_rdms_to_json(nlohmann::json& j) const {
  QDK_LOG_TRACE_ENTERING();

  // If rdms are available, store them
  if (has_one_rdm_spin_dependent() || has_two_rdm_spin_dependent()) {
    nlohmann::json rdm_json;

    if (has_one_rdm_spin_dependent()) {
      // For restricted systems, only store one spin component
      if (get_orbitals()->is_restricted()) {
        if (_one_rdm_spin_dependent_aa != nullptr) {
          bool is_one_rdm_complex =
              detail::is_matrix_variant_complex(*_one_rdm_spin_dependent_aa);
          rdm_json["is_one_rdm_aa_complex"] = is_one_rdm_complex;
          rdm_json["one_rdm_aa"] = matrix_variant_to_json(
              *_one_rdm_spin_dependent_aa, is_one_rdm_complex);
        } else if (_one_rdm_spin_dependent_bb != nullptr) {
          bool is_one_rdm_complex =
              detail::is_matrix_variant_complex(*_one_rdm_spin_dependent_bb);
          rdm_json["is_one_rdm_aa_complex"] = is_one_rdm_complex;
          rdm_json["one_rdm_aa"] = matrix_variant_to_json(
              *_one_rdm_spin_dependent_bb, is_one_rdm_complex);
        } else if (_one_rdm_spin_traced != nullptr) {
          // Only spin-traced RDM available - derive spin component
          auto derived_one_rdm_aa =
              detail::multiply_matrix_variant(*_one_rdm_spin_traced, 0.5);
          bool is_one_rdm_complex =
              detail::is_matrix_variant_complex(*derived_one_rdm_aa);
          rdm_json["is_one_rdm_aa_complex"] = is_one_rdm_complex;
          rdm_json["one_rdm_aa"] =
              matrix_variant_to_json(*derived_one_rdm_aa, is_one_rdm_complex);
        }
      } else {
        // Unrestricted - store both spin components
        if (_one_rdm_spin_dependent_aa != nullptr) {
          bool is_aa_complex =
              detail::is_matrix_variant_complex(*_one_rdm_spin_dependent_aa);
          rdm_json["is_one_rdm_aa_complex"] = is_aa_complex;
          rdm_json["one_rdm_aa"] = matrix_variant_to_json(
              *_one_rdm_spin_dependent_aa, is_aa_complex);
        }
        if (_one_rdm_spin_dependent_bb != nullptr) {
          bool is_bb_complex =
              detail::is_matrix_variant_complex(*_one_rdm_spin_dependent_bb);
          rdm_json["is_one_rdm_bb_complex"] = is_bb_complex;
          rdm_json["one_rdm_bb"] = matrix_variant_to_json(
              *_one_rdm_spin_dependent_bb, is_bb_complex);
        }
      }
    }

    if (has_two_rdm_spin_dependent()) {
      // Store aabb and aaaa for both restricted and unrestricted
      if (_two_rdm_spin_dependent_aabb != nullptr) {
        bool is_aabb_complex =
            detail::is_vector_variant_complex(*_two_rdm_spin_dependent_aabb);
        rdm_json["is_two_rdm_aabb_complex"] = is_aabb_complex;
        rdm_json["two_rdm_aabb"] = vector_variant_to_json(
            *_two_rdm_spin_dependent_aabb, is_aabb_complex);
      }
      if (_two_rdm_spin_dependent_aaaa != nullptr) {
        bool is_aaaa_complex =
            detail::is_vector_variant_complex(*_two_rdm_spin_dependent_aaaa);
        rdm_json["is_two_rdm_aaaa_complex"] = is_aaaa_complex;
        rdm_json["two_rdm_aaaa"] = vector_variant_to_json(
            *_two_rdm_spin_dependent_aaaa, is_aaaa_complex);
      }

      // For unrestricted, also store bbbb
      if (get_orbitals()->is_unrestricted() &&
          _two_rdm_spin_dependent_bbbb != nullptr) {
        bool is_bbbb_complex =
            detail::is_vector_variant_complex(*_two_rdm_spin_dependent_bbbb);
        rdm_json["is_two_rdm_bbbb_complex"] = is_bbbb_complex;
        rdm_json["two_rdm_bbbb"] = vector_variant_to_json(
            *_two_rdm_spin_dependent_bbbb, is_bbbb_complex);
      }
    }

    j["rdms"] = rdm_json;
  }
}

std::tuple<std::optional<ContainerTypes::MatrixVariant>,
           std::optional<ContainerTypes::MatrixVariant>,
           std::optional<ContainerTypes::VectorVariant>,
           std::optional<ContainerTypes::VectorVariant>,
           std::optional<ContainerTypes::VectorVariant>>
WavefunctionContainer::_deserialize_rdms_from_json_restricted(
    const nlohmann::json& j) {
  QDK_LOG_TRACE_ENTERING();

  std::optional<MatrixVariant> one_rdm_aa;
  std::optional<MatrixVariant> one_rdm_bb;
  std::optional<VectorVariant> two_rdm_aabb;
  std::optional<VectorVariant> two_rdm_aaaa;
  std::optional<VectorVariant> two_rdm_bbbb;

  if (j.contains("rdms")) {
    const auto& rdm_json = j["rdms"];

    // Load one-RDMs if available
    if (rdm_json.contains("one_rdm_aa")) {
      bool is_complex = rdm_json.value("is_one_rdm_aa_complex", false);
      one_rdm_aa = json_to_matrix_variant(rdm_json["one_rdm_aa"], is_complex);
      // For restricted, both spins are the same
      one_rdm_bb = one_rdm_aa;
    }

    // Load two-RDMs if available
    if (rdm_json.contains("two_rdm_aabb")) {
      bool is_aabb_complex = rdm_json.value("is_two_rdm_aabb_complex", false);
      two_rdm_aabb =
          json_to_vector_variant(rdm_json["two_rdm_aabb"], is_aabb_complex);
    }
    if (rdm_json.contains("two_rdm_aaaa")) {
      bool is_aaaa_complex = rdm_json.value("is_two_rdm_aaaa_complex", false);
      two_rdm_aaaa =
          json_to_vector_variant(rdm_json["two_rdm_aaaa"], is_aaaa_complex);
      // For restricted, bbbb is the same as aaaa
      two_rdm_bbbb = two_rdm_aaaa;
    }
  }

  return std::make_tuple(one_rdm_aa, one_rdm_bb, two_rdm_aabb, two_rdm_aaaa,
                         two_rdm_bbbb);
}

std::tuple<std::optional<ContainerTypes::MatrixVariant>,
           std::optional<ContainerTypes::MatrixVariant>,
           std::optional<ContainerTypes::VectorVariant>,
           std::optional<ContainerTypes::VectorVariant>,
           std::optional<ContainerTypes::VectorVariant>>
WavefunctionContainer::_deserialize_rdms_from_json_unrestricted(
    const nlohmann::json& j) {
  QDK_LOG_TRACE_ENTERING();

  std::optional<MatrixVariant> one_rdm_aa;
  std::optional<MatrixVariant> one_rdm_bb;
  std::optional<VectorVariant> two_rdm_aabb;
  std::optional<VectorVariant> two_rdm_aaaa;
  std::optional<VectorVariant> two_rdm_bbbb;

  if (j.contains("rdms")) {
    const auto& rdm_json = j["rdms"];

    if (rdm_json.contains("one_rdm_aa")) {
      bool is_aa_complex = rdm_json.value("is_one_rdm_aa_complex", false);
      one_rdm_aa =
          json_to_matrix_variant(rdm_json["one_rdm_aa"], is_aa_complex);
    }
    if (rdm_json.contains("one_rdm_bb")) {
      bool is_bb_complex = rdm_json.value("is_one_rdm_bb_complex", false);
      one_rdm_bb =
          json_to_matrix_variant(rdm_json["one_rdm_bb"], is_bb_complex);
    }
    if (rdm_json.contains("two_rdm_aabb")) {
      bool is_aabb_complex = rdm_json.value("is_two_rdm_aabb_complex", false);
      two_rdm_aabb =
          json_to_vector_variant(rdm_json["two_rdm_aabb"], is_aabb_complex);
    }
    if (rdm_json.contains("two_rdm_aaaa")) {
      bool is_aaaa_complex = rdm_json.value("is_two_rdm_aaaa_complex", false);
      two_rdm_aaaa =
          json_to_vector_variant(rdm_json["two_rdm_aaaa"], is_aaaa_complex);
    }
    if (rdm_json.contains("two_rdm_bbbb")) {
      bool is_bbbb_complex = rdm_json.value("is_two_rdm_bbbb_complex", false);
      two_rdm_bbbb =
          json_to_vector_variant(rdm_json["two_rdm_bbbb"], is_bbbb_complex);
    }
  }

  return std::make_tuple(one_rdm_aa, one_rdm_bb, two_rdm_aabb, two_rdm_aaaa,
                         two_rdm_bbbb);
}

std::unique_ptr<WavefunctionContainer> WavefunctionContainer::from_json(
    const nlohmann::json& j) {
  QDK_LOG_TRACE_ENTERING();

  try {
    // Check version first
    if (!j.contains("version")) {
      throw std::runtime_error("Invalid JSON: missing version field");
    }
    validate_serialization_version(SERIALIZATION_VERSION, j["version"]);

    // Load wavefunction type
    WavefunctionType type = WavefunctionType::SelfDual;
    if (j.contains("wavefunction_type")) {
      std::string type_str = j["wavefunction_type"];
      type = (type_str == "self_dual") ? WavefunctionType::SelfDual
                                       : WavefunctionType::NotSelfDual;
    }

    // Load container type
    if (!j.contains("container_type")) {
      throw std::runtime_error("JSON missing 'container_type' field");
    }
    std::string container_type = j["container_type"];

    // Load restrictedness flag (may be overridden by orbitals later)
    bool is_restricted = j.value("is_restricted", false);

    // Load coefficients if they exist
    VectorVariant coefficients;
    if (j.contains("coefficients")) {
      bool is_complex = j.value("is_complex", false);
      const auto& coeffs_json = j["coefficients"];

      if (is_complex) {
        if (!coeffs_json.is_array() || coeffs_json.empty() ||
            !coeffs_json[0].is_array()) {
          throw std::runtime_error(
              "Invalid complex coefficient format: expected array of [real, "
              "imag] pairs");
        }
        Eigen::VectorXcd coeffs_complex(coeffs_json.size());
        for (size_t i = 0; i < coeffs_json.size(); ++i) {
          if (coeffs_json[i].size() != 2) {
            throw std::runtime_error(
                "Invalid complex coefficient format: expected [real, imag] "
                "pairs");
          }
          coeffs_complex(i) =
              std::complex<double>(coeffs_json[i][0], coeffs_json[i][1]);
        }
        coefficients = coeffs_complex;
      } else {
        std::vector<double> coeff_data = coeffs_json;
        Eigen::VectorXd coeffs_real =
            Eigen::Map<Eigen::VectorXd>(coeff_data.data(), coeff_data.size());
        coefficients = coeffs_real;
      }
    }

    // Load configuration set (delegates to ConfigurationSet deserialization)
    DeterminantVector determinants;
    std::shared_ptr<Orbitals> orbitals;
    if (j.contains("configuration_set")) {
      auto config_set = ConfigurationSet::from_json(j["configuration_set"]);
      determinants = config_set.get_configurations();
      orbitals = config_set.get_orbitals();
    } else {
      determinants = {};
      orbitals = nullptr;
    }

    // Load RDMs if they are available
    if (j.contains("rdms")) {
      if (orbitals != nullptr) {
        // Determine if we're dealing with restricted or unrestricted orbitals
        is_restricted = orbitals->is_restricted();
      }

      // Deserialize RDMs using helper functions
      std::optional<MatrixVariant> one_rdm_aa;
      std::optional<MatrixVariant> one_rdm_bb;
      std::optional<VectorVariant> two_rdm_aabb;
      std::optional<VectorVariant> two_rdm_aaaa;
      std::optional<VectorVariant> two_rdm_bbbb;

      if (is_restricted) {
        std::tie(one_rdm_aa, one_rdm_bb, two_rdm_aabb, two_rdm_aaaa,
                 two_rdm_bbbb) = _deserialize_rdms_from_json_restricted(j);
      } else {
        std::tie(one_rdm_aa, one_rdm_bb, two_rdm_aabb, two_rdm_aaaa,
                 two_rdm_bbbb) = _deserialize_rdms_from_json_unrestricted(j);
      }

      // Return appropriate container with RDMs
      bool has_one_rdm = one_rdm_aa.has_value();
      bool has_two_rdm = two_rdm_aabb.has_value() && two_rdm_aaaa.has_value();

      if (has_one_rdm && !has_two_rdm) {
        if (container_type == "cas") {
          return std::make_unique<CasWavefunctionContainer>(
              coefficients, determinants, orbitals, std::nullopt, one_rdm_aa,
              one_rdm_bb, std::nullopt, std::nullopt, std::nullopt,
              std::nullopt, type);
        } else if (container_type == "sci") {
          return std::make_unique<SciWavefunctionContainer>(
              coefficients, determinants, orbitals, std::nullopt, one_rdm_aa,
              one_rdm_bb, std::nullopt, std::nullopt, std::nullopt,
              std::nullopt, type);
        }
      } else if (!has_one_rdm && has_two_rdm) {
        if (container_type == "cas") {
          return std::make_unique<CasWavefunctionContainer>(
              coefficients, determinants, orbitals, std::nullopt, std::nullopt,
              std::nullopt, std::nullopt, two_rdm_aabb, two_rdm_aaaa,
              two_rdm_bbbb, type);
        } else if (container_type == "sci") {
          return std::make_unique<SciWavefunctionContainer>(
              coefficients, determinants, orbitals, std::nullopt, std::nullopt,
              std::nullopt, std::nullopt, two_rdm_aabb, two_rdm_aaaa,
              two_rdm_bbbb, type);
        }
      } else if (has_one_rdm && has_two_rdm) {
        if (container_type == "cas") {
          return std::make_unique<CasWavefunctionContainer>(
              coefficients, determinants, orbitals, std::nullopt, one_rdm_aa,
              one_rdm_bb, std::nullopt, two_rdm_aabb, two_rdm_aaaa,
              two_rdm_bbbb, type);
        } else if (container_type == "sci") {
          return std::make_unique<SciWavefunctionContainer>(
              coefficients, determinants, orbitals, std::nullopt, one_rdm_aa,
              one_rdm_bb, std::nullopt, two_rdm_aabb, two_rdm_aaaa,
              two_rdm_bbbb, type);
        }
      }
      // Throw if RDMs are present but the container type is not supported
      if ((has_one_rdm || has_two_rdm) &&
          (container_type != "cas" && container_type != "sci")) {
        throw std::runtime_error(
            "RDMs are only supported for CAS and SCI containers in "
            "WavefunctionContainer::from_json. Container type: " +
            container_type);
      }
    }

    if (container_type == "cas") {
      return std::make_unique<CasWavefunctionContainer>(
          coefficients, determinants, orbitals, type);
    } else if (container_type == "sci") {
      return std::make_unique<SciWavefunctionContainer>(
          coefficients, determinants, orbitals, type);
    } else {
      throw std::runtime_error(
          "Did not expect to get here for containers other than cas/sci. "
          "Expected delegation to container-specific methods.");
    }
  } catch (const std::exception& e) {
    throw std::runtime_error("JSON error: " + std::string(e.what()));
  }
}

WavefunctionType WavefunctionContainer::get_type() const {
  QDK_LOG_TRACE_ENTERING();
  return _type;
}

// Wavefunction implementations
Wavefunction::Wavefunction(std::unique_ptr<WavefunctionContainer> container)
    : _container(std::move(container)) {
  QDK_LOG_TRACE_ENTERING();
}

// Copy constructor
Wavefunction::Wavefunction(const Wavefunction& other)
    : _container(other._container->clone()) {
  QDK_LOG_TRACE_ENTERING();
}

// Copy assignment operator
Wavefunction& Wavefunction::operator=(const Wavefunction& other) {
  QDK_LOG_TRACE_ENTERING();
  if (this != &other) {
    _container = other._container->clone();
  }
  return *this;
}

std::shared_ptr<Orbitals> Wavefunction::get_orbitals() const {
  QDK_LOG_TRACE_ENTERING();
  return _container->get_orbitals();
}

std::string Wavefunction::get_container_type() const {
  QDK_LOG_TRACE_ENTERING();
  return _container->get_container_type();
}

std::pair<size_t, size_t> Wavefunction::get_total_num_electrons() const {
  QDK_LOG_TRACE_ENTERING();
  return _container->get_total_num_electrons();
}

std::pair<size_t, size_t> Wavefunction::get_active_num_electrons() const {
  QDK_LOG_TRACE_ENTERING();
  return _container->get_active_num_electrons();
}

std::pair<Eigen::VectorXd, Eigen::VectorXd>
Wavefunction::get_total_orbital_occupations() const {
  QDK_LOG_TRACE_ENTERING();
  return _container->get_total_orbital_occupations();
}

std::pair<Eigen::VectorXd, Eigen::VectorXd>
Wavefunction::get_active_orbital_occupations() const {
  QDK_LOG_TRACE_ENTERING();
  return _container->get_active_orbital_occupations();
}

Wavefunction::ScalarVariant Wavefunction::get_coefficient(
    const Configuration& det) const {
  QDK_LOG_TRACE_ENTERING();
  return _container->get_coefficient(det);
}

const Wavefunction::VectorVariant& Wavefunction::get_coefficients() const {
  QDK_LOG_TRACE_ENTERING();
  return _container->get_coefficients();
}

const Wavefunction::DeterminantVector& Wavefunction::get_active_determinants()
    const {
  QDK_LOG_TRACE_ENTERING();
  return _container->get_active_determinants();
}

Wavefunction::DeterminantVector Wavefunction::get_total_determinants() const {
  QDK_LOG_TRACE_ENTERING();
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
  QDK_LOG_TRACE_ENTERING();
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
  QDK_LOG_TRACE_ENTERING();
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

size_t Wavefunction::size() const {
  QDK_LOG_TRACE_ENTERING();
  return _container->size();
}

std::pair<Wavefunction::DeterminantVector, Wavefunction::VectorVariant>
Wavefunction::get_top_determinants(
    std::optional<size_t> max_determinants) const {
  QDK_LOG_TRACE_ENTERING();
  const auto& determinants = get_active_determinants();
  const auto& coeffs = get_coefficients();

  // Create indices and sort by coefficient magnitude
  std::vector<size_t> indices(determinants.size());
  std::iota(indices.begin(), indices.end(), 0);

  std::visit(
      [&indices](const auto& vec) {
        std::sort(indices.begin(), indices.end(), [&vec](size_t i1, size_t i2) {
          return std::abs(vec[i1]) > std::abs(vec[i2]);
        });
      },
      coeffs);

  // Determine how many determinants to return
  size_t n = determinants.size();
  if (max_determinants.has_value()) {
    n = std::min(max_determinants.value(), n);
  }

  // Build result as pair of vectors preserving the original coefficient type
  std::vector<Configuration> configs;
  configs.reserve(n);
  for (size_t i = 0; i < n; ++i) {
    configs.push_back(determinants[indices[i]]);
  }

  auto coeff_vec = std::visit(
      [&indices, n](const auto& vec) -> VectorVariant {
        using VecType = std::decay_t<decltype(vec)>;
        VecType result(static_cast<Eigen::Index>(n));
        for (size_t i = 0; i < n; ++i) {
          result[static_cast<Eigen::Index>(i)] = vec[indices[i]];
        }
        return result;
      },
      coeffs);

  return {std::move(configs), std::move(coeff_vec)};
}

std::shared_ptr<Wavefunction> Wavefunction::truncate(
    std::optional<size_t> max_determinants) const {
  QDK_LOG_TRACE_ENTERING();

  // Get top determinants (sorted by absolute coefficient value)
  auto [top_configs, top_coeffs] = get_top_determinants(max_determinants);

  // Renormalize coefficients
  auto normalized_coeffs = std::visit(
      [](const auto& vec) -> VectorVariant {
        using VecType = std::decay_t<decltype(vec)>;
        double norm_val = vec.norm();
        if (norm_val > 0.0) {
          return VecType(vec / norm_val);
        }
        return vec;
      },
      top_coeffs);

  // Create new wavefunction with SciWavefunctionContainer
  return std::make_shared<Wavefunction>(
      std::make_unique<SciWavefunctionContainer>(normalized_coeffs, top_configs,
                                                 get_orbitals(), get_type()));
}

double Wavefunction::norm() const {
  QDK_LOG_TRACE_ENTERING();
  return _container->norm();
}

Wavefunction::ScalarVariant Wavefunction::overlap(
    const Wavefunction& other) const {
  QDK_LOG_TRACE_ENTERING();
  return _container->overlap(*other._container);
}

std::tuple<const Wavefunction::MatrixVariant&,
           const Wavefunction::MatrixVariant&>
Wavefunction::get_active_one_rdm_spin_dependent() const {
  QDK_LOG_TRACE_ENTERING();
  return _container->get_active_one_rdm_spin_dependent();
}

std::tuple<const Wavefunction::VectorVariant&,
           const Wavefunction::VectorVariant&,
           const Wavefunction::VectorVariant&>
Wavefunction::get_active_two_rdm_spin_dependent() const {
  QDK_LOG_TRACE_ENTERING();
  return _container->get_active_two_rdm_spin_dependent();
}

const Wavefunction::MatrixVariant&
Wavefunction::get_active_one_rdm_spin_traced() const {
  QDK_LOG_TRACE_ENTERING();
  return _container->get_active_one_rdm_spin_traced();
}

const Wavefunction::VectorVariant&
Wavefunction::get_active_two_rdm_spin_traced() const {
  QDK_LOG_TRACE_ENTERING();
  return _container->get_active_two_rdm_spin_traced();
}

bool Wavefunction::has_single_orbital_entropies() const {
  QDK_LOG_TRACE_ENTERING();
  return _container->has_single_orbital_entropies();
}

Eigen::VectorXd Wavefunction::get_single_orbital_entropies() const {
  QDK_LOG_TRACE_ENTERING();
  return _container->get_single_orbital_entropies();
}

bool Wavefunction::has_one_rdm_spin_dependent() const {
  QDK_LOG_TRACE_ENTERING();
  return _container->has_one_rdm_spin_dependent();
}

bool Wavefunction::has_one_rdm_spin_traced() const {
  QDK_LOG_TRACE_ENTERING();
  return _container->has_one_rdm_spin_traced();
}

bool Wavefunction::has_two_rdm_spin_dependent() const {
  QDK_LOG_TRACE_ENTERING();
  return _container->has_two_rdm_spin_dependent();
}

bool Wavefunction::has_two_rdm_spin_traced() const {
  QDK_LOG_TRACE_ENTERING();
  return _container->has_two_rdm_spin_traced();
}

// Cache management
void Wavefunction::_clear_caches() const {
  QDK_LOG_TRACE_ENTERING();
  _container->clear_caches();
}

// Type access
WavefunctionType Wavefunction::get_type() const {
  QDK_LOG_TRACE_ENTERING();
  return _container->get_type();
}

bool Wavefunction::is_complex() const {
  QDK_LOG_TRACE_ENTERING();
  return _container->is_complex();
}

nlohmann::json Wavefunction::to_json() const {
  QDK_LOG_TRACE_ENTERING();
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
  QDK_LOG_TRACE_ENTERING();
  try {
    // Validate version first
    if (!j.contains("version")) {
      throw std::runtime_error("Invalid JSON: missing version field");
    }
    validate_serialization_version(SERIALIZATION_VERSION, j["version"]);

    // Load container type identifier
    if (!j.contains("container_type")) {
      throw std::runtime_error("JSON missing required 'container_type' field");
    }
    std::string container_type = j["container_type"];

    // Load container using factory method (orbitals are loaded internally by
    // the container)
    if (!j.contains("container")) {
      throw std::runtime_error("JSON missing required 'container' field");
    }

    // Dispatch to appropriate container implementation
    std::unique_ptr<WavefunctionContainer> container;
    if (container_type == "cas") {
      container = CasWavefunctionContainer::from_json(j["container"]);
    } else if (container_type == "sci") {
      container = SciWavefunctionContainer::from_json(j["container"]);
    } else if (container_type == "sd") {
      container = SlaterDeterminantContainer::from_json(j["container"]);
    } else if (container_type == "coupled_cluster") {
      container = CoupledClusterContainer::from_json(j["container"]);
    } else if (container_type == "mp2") {
      container = MP2Container::from_json(j["container"]);
    } else {
      throw std::runtime_error("Unknown container type: " + container_type);
    }

    return std::make_shared<Wavefunction>(std::move(container));

  } catch (const std::exception& e) {
    throw std::runtime_error("Failed to parse Wavefunction from JSON: " +
                             std::string(e.what()));
  }
}

void Wavefunction::to_json_file(const std::string& filename) const {
  QDK_LOG_TRACE_ENTERING();
  if (filename.empty()) {
    throw std::invalid_argument("Filename cannot be empty");
  }
  DataTypeFilename::validate_write_suffix(
      filename, DATACLASS_TO_SNAKE_CASE(Wavefunction));
  _to_json_file(filename);
}

std::shared_ptr<Wavefunction> Wavefunction::from_json_file(
    const std::string& filename) {
  QDK_LOG_TRACE_ENTERING();
  if (filename.empty()) {
    throw std::invalid_argument("Filename cannot be empty");
  }
  DataTypeFilename::validate_read_suffix(filename, "wavefunction");
  return _from_json_file(filename);
}

void Wavefunction::to_hdf5(H5::Group& group) const {
  QDK_LOG_TRACE_ENTERING();
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
  QDK_LOG_TRACE_ENTERING();
  try {
    // Check version first
    H5::StrType string_type(H5::PredType::C_S1, H5T_VARIABLE);
    H5::Attribute version_attr = group.openAttribute("version");
    std::string version;
    version_attr.read(string_type, version);
    validate_serialization_version(SERIALIZATION_VERSION, version);

    // Load container type and dispatch to appropriate container implementation
    if (!group.nameExists("container")) {
      throw std::runtime_error(
          "HDF5 group missing required 'container' subgroup");
    }

    H5::Group container_group = group.openGroup("container");

    // Read container type from the container group
    if (!container_group.attrExists("container_type")) {
      throw std::runtime_error(
          "Container group missing 'container_type' attribute");
    }

    H5::Attribute type_attr = container_group.openAttribute("container_type");
    std::string container_type;
    type_attr.read(string_type, container_type);

    // Dispatch to appropriate container implementation
    std::unique_ptr<WavefunctionContainer> container;
    if (container_type == "cas") {
      container = CasWavefunctionContainer::from_hdf5(container_group);
    } else if (container_type == "sci") {
      container = SciWavefunctionContainer::from_hdf5(container_group);
    } else if (container_type == "sd") {
      container = SlaterDeterminantContainer::from_hdf5(container_group);
    } else if (container_type == "coupled_cluster") {
      container = CoupledClusterContainer::from_hdf5(container_group);
    } else if (container_type == "mp2") {
      container = MP2Container::from_hdf5(container_group);
    } else {
      throw std::runtime_error("Unknown container type: " + container_type);
    }

    return std::make_shared<Wavefunction>(std::move(container));
  } catch (const H5::Exception& e) {
    throw std::runtime_error("HDF5 error: " + std::string(e.getCDetailMsg()));
  }
}

void Wavefunction::to_hdf5_file(const std::string& filename) const {
  QDK_LOG_TRACE_ENTERING();
  if (filename.empty()) {
    throw std::invalid_argument("Filename cannot be empty");
  }
  DataTypeFilename::validate_write_suffix(
      filename, DATACLASS_TO_SNAKE_CASE(Wavefunction));
  _to_hdf5_file(filename);
}

std::shared_ptr<Wavefunction> Wavefunction::from_hdf5_file(
    const std::string& filename) {
  QDK_LOG_TRACE_ENTERING();
  if (filename.empty()) {
    throw std::invalid_argument("Filename cannot be empty");
  }
  DataTypeFilename::validate_read_suffix(filename, "wavefunction");
  return _from_hdf5_file(filename);
}

void Wavefunction::to_file(const std::string& filename,
                           const std::string& type) const {
  QDK_LOG_TRACE_ENTERING();
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
  QDK_LOG_TRACE_ENTERING();
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
  QDK_LOG_TRACE_ENTERING();
  std::ofstream file(filename);
  if (!file.is_open()) {
    throw std::runtime_error("Cannot open file for writing: " + filename);
  }
  nlohmann::json j = to_json();
  file << j.dump(2);  // Pretty print with 2-space indentation
  file.close();
}

void WavefunctionContainer::to_hdf5(H5::Group& group) const {
  QDK_LOG_TRACE_ENTERING();

  try {
    H5::StrType string_type(H5::PredType::C_S1, H5T_VARIABLE);

    // Add version attribute
    H5::Attribute version_attr = group.createAttribute(
        "version", string_type, H5::DataSpace(H5S_SCALAR));
    std::string version_str(SERIALIZATION_VERSION);
    version_attr.write(string_type, version_str);
    version_attr.close();

    // Store container type
    std::string container_type = get_container_type();
    H5::Attribute type_attr = group.createAttribute(
        "container_type", string_type, H5::DataSpace(H5S_SCALAR));
    type_attr.write(string_type, container_type);

    // Store wavefunction type
    std::string wf_type =
        (_type == WavefunctionType::SelfDual) ? "self_dual" : "not_self_dual";
    H5::Attribute wf_type_attr = group.createAttribute(
        "wavefunction_type", string_type, H5::DataSpace(H5S_SCALAR));
    wf_type_attr.write(string_type, wf_type);

    // Store restrictedness flag
    bool is_restricted = get_orbitals()->is_restricted();
    H5::Attribute restricted_attr = group.createAttribute(
        "is_restricted", H5::PredType::NATIVE_HBOOL, H5::DataSpace(H5S_SCALAR));
    hbool_t is_restricted_flag = is_restricted ? 1 : 0;
    restricted_attr.write(H5::PredType::NATIVE_HBOOL, &is_restricted_flag);

    // Store complexity flag for coefficients
    // Check if coefficients exist before accessing
    if (has_coefficients()) {
      bool is_complex = detail::is_vector_variant_complex(get_coefficients());
      H5::Attribute complex_attr = group.createAttribute(
          "is_complex", H5::PredType::NATIVE_HBOOL, H5::DataSpace(H5S_SCALAR));
      hbool_t is_complex_flag = is_complex ? 1 : 0;
      complex_attr.write(H5::PredType::NATIVE_HBOOL, &is_complex_flag);

      // Store coefficients
      if (is_complex) {
        const auto& coeffs_complex =
            std::get<Eigen::VectorXcd>(get_coefficients());
        hsize_t coeff_dims = coeffs_complex.size();
        H5::DataSpace coeff_space(1, &coeff_dims);

        // Use HDF5's native complex number support - no data copying
        // Create compound type for complex numbers (real, imag)
        H5::CompType complex_type(sizeof(std::complex<double>));
        complex_type.insertMember("r", 0, H5::PredType::NATIVE_DOUBLE);
        complex_type.insertMember("i", sizeof(double),
                                  H5::PredType::NATIVE_DOUBLE);

        H5::DataSet complex_dataset =
            group.createDataSet("coefficients", complex_type, coeff_space);
        // Write directly from Eigen's memory layout without copying
        complex_dataset.write(coeffs_complex.data(), complex_type);
      } else {
        const auto& coeffs_real = std::get<Eigen::VectorXd>(get_coefficients());
        hsize_t coeff_dims = coeffs_real.size();
        H5::DataSpace coeff_space(1, &coeff_dims);
        H5::DataSet coeff_dataset = group.createDataSet(
            "coefficients", H5::PredType::NATIVE_DOUBLE, coeff_space);
        // Write directly from Eigen's memory without copying
        coeff_dataset.write(coeffs_real.data(), H5::PredType::NATIVE_DOUBLE);
      }
    }

    if (has_configuration_set()) {
      // Store configuration set (delegates to ConfigurationSet serialization)
      H5::Group config_set_group = group.createGroup("configuration_set");
      get_configuration_set().to_hdf5(config_set_group);
    }

    // Serialize RDMs if available
    _serialize_rdms_to_hdf5(group);

  } catch (const H5::Exception& e) {
    throw std::runtime_error("HDF5 error: " + std::string(e.getCDetailMsg()));
  }
}

void Wavefunction::_to_hdf5_file(const std::string& filename) const {
  QDK_LOG_TRACE_ENTERING();
  try {
    H5::H5File file(filename, H5F_ACC_TRUNC);
    H5::Group wavefunction_group = file.createGroup("/wavefunction");
    to_hdf5(wavefunction_group);
  } catch (const H5::Exception& e) {
    throw std::runtime_error("HDF5 error: " + std::string(e.getCDetailMsg()));
  }
}

std::unique_ptr<WavefunctionContainer> WavefunctionContainer::from_hdf5(
    H5::Group& group) {
  QDK_LOG_TRACE_ENTERING();

  try {
    // Check version first
    H5::StrType string_type(H5::PredType::C_S1, H5T_VARIABLE);
    H5::Attribute version_attr = group.openAttribute("version");
    std::string version;
    version_attr.read(string_type, version);
    validate_serialization_version(SERIALIZATION_VERSION, version);

    // Load wavefunction type
    WavefunctionType type = WavefunctionType::SelfDual;
    if (group.attrExists("wavefunction_type")) {
      H5::Attribute wf_type_attr = group.openAttribute("wavefunction_type");
      std::string type_str;
      wf_type_attr.read(string_type, type_str);
      type = (type_str == "self_dual") ? WavefunctionType::SelfDual
                                       : WavefunctionType::NotSelfDual;
    }

    // Load container type
    if (!group.attrExists("container_type")) {
      throw std::runtime_error("HDF5 group missing 'container_type' attribute");
    }
    H5::Attribute type_attr = group.openAttribute("container_type");
    std::string container_type;
    type_attr.read(string_type, container_type);

    // Load restrictedness flag
    bool is_restricted = false;
    if (group.attrExists("is_restricted")) {
      H5::Attribute restrictedness_attr = group.openAttribute("is_restricted");
      hbool_t is_restricted_flag;
      restrictedness_attr.read(H5::PredType::NATIVE_HBOOL, &is_restricted_flag);
      is_restricted = (is_restricted_flag != 0);
    }

    // Load coefficients if they exist
    VectorVariant coefficients;
    if (group.nameExists("coefficients")) {
      // Load complexity flag
      bool is_complex = false;
      if (group.attrExists("is_complex")) {
        H5::Attribute complex_attr = group.openAttribute("is_complex");
        hbool_t is_complex_flag;
        complex_attr.read(H5::PredType::NATIVE_HBOOL, &is_complex_flag);
        is_complex = (is_complex_flag != 0);
      }

      H5::DataSet coeff_dataset = group.openDataSet("coefficients");
      H5::DataSpace coeff_space = coeff_dataset.getSpace();
      hsize_t coeff_size = coeff_space.getSimpleExtentNpoints();

      if (is_complex) {
        // Check if it's complex compound type
        H5::DataType dtype = coeff_dataset.getDataType();
        if (dtype.getClass() != H5T_COMPOUND) {
          throw std::runtime_error(
              "Expected complex compound type in HDF5 coefficients dataset");
        }

        // Native complex compound type
        H5::CompType complex_type(sizeof(std::complex<double>));
        complex_type.insertMember("r", 0, H5::PredType::NATIVE_DOUBLE);
        complex_type.insertMember("i", sizeof(double),
                                  H5::PredType::NATIVE_DOUBLE);

        Eigen::VectorXcd coeffs_complex(coeff_size);
        // Read directly into Eigen's memory without intermediate copying
        coeff_dataset.read(coeffs_complex.data(), complex_type);
        coefficients = coeffs_complex;
      } else {
        Eigen::VectorXd coeffs_real(coeff_size);
        // Read directly into Eigen's memory without copying
        coeff_dataset.read(coeffs_real.data(), H5::PredType::NATIVE_DOUBLE);
        coefficients = coeffs_real;
      }
    }

    // Load configuration set (delegates to ConfigurationSet deserialization)
    DeterminantVector determinants;
    std::shared_ptr<Orbitals> orbitals;
    if (group.nameExists("configuration_set")) {
      H5::Group config_set_group = group.openGroup("configuration_set");
      auto config_set = ConfigurationSet::from_hdf5(config_set_group);
      determinants = config_set.get_configurations();
      orbitals = config_set.get_orbitals();
    } else {
      determinants = {};
      orbitals = nullptr;
    }

    // Load RDMs if they are available
    if (group.nameExists("rdms")) {
      H5::Group rdm_group = group.openGroup("rdms");

      if (orbitals != nullptr) {
        // Determine if we're dealing with restricted or unrestricted orbitals
        is_restricted = orbitals->is_restricted();
      }

      // Deserialize RDMs using helper functions
      std::optional<MatrixVariant> one_rdm_aa;
      std::optional<MatrixVariant> one_rdm_bb;
      std::optional<VectorVariant> two_rdm_aabb;
      std::optional<VectorVariant> two_rdm_aaaa;
      std::optional<VectorVariant> two_rdm_bbbb;
      std::optional<MatrixVariant> one_rdm_spin_traced;
      std::optional<VectorVariant> two_rdm_spin_traced;

      if (is_restricted) {
        std::tie(one_rdm_aa, one_rdm_bb, two_rdm_aabb, two_rdm_aaaa,
                 two_rdm_bbbb, one_rdm_spin_traced, two_rdm_spin_traced) =
            _deserialize_rdms_from_hdf5_restricted(rdm_group, orbitals);
      } else {
        std::tie(one_rdm_aa, one_rdm_bb, two_rdm_aabb, two_rdm_aaaa,
                 two_rdm_bbbb, one_rdm_spin_traced, two_rdm_spin_traced) =
            _deserialize_rdms_from_hdf5_unrestricted(rdm_group, orbitals);
      }

      // Return appropriate container with RDMs
      bool has_one_rdm = one_rdm_aa.has_value();
      bool has_two_rdm = two_rdm_aabb.has_value() && two_rdm_aaaa.has_value();

      if (has_one_rdm && !has_two_rdm) {
        if (container_type == "cas") {
          return std::make_unique<CasWavefunctionContainer>(
              coefficients, determinants, orbitals, one_rdm_spin_traced,
              one_rdm_aa, one_rdm_bb, std::nullopt, std::nullopt, std::nullopt,
              std::nullopt, type);
        } else if (container_type == "sci") {
          return std::make_unique<SciWavefunctionContainer>(
              coefficients, determinants, orbitals, one_rdm_spin_traced,
              one_rdm_aa, one_rdm_bb, std::nullopt, std::nullopt, std::nullopt,
              std::nullopt, type);
        }
      } else if (!has_one_rdm && has_two_rdm) {
        if (container_type == "cas") {
          return std::make_unique<CasWavefunctionContainer>(
              coefficients, determinants, orbitals, std::nullopt, std::nullopt,
              std::nullopt, two_rdm_spin_traced, two_rdm_aabb, two_rdm_aaaa,
              two_rdm_bbbb, type);
        } else if (container_type == "sci") {
          return std::make_unique<SciWavefunctionContainer>(
              coefficients, determinants, orbitals, std::nullopt, std::nullopt,
              std::nullopt, two_rdm_spin_traced, two_rdm_aabb, two_rdm_aaaa,
              two_rdm_bbbb, type);
        }
      } else if (has_one_rdm && has_two_rdm) {
        if (container_type == "cas") {
          return std::make_unique<CasWavefunctionContainer>(
              coefficients, determinants, orbitals, one_rdm_spin_traced,
              one_rdm_aa, one_rdm_bb, two_rdm_spin_traced, two_rdm_aabb,
              two_rdm_aaaa, two_rdm_bbbb, type);
        } else if (container_type == "sci") {
          return std::make_unique<SciWavefunctionContainer>(
              coefficients, determinants, orbitals, one_rdm_spin_traced,
              one_rdm_aa, one_rdm_bb, two_rdm_spin_traced, two_rdm_aabb,
              two_rdm_aaaa, two_rdm_bbbb, type);
        }
      }
    }

    if (container_type == "cas") {
      return std::make_unique<CasWavefunctionContainer>(
          coefficients, determinants, orbitals, type);
    } else if (container_type == "sci") {
      return std::make_unique<SciWavefunctionContainer>(
          coefficients, determinants, orbitals, type);
    } else {
      throw std::runtime_error(
          "Did not expect to get here for containers other than cas/sci. "
          "Expected delegation to container-specific methods.");
    }
  } catch (const H5::Exception& e) {
    throw std::runtime_error("HDF5 error: " + std::string(e.getCDetailMsg()));
  }
}

std::shared_ptr<Wavefunction> Wavefunction::_from_json_file(
    const std::string& filename) {
  QDK_LOG_TRACE_ENTERING();
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
  QDK_LOG_TRACE_ENTERING();
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
  QDK_LOG_TRACE_ENTERING();
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
