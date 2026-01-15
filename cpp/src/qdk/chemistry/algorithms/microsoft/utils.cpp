// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include "utils.hpp"

#include <qdk/chemistry/scf/config.h>
#ifdef QDK_CHEMISTRY_ENABLE_MPI
#include <mpi.h>
#endif
#ifdef QDK_CHEMISTRY_ENABLE_GPU
#include <qdk/chemistry/scf/util/gpu/cuda_helper.h>
#endif

#include <libint2.hpp>
#include <qdk/chemistry/utils/logger.hpp>

namespace qdk::chemistry::utils::microsoft {

using qdk::chemistry::data::MAX_ORBITAL_ANGULAR_MOMENTUM;

namespace qcs = qdk::chemistry::scf;

void _norm_psi4_mode(std::vector<qcs::Shell>& shells) {
  QDK_LOG_TRACE_ENTERING();

  // Precompute the double factorial up to n=2*MAX_ORBITAL_ANGULAR_MOMENTUM
  std::vector<double> double_factorial(2 * MAX_ORBITAL_ANGULAR_MOMENTUM + 1, 1);

  // Start at i = 3 since 0!! = 1 and 1!! = 1 and 2!! = 2 (covered by the first
  // iteration)
  for (int i = 3; i < double_factorial.size(); i++) {
    double_factorial[i] = double_factorial[i - 2] * (i - 1);
  }

  const double sqrt_PI_cubed = std::sqrt(std::pow(std::acos(-1.0), 3.0));

  for (auto& shell : shells) {
    int am = shell.angular_momentum;

    // Check if angular momentum is within the supported range
    if (am > MAX_ORBITAL_ANGULAR_MOMENTUM) {
      throw std::runtime_error(
          "Shell angular momentum exceeds MAX_ORBITAL_ANGULAR_MOMENTUM");
    }

    // Check for zero exponents ahead of time
    for (size_t i = 0; i < shell.contraction; i++) {
      if (shell.exponents[i] <= 0) {
        throw std::runtime_error(
            "Shell exponents must be positive (found a zero or negative "
            "value)");
      }
    }

    for (size_t i = 0; i < shell.contraction; i++) {
      const double exp2 = 2 * shell.exponents[i];
      const double exp2_to_am32 = std::pow(exp2, am + 1) * std::sqrt(exp2);
      const double normalization_factor =
          std::sqrt(std::pow(2, am) * exp2_to_am32 /
                    (sqrt_PI_cubed * double_factorial[2 * am]));

      shell.coefficients[i] *= normalization_factor;
    }
    double norm = 0.0;
    for (size_t i = 0; i < shell.contraction; i++) {
      for (size_t j = 0; j <= i; j++) {
        auto gamma = shell.exponents[i] + shell.exponents[j];
        norm += (i == j ? 1 : 2) * double_factorial[2 * am] * sqrt_PI_cubed *
                shell.coefficients[i] * shell.coefficients[j] /
                (pow(2, am) * pow(gamma, am + 1) * sqrt(gamma));
      }
    }
    double normalization_factor = 1 / sqrt(norm);
    for (size_t i = 0; i < shell.contraction; i++) {
      shell.coefficients[i] *= normalization_factor;
    }
  }
}
static int _backed_initialized = false;
static int _qdk_initialized_mpi = false;
static int _qdk_initialized_libint2 = false;

void initialize_backend() {
  QDK_LOG_TRACE_ENTERING();

  if (_backed_initialized) return;

#ifdef QDK_CHEMISTRY_ENABLE_MPI
  int initialized;
  MPI_Initialized(&initialized);
  if (!initialized) {
    _qdk_initialized_mpi = true;
    int req = MPI_THREAD_SERIALIZED, prov;
    MPI_Init_thread(nullptr, nullptr, req, &prov);
    if (prov != req) {
      throw std::runtime_error("MPI does not support THREAD_SERIALIZED");
    }
  }

  int world_size;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  if (world_size > 1) {
    throw std::runtime_error("QDK Does Not Currently Support MPI");
  }
#endif

  if (not libint2::initialized()) {
    _qdk_initialized_libint2 = true;
    libint2::initialize();
  }

#ifdef QDK_CHEMISTRY_ENABLE_GPU
  CUDA_CHECK(cudaSetDevice(0));
  qdk::chemistry::scf::cuda::init_memory_pool(0);
#endif /* QDK_CHEMISTRY_ENABLE_GPU */

  _backed_initialized = true;
}

void finalize_backend() {
  QDK_LOG_TRACE_ENTERING();

  if (!_backed_initialized) return;
  if (_qdk_initialized_libint2) libint2::finalize();
#ifdef QDK_CHEMISTRY_ENABLE_MPI
  if (_qdk_initialized_mpi) MPI_Finalize();
#endif
}

qdk::chemistry::data::Structure convert_to_structure(
    const qcs::Molecule& molecule) {
  QDK_LOG_TRACE_ENTERING();

  // Convert the Molecule to a Structure
  Eigen::MatrixXd coordinates(molecule.n_atoms, 3);
  std::vector<qdk::chemistry::data::Element> elements(molecule.n_atoms);

  for (uint64_t i = 0; i < molecule.n_atoms; ++i) {
    // Structure expects Bohr, Molecule is in Bohr
    coordinates(i, 0) = molecule.coords[i][0];
    coordinates(i, 1) = molecule.coords[i][1];
    coordinates(i, 2) = molecule.coords[i][2];
    elements[i] = qdk::chemistry::data::Structure::nuclear_charge_to_element(
        molecule.atomic_nums[i]);
  }

  return qdk::chemistry::data::Structure(coordinates, elements);
}

std::shared_ptr<qcs::Molecule> convert_to_molecule(
    const qdk::chemistry::data::Structure& structure, int64_t charge,
    int64_t multiplicity) {
  QDK_LOG_TRACE_ENTERING();

  // Convert the Structure to a Molecule
  const auto& coordinates = structure.get_coordinates();
  const auto& nuclear_charges = structure.get_nuclear_charges();

  auto molecule_ptr = std::make_shared<qcs::Molecule>();
  auto& molecule = *molecule_ptr;
  molecule.n_atoms = static_cast<uint64_t>(coordinates.rows());
  molecule.total_nuclear_charge =
      std::accumulate(nuclear_charges.begin(), nuclear_charges.end(), 0u);
  molecule.charge = charge;
  molecule.multiplicity = multiplicity;
  molecule.n_electrons = molecule.total_nuclear_charge - molecule.charge;
  molecule.atomic_nums.resize(structure.get_num_atoms());
  molecule.atomic_charges.resize(structure.get_num_atoms());
  molecule.coords.resize(structure.get_num_atoms());

  for (uint64_t i = 0; i < molecule.n_atoms; ++i) {
    molecule.atomic_nums[i] = nuclear_charges[i];
    molecule.atomic_charges[i] = nuclear_charges[i];
    molecule.coords[i] = {coordinates(i, 0), coordinates(i, 1),
                          coordinates(i, 2)};
  }

  return molecule_ptr;
}

qdk::chemistry::data::BasisSet convert_basis_set_to_qdk(
    const qcs::BasisSet& basis_set) {
  QDK_LOG_TRACE_ENTERING();

  // Check if we're encountering edge cases
  if (not basis_set.pure) {
    // TODO (NAB):  is basis_set.pure always false for cartesian basis sets?
    // Does it mean anything else? Work item: 41332
    throw std::runtime_error("QDK Does Not Support Cartesian Atomic Orbitals");
  }

  // Convert the BasisSet to a qdk::chemistry::data::BasisSet
  auto structure = convert_to_structure(*basis_set.mol);

  // Collect shells
  std::vector<qdk::chemistry::data::Shell> qdk_shells;
  for (const auto& shell : basis_set.shells) {
    Eigen::VectorXd exponents(shell.contraction);
    Eigen::VectorXd coefficients(shell.contraction);
    std::memcpy(exponents.data(), shell.exponents,
                exponents.size() * sizeof(double));
    std::memcpy(coefficients.data(), shell.coefficients,
                coefficients.size() * sizeof(double));

    qdk_shells.emplace_back(
        shell.atom_index,
        static_cast<qdk::chemistry::data::OrbitalType>(shell.angular_momentum),
        exponents, coefficients);
  }

  // Collect ECP shells
  std::vector<qdk::chemistry::data::Shell> qdk_ecp_shells;
  for (const auto& ecp_shell : basis_set.ecp_shells) {
    Eigen::VectorXd exponents(ecp_shell.contraction);
    Eigen::VectorXd coefficients(ecp_shell.contraction);
    Eigen::VectorXi rpowers(ecp_shell.contraction);

    std::memcpy(exponents.data(), ecp_shell.exponents,
                exponents.size() * sizeof(double));
    std::memcpy(coefficients.data(), ecp_shell.coefficients,
                coefficients.size() * sizeof(double));
    std::memcpy(rpowers.data(), ecp_shell.rpowers,
                rpowers.size() * sizeof(int));

    qdk_ecp_shells.emplace_back(ecp_shell.atom_index,
                                static_cast<qdk::chemistry::data::OrbitalType>(
                                    ecp_shell.angular_momentum),
                                exponents, coefficients, rpowers);
  }

  // Handle ECP (Effective Core Potential) information if present
  if (basis_set.n_ecp_electrons != 0 || !basis_set.ecp_shells.empty() ||
      !basis_set.element_ecp_electrons.empty()) {
    // Use basis set name as ECP name
    std::string qdk_ecp_name = basis_set.name;

    // Build ECP electrons per atom vector
    std::vector<size_t> qdk_ecp_electrons(basis_set.mol->n_atoms, 0);
    for (size_t i = 0; i < basis_set.mol->n_atoms; ++i) {
      int atomic_num = basis_set.mol->atomic_nums[i];
      auto it = basis_set.element_ecp_electrons.find(atomic_num);
      if (it != basis_set.element_ecp_electrons.end()) {
        qdk_ecp_electrons[i] = static_cast<size_t>(it->second);
      }
    }

    // Create the BasisSet with shells, ECP shells, ECP name, ECP electrons, and
    // structure
    qdk::chemistry::data::BasisSet qdk_basis_set(basis_set.name, qdk_shells,
                                                 qdk_ecp_name, qdk_ecp_shells,
                                                 qdk_ecp_electrons, structure);
    return qdk_basis_set;
  } else {
    // Create the BasisSet with shells, ECP shells, and structure (no ECP
    // name/electrons)
    qdk::chemistry::data::BasisSet qdk_basis_set(basis_set.name, qdk_shells,
                                                 qdk_ecp_shells, structure);
    return qdk_basis_set;
  }
}

std::shared_ptr<qcs::BasisSet> convert_basis_set_from_qdk(
    const qdk::chemistry::data::BasisSet& qdk_basis_set, bool normalize) {
  QDK_LOG_TRACE_ENTERING();
  // Create internal Molecule from the structure
  auto structure = qdk_basis_set.get_structure();
  auto mol = convert_to_molecule(*structure, 0,
                                 1);  // Default charge=0, multiplicity=1

  // remove number of ecp electrons from atomic charges
  auto ecp_electrons = qdk_basis_set.get_ecp_electrons();
  for (size_t i = 0; i < mol->n_atoms; ++i) {
    int n_core_electrons = static_cast<int>(ecp_electrons[i]);
    mol->atomic_charges[i] = mol->atomic_nums[i] - n_core_electrons;
  }

  auto basis_json = convert_to_json(qdk_basis_set);
  auto internal_basis_set =
      qcs::BasisSet::from_serialized_json(mol, basis_json);

  if (internal_basis_set->mode == qcs::BasisMode::RAW && normalize) {
    _norm_psi4_mode(internal_basis_set->shells);
    internal_basis_set->mode = qcs::BasisMode::PSI4;
  }

  return internal_basis_set;
}

nlohmann::ordered_json convert_to_json(
    const qdk::chemistry::data::Shell& shell) {
  QDK_LOG_TRACE_ENTERING();

  std::vector<double> exponents;
  std::vector<double> coefficients;
  size_t contraction = shell.get_num_primitives();
  exponents.resize(contraction);
  coefficients.resize(contraction);
  std::memcpy(exponents.data(), shell.exponents.data(),
              contraction * sizeof(double));
  std::memcpy(coefficients.data(), shell.coefficients.data(),
              contraction * sizeof(double));

  nlohmann::ordered_json record = {
      {"atom", shell.atom_index},
      {"am", static_cast<unsigned>(shell.orbital_type)},
      {"exp", exponents},
      {"coeff", coefficients}};

  // Add rpowers for ECP shells
  if (shell.rpowers.size() > 0) {
    std::vector<int> rpowers(contraction);
    std::memcpy(rpowers.data(), shell.rpowers.data(),
                contraction * sizeof(int));
    record["rpowers"] = rpowers;
  }

  return record;
}

nlohmann::ordered_json convert_to_json(
    const qdk::chemistry::data::BasisSet& basis_set) {
  QDK_LOG_TRACE_ENTERING();

  nlohmann::ordered_json j;

  std::vector<nlohmann::ordered_json> json_shells;
  for (const auto& sh : basis_set.get_shells()) {
    json_shells.push_back(convert_to_json(sh));
  }

  // Handle ECP
  std::vector<nlohmann::ordered_json> json_ecp_shells;
  if (basis_set.has_ecp_shells()) {
    for (const auto& ecp_shell : basis_set.get_ecp_shells()) {
      json_ecp_shells.push_back(convert_to_json(ecp_shell));
    }
  }

  // Build element_ecp_electrons map from ecp_electrons vector
  auto& structure = basis_set.get_structure();
  auto nuclear_charges = structure->get_nuclear_charges();
  auto ecp_electrons = basis_set.get_ecp_electrons();

  std::map<int, int> element_ecp_electrons;
  for (size_t i = 0; i < ecp_electrons.size(); ++i) {
    if (ecp_electrons[i] > 0) {
      int atomic_num = static_cast<int>(nuclear_charges[i]);
      element_ecp_electrons[atomic_num] = static_cast<int>(ecp_electrons[i]);
    }
  }

  // Serialize element_ecp_electrons as flat list
  std::vector<int> json_element_ecp_electrons;
  for (const auto& [k, v] : element_ecp_electrons) {
    json_element_ecp_electrons.push_back(k);
    json_element_ecp_electrons.push_back(v);
  }

  std::vector<unsigned> nuclear_charges_unsigned(nuclear_charges.size());
  std::transform(nuclear_charges.begin(), nuclear_charges.end(),
                 nuclear_charges_unsigned.begin(),
                 [](double z) { return static_cast<unsigned>(z); });

  j = nlohmann::ordered_json(
      {{"name", basis_set.get_name()},
       {"pure", true},
       {"mode", "RAW"},
       {"atoms", nuclear_charges_unsigned},
       {"num_atomic_orbitals", basis_set.get_num_atomic_orbitals()},
       {"electron_shells", json_shells},
       {"ecp_shells", json_ecp_shells},
       {"element_ecp_electrons", json_element_ecp_electrons}});

  return j;
}

std::vector<unsigned> compute_shell_map(
    const qdk::chemistry::data::BasisSet& qdk_basis_set,
    const qcs::BasisSet& itrn_basis_set) {
  QDK_LOG_TRACE_ENTERING();

  const size_t nshells = qdk_basis_set.get_num_shells();

  if (nshells != itrn_basis_set.shells.size()) {
    throw std::runtime_error(
        "QDK Basis size is inconsistent with internal representation");
  }
  std::vector<int> qdk_to_internal_shells(nshells, -1);

  const double shell_data_compare_tol = 1e-10;
  auto qdk_shells = qdk_basis_set.get_shells();
  const auto& itrn_shells = itrn_basis_set.shells;
  for (size_t i = 0; i < nshells; ++i) {
    const auto& qdk_shell = qdk_shells[i];
    const auto nprim = qdk_shell.exponents.size();
    const auto l = static_cast<unsigned>(qdk_shell.orbital_type);
    for (size_t j = 0; j < nshells; ++j) {
      const auto& itrn_shell = itrn_shells[j];
      if (qdk_shell.atom_index != itrn_shell.atom_index) continue;
      if (l != itrn_shell.angular_momentum) continue;
      if (nprim != itrn_shell.contraction) continue;
      bool exp_equiv = true;
      for (size_t k = 0; k < nprim; ++k) {
        exp_equiv &= std::abs(qdk_shell.exponents[k] -
                              itrn_shell.exponents[k]) < shell_data_compare_tol;
      }
      if (!exp_equiv) continue;

      bool coeff_equiv = true;
      for (size_t k = 0; k < nprim; ++k) {
        coeff_equiv &=
            std::abs(qdk_shell.coefficients[k] - itrn_shell.coefficients[k]) <
            shell_data_compare_tol;
      }
      if (!coeff_equiv) continue;

      // The shells are equivalent if I reach here
      qdk_to_internal_shells[i] = j;
      break;
    }
  }

  if (std::any_of(qdk_to_internal_shells.begin(), qdk_to_internal_shells.end(),
                  [](auto i) { return i == -1; }))
    throw std::runtime_error(
        "Could Not Determine a 1:1 Mapping Between QDK and Internal Shell "
        "Representations");

  std::vector<unsigned> ret_val(nshells);
  std::transform(qdk_to_internal_shells.begin(), qdk_to_internal_shells.end(),
                 ret_val.begin(), [](auto i) { return i; });
  return ret_val;
}

size_t factorial(size_t n) {
  size_t result = 1;
  for (size_t i = 2; i <= n; ++i) {
    result *= i;
  }
  return result;
}

size_t binomial_coefficient(size_t n, size_t k) {
  if (k > n) return 0ul;
  if (k == 0 || k == n) return 1ul;
  // Take advantage of symmetry: C(n, k) == C(n, n - k)
  if (k > n - k) {
    k = n - k;
  }
  size_t result = 1;
  for (size_t i = 1; i <= k; ++i) {
    // At each step: result *= (n - (k - i)); result /= i;
    // This keeps intermediate values smaller than full factorials.
    result = (result * (n - (k - i))) / i;
  }
  return result;
}

}  // namespace qdk::chemistry::utils::microsoft
