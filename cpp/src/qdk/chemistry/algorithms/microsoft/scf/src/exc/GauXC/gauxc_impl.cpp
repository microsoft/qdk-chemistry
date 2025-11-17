// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <qdk/chemistry/scf/config.h>
#include <qdk/chemistry/scf/exc/gauxc_impl.h>
#include <qdk/chemistry/scf/util/gauxc_util.h>
#include <spdlog/spdlog.h>

#include <gauxc/molecular_weights.hpp>
#include <gauxc/molgrid/defaults.hpp>
#include <gauxc/xc_integrator/impl.hpp>
#include <gauxc/xc_integrator/integrator_factory.hpp>
#include <gauxc/xc_integrator_settings.hpp>
#include <map>

#include "util/timer.h"
#ifdef QDK_CHEMISTRY_ENABLE_GPU
#include <qdk/chemistry/scf/util/gpu/cuda_helper.h>
#endif

namespace qdk::chemistry::scf::impl {

// Map string to GauXC preset grid defaults
std::map<std::string, GauXC::AtomicGridSizeDefault> mg_map = {
    {"FINE", GauXC::AtomicGridSizeDefault::FineGrid},
    {"ULTRAFINE", GauXC::AtomicGridSizeDefault::UltraFineGrid},
    {"SUPERFINE", GauXC::AtomicGridSizeDefault::SuperFineGrid},
    {"GM3", GauXC::AtomicGridSizeDefault::GM3},
    {"GM5", GauXC::AtomicGridSizeDefault::GM5}};

// Map string to GauXC grid pruning scheme
std::map<std::string, GauXC::PruningScheme> prune_map = {
    {"UNPRUNED", GauXC::PruningScheme::Unpruned},
    {"ROBUST", GauXC::PruningScheme::Robust},
    {"TREUTLER", GauXC::PruningScheme::Treutler}};

// Map string to GauXC radial quadrature
std::map<std::string, GauXC::RadialQuad> rad_quad_map = {
    {"MURAKNOWLES", GauXC::RadialQuad::MuraKnowles},
    {"TREUTLERAHLRICHS", GauXC::RadialQuad::TreutlerAhlrichs},
    {"MURRAYHANDYLAMING", GauXC::RadialQuad::MurrayHandyLaming},
    {"MK", GauXC::RadialQuad::MuraKnowles},
    {"TA", GauXC::RadialQuad::TreutlerAhlrichs},
    {"MHL", GauXC::RadialQuad::MurrayHandyLaming}};

// Convert QDK-SCF functional name to GauXC functional name
ExchCXX::BidirectionalMap<std::string, std::string> gauxc_func_alias{
    {{"SLATER", "LDA"},
     {"PBEH", "PBE0"},
     {"MPWPW", "MPW91"},
     {"X3LYPG", "X3LYP"},
     {"B3LYPG", "B3LYP"},
     {"REVB3LYPG", "REVB3LYP"},
     {"B3PW91G", "B3PW91"},
     {"WPBE", "LCWPBE"},
     {"SVWN", "SVWN5"}}};

/**
 * @brief Convert QDK-SCF Molecule to GauXC Molecule
 * @param mol QDK Molecule
 * @returns GauXC Molecule
 */
GauXC::Molecule to_gauxc_molecule(const Molecule& mol) {
  GauXC::Molecule gauxc_molecule;

  for (size_t iatom = 0; iatom != mol.n_atoms; ++iatom) {
    auto atomic_number = mol.atomic_nums[iatom];
    auto x_coord = mol.coords[iatom][0];
    auto y_coord = mol.coords[iatom][1];
    auto z_coord = mol.coords[iatom][2];

    gauxc_molecule.emplace_back(GauXC::AtomicNumber(atomic_number), x_coord,
                                y_coord, z_coord);
  }
  return gauxc_molecule;
}

/**
 * @brief Convert QDK-SCF BasisSet to GauXC BasisSet
 * @tparam T Datatype underlying the GauXC storage (double, float, etc)
 * @param aimd_basisset QDK-SCF BasisSet
 * @returns GauXC BasisSet
 */
template <typename T>
GauXC::BasisSet<T> to_gauxc_basisset(const BasisSet& aimd_basisset) {
  using prim_array = typename GauXC::Shell<T>::prim_array;
  using cart_array = typename GauXC::Shell<T>::cart_array;

  int nshell = aimd_basisset.shells.size();
  GauXC::BasisSet<T> gauxc_basisset(aimd_basisset.shells.size());

  for (size_t ishell = 0; ishell != nshell; ++ishell) {
    auto aimd_shell = aimd_basisset.shells[ishell];
    prim_array exponents;
    prim_array coefficients;

    for (size_t iprim = 0; iprim != aimd_shell.contraction; ++iprim) {
      exponents.at(iprim) = aimd_shell.exponents[iprim];
      coefficients.at(iprim) = aimd_shell.coefficients[iprim];
    }
    cart_array center = aimd_shell.O;

    gauxc_basisset[ishell] =
        GauXC::Shell<T>(GauXC::PrimSize(aimd_shell.contraction),
                        GauXC::AngularMomentum(aimd_shell.angular_momentum),
                        aimd_shell.angular_momentum > 1
                            ? GauXC::SphericalType(aimd_basisset.pure)
                            : GauXC::SphericalType(false),
                        exponents, coefficients, center,
                        false  // Do not normalize shell via GauXC
        );
  }
  return gauxc_basisset;
}

GAUXC::GAUXC(BasisSet& basis_set, const GAUXCInput& gauxc_input,
             bool unrestricted, const std::string& xc_name) {
  // spdlog::trace("TOP GauXC::GauXC");

  // Unpack the input options directly from GAUXCInput
  const auto& grid_spec = gauxc_input.grid_spec;
  const auto& rad_quad_spec = gauxc_input.rad_quad_spec;
  const auto& prune_spec = gauxc_input.prune_spec;
  auto batch_size = gauxc_input.batch_size;
  auto basis_tol = gauxc_input.basis_tol;
  unrestricted_ = unrestricted;
  const auto& integrator_kernel = gauxc_input.integrator_kernel;
  const auto& lwd_kernel = gauxc_input.lwd_kernel;
  const auto& reduction_kernel = gauxc_input.reduction_kernel;
  const auto& integrator_ex_spec = gauxc_input.integrator_ex;
  const auto& loadbalancer_ex_spec = gauxc_input.loadbalancer_ex;
  const auto& weights_ex_spec = gauxc_input.weights_ex;

  spdlog::trace("GauXC Settings:");
  spdlog::trace(
      "  MolGrid={}, RadQuad={}, PruneSpec={}, BatchSz={}, BasisTol={}",
      gauxc_util::to_string(grid_spec), gauxc_util::to_string(rad_quad_spec),
      gauxc_util::to_string(prune_spec), batch_size, basis_tol);
  spdlog::trace("  IntExSpace={}, LBExSpace={}, MolWeightsExSpace={}",
                gauxc_util::to_string(integrator_ex_spec),
                gauxc_util::to_string(loadbalancer_ex_spec),
                gauxc_util::to_string(weights_ex_spec));

  // Create GauXC Runtime instance
#ifdef QDK_CHEMISTRY_ENABLE_GPU
  rt_ = std::make_shared<GauXC::DeviceRuntimeEnvironment>(
      GAUXC_MPI_CODE(MPI_COMM_WORLD, ) nullptr, 0);
#else
  rt_ = std::make_shared<GauXC::RuntimeEnvironment>(
      GAUXC_MPI_CODE(MPI_COMM_WORLD));
#endif

  // Convert internal Molecule to GauXC Molecule
  if (!basis_set.mol) {
    throw std::runtime_error(
        "BasisSet received in GauXC contains invalid Molecule");
  }
  auto mol = to_gauxc_molecule(*basis_set.mol);

  // Convert internal BasisSet to GauXC BasisSet
  auto basis = to_gauxc_basisset<double>(basis_set);
  for (auto& sh : basis) {
    sh.set_shell_tolerance(basis_tol);
  }

  // Setup the XC functional
  std::string processed_xc_name = xc_name;
  auto all_upper = [](auto& s) {
    std::transform(s.begin(), s.end(), s.begin(), ::toupper);
  };

  all_upper(processed_xc_name);
  // remove any "-" and "_" from the functional name
  processed_xc_name.erase(
      std::remove(processed_xc_name.begin(), processed_xc_name.end(), '-'),
      processed_xc_name.end());
  processed_xc_name.erase(
      std::remove(processed_xc_name.begin(), processed_xc_name.end(), '_'),
      processed_xc_name.end());

  if (gauxc_func_alias.key_exists(processed_xc_name)) {
    processed_xc_name = gauxc_func_alias.value(processed_xc_name);
  }

  if (!ExchCXX::functional_map.key_exists(processed_xc_name)) {
    throw std::runtime_error("GauXC does not support functional " +
                             processed_xc_name);
  }

  GauXC::functional_type func(
      ExchCXX::Backend::builtin,
      ExchCXX::functional_map.value(processed_xc_name),
      unrestricted_ ? ExchCXX::Spin::Polarized : ExchCXX::Spin::Unpolarized);
  // read hyb_coeff
  auto hyb_coeff = func.hyb_exx();
  x_alpha = hyb_coeff.alpha;
  x_beta = hyb_coeff.beta;
  x_omega = hyb_coeff.omega;

  // Create the MolGrid specification - directly use enum values
  auto mg = GauXC::MolGridFactory::create_default_molgrid(
      mol, prune_spec, GauXC::BatchSize(batch_size), rad_quad_spec, grid_spec);

  // Calculate GauXC Device buffer size
#ifdef QDK_CHEMISTRY_ENABLE_GPU
  {
    size_t available_mem, total_mem;
    CUDA_CHECK(cudaMemGetInfo(&available_mem, &total_mem));
    device_buffer_sz_ = 0.5 * available_mem;
    spdlog::trace("  DeviceBufferSz={}", device_buffer_sz_);
  }
#endif

  // Allocate a large temporary buffer
#ifdef QDK_CHEMISTRY_ENABLE_GPU
  allocate_device_buffer_async_(device_buffer_sz_, 0);
#endif

  // Create the LoadBalancer
  GauXC::LoadBalancerFactory lb_factory(loadbalancer_ex_spec, "Replicated");
  auto lb = lb_factory.get_shared_instance(*rt_, mol, mg, basis);

  // Apply Partition Weights
  GauXC::MolecularWeightsFactory mw_factory(weights_ex_spec, "Default",
                                            GauXC::MolecularWeightsSettings{});
  mw_factory.get_instance().modify_weights(*lb);
#ifdef QDK_CHEMISTRY_ENABLE_GPU
  cudaDeviceSynchronize();
#endif

  // Free Device Buffer
#ifdef QDK_CHEMISTRY_ENABLE_GPU
  free_device_buffer_async_(0);
#endif

  // Setup Integrator
  using matrix_type = Eigen::MatrixXd;
  GauXC::XCIntegratorFactory<matrix_type> integrator_factory(
      integrator_ex_spec, "Replicated", integrator_kernel, lwd_kernel,
      reduction_kernel);
  integrator_ = integrator_factory.get_shared_instance(func, lb);
}

void GAUXC::eval_dd_psi(int lmax, const double* D, double* dd_psi) {
  auto num_basis_funcs = integrator_->load_balancer().basis().nbf();
  auto natom = integrator_->load_balancer().molecule().size();
  auto nharmonics = (lmax + 1) * (lmax + 1);
  Eigen::MatrixXd D_eigen =
      Eigen::Map<const Eigen::MatrixXd>(D, num_basis_funcs, num_basis_funcs);
  auto dd_psi_vec = integrator_->eval_dd_psi(D_eigen, lmax);
  std::copy(dd_psi_vec.begin(), dd_psi_vec.end(), dd_psi);
}

void GAUXC::eval_dd_psi_potential(int lmax, const double* x,
                                  double* dd_psi_potential) {
  auto num_basis_funcs = integrator_->load_balancer().basis().nbf();
  auto natom = integrator_->load_balancer().molecule().size();
  auto nharmonics = (lmax + 1) * (lmax + 1);
  Eigen::Map<const Eigen::MatrixXd> x_eigen(x, nharmonics, natom);
  Eigen::Map<Eigen::MatrixXd> dd_psi_potential_eigen(
      dd_psi_potential, num_basis_funcs, num_basis_funcs);
  dd_psi_potential_eigen = integrator_->eval_dd_psi_potential(x_eigen, lmax);
}

GAUXC::~GAUXC() noexcept = default;

#ifdef QDK_CHEMISTRY_ENABLE_GPU
void GAUXC::allocate_device_buffer_async_(size_t sz, cudaStream_t stream) {
  auto dev_rt_ =
      std::dynamic_pointer_cast<GauXC::DeviceRuntimeEnvironment>(rt_);
  void* p;
  CUDA_CHECK(cudaMallocAsync(&p, sz, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  dev_rt_->set_buffer(p, sz);
}

void GAUXC::free_device_buffer_async_(cudaStream_t stream) {
  auto dev_rt_ =
      std::dynamic_pointer_cast<GauXC::DeviceRuntimeEnvironment>(rt_);
  CUDA_CHECK(cudaFreeAsync(dev_rt_->device_memory(), stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
}
#endif

void GAUXC::build_XC(const double* D, double* XC, double* xc_energy) {
  // Allocate a large temporary buffer
#ifdef QDK_CHEMISTRY_ENABLE_GPU
  allocate_device_buffer_async_(device_buffer_sz_, 0);
#endif

  auto num_basis_funcs = integrator_->load_balancer().basis().nbf();

  if (not unrestricted_) {
    // GauXC expects P[alpha] not P[total] for RKS
    Eigen::MatrixXd D_eigen = 0.5 * Eigen::Map<const Eigen::MatrixXd>(
                                        D, num_basis_funcs, num_basis_funcs);
    Eigen::Map<Eigen::MatrixXd> VXC(XC, num_basis_funcs, num_basis_funcs);

    std::tie(*xc_energy, VXC) = integrator_->eval_exc_vxc(D_eigen);
  } else {
    Eigen::Map<const Eigen::MatrixXd> D_alpha(D, num_basis_funcs,
                                              num_basis_funcs);
    Eigen::Map<const Eigen::MatrixXd> D_beta(
        D + num_basis_funcs * num_basis_funcs, num_basis_funcs,
        num_basis_funcs);
    Eigen::Map<Eigen::MatrixXd> VXC_scalar(XC, num_basis_funcs,
                                           num_basis_funcs);
    Eigen::Map<Eigen::MatrixXd> VXC_z(XC + num_basis_funcs * num_basis_funcs,
                                      num_basis_funcs, num_basis_funcs);

    // GauXC expects Pauli basis for UKS
    Eigen::MatrixXd D_scalar = D_alpha + D_beta;
    Eigen::MatrixXd D_z = D_alpha - D_beta;

    std::tie(*xc_energy, VXC_scalar, VXC_z) =
        integrator_->eval_exc_vxc(D_scalar, D_z);

    for (size_t i = 0; i < num_basis_funcs * num_basis_funcs; ++i) {
      const auto v_scalar = XC[i];
      const auto v_z = XC[i + num_basis_funcs * num_basis_funcs];

      const auto v_alpha = (v_scalar + v_z);
      const auto v_beta = (v_scalar - v_z);

      XC[i] = v_alpha;
      XC[i + num_basis_funcs * num_basis_funcs] = v_beta;
    }
  }

  // Free Device Buffer
#ifdef QDK_CHEMISTRY_ENABLE_GPU
  free_device_buffer_async_(0);
#endif
}

void GAUXC::get_gradients(const double* D, double* dXC) {
  // Allocate a large temporary buffer
#ifdef QDK_CHEMISTRY_ENABLE_GPU
  allocate_device_buffer_async_(device_buffer_sz_, 0);
#endif

  GauXC::IntegratorSettingsEXC_GRAD settings;
  settings.include_weight_derivatives = false;
  auto num_basis_funcs = integrator_->load_balancer().basis().nbf();
  auto nat = integrator_->load_balancer().molecule().size();
  std::vector<double> exc_grad;

  if (not unrestricted_) {
    // GauXC expects P[alpha] not P[total] for RKS
    Eigen::MatrixXd D_eigen = 0.5 * Eigen::Map<const Eigen::MatrixXd>(
                                        D, num_basis_funcs, num_basis_funcs);
    exc_grad = integrator_->eval_exc_grad(D_eigen, settings);
  } else {
    Eigen::Map<const Eigen::MatrixXd> D_alpha(D, num_basis_funcs,
                                              num_basis_funcs);
    Eigen::Map<const Eigen::MatrixXd> D_beta(
        D + num_basis_funcs * num_basis_funcs, num_basis_funcs,
        num_basis_funcs);

    // GauXC expects Pauli basis for UKS
    Eigen::MatrixXd D_scalar = D_alpha + D_beta;
    Eigen::MatrixXd D_z = D_alpha - D_beta;

    exc_grad = integrator_->eval_exc_grad(D_scalar, D_z, settings);
  }

  // Gradients are expected as a (3,NATOM) *ROW-MAJOR* matrix on return
  for (auto i = 0; i < nat; ++i)
    for (auto w = 0; w < 3; ++w) {
      dXC[w * nat + i] = exc_grad[3 * i + w];
    }

  // Free Device Buffer
#ifdef QDK_CHEMISTRY_ENABLE_GPU
  free_device_buffer_async_(0);
#endif
}

void GAUXC::build_snK(const double* D, double* K) {
#ifdef QDK_CHEMISTRY_ENABLE_GPU
  allocate_device_buffer_async_(device_buffer_sz_, 0);
#endif

  auto num_basis_funcs = integrator_->load_balancer().basis().nbf();
  Eigen::MatrixXd D_eigen =
      Eigen::Map<const Eigen::MatrixXd>(D, num_basis_funcs, num_basis_funcs);
  Eigen::Map<Eigen::MatrixXd> K_eigen(K, num_basis_funcs, num_basis_funcs);

  K_eigen = integrator_->eval_exx(D_eigen);

  if (unrestricted_) {
    Eigen::Map<Eigen::MatrixXd> K_beta_eigen(
        K + num_basis_funcs * num_basis_funcs, num_basis_funcs,
        num_basis_funcs);
    D_eigen =
        Eigen::Map<const Eigen::MatrixXd>(D + num_basis_funcs * num_basis_funcs,
                                          num_basis_funcs, num_basis_funcs);
    K_beta_eigen = integrator_->eval_exx(D_eigen);
  }

#ifdef QDK_CHEMISTRY_ENABLE_GPU
  free_device_buffer_async_(0);
#endif
}

void GAUXC::eval_fxc_contraction(const double* D, const double* tD,
                                 double* Fxc) {
  AutoTimer __timer("polarizability::  GAUXC::eval_fxc_contraction");
  // Allocate a large temporary buffer
#ifdef QDK_CHEMISTRY_ENABLE_GPU
  allocate_device_buffer_async_(device_buffer_sz_, 0);
#endif

  auto num_basis_funcs = integrator_->load_balancer().basis().nbf();

  if (not unrestricted_) {
    // GauXC expects P[alpha] not P[total] for RKS
    Eigen::MatrixXd D_eigen = 0.5 * Eigen::Map<const Eigen::MatrixXd>(
                                        D, num_basis_funcs, num_basis_funcs);
    Eigen::MatrixXd tD_eigen =
        Eigen::Map<const Eigen::MatrixXd>(tD, num_basis_funcs, num_basis_funcs);
    Eigen::Map<Eigen::MatrixXd> FXC(Fxc, num_basis_funcs, num_basis_funcs);

    FXC = integrator_->eval_fxc_contraction(D_eigen, tD_eigen);

  } else {
    Eigen::Map<const Eigen::MatrixXd> D_alpha(D, num_basis_funcs,
                                              num_basis_funcs);
    Eigen::Map<const Eigen::MatrixXd> D_beta(
        D + num_basis_funcs * num_basis_funcs, num_basis_funcs,
        num_basis_funcs);
    Eigen::Map<const Eigen::MatrixXd> tD_alpha(tD, num_basis_funcs,
                                               num_basis_funcs);
    Eigen::Map<const Eigen::MatrixXd> tD_beta(
        tD + num_basis_funcs * num_basis_funcs, num_basis_funcs,
        num_basis_funcs);
    Eigen::Map<Eigen::MatrixXd> FXCa(Fxc, num_basis_funcs, num_basis_funcs);
    Eigen::Map<Eigen::MatrixXd> FXCb(Fxc + num_basis_funcs * num_basis_funcs,
                                     num_basis_funcs, num_basis_funcs);

    // GauXC expects Pauli basis for UKS
    Eigen::MatrixXd D_scalar = D_alpha + D_beta;
    Eigen::MatrixXd D_z = D_alpha - D_beta;
    Eigen::MatrixXd tD_scalar = tD_alpha + tD_beta;
    Eigen::MatrixXd tD_z = tD_alpha - tD_beta;

    // Get FXCs and FXCz from integrator
    std::tie(FXCa, FXCb) =
        integrator_->eval_fxc_contraction(D_scalar, D_z, tD_scalar, tD_z);

    // Convert from scalar/z to alpha/beta
    for (size_t i = 0; i < num_basis_funcs * num_basis_funcs; ++i) {
      const auto fxc_scalar = Fxc[i];
      const auto fxc_z = Fxc[i + num_basis_funcs * num_basis_funcs];
      Fxc[i] = fxc_scalar + fxc_z;
      Fxc[i + num_basis_funcs * num_basis_funcs] = fxc_scalar - fxc_z;
    }
  }

  // Free Device Buffer
#ifdef QDK_CHEMISTRY_ENABLE_GPU
  free_device_buffer_async_(0);
#endif
}
}  // namespace qdk::chemistry::scf::impl
