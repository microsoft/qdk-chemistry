// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <qdk/chemistry/scf/config.h>

#include <nlohmann/json.hpp>
#ifdef QDK_CHEMISTRY_ENABLE_GPU
#include <cuda_runtime.h>
#include <qdk/chemistry/scf/util/gpu/cuda_helper.h>
#endif
#include <qdk/chemistry/scf/core/eri.h>
#include <qdk/chemistry/scf/core/exc.h>
#include <qdk/chemistry/scf/core/molecule.h>
#include <qdk/chemistry/scf/scf/scf_solver.h>
#include <qdk/chemistry/scf/util/gauxc_registry.h>

#include <filesystem>
#include <fstream>
#include <qdk/chemistry/utils/logger.hpp>
#include <vector>

#include "test_common.h"
#include "test_config.h"

using ::testing::FloatNear;
using ::testing::Pointwise;
using namespace qdk::chemistry::scf;

class SCFTest
    : public ::testing::TestWithParam<std::tuple<
          std::string /*file*/, std::string /*test_case*/, std::string /*eri*/,
          std::string /*k_eri*/, std::string /*exc*/, double /*energy_tol*/,
          double /*grad_tol*/>> {
 protected:
  void SetUp() override {
    auto [file_name, test_case, eri, k_eri, exc, energy_tolerance,
          grad_tolerance] = GetParam();
    auto path = std::filesystem::path(TEST_DATA_DIR) / (file_name + ".json");
    auto json = nlohmann::json::parse(std::ifstream(path))[test_case];

    if (eri == "DEFAULT") {
#ifdef QDK_CHEMISTRY_ENABLE_HGP
      eri = "HGP";
#else
      eri = "CPU";
#endif
    }
    if (k_eri == "DEFAULT") {
#ifdef QDK_CHEMISTRY_ENABLE_HGP
      k_eri = "HGP";
#else
      k_eri = "CPU";
#endif
    }

    mol = make_molecule(json["mol"].get<std::string>());
#ifdef QDK_CHEMISTRY_ENABLE_QMMM
    if (json.contains("mm_charges")) {
      std::vector<double> mm_charges =
          json.value("mm_charges", std::vector<double>{});
      std::vector<double> mm_geometry =
          json.value("mm_geometry", std::vector<double>{});
      cfg.pointcharges = std::make_unique<PointCharges>();
      auto& pointcharges = cfg.pointcharges;
      pointcharges->n_points = mm_charges.size();
      pointcharges->charges = mm_charges;
      pointcharges->coords.resize(pointcharges->n_points);
      for (int i = 0; i < pointcharges->n_points; i++) {
        pointcharges->coords[i] = {mm_geometry[i * 3], mm_geometry[i * 3 + 1],
                                   mm_geometry[i * 3 + 2]};
      }
    }
#endif

#ifdef QDK_CHEMISTRY_ENABLE_PCM
    cfg.enable_pcm = json.value("pcm", false);
    cfg.use_ddx = json.value("ddx", false);
    if (cfg.use_ddx) {
      cfg.enable_pcm = true;
      auto& ddx = cfg.ddx_input;
      ddx.solvent = json.value("ddx_solvent", ddx.solvent);
      ddx.lmax = json.value("ddx_lmax", ddx.lmax);
      ddx.incore = json.value("ddx_incore", ddx.incore);
      ddx.radius_type = json.value("ddx_radius_type", ddx.radius_type);
      ddx.model = json.value("ddx_model", ddx.model);
      if (ddx.model == 1) {
        ddx.shift = -1.0;
      }
    } else if (cfg.enable_pcm) {
      cfg.pcm_input = pcm_default_input();
      std::string solverTypeValue =
          json.value("pcm_solver", std::string(cfg.pcm_input.solver_type));
      std::string solventValue =
          json.value("solvent", std::string(cfg.pcm_input.solvent));
      strncpy(cfg.pcm_input.solver_type, solverTypeValue.c_str(), 7);
      strncpy(cfg.pcm_input.solvent, solventValue.c_str(), 7);
    }
#endif

    mol->n_electrons -= json.value("charge", 0);
    mol->multiplicity = json.value("multiplicity", 1);

    scf_type = json["method"];
    std::transform(scf_type.begin(), scf_type.end(), scf_type.begin(),
                   ::tolower);
    ref_energy = json["energy"];
    ref_grads = json.value("gradient", std::vector<double>{});
#if defined(QDK_CHEMISTRY_ENABLE_HGP) && defined(QDK_CHEMISTRY_ENABLE_RYS) && \
    defined(QDK_CHEMISTRY_ENABLE_LIBINTX)
    if (eri == "CPU")
      std::vector<double>().swap(ref_grads);  // Disable gradient check for CPU
#else
    std::vector<double>().swap(ref_grads);  // Disable gradient check for CPU
#endif
    energy_tol = energy_tolerance;
    grad_tol = grad_tolerance;

    cfg.require_gradient = ref_grads.size() > 0 && grad_tol > 0;
    cfg.basis = json["basis"];
    cfg.do_dfj = json.value("dfj", false);
    if (cfg.do_dfj) {
      cfg.aux_basis = json["aux_basis"];
    }
    cfg.cartesian = json.value("cart", true);
    cfg.unrestricted = scf_type == "uhf" || scf_type == "uks";
    cfg.scf_algorithm.max_iteration = 100;
    cfg.scf_algorithm.og_threshold = json.value("og_threshold", 1e-7);
    cfg.scf_algorithm.density_threshold = json.value("density_threshold", 1e-5);
    cfg.lindep_threshold = json.value("lindep_threshold", 1e-6);
    cfg.fock_reset_steps = 9999;
    cfg.incremental_fock_start_step = 3;
    cfg.scf_algorithm.enable_damping = false;
    cfg.mpi = ParallelConfig{1, 0, 1, 0};
    cfg.verbose = 5;
#if defined(QDK_CHEMISTRY_ENABLE_HGP) && defined(QDK_CHEMISTRY_ENABLE_RYS) && \
    defined(QDK_CHEMISTRY_ENABLE_LIBINTX)
    cfg.eri.method = eri == "HGP"       ? ERIMethod::HGP
                     : eri == "RYS"     ? ERIMethod::Rys
                     : eri == "LibintX" ? ERIMethod::LibintX
                     : eri == "CPU"     ? ERIMethod::Libint2Direct
                                        : ERIMethod::Incore;
    cfg.k_eri.method = k_eri == "HGP"   ? ERIMethod::HGP
                       : k_eri == "RYS" ? ERIMethod::Rys
                       : k_eri == "SNK" ? ERIMethod::SnK
                       : eri == "CPU"   ? ERIMethod::Libint2Direct
                                        : ERIMethod::Incore;
    cfg.grad_eri.method = (cfg.eri.method == ERIMethod::Incore && !cfg.do_dfj)
                              ? ERIMethod::HGP
                              : cfg.eri.method;
#else
    cfg.eri.method =
        eri == "CPU" ? ERIMethod::Libint2Direct : ERIMethod::Incore;
    cfg.k_eri.method =
        eri == "CPU" ? ERIMethod::Libint2Direct : ERIMethod::Incore;
    cfg.grad_eri.method = (cfg.eri.method == ERIMethod::Incore && !cfg.do_dfj)
                              ? ERIMethod::Libint2Direct
                              : cfg.eri.method;
#endif
    cfg.exc.method = EXCMethod::GauXC;
    cfg.basis_mode = BasisMode::PSI4;
    cfg.scf_algorithm.diis_subspace_size = 8;
    cfg.exc.xc_name = json.value("functional", "HF");
#ifdef QDK_CHEMISTRY_ENABLE_DFTD3
    cfg.disp = dispersion_from_string(json.value("disp", "none"));
#endif
    cfg.density_init_method =
        json.value("density-init-method", DensityInitializationMethod::Atom);
    if (cfg.density_init_method == DensityInitializationMethod::File) {
      if (json.contains("density-init-file")) {
        cfg.density_init_file = std::string(TEST_DATA_DIR) + "/" +
                                json.value("density-init-file", "");
      } else {
        QDK_LOGGER().error("SCFTest: No density-init-file provided");
        throw std::runtime_error("SCFTest: No density-init-file provided");
      }
    }

    cfg.scf_algorithm.level_shift = json.value("level_shift", -1.0);
    // For polarizability (CPSCF)
    ref_polarizability = json.value(
        "polarizability",
        std::array<double, 9>{{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}});
    // Isotropic polarizability is the average of the diagonal elements: (xx +
    // yy + zz) / 3
    ref_iso_polarizability = json.value("isotropic_polarizability", 0.0);
    if (ref_iso_polarizability == 0.0) {
      ref_iso_polarizability = (ref_polarizability[0] + ref_polarizability[4] +
                                ref_polarizability[8]) /
                               3.0;
    }
    cfg.require_polarizability = ref_iso_polarizability != 0.0;
    // Tolerance for comparing polarizability values
    polarizability_tol =
        json.value("polarizability_tolerance", energy_tol * 1e3);
    if (cfg.require_polarizability) {
      cfg.cpscf_input.max_iteration = json.value("cpscf_max_iteration", 30);
      cfg.cpscf_input.tolerance =
          json.value("cpscf_tolerance", polarizability_tol * 1e-3);
      cfg.cpscf_input.max_restart = json.value("cpscf_max_restart", 5);
    }
  }
  // void TearDown() override {}

  void Run() {
#ifdef QDK_CHEMISTRY_ENABLE_GPU
    cudaSetDevice(0);
    cuda::init_memory_pool(0);
#endif

    auto scf = scf_type == "rhf" or scf_type == "uhf"
                   ? SCF::make_hf_solver(mol, cfg)
                   : SCF::make_ks_solver(mol, cfg);
    const auto& ctx = scf->run();
    auto res = ctx.result;

    // Print JSON line for reference data
#if 1
    nlohmann::json json;
    json["energy"] = res.scf_total_energy;
    if (ref_grads.size()) json["gradient"] = res.scf_total_gradient;
    std::cout << json.dump(0);
#endif
    ASSERT_NEAR(res.scf_total_energy, ref_energy, energy_tol);

    if (grad_tol > 0 && ref_grads.size() > 0) {
      const auto& grads = res.scf_total_gradient;
      double max_err = 0;
      for (size_t i = 0; i < grads.size(); i++) {
        max_err = std::max(max_err, std::abs(grads[i] - ref_grads[i]));
      }
      ASSERT_LE(max_err, grad_tol);
    }

    if (cfg.require_polarizability) {
      const auto& polarizability = res.scf_polarizability;
      // if ref_polarizability is not all zeros, check the polarizability tensor
      if (!std::all_of(ref_polarizability.begin(), ref_polarizability.end(),
                       [](double val) { return val == 0.0; })) {
        for (size_t i = 0; i < polarizability.size(); i++)
          ASSERT_NEAR(ref_polarizability[i], polarizability[i],
                      polarizability_tol);
      } else
        ASSERT_NEAR(ref_iso_polarizability, res.scf_isotropic_polarizability,
                    polarizability_tol);
    }

    util::GAUXCRegistry::clear();
  }

  SCFConfig cfg;
  std::shared_ptr<Molecule> mol;
  std::string scf_type;
  double ref_energy;
  std::vector<double> ref_grads;
  double energy_tol;
  double grad_tol;
  double ref_lmo_objective_function_occupied;
  double ref_lmo_objective_function_virtual;
  std::array<double, 9> ref_polarizability;
  double ref_iso_polarizability;
  double polarizability_tol;
};
