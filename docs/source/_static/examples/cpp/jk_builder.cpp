// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

// J/K builder usage examples.
// --------------------------------------------------------------------------------------------
// start-cell-custom
#include <chrono>
#include <iostream>
#include <qdk/chemistry.hpp>

using namespace qdk::chemistry::algorithms;
using namespace qdk::chemistry::data;

// J/K builder timing the builds of the qdk builder
class TimedJKBuilder : public JKBuilder {
 public:
  std::string name() const override { return "timed"; }

  static inline double seconds = 0.0;

 protected:
  std::pair<std::shared_ptr<const SymmetryBlockedTensor<2>>,
            std::shared_ptr<const SymmetryBlockedTensor<2>>>
  _run_impl(std::shared_ptr<BasisSet> basis_set,
            std::shared_ptr<const SymmetryBlockedTensor<2>> densities)
      const override {
    if (!_qdk) {
      _qdk = JKBuilderFactory::create("qdk");
      _qdk->settings().update(*_settings);
    }
    const auto start = std::chrono::steady_clock::now();
    auto result = _qdk->run(basis_set, densities);
    seconds +=
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
            .count();
    return result;
  }

 private:
  mutable std::unique_ptr<JKBuilder> _qdk;
};
// end-cell-custom
// --------------------------------------------------------------------------------------------

int main() {
  // --------------------------------------------------------------------------------------------
  // docs:xyz ../data/water.structure.xyz
  // start-cell-scf
  auto structure = Structure::from_xyz(R"(3
Water molecule
O    0.000000    0.000000    0.000000
H    0.758602    0.000000    0.504284
H   -0.758602    0.000000    0.504284
)");

  // J/K builders available to the SCF solver ("qdk" is the default)
  for (const auto& name : JKBuilderFactory::available()) {
    std::cout << name << std::endl;
  }

  // The qdk builder with its integrals stored in memory
  AlgorithmRef jk_builder("jk_builder", "qdk");
  jk_builder.get_settings()->set("eri_method", "incore");
  auto scf_solver = ScfSolverFactory::create();
  scf_solver->settings().set("jk_builder", jk_builder);
  auto [energy, wavefunction] = scf_solver->run(structure, 0, 1, "cc-pvdz");
  // end-cell-scf
  // --------------------------------------------------------------------------------------------

  // --------------------------------------------------------------------------------------------
  // start-cell-build
  auto orbitals = wavefunction->get_orbitals();
  auto basis_set = orbitals->get_basis_set();
  Eigen::VectorXd occupations =
      Eigen::VectorXd::Zero(orbitals->get_num_molecular_orbitals());
  occupations.head(5).setConstant(1.0);
  const Eigen::MatrixXd P_alpha =
      orbitals->calculate_ao_density_matrix(occupations);

  // J and K of the closed-shell spin densities (restricted: one alpha block)
  auto densities = std::make_shared<const SymmetryBlockedTensor<2>>(
      make_spin_diagonal_rank2_sbt(P_alpha, P_alpha, true));
  auto [J, K] = JKBuilderFactory::create()->run(basis_set, densities);
  const auto& J_alpha = J->block({axes::alpha(), axes::alpha()});
  std::cout << "Coulomb energy: "
            << 2.0 * (P_alpha.array() * J_alpha.array()).sum() << " Hartree"
            << std::endl;
  // end-cell-build
  // --------------------------------------------------------------------------------------------

  // --------------------------------------------------------------------------------------------
  // start-cell-register
  JKBuilderFactory::register_instance(
      [] { return std::make_unique<TimedJKBuilder>(); });

  auto timed_solver = ScfSolverFactory::create();
  timed_solver->settings().set("jk_builder",
                               AlgorithmRef("jk_builder", "timed"));
  auto [timed_energy, timed_wavefunction] =
      timed_solver->run(structure, 0, 1, "cc-pvdz");
  std::cout << "J/K builds: " << TimedJKBuilder::seconds << " s" << std::endl;
  // end-cell-register
  // --------------------------------------------------------------------------------------------
  return 0;
}
