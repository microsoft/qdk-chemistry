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

// J/K engine timing the builds of another engine
class TimedJKEngine : public JKEngine {
 public:
  explicit TimedJKEngine(std::shared_ptr<JKEngine> engine)
      : engine_(std::move(engine)) {}

  void build(const double* P, size_t num_atomic_orbitals, size_t num_densities,
             double* J, double* K, double alpha, double beta,
             double omega) override {
    const auto start = std::chrono::steady_clock::now();
    engine_->build(P, num_atomic_orbitals, num_densities, J, K, alpha, beta,
                   omega);
    seconds +=
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
            .count();
  }

  static inline double seconds = 0.0;

 private:
  std::shared_ptr<JKEngine> engine_;
};

// J/K builder timing the builds of the qdk engine
class TimedJKBuilder : public JKBuilder {
 public:
  std::string name() const override { return "timed"; }

 protected:
  std::shared_ptr<JKEngine> _run_impl(std::shared_ptr<BasisSet> basis_set,
                                      double eri_threshold) const override {
    return std::make_shared<TimedJKEngine>(
        JKBuilderFactory::create("qdk")->run(basis_set, eri_threshold));
  }
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
  // start-cell-engine
  auto orbitals = wavefunction->get_orbitals();
  auto basis_set = orbitals->get_basis_set();
  const size_t n = basis_set->get_num_atomic_orbitals();
  Eigen::VectorXd occupations =
      Eigen::VectorXd::Zero(orbitals->get_num_molecular_orbitals());
  occupations.head(5).setConstant(2.0);
  const Eigen::MatrixXd P = orbitals->calculate_ao_density_matrix(occupations);

  // An engine for the basis set; J and K are filled in place
  auto engine = JKBuilderFactory::create()->run(basis_set, 1e-12);
  Eigen::MatrixXd J(n, n), K(n, n);
  engine->build(P.data(), n, 1, J.data(), K.data(), 1.0, 0.0, 0.0);
  std::cout << "Coulomb energy: " << 0.5 * (P.array() * J.array()).sum()
            << " Hartree" << std::endl;
  // end-cell-engine
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
  std::cout << "J/K builds: " << TimedJKEngine::seconds << " s" << std::endl;
  // end-cell-register
  // --------------------------------------------------------------------------------------------
  return 0;
}
