// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

// Model Hamiltonian construction examples.
// --------------------------------------------------------------------------------------------
#include <qdk/chemistry.hpp>
using namespace qdk::chemistry::data;
using namespace qdk::chemistry::utils::model_hamiltonians;

int main() {
  // start-cell-create-huckel
  // Create a 6-site chain for the Hückel model
  auto lattice = LatticeGraph::chain(6);

  // Uniform parameters: all sites have the same on-site energy and hopping
  auto hamiltonian =
      create_huckel_hamiltonian(lattice, /*epsilon=*/0.0, /*t=*/1.0);

  bool has_h1 = hamiltonian.has_one_body_integrals();  // true
  bool has_h2 = hamiltonian.has_two_body_integrals();  // false
  // end-cell-create-huckel

  // --------------------------------------------------------------------------------------------
  // start-cell-create-hubbard
  // Create a 4-site Hubbard chain
  auto hub_lattice = LatticeGraph::chain(4);
  auto hub_hamiltonian = create_hubbard_hamiltonian(
      hub_lattice, /*epsilon=*/0.0, /*t=*/1.0, /*U=*/4.0);

  bool hub_has_h1 = hub_hamiltonian.has_one_body_integrals();  // true
  bool hub_has_h2 = hub_hamiltonian.has_two_body_integrals();  // true

  // Verify the on-site repulsion
  for (int i = 0; i < hub_lattice.num_sites(); ++i) {
    double Uii = hub_hamiltonian.get_two_body_element(i, i, i, i);  // 4.0
  }
  // end-cell-create-hubbard

  // --------------------------------------------------------------------------------------------
  // start-cell-create-hubbard-2d
  // Create a 4x4 square lattice Hubbard model with periodic boundaries
  auto square_lattice =
      LatticeGraph::square(4, 4, /*periodic_x=*/true, /*periodic_y=*/true);
  auto square_hamiltonian = create_hubbard_hamiltonian(
      square_lattice, /*epsilon=*/0.0, /*t=*/1.0, /*U=*/4.0);

  bool sq_has_h2 = square_hamiltonian.has_two_body_integrals();  // true
  // end-cell-create-hubbard-2d

  // --------------------------------------------------------------------------------------------
  // start-cell-create-ppp
  // Create a 6-site ring for the PPP model (benzene-like)
  auto ring = LatticeGraph::chain(6, /*periodic=*/true);

  // Compute interpair Coulomb repulsion with the Ohno potential
  auto V = ohno_potential(ring, /*U=*/0.414, /*R=*/2.65);

  // Create the PPP Hamiltonian
  auto ppp_hamiltonian = create_ppp_hamiltonian(ring,
                                                /*epsilon=*/0.0,
                                                /*t=*/0.088,
                                                /*U=*/0.414,
                                                /*V=*/V,
                                                /*z=*/1.0);

  bool ppp_has_h1 = ppp_hamiltonian.has_one_body_integrals();  // true
  bool ppp_has_h2 = ppp_hamiltonian.has_two_body_integrals();  // true
  double core_e = ppp_hamiltonian.get_core_energy();           // > 0
  // end-cell-create-ppp

  // --------------------------------------------------------------------------------------------
  // start-cell-site-dependent
  // Site-dependent Hubbard model: different U on each site
  auto chain = LatticeGraph::chain(4);
  Eigen::VectorXd U_values(4);
  U_values << 2.0, 4.0, 4.0, 2.0;  // Weaker repulsion at edges
  auto site_dep_hamiltonian = create_hubbard_hamiltonian(
      chain, /*epsilon=*/0.0, /*t=*/1.0, /*U=*/U_values);
  // end-cell-site-dependent

  // --------------------------------------------------------------------------------------------
  // start-cell-potentials
  auto pot_lattice = LatticeGraph::chain(4);

  // Ohno potential: V_ij = U_ij / sqrt(1 + (U_ij * epsilon_r * R_ij)^2)
  auto V_ohno =
      ohno_potential(pot_lattice, /*U=*/0.414, /*R=*/2.65, /*epsilon_r=*/1.0);

  // Mataga-Nishimoto potential: V_ij = U_ij / (1 + U_ij * epsilon_r * R_ij)
  auto V_mn = mataga_nishimoto_potential(pot_lattice, /*U=*/0.414, /*R=*/2.65,
                                         /*epsilon_r=*/1.0);

  // Custom pairwise potential using a user-defined function
  auto V_custom = pairwise_potential(
      pot_lattice,
      /*U=*/0.414,
      /*R=*/2.65,
      [](int, int, double Uij, double Rij) { return Uij / (1.0 + Rij); });
  // end-cell-potentials

  // --------------------------------------------------------------------------------------------
  // start-cell-solve-hubbard
  // Create a 4-site half-filled Hubbard chain
  auto solve_lattice = LatticeGraph::chain(4);
  auto solve_hamiltonian = std::make_shared<Hamiltonian>(
      create_hubbard_hamiltonian(solve_lattice,
                                 /*epsilon=*/0.0, /*t=*/1.0, /*U=*/4.0));

  // Run exact diagonalization (CASCI) with half filling (2 alpha + 2 beta)
  auto mc_calculator =
      qdk::chemistry::algorithms::MultiConfigurationCalculatorFactory::create(
          "macis_cas");
  auto [energy, wavefunction] = mc_calculator->run(solve_hamiltonian, 2, 2);
  // end-cell-solve-hubbard

  return 0;
}
