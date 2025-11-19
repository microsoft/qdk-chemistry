// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once
#include <qdk/chemistry/scf/core/molecule.h>

#include <memory>
using namespace qdk::chemistry::scf;

inline std::shared_ptr<Molecule> make_h2o() {
  std::shared_ptr<Molecule> mol = std::make_shared<Molecule>();
  mol->atomic_nums = {1, 1, 8};

  mol->n_atoms = mol->atomic_nums.size();
  mol->atomic_charges = mol->atomic_nums;
  ;
  mol->total_nuclear_charge =
      std::accumulate(mol->atomic_nums.begin(), mol->atomic_nums.end(), 0ul);
  mol->n_electrons = mol->total_nuclear_charge;

  mol->coords = {{0.00, 0.49, -0.79}, {0.00, 0.49, 0.79}, {0.00, -0.12, 0.00}};

  const auto bohr_to_ang = 0.52917721092;  // Local def of conversion factor
                                           // consistent with unit tests
  for (auto& c : mol->coords)
    for (auto& p : c) {
      p /= bohr_to_ang;
    }

  return mol;
}

inline std::shared_ptr<Molecule> make_TeH2_HI() {
  std::shared_ptr<Molecule> mol = std::make_shared<Molecule>();
  mol->atomic_nums = {52, 1, 1, 1, 53};

  mol->n_atoms = mol->atomic_nums.size();
  mol->atomic_charges = mol->atomic_nums;
  ;
  mol->total_nuclear_charge =
      std::accumulate(mol->atomic_nums.begin(), mol->atomic_nums.end(), 0ul);
  mol->n_electrons = mol->total_nuclear_charge;

  mol->coords = {{0.03866320, 2.16215120, 0.00000000},
                 {-0.10453280, 0.53339420, 0.00000000},
                 {-1.58803480, 2.30795520, 0.00000000},
                 {1.61524120, -2.39760280, 0.00000000},
                 {0.03866320, -2.60589780, 0.00000000}};

  const auto bohr_to_ang = 0.52917721092;  // local def of conversion factor
                                           // consistent with unit tests
  for (auto& c : mol->coords)
    for (auto& p : c) {
      p /= bohr_to_ang;
    }

  return mol;
}

inline std::shared_ptr<Molecule> make_acetylene() {
  std::shared_ptr<Molecule> mol = std::make_shared<Molecule>();
  mol->atomic_nums = {6, 6, 1, 1};

  mol->n_atoms = mol->atomic_nums.size();
  mol->atomic_charges = mol->atomic_nums;
  ;
  mol->total_nuclear_charge =
      std::accumulate(mol->atomic_nums.begin(), mol->atomic_nums.end(), 0ul);
  mol->n_electrons = mol->total_nuclear_charge;

  mol->coords = {
      {-0.6000, 0.0000, 0.00000},
      {0.6000, 0.0000, 0.00000},
      {-1.6650, 0.0000, -0.00010},
      {1.6650, 0.0000, 0.00010},
  };

  const auto bohr_to_ang = 0.52917721092;  // Local def of conversion factor
                                           // consistent with unit tests
  for (auto& c : mol->coords)
    for (auto& p : c) {
      p /= bohr_to_ang;
    }

  return mol;
}

inline std::shared_ptr<Molecule> make_benzene() {
  std::shared_ptr<Molecule> mol = std::make_shared<Molecule>();
  mol->atomic_nums = {6, 6, 6, 6, 6, 6, 1, 1, 1, 1, 1, 1};

  mol->n_atoms = mol->atomic_nums.size();
  mol->atomic_charges = mol->atomic_nums;
  ;
  mol->total_nuclear_charge =
      std::accumulate(mol->atomic_nums.begin(), mol->atomic_nums.end(), 0ul);
  mol->n_electrons = mol->total_nuclear_charge;

  mol->coords = {
      {-1.2131, -0.6884, 0.0000},  {-1.2028, 0.7064, 0.0001},
      {-0.0103, -1.3948, 0.0000},  {0.0104, 1.3948, -0.0001},
      {1.2028, -0.7063, 0.0000},   {1.2131, 0.6884, 0.0000},
      {-2.1577, -1.2244, 0.0000},  {-2.1393, 1.2564, 0.0001},
      {-0.0184, -2.4809, -0.0001}, {0.0184, 2.4808, 0.0000},
      {2.1394, -1.2563, 0.0001},   {2.1577, 1.2245, 0.0000},
  };

  const auto bohr_to_ang = 0.52917721092;  // Local def of conversion factor
                                           // consistent with unit tests
  for (auto& c : mol->coords)
    for (auto& p : c) {
      p /= bohr_to_ang;
    }

  return mol;
}

inline std::shared_ptr<Molecule> make_bf() {
  std::shared_ptr<Molecule> mol = std::make_shared<Molecule>();
  mol->atomic_nums = {9, 5};

  mol->n_atoms = mol->atomic_nums.size();
  mol->atomic_charges = mol->atomic_nums;
  ;
  mol->total_nuclear_charge =
      std::accumulate(mol->atomic_nums.begin(), mol->atomic_nums.end(), 0ul);
  mol->n_electrons = mol->total_nuclear_charge;

  mol->coords = {
      {0.0, 0.0, 0.85543933},
      {0.0, 0.0, -1.53978853},
  };

  return mol;
}

inline std::shared_ptr<Molecule> make_c2h4() {
  std::shared_ptr<Molecule> mol = std::make_shared<Molecule>();
  mol->atomic_nums = {6, 6, 1, 1, 1, 1};

  mol->n_atoms = mol->atomic_nums.size();
  mol->atomic_charges = mol->atomic_nums;
  ;
  mol->total_nuclear_charge =
      std::accumulate(mol->atomic_nums.begin(), mol->atomic_nums.end(), 0ul);
  mol->n_electrons = mol->total_nuclear_charge;

  mol->coords = {{0.0000000, 0.0000000, 0.66766900},
                 {0.0000000, 0.0000000, -0.66766900},
                 {0.0000000, 0.9287200, 1.24085600},
                 {0.0000000, -0.9287200, 1.24085600},
                 {0.0000000, -0.9287200, -1.24085600},
                 {0.0000000, 0.9287200, -1.24085600}};

  const auto bohr_to_ang = 0.52917721092;  // local def of conversion factor
                                           // consistent with unit tests
  for (auto& c : mol->coords)
    for (auto& p : c) {
      p /= bohr_to_ang;
    }

  return mol;
}

inline std::shared_ptr<Molecule> make_o2() {
  std::shared_ptr<Molecule> mol = std::make_shared<Molecule>();
  mol->atomic_nums = {8, 8};

  mol->n_atoms = mol->atomic_nums.size();
  mol->atomic_charges = mol->atomic_nums;
  ;
  mol->total_nuclear_charge =
      std::accumulate(mol->atomic_nums.begin(), mol->atomic_nums.end(), 0ul);
  mol->n_electrons = mol->total_nuclear_charge;

  mol->coords = {
      {0.0, 0.0, 0.0},
      {0.0, 0.0, 1.21},
  };

  const auto bohr_to_ang = 0.52917721092;  // local def of conversion factor
                                           // consistent with unit tests
  for (auto& c : mol->coords)
    for (auto& p : c) {
      p /= bohr_to_ang;
    }

  return mol;
}

inline std::shared_ptr<Molecule> make_o_clean() {
  std::shared_ptr<Molecule> mol = std::make_shared<Molecule>();
  mol->atomic_nums = {8};

  mol->n_atoms = mol->atomic_nums.size();
  mol->atomic_charges = mol->atomic_nums;
  ;
  mol->total_nuclear_charge =
      std::accumulate(mol->atomic_nums.begin(), mol->atomic_nums.end(), 0ul);
  mol->n_electrons = mol->total_nuclear_charge;

  mol->coords = {{1.0, 1.0, 1.0}};

  const auto bohr_to_ang = 0.52917721092;  // local def of conversion factor
                                           // consistent with unit tests
  for (auto& c : mol->coords)
    for (auto& p : c) {
      p /= bohr_to_ang;
    }

  return mol;
}

inline std::shared_ptr<Molecule> make_water10() {
  std::shared_ptr<Molecule> mol = std::make_shared<Molecule>();
  mol->atomic_nums = {8, 1, 1, 8, 1, 1, 8, 1, 1, 8, 1, 1, 8, 1, 1,
                      8, 1, 1, 8, 1, 1, 8, 1, 1, 8, 1, 1, 8, 1, 1};

  mol->n_atoms = mol->atomic_nums.size();
  mol->atomic_charges = mol->atomic_nums;
  ;
  mol->total_nuclear_charge =
      std::accumulate(mol->atomic_nums.begin(), mol->atomic_nums.end(), 0ul);
  mol->n_electrons = mol->total_nuclear_charge;

  mol->coords = {
      {97.873900000, 103.017000000, 100.816000000},
      {98.128600000, 103.038000000, 99.848800000},
      {97.173800000, 102.317000000, 100.960000000},
      {99.814000000, 100.835000000, 101.232000000},
      {99.329200000, 99.976800000, 101.063000000},
      {99.151600000, 101.561000000, 101.414000000},
      {98.804000000, 98.512200000, 97.758100000},
      {99.782100000, 98.646900000, 97.916700000},
      {98.421800000, 99.326500000, 97.321300000},
      {100.747000000, 100.164000000, 103.736000000},
      {100.658000000, 100.628000000, 102.855000000},
      {100.105000000, 99.398600000, 103.776000000},
      {98.070300000, 98.516900000, 100.438000000},
      {97.172800000, 98.878600000, 100.690000000},
      {98.194000000, 98.592200000, 99.448100000},
      {98.548000000, 101.265000000, 97.248600000},
      {98.688900000, 102.140000000, 97.711000000},
      {97.919900000, 101.391000000, 96.480800000},
      {102.891000000, 100.842000000, 97.477600000},
      {103.837000000, 100.662000000, 97.209700000},
      {102.868000000, 101.166000000, 98.423400000},
      {102.360000000, 101.551000000, 99.964500000},
      {102.675000000, 102.370000000, 100.444000000},
      {101.556000000, 101.180000000, 100.430000000},
      {101.836000000, 97.446700000, 102.110000000},
      {100.860000000, 97.397400000, 101.898000000},
      {101.991000000, 97.133400000, 103.047000000},
      {101.665000000, 98.316100000, 98.319400000},
      {101.904000000, 99.233800000, 98.002000000},
      {102.224000000, 97.640900000, 97.837700000},
  };

  const auto bohr_to_ang = 0.52917721092;  // local def of conversion factor
                                           // consistent with unit tests
  for (auto& c : mol->coords)
    for (auto& p : c) {
      p /= bohr_to_ang;
    }

  return mol;
}

inline std::shared_ptr<Molecule> make_molecule(std::string name) {
  if (name == "h2o") return make_h2o();
  if (name == "TeH2-HI") return make_TeH2_HI();
  if (name == "acetylene") return make_acetylene();
  if (name == "benzene") return make_benzene();
  if (name == "bf") return make_bf();
  if (name == "c2h4") return make_c2h4();
  if (name == "o2") return make_o2();
  if (name == "o_clean") return make_o_clean();
  if (name == "water10") return make_water10();

  throw std::invalid_argument("Unknown Molecule");
  return nullptr;
}
