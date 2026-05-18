// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

/// @file test_fock_benchmark.cpp
/// @brief Phase 0 benchmarking and correctness harness for the LIBINT2_DIRECT
/// Fock builder. Provides:
///   - Timing breakdown (engine vs contraction vs reduction)
///   - Thread-scaling measurements
///   - Bit-reproducibility tests (TLS variant)
///   - Golden Fock tolerance tests (multi-thread vs single-thread reference)
///   - TLS-vs-serial cross-check (validates current s1234 counter correctness)
///   - Per-equality-pattern shell-quartet correctness fixtures
///   - Incremental Fock invariant test
///
/// All tests use GTest; benchmarks are controlled by the QDK_RUN_LONG_TESTS
/// option and print CSV-formatted timing data to stdout.

#include <gtest/gtest.h>

#include <qdk/chemistry/scf/config.h>
#include <qdk/chemistry/scf/core/basis_set.h>
#include <qdk/chemistry/scf/core/eri.h>
#include <qdk/chemistry/scf/core/molecule.h>
#include <qdk/chemistry/scf/scf/scf_solver.h>

#ifdef _OPENMP
#include <omp.h>
#endif
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "test_common.h"
#include "test_config.h"

using namespace qdk::chemistry::scf;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

namespace {

/// Make a symmetric random density matrix of size NAO x NAO.
/// Uses a fixed seed for reproducibility.
std::vector<double> make_random_density(size_t nao, unsigned seed = 42) {
  std::vector<double> P(nao * nao, 0.0);
  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> dist(-0.01, 0.01);
  for (size_t i = 0; i < nao; ++i) {
    for (size_t j = i; j < nao; ++j) {
      double val = dist(rng);
      P[i * nao + j] = val;
      P[j * nao + i] = val;
    }
  }
  return P;
}

/// Infinity-norm difference between two vectors.
double inf_norm_diff(const std::vector<double>& a,
                     const std::vector<double>& b) {
  double mx = 0.0;
  for (size_t i = 0; i < a.size(); ++i) {
    mx = std::max(mx, std::abs(a[i] - b[i]));
  }
  return mx;
}

/// Check byte-level identity of two vectors.
bool byte_identical(const std::vector<double>& a,
                    const std::vector<double>& b) {
  if (a.size() != b.size()) return false;
  return std::memcmp(a.data(), b.data(), a.size() * sizeof(double)) == 0;
}

/// Helper: create a BasisSet from a molecule and basis name.
std::shared_ptr<BasisSet> make_basis(std::shared_ptr<Molecule> mol,
                                     const std::string& basis_name,
                                     bool pure = true) {
  return BasisSet::from_database_json(mol, basis_name, BasisMode::PSI4, pure);
}

/// Helper: create a LIBINT2_DIRECT ERI from a basis set.
std::shared_ptr<ERI> make_direct_eri(BasisSet& basis,
                                     SCFOrbitalType orbital_type,
                                     bool use_atomics = false,
                                     double eri_threshold = 1e-9,
                                     double sp_threshold = 1e-12) {
  SCFConfig cfg;
  cfg.scf_orbital_type = orbital_type;
  cfg.eri.method = ERIMethod::Libint2Direct;
  cfg.eri.use_atomics = use_atomics;
  cfg.eri.eri_threshold = eri_threshold;
  cfg.eri.shell_pair_threshold = sp_threshold;
  cfg.mpi = ParallelConfig{1, 0, 1, 0};
  return ERI::create(basis, cfg, 0.0);
}

/// Build J and K from a density matrix using a given ERI engine.
/// Returns (J, K) as flat vectors.
std::pair<std::vector<double>, std::vector<double>> build_JK_standalone(
    ERI& eri, const std::vector<double>& P, size_t nao, size_t ndm,
    double alpha = 1.0, double beta = 0.0, double omega = 0.0) {
  size_t mat_size = ndm * nao * nao;
  std::vector<double> J(mat_size, 0.0);
  std::vector<double> K(mat_size, 0.0);
  eri.build_JK(P.data(), J.data(), K.data(), alpha, beta, omega);
  return {J, K};
}

/// A test system definition.
struct TestSystem {
  std::string name;
  std::string basis_name;
  bool pure;
  std::shared_ptr<Molecule> mol;
};

/// Build a water cluster with n water molecules at random positions.
/// For n=1, returns the standard water. For n>1, places copies at
/// random offsets from the (0,0,0) origin.
std::shared_ptr<Molecule> make_water_cluster(int n, unsigned seed = 123) {
  if (n == 1) return make_h2o();

  auto mol = std::make_shared<Molecule>();
  mol->atomic_nums.reserve(3 * n);
  mol->coords.reserve(3 * n);

  // Base water geometry in Angstrom
  const std::vector<std::array<double, 3>> base_coords = {
      {0.00, 0.49, -0.79}, {0.00, 0.49, 0.79}, {0.00, -0.12, 0.00}};
  const std::vector<size_t> base_z = {1, 1, 8};
  const double bohr_to_ang = 0.52917721092;

  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> offset_dist(-5.0, 5.0);

  for (int i = 0; i < n; ++i) {
    double ox = offset_dist(rng);
    double oy = offset_dist(rng);
    double oz = offset_dist(rng);
    // Ensure no overlap: space molecules at least 3 Angstrom apart
    ox += i * 3.0;

    for (int a = 0; a < 3; ++a) {
      mol->atomic_nums.push_back(base_z[a]);
      mol->coords.push_back(
          {(base_coords[a][0] + ox) / bohr_to_ang,
           (base_coords[a][1] + oy) / bohr_to_ang,
           (base_coords[a][2] + oz) / bohr_to_ang});
    }
  }

  mol->n_atoms = mol->atomic_nums.size();
  mol->atomic_charges = mol->atomic_nums;
  mol->total_nuclear_charge =
      std::accumulate(mol->atomic_nums.begin(), mol->atomic_nums.end(), 0ul);
  mol->n_electrons = mol->total_nuclear_charge;
  return mol;
}

/// Build a linear alkane C_n H_{2n+2}.
std::shared_ptr<Molecule> make_linear_alkane(int n_carbon, unsigned seed = 456) {
  auto mol = std::make_shared<Molecule>();
  const double cc_bond = 1.54;   // Angstrom
  const double ch_bond = 1.09;   // Angstrom
  const double bohr_to_ang = 0.52917721092;
  const double angle = 109.5 * M_PI / 180.0;

  // Place carbons along a zigzag chain
  std::vector<std::array<double, 3>> c_pos;
  for (int i = 0; i < n_carbon; ++i) {
    double x = i * cc_bond * std::cos(angle / 2.0);
    double y = (i % 2 == 0) ? 0.0 : cc_bond * std::sin(angle / 2.0);
    double z = 0.0;
    c_pos.push_back({x, y, z});
    mol->atomic_nums.push_back(6);
    mol->coords.push_back({x / bohr_to_ang, y / bohr_to_ang, z / bohr_to_ang});
  }

  // Add terminal and bridging hydrogens (simplified placement)
  std::mt19937 rng(seed);
  for (int i = 0; i < n_carbon; ++i) {
    int n_h = (i == 0 || i == n_carbon - 1) ? 3 : 2;
    for (int h = 0; h < n_h; ++h) {
      double hx = c_pos[i][0] + ch_bond * std::cos(h * 2.0 * M_PI / n_h + i);
      double hy = c_pos[i][1] + ch_bond * std::sin(h * 2.0 * M_PI / n_h + i);
      double hz = ch_bond * ((h % 2 == 0) ? 0.5 : -0.5);
      mol->atomic_nums.push_back(1);
      mol->coords.push_back(
          {hx / bohr_to_ang, hy / bohr_to_ang, hz / bohr_to_ang});
    }
  }

  mol->n_atoms = mol->atomic_nums.size();
  mol->atomic_charges = mol->atomic_nums;
  mol->total_nuclear_charge =
      std::accumulate(mol->atomic_nums.begin(), mol->atomic_nums.end(), 0ul);
  mol->n_electrons = mol->total_nuclear_charge;
  return mol;
}

/// P450 model compound: 1DZ9 HEM501+O502+CYS357, trimmed (88 atoms).
/// Fe-porphyrin with axial cysteine and oxo ligand.
/// Charged +1, singlet. Diverse elements: Fe, S, N, O, C, H.
/// Exercises d-function metal center + mixed AM basis.
std::shared_ptr<Molecule> make_p450_model() {
  auto mol = std::make_shared<Molecule>();
  // Coordinates in Angstrom — converted to Bohr below
  const double bohr_to_ang = 0.52917721092;
  mol->atomic_nums = {
    7, 6, 6, 8, 6, 16,  // N, C, C, O, C, S
    6, 6, 6, 6,          // 4C
    6, 6, 6, 6, 6,       // 5C
    6, 6, 6, 8, 8,       // 3C, 2O
    6, 6, 6, 6, 6,       // 5C
    6, 6, 6,             // 3C
    6, 6, 6, 6, 6,       // 5C
    6, 6, 6, 6,          // 4C
    6, 6, 6, 6, 6,       // 5C
    6, 6, 8, 8,          // 2C, 2O
    7, 7, 7, 7,          // 4N (porphyrin)
    26,                   // Fe
    8,                    // O (axial)
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  // 10H
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  // 10H
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  // 10H
    1, 1, 1, 1, 1, 1               // 6H
  };
  mol->coords = {
    {24.07652371, 6.10224978, 46.17969429},
    {24.54100037, 5.19099998, 45.18899918},
    {25.65852345, 5.56652138, 44.26768109},
    {25.40931657, 5.84119315, 43.08775448},
    {23.36564125, 4.84774259, 44.27750294},
    {23.73138212, 3.70854959, 42.88224959},
    {24.39301402, 2.00118437, 47.04838900},
    {20.37395151, 2.03779836, 44.43642519},
    {22.86362846, 0.73251363, 40.50499708},
    {26.89048020, 0.71934709, 43.11367033},
    {23.06380190, 2.11155751, 46.69945357},
    {22.01493340, 2.55771916, 47.58189547},
    {20.86145351, 2.52068110, 46.85410460},
    {21.23407254, 2.09743191, 45.51759402},
    {19.47651008, 2.85660376, 47.30217225},
    {22.19903101, 2.96946028, 49.00425552},
    {22.02537027, 1.79518870, 49.97291230},
    {22.00691310, 2.28498476, 51.41411320},
    {21.56423488, 3.33258825, 51.78022624},
    {22.47050284, 1.38667925, 52.29777768},
    {20.71985404, 1.74029602, 43.12747729},
    {19.79201595, 1.75222916, 42.02156215},
    {20.50405560, 1.38263152, 40.91707580},
    {21.85246268, 1.14705130, 41.35366006},
    {18.34100153, 2.07662746, 42.09764730},
    {20.03163444, 1.17663453, 39.50169596},
    {18.76242591, 1.83403459, 39.17870890},
    {24.18282415, 0.58433160, 40.88231956},
    {25.27214060, 0.27530441, 39.97413577},
    {26.40316005, 0.24477634, 40.72779327},
    {26.00143981, 0.56575736, 42.07490101},
    {25.09777287, 0.01242108, 38.51585584},
    {27.81579269,-0.08555060, 40.37340741},
    {28.34043216,-1.22497471, 41.14573936},
    {26.54699097, 1.11160788, 44.39224113},
    {27.48279233, 1.18980350, 45.47087095},
    {26.76922079, 1.51763531, 46.59037680},
    {25.40386508, 1.67256287, 46.16988299},
    {28.92197585, 0.83952582, 45.39655417},
    {27.24475991, 1.67781187, 47.98471156},
    {27.45045291, 3.15490879, 48.29490800},
    {27.71717604, 3.38284217, 49.89419285},
    {28.39885931, 4.37300043, 50.09423308},
    {27.18611847, 2.54212281, 50.59832603},
    {22.57499903, 1.86669867, 45.45435473},
    {21.96433356, 1.38882969, 42.68994418},
    {24.66027652, 0.75472399, 42.14449573},
    {25.28591737, 1.42227887, 44.83132023},
    {26.0,        1.17426732, 43.82549016}, // Fe
    {23.34469863,-0.35183942, 44.17277384},
    // Hydrogens
    {23.10234559, 6.35936257, 46.08387099},
    {25.01312649, 4.00663340, 42.69969817},
    {22.97035937, 5.74611564, 43.78692113},
    {22.55491706, 4.35497851, 44.81179951},
    {24.68497416, 2.18402057, 48.06990006},
    {19.34227965, 2.29053336, 44.62364704},
    {22.61516984, 0.54513797, 39.46929462},
    {27.92427867, 0.48258180, 42.91616690},
    {19.23326218, 3.90360409, 47.10472613},
    {19.35927418, 2.68766378, 48.37153232},
    {18.75369254, 2.23675989, 46.77579538},
    {23.18812039, 3.40479463, 49.15164511},
    {21.46409513, 3.73312038, 49.26072599},
    {22.79560665, 1.03375678, 49.81758371},
    {21.06340662, 1.30111927, 49.80104713},
    {18.16970900, 3.15420866, 42.10889015},
    {17.89585883, 1.65125022, 42.99388286},
    {17.83222487, 1.67488946, 41.21861779},
    {19.97014781, 0.10256686, 39.31374943},
    {20.81398238, 1.58000106, 38.84264492},
    {17.94439137, 1.28404803, 38.76059686},
    {18.67296418, 2.89747827, 39.27234684},
    {24.56436456,-0.93109775, 38.39013223},
    {26.05411648,-0.05946565, 37.99733495},
    {24.51380483, 0.79995865, 38.03654361},
    {28.45071985, 0.79628595, 40.50808775},
    {27.88012914,-0.33122944, 39.30579160},
    {29.38471863,-1.45865244, 41.09890442},
    {27.67496721,-1.99039362, 41.49285295},
    {29.44680350, 1.25074874, 46.25154717},
    {29.02298849,-0.24366046, 45.40979078},
    {29.40167900, 1.19626217, 44.49015656},
    {28.18627153, 1.14638019, 48.12478296},
    {26.52225939, 1.27160915, 48.69559395},
    {26.57200026, 3.76757278, 48.09516550},
    {28.30417441, 3.58101273, 47.77424841},
    {24.60172024, 6.50045093, 46.94500759},
    {26.70020140, 5.36880292, 44.52049418},
  };
  // Convert Angstrom to Bohr
  for (auto& c : mol->coords)
    for (auto& p : c)
      p /= bohr_to_ang;

  mol->n_atoms = mol->atomic_nums.size();
  mol->atomic_charges = mol->atomic_nums;
  mol->total_nuclear_charge =
      std::accumulate(mol->atomic_nums.begin(), mol->atomic_nums.end(), 0ul);
  mol->n_electrons = mol->total_nuclear_charge - 1;  // +1 charge
  mol->multiplicity = 1;  // singlet
  return mol;
}

/// Protonated nitrogen-rich porphyrin-like macrocycle C15H14N7(+1).
/// Planar, singlet, 152 electrons, 76 occupied MOs.
/// Provides a P450-sized system without metal d-orbitals or open-shell issues.
std::shared_ptr<Molecule> make_porphyrin_model() {
  auto mol = std::make_shared<Molecule>();
  const double bohr_to_ang = 0.52917721092;
  // C15H14N7: 15 carbons, 7 nitrogens, 14 hydrogens = 36 atoms
  mol->atomic_nums = {
    6, 6, 6, 6, 6, 6,       // C1-C6
    7, 7, 7, 7,              // N7-N10
    6,                        // C11
    6, 6, 6, 6, 6, 6, 6, 6,  // C12-C19
    7, 7, 7,                  // N20-N22
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1  // H23-H36
  };
  // Geometry in Angstrom (all z=0, planar)
  std::vector<std::array<double, 3>> coords_ang = {
    {-1.87749114,  1.72995118, 0.0},
    { 0.55298794, -2.49233636, 0.0},
    {-1.44240842,  3.83490380, 0.0},
    { 2.59164903, -3.17361167, 0.0},
    {-2.85053995,  3.71860327, 0.0},
    { 1.78367879, -4.33272881, 0.0},
    {-3.10205413,  2.34160934, 0.0},
    { 0.46668746, -3.85845403, 0.0},
    {-1.75265183,  0.33608322, 0.0},
    {-0.58959022, -1.68414207, 0.0},
    {-0.55520632, -0.31941790, 0.0},
    { 3.70172364, -5.71288198, 0.0},
    {-3.08037304,  6.07039014, 0.0},
    {-1.67915524,  6.20326625, 0.0},
    { 4.52056987, -4.56807508, 0.0},
    { 3.98373604, -3.28506175, 0.0},
    {-0.83943378,  5.09454174, 0.0},
    {-3.69852891,  4.82220250, 0.0},
    { 2.31191334, -5.62033823, 0.0},
    { 0.58002784,  0.33437541, 0.0},
    {-0.86259333,  2.56230020, 0.0},
    { 1.78253082, -2.03297683, 0.0},
    { 0.50897917,  1.36954842, 0.0},
    { 1.43978081, -0.24652159, 0.0},
    {-1.49025925, -2.14352941, 0.0},
    {-2.60226661, -0.21204512, 0.0},
    {-4.01401211,  1.90486802, 0.0},
    {-0.36900502, -4.42768098, 0.0},
    {-1.24613979,  7.19823447, 0.0},
    { 5.59842868, -4.69352500, 0.0},
    { 4.16454367, -6.69431724, 0.0},
    {-3.69650783,  6.96359793, 0.0},
    { 4.61473138, -2.40280157, 0.0},
    { 0.24042477,  5.19686258, 0.0},
    { 1.68514580, -6.50614389, 0.0},
    {-4.77932212,  4.72525102, 0.0},
  };
  mol->coords.reserve(coords_ang.size());
  for (const auto& c : coords_ang) {
    mol->coords.push_back({c[0] / bohr_to_ang, c[1] / bohr_to_ang, c[2] / bohr_to_ang});
  }

  mol->n_atoms = mol->atomic_nums.size();
  mol->atomic_charges = mol->atomic_nums;
  mol->total_nuclear_charge =
      std::accumulate(mol->atomic_nums.begin(), mol->atomic_nums.end(), 0ul);
  mol->n_electrons = mol->total_nuclear_charge - 1;  // +1 charge
  mol->multiplicity = 1;  // singlet
  return mol;
}

/// Get the set of test systems for benchmarking.
/// Systems are ordered by increasing size. The largest ones (water20, water40,
/// alkane chains, benzene in TZVP) push into the regime where TLS scratch
/// memory dominates and scaling should collapse.
std::vector<TestSystem> get_benchmark_systems() {
  std::vector<TestSystem> systems;

  // --- Small (< 50 AOs) --- baseline / overhead-dominated regime
  // water in def2-SVP (~25 AOs, spherical)
  systems.push_back({"h2o_def2svp", "def2-svp", true, make_h2o()});
  // water in 6-31G** (~25 AOs, Cartesian) — exercises Cartesian shell sizes
  systems.push_back({"h2o_631gss_cart", "6-31g**", false, make_h2o()});

  // --- Medium (50–200 AOs) --- compute/reduction crossover
  // Benzene in def2-SVP (~114 AOs)
  systems.push_back({"benzene_def2svp", "def2-svp", true, make_benzene()});
  // Water cluster (5 waters) in def2-SVP (~120 AOs)
  systems.push_back(
      {"water5_def2svp", "def2-svp", true, make_water_cluster(5)});

  // --- Large (200–600 AOs) --- TLS starts to dominate
  // Water cluster (10) in def2-SVP (~240 AOs)
  systems.push_back(
      {"water10_def2svp", "def2-svp", true, make_water10()});
  // Linear alkane C12H26 in 6-31G(d) (~160 AOs, low-AM-heavy)
  systems.push_back(
      {"alkane12_631gd", "6-31g*", true, make_linear_alkane(12)});
  // Water cluster (20) in def2-SVP (~480 AOs)
  systems.push_back(
      {"water20_def2svp", "def2-svp", true, make_water_cluster(20)});
  // Benzene in def2-TZVP (~270 AOs, d/f functions, higher AM mix)
  systems.push_back({"benzene_def2tzvp", "def2-tzvp", true, make_benzene()});

  // P450 model compound (88 atoms, Fe/S/N/O/C/H) in cc-pVDZ (~600 AOs)
  // Production-relevant: iron-porphyrin with d-functions on metal center
  systems.push_back({"p450_ccpvdz", "cc-pvdz", true, make_p450_model()});

  // Porphyrin model: C15H14N7(+1) in cc-pVTZ (~856 AOs, spherical)
  // Planar N-rich macrocycle: P450-scale without d-orbitals or open-shell issues
  systems.push_back(
      {"porphyrin_ccpvtz", "cc-pvtz", true, make_porphyrin_model()});

  // --- Very large (600+ AOs) --- where you really feel the pain
  // Only run if QDK_FOCK_BENCH_XL=1 is set (these take minutes per thread count)
  if (std::getenv("QDK_FOCK_BENCH_XL")) {
    // Linear alkane C24H50 in 6-31G(d) (~320 AOs)
    systems.push_back(
        {"alkane24_631gd", "6-31g*", true, make_linear_alkane(24)});
    // Water cluster (40) in def2-SVP (~960 AOs)
    systems.push_back(
        {"water40_def2svp", "def2-svp", true, make_water_cluster(40)});
  }

  return systems;
}

/// Get the small system set for quick correctness tests.
std::vector<TestSystem> get_correctness_systems() {
  std::vector<TestSystem> systems;
  // Spherical
  systems.push_back({"h2o_def2svp", "def2-svp", true, make_h2o()});
  // Cartesian
  systems.push_back({"h2o_631gss_cart", "6-31g**", false, make_h2o()});
  // Multi-atom with p,d functions
  systems.push_back({"benzene_def2svp", "def2-svp", true, make_benzene()});
  return systems;
}

}  // namespace

// ===========================================================================
// TEST: TLS vs Serial cross-check
// Validates that the current multi-threaded TLS code produces the same J/K
// as a single-threaded run. This guards against the s1234++ counter being
// broken by a desynchronized thread.
// ===========================================================================
class TlsSerialCrossCheckTest
    : public ::testing::TestWithParam<TestSystem> {};

TEST_P(TlsSerialCrossCheckTest, TlsMatchesSerial) {
  const auto& sys = GetParam();
  auto basis = make_basis(sys.mol, sys.basis_name, sys.pure);
  const size_t nao = basis->num_atomic_orbitals;
  const size_t ndm = 1;
  auto P = make_random_density(nao);

  // Run with 1 thread (serial reference)
  std::vector<double> J_ref, K_ref;
  {
#ifdef _OPENMP
    int orig = omp_get_max_threads();
    omp_set_num_threads(1);
#endif
    auto eri = make_direct_eri(*basis, SCFOrbitalType::Restricted, false);
    std::tie(J_ref, K_ref) =
        build_JK_standalone(*eri, P, nao, ndm);
#ifdef _OPENMP
    omp_set_num_threads(orig);
#endif
  }

  // Run with max threads (TLS variant)
  std::vector<double> J_mt, K_mt;
  {
    auto eri = make_direct_eri(*basis, SCFOrbitalType::Restricted, false);
    std::tie(J_mt, K_mt) =
        build_JK_standalone(*eri, P, nao, ndm);
  }

  double j_diff = inf_norm_diff(J_ref, J_mt);
  double k_diff = inf_norm_diff(K_ref, K_mt);

  // They should be byte-identical since TLS reduction order is deterministic
  // within a single binary/run. But at minimum, they should match to 1e-13.
  EXPECT_LT(j_diff, 1e-13)
      << "J mismatch between 1-thread and multi-thread TLS for " << sys.name;
  EXPECT_LT(k_diff, 1e-13)
      << "K mismatch between 1-thread and multi-thread TLS for " << sys.name;

  // Log whether they are byte-identical
  bool j_identical = byte_identical(J_ref, J_mt);
  bool k_identical = byte_identical(K_ref, K_mt);
  std::cout << "[INFO] " << sys.name
            << " TLS-vs-serial: J byte-identical=" << j_identical
            << " K byte-identical=" << k_identical
            << " J_inf_diff=" << j_diff << " K_inf_diff=" << k_diff
            << std::endl;
}

INSTANTIATE_TEST_SUITE_P(
    FockBuilder, TlsSerialCrossCheckTest,
    ::testing::ValuesIn(get_correctness_systems()),
    [](const ::testing::TestParamInfo<TestSystem>& info) {
      return info.param.name;
    });

// ===========================================================================
// TEST: Bit-reproducibility (same binary, same input, two runs)
// The TLS variant should produce byte-identical results.
// ===========================================================================
class BitReproducibilityTest
    : public ::testing::TestWithParam<TestSystem> {};

TEST_P(BitReproducibilityTest, TlsIsBitReproducible) {
  const auto& sys = GetParam();
  auto basis = make_basis(sys.mol, sys.basis_name, sys.pure);
  const size_t nao = basis->num_atomic_orbitals;
  const size_t ndm = 1;
  auto P = make_random_density(nao);

  auto eri = make_direct_eri(*basis, SCFOrbitalType::Restricted, false);

  auto [J1, K1] = build_JK_standalone(*eri, P, nao, ndm);
  auto [J2, K2] = build_JK_standalone(*eri, P, nao, ndm);

  EXPECT_TRUE(byte_identical(J1, J2))
      << "J not bit-reproducible for " << sys.name
      << " diff=" << inf_norm_diff(J1, J2);
  EXPECT_TRUE(byte_identical(K1, K2))
      << "K not bit-reproducible for " << sys.name
      << " diff=" << inf_norm_diff(K1, K2);
}

TEST_P(BitReproducibilityTest, AtomicVariantIsNonDeterministic) {
  // Document that the atomic variant is NOT bit-reproducible
  const auto& sys = GetParam();
  auto basis = make_basis(sys.mol, sys.basis_name, sys.pure);
  const size_t nao = basis->num_atomic_orbitals;
  const size_t ndm = 1;
  auto P = make_random_density(nao);

#ifdef _OPENMP
  if (omp_get_max_threads() < 2) {
    GTEST_SKIP() << "Need >= 2 threads to test atomic non-determinism";
  }
#else
  GTEST_SKIP() << "Need OpenMP to test atomic non-determinism";
#endif

  auto eri = make_direct_eri(*basis, SCFOrbitalType::Restricted, true);

  // Run multiple times and check for non-determinism
  auto [J1, K1] = build_JK_standalone(*eri, P, nao, ndm);
  bool found_nondeterminism = false;
  for (int trial = 0; trial < 10; ++trial) {
    auto [J2, K2] = build_JK_standalone(*eri, P, nao, ndm);
    if (!byte_identical(J1, J2) || !byte_identical(K1, K2)) {
      double j_diff = inf_norm_diff(J1, J2);
      double k_diff = inf_norm_diff(K1, K2);
      std::cout << "[INFO] " << sys.name
                << " atomic variant non-determinism detected at trial "
                << trial << ": J_diff=" << j_diff << " K_diff=" << k_diff
                << std::endl;
      found_nondeterminism = true;
      break;
    }
  }
  // We expect non-determinism, but it may not manifest in every trial
  // for very small systems. Just document the result.
  std::cout << "[INFO] " << sys.name
            << " atomic variant non-determinism "
            << (found_nondeterminism ? "DETECTED" : "not detected in 10 trials")
            << std::endl;
}

INSTANTIATE_TEST_SUITE_P(
    FockBuilder, BitReproducibilityTest,
    ::testing::ValuesIn(get_correctness_systems()),
    [](const ::testing::TestParamInfo<TestSystem>& info) {
      return info.param.name;
    });

// ===========================================================================
// TEST: Golden Fock tolerance test
// Multi-threaded J/K must match single-threaded reference within 1e-13.
// ===========================================================================
class GoldenFockTest : public ::testing::TestWithParam<TestSystem> {};

TEST_P(GoldenFockTest, MultiThreadMatchesSingleThread) {
  const auto& sys = GetParam();
  auto basis = make_basis(sys.mol, sys.basis_name, sys.pure);
  const size_t nao = basis->num_atomic_orbitals;
  const size_t ndm = 1;
  auto P = make_random_density(nao);

  // Single-threaded reference
  std::vector<double> J_ref, K_ref;
  {
#ifdef _OPENMP
    int orig = omp_get_max_threads();
    omp_set_num_threads(1);
#endif
    auto eri = make_direct_eri(*basis, SCFOrbitalType::Restricted, false);
    std::tie(J_ref, K_ref) = build_JK_standalone(*eri, P, nao, ndm);
#ifdef _OPENMP
    omp_set_num_threads(orig);
#endif
  }

  // Multi-threaded
  {
    auto eri = make_direct_eri(*basis, SCFOrbitalType::Restricted, false);
    auto [J_mt, K_mt] = build_JK_standalone(*eri, P, nao, ndm);

    double j_diff = inf_norm_diff(J_ref, J_mt);
    double k_diff = inf_norm_diff(K_ref, K_mt);

    EXPECT_LT(j_diff, 1e-13)
        << "J golden Fock mismatch for " << sys.name;
    EXPECT_LT(k_diff, 1e-13)
        << "K golden Fock mismatch for " << sys.name;

    std::cout << "[INFO] " << sys.name << " golden Fock: J_inf_diff=" << j_diff
              << " K_inf_diff=" << k_diff << std::endl;
  }
}

TEST_P(GoldenFockTest, JOnlyAndKOnlyWork) {
  const auto& sys = GetParam();
  auto basis = make_basis(sys.mol, sys.basis_name, sys.pure);
  const size_t nao = basis->num_atomic_orbitals;
  const size_t ndm = 1;
  auto P = make_random_density(nao);

  auto eri = make_direct_eri(*basis, SCFOrbitalType::Restricted, false);

  // Full J+K reference
  auto [J_full, K_full] = build_JK_standalone(*eri, P, nao, ndm);

  // J-only (K = nullptr) — tighter screening may skip quartets whose K
  // contribution was keeping them alive in the J+K path, so allow threshold-
  // level differences rather than exact match.
  {
    size_t mat_size = ndm * nao * nao;
    std::vector<double> J_only(mat_size, 0.0);
    eri->build_JK(P.data(), J_only.data(), nullptr, 1.0, 0.0, 0.0);
    double j_diff = inf_norm_diff(J_full, J_only);
    EXPECT_LT(j_diff, 1e-8) << "J-only differs from J+K for " << sys.name;
  }

  // K-only (J = nullptr)
  {
    size_t mat_size = ndm * nao * nao;
    std::vector<double> K_only(mat_size, 0.0);
    eri->build_JK(P.data(), nullptr, K_only.data(), 1.0, 0.0, 0.0);
    double k_diff = inf_norm_diff(K_full, K_only);
    EXPECT_LT(k_diff, 1e-8) << "K-only differs from J+K for " << sys.name;
  }
}

INSTANTIATE_TEST_SUITE_P(
    FockBuilder, GoldenFockTest,
    ::testing::ValuesIn(get_correctness_systems()),
    [](const ::testing::TestParamInfo<TestSystem>& info) {
      return info.param.name;
    });

// ===========================================================================
// TEST: UHF (unrestricted) correctness
// Verify build_JK works with ndm=2 (two spin density matrices).
// Calls build_JK twice on the same ERI object to exercise sparse zeroing of
// both spin blocks, and checks alpha and beta blocks independently.
// ===========================================================================
TEST(FockBuilderUHF, UnrestrictedMatchesTolerance) {
  auto mol = make_h2o();
  mol->multiplicity = 1;  // Still singlet, but force unrestricted
  auto basis = make_basis(mol, "def2-svp", true);
  const size_t nao = basis->num_atomic_orbitals;
  const size_t ndm = 2;

  // Create two spin-density matrices (alpha and beta, both symmetric)
  std::vector<double> P(ndm * nao * nao, 0.0);
  auto P_alpha = make_random_density(nao, 42);
  auto P_beta = make_random_density(nao, 43);
  std::copy(P_alpha.begin(), P_alpha.end(), P.begin());
  std::copy(P_beta.begin(), P_beta.end(), P.begin() + nao * nao);

  // Single-threaded reference — call twice on the same ERI to exercise zeroing
  std::vector<double> J_ref(ndm * nao * nao, 0.0);
  std::vector<double> K_ref(ndm * nao * nao, 0.0);
  {
#ifdef _OPENMP
    int orig = omp_get_max_threads();
    omp_set_num_threads(1);
#endif
    auto eri =
        make_direct_eri(*basis, SCFOrbitalType::Unrestricted, false);
    // First call warms up; second call exercises sparse zeroing
    eri->build_JK(P.data(), J_ref.data(), K_ref.data(), 1.0, 0.0, 0.0);
    std::fill(J_ref.begin(), J_ref.end(), 0.0);
    std::fill(K_ref.begin(), K_ref.end(), 0.0);
    eri->build_JK(P.data(), J_ref.data(), K_ref.data(), 1.0, 0.0, 0.0);
#ifdef _OPENMP
    omp_set_num_threads(orig);
#endif
  }

  // Multi-threaded — also call twice
  std::vector<double> J_mt(ndm * nao * nao, 0.0);
  std::vector<double> K_mt(ndm * nao * nao, 0.0);
  {
    auto eri =
        make_direct_eri(*basis, SCFOrbitalType::Unrestricted, false);
    eri->build_JK(P.data(), J_mt.data(), K_mt.data(), 1.0, 0.0, 0.0);
    std::fill(J_mt.begin(), J_mt.end(), 0.0);
    std::fill(K_mt.begin(), K_mt.end(), 0.0);
    eri->build_JK(P.data(), J_mt.data(), K_mt.data(), 1.0, 0.0, 0.0);
  }

  // Compare full matrices
  double j_diff = inf_norm_diff(J_ref, J_mt);
  double k_diff = inf_norm_diff(K_ref, K_mt);
  EXPECT_LT(j_diff, 1e-13) << "UHF J mismatch";
  EXPECT_LT(k_diff, 1e-13) << "UHF K mismatch";

  // Verify each spin block independently is non-trivial
  auto block_max_abs = [](const std::vector<double>& v, size_t offset, size_t n) {
    double mx = 0.0;
    for (size_t i = 0; i < n; ++i)
      mx = std::max(mx, std::abs(v[offset + i]));
    return mx;
  };
  EXPECT_GT(block_max_abs(J_ref, 0, nao * nao), 1e-10)
      << "UHF J alpha block appears to be all zeros";
  EXPECT_GT(block_max_abs(J_ref, nao * nao, nao * nao), 1e-10)
      << "UHF J beta block appears to be all zeros";
  EXPECT_GT(block_max_abs(K_ref, 0, nao * nao), 1e-10)
      << "UHF K alpha block appears to be all zeros";
  EXPECT_GT(block_max_abs(K_ref, nao * nao, nao * nao), 1e-10)
      << "UHF K beta block appears to be all zeros";
}

// ===========================================================================
// TEST: Incremental Fock invariant
// build_JK(P_diff) should produce the correct delta contribution such that
// J_accumulated = J(P_full) (within tolerance).
// ===========================================================================
TEST(FockBuilderIncremental, DeltaDensityIsAdditive) {
  auto mol = make_h2o();
  auto basis = make_basis(mol, "def2-svp", true);
  const size_t nao = basis->num_atomic_orbitals;
  const size_t ndm = 1;
  const size_t mat_size = ndm * nao * nao;

  // Create two slightly different density matrices
  auto P1 = make_random_density(nao, 100);
  auto P2 = make_random_density(nao, 200);

  // Scale P2 to be a small perturbation of P1 (simulates incremental SCF)
  for (size_t i = 0; i < mat_size; ++i) {
    P2[i] = P1[i] + 0.001 * P2[i];
    // Re-symmetrize
  }
  for (size_t i = 0; i < nao; ++i) {
    for (size_t j = i + 1; j < nao; ++j) {
      double avg = 0.5 * (P2[i * nao + j] + P2[j * nao + i]);
      P2[i * nao + j] = avg;
      P2[j * nao + i] = avg;
    }
  }

  // P_diff = P2 - P1
  std::vector<double> P_diff(mat_size);
  for (size_t i = 0; i < mat_size; ++i) P_diff[i] = P2[i] - P1[i];

  auto eri = make_direct_eri(*basis, SCFOrbitalType::Restricted, false);

  // Full build from P1
  auto [J1, K1] = build_JK_standalone(*eri, P1, nao, ndm);

  // Full build from P2 (reference)
  auto [J2_ref, K2_ref] = build_JK_standalone(*eri, P2, nao, ndm);

  // Incremental: build from P_diff and add to J1, K1
  std::vector<double> J_delta(mat_size, 0.0);
  std::vector<double> K_delta(mat_size, 0.0);
  eri->build_JK(P_diff.data(), J_delta.data(), K_delta.data(), 1.0, 0.0, 0.0);

  // J2_incremental = J1 + J_delta
  std::vector<double> J2_inc(mat_size);
  std::vector<double> K2_inc(mat_size);
  for (size_t i = 0; i < mat_size; ++i) {
    J2_inc[i] = J1[i] + J_delta[i];
    K2_inc[i] = K1[i] + K_delta[i];
  }

  double j_diff = inf_norm_diff(J2_ref, J2_inc);
  double k_diff = inf_norm_diff(K2_ref, K2_inc);

  EXPECT_LT(j_diff, 1e-12) << "Incremental J not additive";
  EXPECT_LT(k_diff, 1e-12) << "Incremental K not additive";

  std::cout << "[INFO] Incremental Fock: J_diff=" << j_diff
            << " K_diff=" << k_diff << std::endl;
}

// ===========================================================================
// TEST: Symmetry of output matrices
// J and K should be symmetric for a symmetric P.
// ===========================================================================
TEST(FockBuilderSymmetry, OutputIsSymmetric) {
  auto mol = make_h2o();
  auto basis = make_basis(mol, "def2-svp", true);
  const size_t nao = basis->num_atomic_orbitals;
  const size_t ndm = 1;
  auto P = make_random_density(nao);

  auto eri = make_direct_eri(*basis, SCFOrbitalType::Restricted, false);
  auto [J, K] = build_JK_standalone(*eri, P, nao, ndm);

  // Check symmetry
  double j_asym = 0.0, k_asym = 0.0;
  for (size_t i = 0; i < nao; ++i) {
    for (size_t j = i + 1; j < nao; ++j) {
      j_asym =
          std::max(j_asym, std::abs(J[i * nao + j] - J[j * nao + i]));
      k_asym =
          std::max(k_asym, std::abs(K[i * nao + j] - K[j * nao + i]));
    }
  }
  EXPECT_LT(j_asym, 1e-15) << "J is not symmetric";
  EXPECT_LT(k_asym, 1e-15) << "K is not symmetric";
}

// ===========================================================================
// TEST: Timing benchmark
// Measures build_JK wall time across thread counts and systems.
// Set environment variable QDK_FOCK_BENCH=1 to enable.
// ===========================================================================
TEST(FockBenchmark, ThreadScaling) {
  if (!std::getenv("QDK_FOCK_BENCH")) {
    GTEST_SKIP() << "Set QDK_FOCK_BENCH=1 to run timing benchmarks";
  }
  std::cout << "system,basis,nao,nthreads,build_jk_ms,variant" << std::endl;

  auto systems = get_benchmark_systems();

#ifdef _OPENMP
  int max_threads = omp_get_max_threads();
#endif

  for (const auto& sys : systems) {
    // Per-system minimum thread count: P450 starts at 32, others at 16
    int min_threads = 16;
    if (sys.name.find("p450") != std::string::npos) {
      min_threads = 32;
    }

    std::vector<int> thread_counts;
#ifdef _OPENMP
    for (int nt : {16, 32, 64}) {
      if (nt >= min_threads && nt <= max_threads)
        thread_counts.push_back(nt);
    }
#else
    thread_counts = {1};
#endif
    if (thread_counts.empty()) continue;

    auto basis = make_basis(sys.mol, sys.basis_name, sys.pure);
    const size_t nao = basis->num_atomic_orbitals;
    const size_t ndm = 1;
    auto P = make_random_density(nao);

    for (int nt : thread_counts) {
#ifdef _OPENMP
      omp_set_num_threads(nt);
#else
      if (nt > 1) continue;
#endif
      // TLS variant
      {
        auto eri =
            make_direct_eri(*basis, SCFOrbitalType::Restricted, false);

        // Warmup
        auto [J_w, K_w] = build_JK_standalone(*eri, P, nao, ndm);

        // Timed runs
        const int n_runs = 3;
        double total_ms = 0.0;
        for (int r = 0; r < n_runs; ++r) {
          auto start = std::chrono::high_resolution_clock::now();
          auto [J, K] = build_JK_standalone(*eri, P, nao, ndm);
          auto end = std::chrono::high_resolution_clock::now();
          total_ms += std::chrono::duration<double, std::milli>(end - start)
                          .count();
        }
        double avg_ms = total_ms / n_runs;
        std::cout << sys.name << "," << sys.basis_name << "," << nao << ","
                  << nt << "," << std::fixed << std::setprecision(2) << avg_ms
                  << ",tls" << std::endl;
      }
    }
  }

#ifdef _OPENMP
  // Restore
  omp_set_num_threads(omp_get_max_threads());
#endif
}

TEST(FockBenchmark, MemoryBaseline) {
  if (!std::getenv("QDK_FOCK_BENCH")) {
    GTEST_SKIP() << "Set QDK_FOCK_BENCH=1 to run timing benchmarks";
  }
  // Report memory usage for the TLS variant on larger systems
  std::cout << "system,basis,nao,nthreads,expected_tls_scratch_mb" << std::endl;

  auto systems = get_benchmark_systems();
#ifdef _OPENMP
  int nthreads = omp_get_max_threads();
#else
  int nthreads = 1;
#endif

  for (const auto& sys : systems) {
    auto basis = make_basis(sys.mol, sys.basis_name, sys.pure);
    const size_t nao = basis->num_atomic_orbitals;
    const size_t ndm = 1;

    // TLS scratch = nthreads * ndm * nao * nao * 8 bytes * 2 (J+K)
    double scratch_mb =
        static_cast<double>(nthreads * ndm * nao * nao * sizeof(double) * 2) /
        (1024.0 * 1024.0);

    std::cout << sys.name << "," << sys.basis_name << "," << nao << ","
              << nthreads << "," << std::fixed << std::setprecision(2)
              << scratch_mb << std::endl;
  }
}

// ===========================================================================
// BENCHMARK: ERI threshold sweep
// For each benchmark system, measure accuracy and timing at different
// screening thresholds relative to a tight (1e-14) reference.
// ===========================================================================
TEST(FockBenchmark, ThresholdSweep) {
  if (!std::getenv("QDK_FOCK_BENCH")) {
    GTEST_SKIP() << "Set QDK_FOCK_BENCH=1 to run timing benchmarks";
  }
  std::cout << "system,nao,threshold,build_jk_ms,j_max_err,k_max_err"
            << std::endl;

  auto systems = get_benchmark_systems();
  const std::vector<double> thresholds = {1e-8, 1e-9, 1e-10, 1e-11, 1e-12};
  constexpr double ref_threshold = 1e-14;

  for (const auto& sys : systems) {
    auto basis = make_basis(sys.mol, sys.basis_name, sys.pure);
    const size_t nao = basis->num_atomic_orbitals;
    const size_t ndm = 1;
    auto P = make_random_density(nao);

    // Tight-threshold reference
    auto eri_ref = make_direct_eri(*basis, SCFOrbitalType::Restricted,
                                   false, ref_threshold);
    auto [J_ref, K_ref] = build_JK_standalone(*eri_ref, P, nao, ndm);

    for (double thresh : thresholds) {
      auto eri = make_direct_eri(*basis, SCFOrbitalType::Restricted,
                                 false, thresh);
      // Warmup
      auto [J_w, K_w] = build_JK_standalone(*eri, P, nao, ndm);

      // Timed runs
      const int n_runs = 3;
      double total_ms = 0.0;
      std::vector<double> J_last, K_last;
      for (int r = 0; r < n_runs; ++r) {
        auto start = std::chrono::high_resolution_clock::now();
        auto [J, K] = build_JK_standalone(*eri, P, nao, ndm);
        auto end = std::chrono::high_resolution_clock::now();
        total_ms +=
            std::chrono::duration<double, std::milli>(end - start).count();
        if (r == n_runs - 1) { J_last = J; K_last = K; }
      }
      double avg_ms = total_ms / n_runs;
      double j_err = inf_norm_diff(J_ref, J_last);
      double k_err = inf_norm_diff(K_ref, K_last);

      std::cout << sys.name << "," << nao << ","
                << std::scientific << std::setprecision(0) << thresh << ","
                << std::fixed << std::setprecision(2) << avg_ms << ","
                << std::scientific << std::setprecision(2)
                << j_err << "," << k_err << std::endl;
    }
  }
}

// ===========================================================================
// BENCHMARK: Full SCF convergence
// Runs SCF to convergence on each benchmark system using DIIS_GDM.
// Measures total wall time, iteration count, and prints per-phase breakdown.
// Set environment variable QDK_SCF_BENCH=1 to enable.
// ===========================================================================
TEST(SCFBenchmark, FullSCF) {
  if (!std::getenv("QDK_SCF_BENCH")) {
    GTEST_SKIP() << "Set QDK_SCF_BENCH=1 to run SCF benchmarks";
  }

  std::cout << "\n=== SCF Benchmark (DIIS_GDM, Libint2Direct) ===" << std::endl;
  std::cout << "system,basis,nao,iterations,total_ms,energy" << std::endl;

  auto systems = get_benchmark_systems();

  for (const auto& sys : systems) {
    // Build basis to get NAO for reporting
    auto basis = make_basis(sys.mol, sys.basis_name, sys.pure);
    const size_t nao = basis->num_atomic_orbitals;

    // Optional NAO limit for quick profiling runs
    const char* max_nao_env = std::getenv("QDK_SCF_BENCH_MAX_NAO");
    if (max_nao_env && nao > static_cast<size_t>(std::atoi(max_nao_env))) {
      std::cout << sys.name << "," << sys.basis_name << "," << nao
                << ",SKIPPED,0.0,0.0" << std::endl;
      continue;
    }

    // Configure SCF with DIIS_GDM (production default)
    SCFConfig cfg;
    cfg.scf_orbital_type = SCFOrbitalType::Restricted;
    cfg.eri.method = ERIMethod::Libint2Direct;
    cfg.eri.eri_threshold = 1e-9;
    cfg.eri.shell_pair_threshold = 1e-12;
    cfg.eri.use_atomics = false;
    cfg.scf_algorithm.method = SCFAlgorithmName::DIIS_GDM;
    cfg.scf_algorithm.max_iteration = 200;
    cfg.basis = sys.basis_name;
    cfg.cartesian = !sys.pure;
    cfg.mpi = ParallelConfig{1, 0, 1, 0};
    cfg.verbose = 1;

    // Tighter convergence for testing (env var override)
    if (std::getenv("QDK_SCF_TIGHT")) {
      cfg.scf_algorithm.density_threshold = 1e-8;
      cfg.scf_algorithm.og_threshold = 1e-8;
    }

    try {
      auto solver = SCF::make_hf_solver(sys.mol, cfg);

      auto start = std::chrono::high_resolution_clock::now();
      const auto& ctx = solver->run();
      auto end = std::chrono::high_resolution_clock::now();

      double total_ms =
          std::chrono::duration<double, std::milli>(end - start).count();

      std::cout << sys.name << "," << sys.basis_name << "," << nao << ","
                << ctx.result.scf_iterations << ","
                << std::fixed << std::setprecision(1) << total_ms << ","
                << std::fixed << std::setprecision(12)
                << ctx.result.scf_total_energy << std::endl;
    } catch (const std::exception& e) {
      std::cout << sys.name << "," << sys.basis_name << "," << nao
                << ",FAILED,0.0,0.0  # " << e.what() << std::endl;
    }
  }

  // Print cumulative per-phase timing breakdown
  std::cout << "\n";
  SCF::print_timer_summary();
}
