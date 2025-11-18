/*
 * MACIS Copyright (c) 2023, The Regents of the University of California,
 * through Lawrence Berkeley National Laboratory (subject to receipt of
 * any required approvals from the U.S. Dept. of Energy). All rights reserved.
 * Portions Copyright (c) Microsoft Corporation.
 *
 * See LICENSE.txt for details
 */

#include <spdlog/cfg/env.h>
#include <spdlog/sinks/null_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>
#include <spdlog/stopwatch.h>

#include <filesystem>
#include <iomanip>
#include <iostream>
#include <macis/asci/grow.hpp>
#include <macis/asci/pt2.hpp>
#include <macis/asci/refine.hpp>
#include <macis/hamiltonian_generator/double_loop.hpp>
#include <macis/hamiltonian_generator/sorted_double_loop.hpp>
#include <macis/mcscf/cas.hpp>
#include <macis/mcscf/fock_matrices.hpp>
#include <macis/util/detail/rdm_files.hpp>
#include <macis/util/fcidump.hpp>
#include <macis/util/memory.hpp>
#include <macis/util/moller_plesset.hpp>
#include <macis/util/mpi.hpp>
#include <macis/util/transform.hpp>
#include <macis/wavefunction_io.hpp>
#include <map>
#include <sparsexx/io/write_dist_mm.hpp>

#include "ini_input.hpp"

using macis::NumActive;
using macis::NumCanonicalOccupied;
using macis::NumCanonicalVirtual;
using macis::NumElectron;
using macis::NumInactive;
using macis::NumOrbital;
using macis::NumVirtual;

enum class Job { CI, MCSCF };

enum class CIExpansion { CAS, ASCI };

std::map<std::string, Job> job_map = {{"CI", Job::CI}, {"MCSCF", Job::MCSCF}};

std::map<std::string, CIExpansion> ci_exp_map = {{"CAS", CIExpansion::CAS},
                                                 {"ASCI", CIExpansion::ASCI}};

template <typename T>
T vec_sum(const std::vector<T>& x) {
  return std::accumulate(x.begin(), x.end(), T(0));
}

int main(int argc, char** argv) {
  using hrt_t = std::chrono::high_resolution_clock;
  using dur_t = std::chrono::duration<double, std::milli>;

  std::cout << std::scientific << std::setprecision(12);
  spdlog::cfg::load_env_levels();
  spdlog::set_pattern("[%n] %v");

  constexpr size_t nwfn_bits = 256;
  using wfn_type = macis::wfn_t<nwfn_bits>;
  using wfn_traits = macis::wavefunction_traits<wfn_type>;

  MACIS_MPI_CODE(int dummy;
                 MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &dummy);
                 if (dummy != MPI_THREAD_MULTIPLE) throw std::runtime_error(
                     "MPI Thread Init Failed");)

#ifdef MACIS_ENABLE_MPI
  auto world_rank = macis::comm_rank(MPI_COMM_WORLD);
  auto world_size = macis::comm_size(MPI_COMM_WORLD);
#else
  int world_rank = 0;
  int world_size = 1;
#endif
  {
    // Create Logger
    auto console = world_rank ? spdlog::null_logger_mt("auto_sci")
                              : spdlog::stdout_color_mt("auto_sci");

    // Read Input Options
    std::vector<std::string> opts(argc);
    for (int i = 0; i < argc; ++i) opts[i] = argv[i];

    auto input_file = opts.at(1);
    INIFile input(input_file);

#define OPT_KEYWORD(STR, RES, DTYPE) \
  if (input.containsData(STR)) {     \
    RES = input.getData<DTYPE>(STR); \
  }

    // Required Keywords
    auto nalpha = input.getData<size_t>("CI.NALPHA");
    auto nbeta = input.getData<size_t>("CI.NBETA");

    std::string reference_data_format =
        input.getData<std::string>("CI.REF_DATA_FORMAT");
    std::string reference_data_file =
        input.getData<std::string>("CI.REF_DATA_FILE");

    if (!std::filesystem::exists(reference_data_file)) {
      throw std::runtime_error("Reference data file does not exist: " +
                               reference_data_file);
    }

    size_t norb, norb2, norb3, norb4;
    std::vector<double> T, V;
    double E_core;
    if (reference_data_format == "FCIDUMP") {
      // Read FCIDUMP File
      norb = macis::read_fcidump_norb(reference_data_file);
      norb2 = norb * norb;
      norb3 = norb2 * norb;
      norb4 = norb2 * norb2;

      // XXX: Consider reading this into shared memory to avoid replication
      T.resize(norb2);
      V.resize(norb4);
      E_core = macis::read_fcidump_core(reference_data_file);
      macis::read_fcidump_1body(reference_data_file, T.data(), norb);
      macis::read_fcidump_2body(reference_data_file, V.data(), norb);
    } else {
      throw std::runtime_error("Unsupported reference data format: " +
                               reference_data_format);
    }

    // Set up active space
    size_t n_inactive = 0;
    OPT_KEYWORD("CI.NINACTIVE", n_inactive, size_t);

    if (n_inactive >= norb) throw std::runtime_error("NINACTIVE >= NORB");

    size_t n_active = norb - n_inactive;
    OPT_KEYWORD("CI.NACTIVE", n_active, size_t);

    if (n_inactive + n_active > norb)
      throw std::runtime_error("NINACTIVE + NACTIVE > NORB");

    size_t num_virtual_orbitals = norb - n_active - n_inactive;

    if (n_active > nwfn_bits / 2) throw std::runtime_error("Not Enough Bits");

    // MCSCF Settings
    macis::MCSCFSettings mcscf_settings;
    OPT_KEYWORD("SOLVER.CI_RES_TOL", mcscf_settings.ci_res_tol, double);
    OPT_KEYWORD("SOLVER.CI_MAX_SUB", mcscf_settings.ci_max_subspace, size_t);
    OPT_KEYWORD("SOLVER.CI_MATEL_TOL", mcscf_settings.ci_matel_tol, double);

    // ASCI Settings
    macis::ASCISettings asci_settings;

    if (!world_rank) {
      console->info("[Auto SCI Driver]:");
      console->info("  * NMPI          = {}", world_size);
      console->info("  * NTHREADS      = {}", omp_get_max_threads());
      console->info("[Wavefunction Data]:");
      console->info("  * REF_FILE_NAME = {}", reference_data_file);
      console->info("  * REF_FILE_FMT  = {}", reference_data_format);
      console->info("  * NORBITAL  = {}", norb);
      console->info("  * NINACTIVE = {}", n_inactive);
      console->info("  * NACTIVE   = {}", n_active);
      console->info("  * NVIRTUAL  = {}", num_virtual_orbitals);

      console->debug("READ {} 1-body integrals and {} 2-body integrals",
                     T.size(), V.size());
      console->info("ECORE = {:.12f}", E_core);
      console->debug("TSUM  = {:.12f}", vec_sum(T));
      console->debug("VSUM  = {:.12f}", vec_sum(V));
      console->info("TMEM   = {:.2e} GiB", macis::to_gib(T));
      console->info("VMEM   = {:.2e} GiB", macis::to_gib(V));
    }

    // Setup printing
    if (world_rank) spdlog::null_logger_mt("davidson");
    if (world_rank) spdlog::null_logger_mt("ci_solver");
    if (world_rank) spdlog::null_logger_mt("mcscf");
    if (world_rank) spdlog::null_logger_mt("diis");
    if (world_rank) spdlog::null_logger_mt("asci_search");

    // Copy integrals into active subsets
    std::vector<double> T_active(n_active * n_active);
    std::vector<double> V_active(n_active * n_active * n_active * n_active);

    // Compute active-space Hamiltonian and inactive Fock matrix
    std::vector<double> F_inactive(norb2);
    macis::active_hamiltonian(NumOrbital(norb), NumActive(n_active),
                              NumInactive(n_inactive), T.data(), norb, V.data(),
                              norb, F_inactive.data(), norb, T_active.data(),
                              n_active, V_active.data(), n_active);

    console->debug("FINACTIVE_SUM = {:.12f}", vec_sum(F_inactive));
    console->debug("VACTIVE_SUM   = {:.12f}", vec_sum(V_active));
    console->debug("TACTIVE_SUM   = {:.12f}", vec_sum(T_active));

    // Compute Inactive energy
    auto E_inactive = macis::inactive_energy(NumInactive(n_inactive), T.data(),
                                             norb, F_inactive.data(), norb);
    console->info("E(inactive) = {:.12f}", E_inactive);

    // CASCI
    using generator_t = macis::SortedDoubleLoopHamiltonianGenerator<wfn_type>;
    std::vector<double> C_casci;
    double E_casci = macis::CASRDMFunctor<generator_t>::rdms(
        mcscf_settings, NumOrbital(n_active), nalpha, nbeta, T_active.data(),
        V_active.data(), nullptr, nullptr,
        C_casci MACIS_MPI_CODE(, MPI_COMM_WORLD));
    std::vector<wfn_type> dets_casci =
        macis::generate_hilbert_space<wfn_type>(n_active, nalpha, nbeta);

    console->info("CASCI Energy = {:.12f}", E_casci);

    // Sort the CASCI wfn on coefficients
    macis::reorder_ci_on_coeff(dets_casci, C_casci);

    // Write CASCI wavefunction to file
    std::string wfn_out_file_casci;
    OPT_KEYWORD("AUTO_SCI.WFN_OUT_FILE_CASCI", wfn_out_file_casci, std::string);
    if (!wfn_out_file_casci.empty() and !world_rank) {
      console->info("Writing CASCI Wavefunction to {}", wfn_out_file_casci);
      macis::write_wavefunction(wfn_out_file_casci, n_active, dets_casci,
                                C_casci);
    }

    // Determine the determinants needed to achieve chemical accuracy
    const double energy_threshold = 1e-3;
    const size_t ndets = dets_casci.size();
    generator_t ham_gen(
        macis::matrix_span<double>(T_active.data(), n_active, n_active),
        macis::rank4_span<double>(V_active.data(), n_active, n_active, n_active,
                                  n_active));

    // Bi-section search for minimal k to achieve chemical accuracy
    size_t low = 2, high = ndets, best_k = ndets;
    std::vector<wfn_type> dets;
    std::vector<double> C_sci;
    double E_sci = 0.0;
    size_t ii = 0;
    while (low < high) {
      size_t mid = (low + high) / 2;
      dets.assign(dets_casci.begin(), dets_casci.begin() + mid);
      C_sci.clear();
      E_sci = macis::selected_ci_diag<int64_t>(
          dets.begin(), dets.end(), ham_gen, mcscf_settings.ci_matel_tol,
          mcscf_settings.ci_max_subspace, mcscf_settings.ci_res_tol,
          C_sci MACIS_MPI_CODE(, MPI_COMM_WORLD));
      console->info(
          "Search iteration {}: k=({}:{}) SCI(k={:13}) Energy = {:.12e} Error "
          "= {:.4e}",
          ii, low, high, mid, E_sci, std::abs(E_sci - E_casci));
      if (std::abs(E_sci - E_casci) < energy_threshold) {
        // Midpoint is good enough to reach chemical accuracy, so set it as the
        // high limit in the range
        best_k = mid;
        high = mid;
      } else {
        // Midpoint is too small, so set it as the low limit in the range
        low = mid + 1;
      }
      ++ii;
    }
    // Linear scan over the final interval to guarantee minimal k
    size_t minimal_k = best_k;
    double minimal_E = E_sci;
    for (size_t k = low; k <= best_k; ++k) {
      dets.assign(dets_casci.begin(), dets_casci.begin() + k);
      C_sci.clear();
      double E_test = macis::selected_ci_diag<int64_t>(
          dets.begin(), dets.end(), ham_gen, mcscf_settings.ci_matel_tol,
          mcscf_settings.ci_max_subspace, mcscf_settings.ci_res_tol,
          C_sci MACIS_MPI_CODE(, MPI_COMM_WORLD));
      double err = std::abs(E_test - E_casci);
      console->info("Final scan: SCI(k={:13}) Energy = {:.12e} Error = {:.4e}",
                    k, E_test, err);
      if (err < energy_threshold) {
        minimal_k = k;
        minimal_E = E_test;
        break;
      }
    }
    best_k = minimal_k;
    E_sci = minimal_E;

    console->info("Found {} determinants to achieve chemical accuracy", best_k);
    const auto p = std::pow(C_casci[best_k - 1], 2);
    console->info("E(sci) = {:.12e} MIN_P = {:.4e} SHOT_COUNT = {}", E_sci, p,
                  std::ceil(1. / p));

    // Write the SCI wavefunction to file
    std::string wfn_out_file_sci;
    OPT_KEYWORD("AUTO_SCI.WFN_OUT_FILE_SCI", wfn_out_file_sci, std::string);
    if (!wfn_out_file_sci.empty() and !world_rank) {
      // Use the last computed dets and C_sci from the loop
      console->info("Writing SCI Wavefunction to {}", wfn_out_file_sci);
      macis::write_wavefunction(wfn_out_file_sci, n_active, dets, C_sci);
    }

  }  // MPI Scope

  MACIS_MPI_CODE(MPI_Finalize();)
}
