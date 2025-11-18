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
    auto console = world_rank ? spdlog::null_logger_mt("standalone_driver")
                              : spdlog::stdout_color_mt("standalone_driver");

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
      throw std::runtime_error("Reference data format not recognized");
    }

    // Set up job
    std::string job_str = "MCSCF";
    OPT_KEYWORD("CI.JOB", job_str, std::string);
    Job job;
    try {
      job = job_map.at(job_str);
    } catch (...) {
      throw std::runtime_error("Job Not Recognized");
    }

    std::string ciexp_str = "CAS";
    OPT_KEYWORD("CI.EXPANSION", ciexp_str, std::string);
    CIExpansion ci_exp;
    try {
      ci_exp = ci_exp_map.at(ciexp_str);
    } catch (...) {
      throw std::runtime_error("CI Expansion Not Recognized");
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

    // Misc optional files
    std::string rdm_fname, fci_out_fname, ci_wfn_out_fname;
    OPT_KEYWORD("CI.RDMFILE", rdm_fname, std::string);
    OPT_KEYWORD("CI.FCIDUMP_OUT", fci_out_fname, std::string);
    OPT_KEYWORD("CI.WFN_OUT_FILE", ci_wfn_out_fname, std::string);

    if (n_active > nwfn_bits / 2) throw std::runtime_error("Not Enough Bits");

    // MCSCF Settings
    macis::MCSCFSettings mcscf_settings;
    OPT_KEYWORD("MCSCF.MAX_MACRO_ITER", mcscf_settings.max_macro_iter, size_t);
    OPT_KEYWORD("MCSCF.MAX_ORB_STEP", mcscf_settings.max_orbital_step, double);
    OPT_KEYWORD("MCSCF.MCSCF_ORB_TOL", mcscf_settings.orb_grad_tol_mcscf,
                double);
    // OPT_KEYWORD("MCSCF.BFGS_TOL",       mcscf_settings.orb_grad_tol_bfgs,
    // double); OPT_KEYWORD("MCSCF.BFGS_MAX_ITER", mcscf_settings.max_bfgs_iter,
    // size_t);
    OPT_KEYWORD("MCSCF.ENABLE_DIIS", mcscf_settings.enable_diis, bool);
    OPT_KEYWORD("MCSCF.DIIS_START_ITER", mcscf_settings.diis_start_iter,
                size_t);
    OPT_KEYWORD("MCSCF.DIIS_NKEEP", mcscf_settings.diis_nkeep, size_t);
    OPT_KEYWORD("MCSCF.CI_RES_TOL", mcscf_settings.ci_res_tol, double);
    OPT_KEYWORD("MCSCF.CI_MAX_SUB", mcscf_settings.ci_max_subspace, size_t);
    OPT_KEYWORD("MCSCF.CI_MATEL_TOL", mcscf_settings.ci_matel_tol, double);

    // ASCI Settings
    macis::ASCISettings asci_settings;
    std::string asci_wfn_fname, asci_ham_out_fname;
    double asci_E0 = 0.0;
    bool compute_asci_E0 = true, pt2 = false;
    bool compute_asci_E0_diag = false;
    OPT_KEYWORD("ASCI.NTDETS_MAX", asci_settings.ntdets_max, size_t);
    OPT_KEYWORD("ASCI.NTDETS_MIN", asci_settings.ntdets_min, size_t);
    OPT_KEYWORD("ASCI.NCDETS_MAX", asci_settings.ncdets_max, size_t);
    OPT_KEYWORD("ASCI.HAM_EL_TOL", asci_settings.h_el_tol, double);
    OPT_KEYWORD("ASCI.RV_PRUNE_TOL", asci_settings.rv_prune_tol, double);
    OPT_KEYWORD("ASCI.PAIR_MAX_LIM", asci_settings.pair_size_max, size_t);
    OPT_KEYWORD("ASCI.GROW_FACTOR", asci_settings.grow_factor, int);
    OPT_KEYWORD("ASCI.MAX_REFINE_ITER", asci_settings.max_refine_iter, size_t);
    OPT_KEYWORD("ASCI.REFINE_ETOL", asci_settings.refine_energy_tol, double);
    OPT_KEYWORD("ASCI.GROW_WITH_ROT", asci_settings.grow_with_rot, bool);
    OPT_KEYWORD("ASCI.ROT_SIZE_START", asci_settings.rot_size_start, size_t);
    // OPT_KEYWORD("ASCI.DIST_TRIP_RAND",  asci_settings.dist_triplet_random,
    // bool );
    OPT_KEYWORD("ASCI.CONSTRAINT_LVL", asci_settings.constraint_level, int);
    OPT_KEYWORD("ASCI.WFN_FILE", asci_wfn_fname, std::string);
    if (input.containsData("ASCI.E0_WFN")) {
      asci_E0 = input.getData<double>("ASCI.E0_WFN");
      compute_asci_E0 = false;
    } else if (input.containsData("ASCI.E0_WFN_DIAG")) {
      compute_asci_E0 = false;
      compute_asci_E0_diag = true;
    }
    OPT_KEYWORD("ASCI.HAM_OUT_FILE", asci_ham_out_fname, std::string);
    OPT_KEYWORD("ASCI.PT2", pt2, bool);
    OPT_KEYWORD("ASCI.PT2_TOL", asci_settings.pt2_tol, double);
    OPT_KEYWORD("ASCI.PT2_RESERVE_COUNT", asci_settings.pt2_reserve_count,
                size_t);
    OPT_KEYWORD("ASCI.PT2_CONSTRAINT_LVL_MAX",
                asci_settings.pt2_max_constraint_level, int);
    OPT_KEYWORD("ASCI.PT2_CONSTRAINT_LVL_MIN",
                asci_settings.pt2_min_constraint_level, int);
    OPT_KEYWORD("ASCI.PT2_CNSTRNT_RFNE_FORCE",
                asci_settings.pt2_constraint_refine_force, int64_t);
    OPT_KEYWORD("ASCI.PT2_PRUNE", asci_settings.pt2_prune, bool);
    OPT_KEYWORD("ASCI.PT2_PRECOMPUTE_EPS", asci_settings.pt2_precompute_eps,
                bool);
    OPT_KEYWORD("ASCI.PT2_PRECOMPUTE_IDX", asci_settings.pt2_precompute_idx,
                bool);
    OPT_KEYWORD("ASCI.PT2_PRINT_PROGRESS", asci_settings.pt2_print_progress,
                bool);
    OPT_KEYWORD("ASCI.PT2_BIGCON_THRESH", asci_settings.pt2_bigcon_thresh,
                size_t);
    OPT_KEYWORD("ASCI.NXTVAL_BCOUNT_THRESH", asci_settings.nxtval_bcount_thresh,
                size_t);
    OPT_KEYWORD("ASCI.NXTVAL_BCOUNT_INC", asci_settings.nxtval_bcount_inc,
                size_t);

    bool mp2_guess = false;
    OPT_KEYWORD("MCSCF.MP2_GUESS", mp2_guess, bool);

    if (!world_rank) {
      console->info("[Standalone MACIS Driver]:");
      console->info("  * NMPI          = {}", world_size);
      console->info("  * NTHREADS      = {}", omp_get_max_threads());
      console->info("[Wavefunction Data]:");
      console->info("  * JOB           = {}", job_str);
      console->info("  * CIEXP         = {}", ciexp_str);
      console->info("  * REF_FILE_NAME = {}", reference_data_file);
      console->info("  * REF_FILE_FMT  = {}", reference_data_format);
      if (fci_out_fname.size())
        console->info("  * FCIDUMP_OUT = {}", fci_out_fname);
      console->info("  * NORBITAL  = {}", norb);
      console->info("  * NINACTIVE = {}", n_inactive);
      console->info("  * NACTIVE   = {}", n_active);
      console->info("  * NVIRTUAL  = {}", num_virtual_orbitals);
      console->info("  * MP2_GUESS = {}", mp2_guess);

      console->debug("READ {} 1-body integrals and {} 2-body integrals",
                     T.size(), V.size());
      console->info("ECORE = {:.12f}", E_core);
      console->debug("TSUM  = {:.12f}", vec_sum(T));
      console->debug("VSUM  = {:.12f}", vec_sum(V));
      console->info("TMEM   = {:.2e} GiB", macis::to_gib(T));
      console->info("VMEM   = {:.2e} GiB", macis::to_gib(V));
    }

    // Setup printing
    bool print_davidson = false, print_ci = false, print_mcscf = true,
         print_diis = false, print_asci_search = false,
         print_determinants = false;
    double determinants_threshold = 1e-2;
    OPT_KEYWORD("PRINT.DAVIDSON", print_davidson, bool);
    OPT_KEYWORD("PRINT.CI", print_ci, bool);
    OPT_KEYWORD("PRINT.MCSCF", print_mcscf, bool);
    OPT_KEYWORD("PRINT.DIIS", print_diis, bool);
    OPT_KEYWORD("PRINT.ASCI_SEARCH", print_asci_search, bool);
    OPT_KEYWORD("PRINT.DETERMINANTS", print_determinants, bool);
    OPT_KEYWORD("PRINT.DETERMINANTS_THRES", determinants_threshold, double);

    if (world_rank or not print_davidson) spdlog::null_logger_mt("davidson");
    if (world_rank or not print_ci) spdlog::null_logger_mt("ci_solver");
    if (world_rank or not print_mcscf) spdlog::null_logger_mt("mcscf");
    if (world_rank or not print_diis) spdlog::null_logger_mt("diis");
    if (world_rank or not print_asci_search)
      spdlog::null_logger_mt("asci_search");

    // MP2 Guess Orbitals
    if (mp2_guess) {
      if (nalpha != nbeta)
        throw std::runtime_error("MP2 Guess only implemented for closed-shell");

      console->info("Calculating MP2 Natural Orbitals");
      size_t num_occupied_orbitals_canon = n_inactive + nalpha;
      size_t nvir_canon = norb - num_occupied_orbitals_canon;

      // Compute MP2 Natural Orbitals
      std::vector<double> MP2_RDM(norb * norb, 0.0);
      std::vector<double> W_occ(norb);
      macis::mp2_natural_orbitals(
          NumOrbital(norb), NumCanonicalOccupied(num_occupied_orbitals_canon),
          NumCanonicalVirtual(nvir_canon), T.data(), norb, V.data(), norb,
          W_occ.data(), MP2_RDM.data(), norb);

      // Transform Hamiltonian
      macis::two_index_transform(norb, norb, T.data(), norb, MP2_RDM.data(),
                                 norb, T.data(), norb);
      macis::four_index_transform(norb, norb, V.data(), norb, MP2_RDM.data(),
                                  norb, V.data(), norb);
    }

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

    // Storage for active RDMs
    std::vector<double> active_ordm(n_active * n_active);
    std::vector<double> active_trdm(active_ordm.size() * active_ordm.size());

    double E0 = 0;
    double EPT2 = 0;
    // CI
    if (job == Job::CI) {
      using generator_t = macis::SortedDoubleLoopHamiltonianGenerator<wfn_type>;
      std::vector<double> C;
      std::vector<wfn_type> dets;
      if (ci_exp == CIExpansion::CAS) {
        E0 = macis::CASRDMFunctor<generator_t>::rdms(
            mcscf_settings, NumOrbital(n_active), nalpha, nbeta,
            T_active.data(), V_active.data(), nullptr /*active_ordm.data()*/,
            nullptr /*active_trdm.data()*/, C MACIS_MPI_CODE(, MPI_COMM_WORLD));
        E0 += E_inactive + E_core;
        dets = macis::generate_hilbert_space<wfn_type>(n_active, nalpha, nbeta);
      } else {
        // Generate the Hamiltonian Generator
        generator_t ham_gen(
            macis::matrix_span<double>(T_active.data(), n_active, n_active),
            macis::rank4_span<double>(V_active.data(), n_active, n_active,
                                      n_active, n_active));

        if (asci_wfn_fname.size()) {
          // Read wave function from standard file
          console->info("Reading Guess Wavefunction From {}", asci_wfn_fname);
          macis::read_wavefunction(asci_wfn_fname, dets, C);
          {
            double input_norm = 0;
            for (auto& x : C) input_norm += x * x;
            input_norm = std::sqrt(input_norm);
            for (auto& x : C) x /= input_norm;
          }
          // std::cout << dets[0].to_ullong() << std::endl;
          if (compute_asci_E0) {
            console->info("*  Calculating E0");
            E0 = 0;
            for (auto ii = 0; ii < dets.size(); ++ii) {
              double tmp = 0.0;
              for (auto jj = 0; jj < dets.size(); ++jj) {
                tmp += ham_gen.matrix_element(dets[ii], dets[jj]) * C[jj];
              }
              E0 += C[ii] * tmp;
            }
          } else if (compute_asci_E0_diag) {
            std::vector<double>().swap(C);  // Clear out guess
            E0 = macis::selected_ci_diag<int64_t>(
                dets.begin(), dets.end(), ham_gen, mcscf_settings.ci_matel_tol,
                mcscf_settings.ci_max_subspace, mcscf_settings.ci_res_tol,
                C MACIS_MPI_CODE(, MPI_COMM_WORLD));
          } else {
            console->info("*  Reading E0");
            E0 = asci_E0 - E_core - E_inactive;
          }
        } else {
          // HF Guess
          console->info("Generating HF Guess for ASCI");
          dets = {wfn_traits::canonical_hf_determinant(nalpha, nbeta)};
          // std::cout << dets[0].to_ullong() << std::endl;
          E0 = ham_gen.matrix_element(dets[0], dets[0]);
          C = {1.0};
        }
        console->info("ASCI Guess Size = {}", dets.size());
        console->info("ASCI E0 = {:.10e}", E0 + E_core + E_inactive);

        // Perform the ASCI calculation
        auto asci_st = hrt_t::now();

        // Growth phase
        std::tie(E0, dets, C) = macis::asci_grow<nwfn_bits, int64_t>(
            asci_settings, mcscf_settings, E0, std::move(dets), std::move(C),
            ham_gen, n_active MACIS_MPI_CODE(, MPI_COMM_WORLD));

        // Refinement phase
        if (asci_settings.max_refine_iter) {
          std::tie(E0, dets, C) = macis::asci_refine<nwfn_bits, int64_t>(
              asci_settings, mcscf_settings, E0, std::move(dets), std::move(C),
              ham_gen, n_active MACIS_MPI_CODE(, MPI_COMM_WORLD));
        }
        E0 += E_inactive + E_core;
        auto asci_en = hrt_t::now();
        dur_t asci_dur = asci_en - asci_st;
        console->info("* ASCI_DUR = {:.2e} ms", asci_dur.count());

#ifdef MACIS_ENABLE_MPI
        // Dump Hamiltonian
        if (asci_ham_out_fname.size()) {
          std::cout << "ASCI HAM FNAME = " << asci_ham_out_fname << std::endl;
          auto H = macis::make_dist_csr_hamiltonian<int64_t>(
              MPI_COMM_WORLD, dets.begin(), dets.end(), ham_gen, 1e-16);
          sparsexx::write_dist_mm(asci_ham_out_fname, H, 1);
        }
        if (pt2) {
          MPI_Barrier(MPI_COMM_WORLD);
          auto pt2_st = hrt_t::now();
          EPT2 = macis::asci_pt2_constraint(
              asci_settings, dets.begin(), dets.end(),
              E0 - (E_inactive + E_core), C, n_active, ham_gen.T(),
              ham_gen.G_red(), ham_gen.V_red(), ham_gen.G(), ham_gen.V(),
              ham_gen MACIS_MPI_CODE(, MPI_COMM_WORLD));
          MPI_Barrier(MPI_COMM_WORLD);
          auto pt2_en = hrt_t::now();
          dur_t pt2_dur = pt2_en - pt2_st;
          console->info("* ASCI_PT2_DUR = {:.2e} ms", pt2_dur.count());
        }
#endif
      }

      console->info("E(CI)     = {:.12f} Eh", E0);

      if (pt2) {
        console->info("E(PT2)    = {:.12f} Eh", EPT2);
        console->info("E(CI+PT2) = {:.12f} Eh", E0 + EPT2);
      }

      macis::reorder_ci_on_coeff(dets, C);
      console->info("WFN FILE{}", ci_wfn_out_fname);
      if (ci_wfn_out_fname.size() and !world_rank) {
        console->info("Writing ASCI Wavefunction to {}", ci_wfn_out_fname);
        console->info("  * Format TEXT");
        macis::write_wavefunction(ci_wfn_out_fname, n_active, dets, C);
      }

      if (print_determinants) {
        auto det_logger = world_rank ? spdlog::null_logger_mt("determinants")
                                     : spdlog::stdout_color_mt("determinants");
        det_logger->info("Print leading determinants > {:.2e}",
                         determinants_threshold);
        for (size_t i = 0; i < dets.size(); ++i) {
          if (std::abs(C[i]) > determinants_threshold) {
            det_logger->info(
                "{:>16.12f}   {}", C[i],
                macis::to_canonical_string(dets[i]).substr(0, n_active));
          }
        }
      }

      // MCSCF
    } else if (job == Job::MCSCF) {
      // Possibly read active RDMs
      if (rdm_fname.size()) {
        console->info("  * RDMFILE = {}", rdm_fname);
        std::vector<double> full_ordm(norb2), full_trdm(norb4);
        macis::read_rdms_binary(rdm_fname, norb, full_ordm.data(), norb,
                                full_trdm.data(), norb);
        macis::active_submatrix_1body(NumActive(n_active),
                                      NumInactive(n_inactive), full_ordm.data(),
                                      norb, active_ordm.data(), n_active);
        macis::active_subtensor_2body(NumActive(n_active),
                                      NumInactive(n_inactive), full_trdm.data(),
                                      norb, active_trdm.data(), n_active);

        // Compute CI energy from RDMs
        double ERDM = blas::dot(active_ordm.size(), active_ordm.data(), 1,
                                T_active.data(), 1);
        ERDM += blas::dot(active_trdm.size(), active_trdm.data(), 1,
                          V_active.data(), 1);
        console->info("E(RDM)  = {:.12f} Eh", ERDM + E_inactive + E_core);
      }

      // CASSCF
      E0 = macis::casscf_diis(
          mcscf_settings, NumElectron(nalpha), NumElectron(nbeta),
          NumOrbital(norb), NumInactive(n_inactive), NumActive(n_active),
          NumVirtual(num_virtual_orbitals), E_core, T.data(), norb, V.data(),
          norb, active_ordm.data(), n_active, active_trdm.data(),
          n_active MACIS_MPI_CODE(, MPI_COMM_WORLD));

      console->info("E(CASSCF)  = {:.12f} Eh", E0);
    }

    // Write FCIDUMP file if requested
    if (fci_out_fname.size()) {
      macis::FCIDumpHeader header;
      header.norb = norb;
      header.nelec = nalpha + nbeta;
      header.ms2 = std::max(nalpha, nbeta) - std::min(nalpha, nbeta);
      header.isym = 1;                            // Assuming A1 symmetry
      header.orbsym = std::vector<int>(norb, 1);  // All orbitals in A1 symmetry
      macis::write_fcidump(fci_out_fname, header, T.data(), norb, V.data(),
                           norb, E_core);
    }

  }  // MPI Scope

  MACIS_MPI_CODE(MPI_Finalize();)
}
