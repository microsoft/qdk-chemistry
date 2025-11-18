/*
 * MACIS Copyright (c) 2023, The Regents of the University of California,
 * through Lawrence Berkeley National Laboratory (subject to receipt of
 * any required approvals from the U.S. Dept. of Energy). All rights reserved.
 * Portions Copyright (c) Microsoft Corporation.
 *
 * See LICENSE.txt for details
 */

#pragma once
#include <macis/asci/alpha_constraint.hpp>
#include <macis/sd_operations.hpp>
#include <macis/types.hpp>
#include <macis/util/mpi.hpp>
#include <variant>

namespace macis {

/**
 * @brief Check if a determinant satisfies a given constraint
 *
 * @tparam WfnType Wavefunction/determinant type
 * @tparam ConType Constraint type
 * @param[in] det Quantum determinant to check
 * @param[in] C Constraint object to test against
 * @return true if determinant satisfies the constraint, false otherwise
 */
template <typename WfnType, typename ConType>
bool satisfies_constraint(WfnType det, ConType C) {
  return C.satisfies_constraint(det);
}

/**
 * @brief Generate single excitation patterns from a determinant within
 * constraint bounds
 *
 * This function generates valid single excitation patterns from a given
 * determinant which satisfy the provided constraint. It returns pairs of
 * occupied and virtual orbital masks that can be used to create single
 * excitations.
 *
 * @tparam WfnType Wavefunction/determinant type
 * @tparam ConType Constraint type
 * @param[in] det Source quantum determinant
 * @param[in] constraint Constraint defining allowed excitation patterns
 * @return Pair of (occupied mask, virtual mask) for valid single excitations
 */
template <typename WfnType, typename ConType>
auto generate_constraint_single_excitations(WfnType det, ConType constraint) {
  using constraint_traits = typename ConType::constraint_traits;
  const auto C = constraint.C();
  const auto B = constraint.B();

  // need to have at most one different from the constraint
  if (constraint.overlap(det) < (constraint.count() - 1))
    return std::make_pair(WfnType(0), WfnType(0));

  auto o = constraint.symmetric_difference(det);
  auto v = constraint.b_mask_union(~det);

  if (constraint_traits::count(o & C) == 1) {
    v = o & C;
    o ^= v;
  }

  const auto o_and_not_b = o & ~B;
  const auto o_and_not_b_count = constraint_traits::count(o_and_not_b);
  if (o_and_not_b_count > 1) return std::make_pair(WfnType(0), WfnType(0));

  if (o_and_not_b_count == 1) o = o_and_not_b;

  return std::make_pair(o, v);
}

/**
 * @brief Generate double excitation patterns from a determinant within
 * constraint bounds
 *
 * This function generates valid double excitation patterns from a given
 * determinant which satisfy the provided constraint. It returns vectors of
 * occupied and virtual orbital pairs that can be used to create double
 * excitations efficiently.
 *
 * @tparam WfnType Wavefunction/determinant type
 * @tparam ConType Constraint type
 * @param[in] det Source quantum determinant
 * @param[in] constraint Constraint defining allowed excitation patterns
 * @return Tuple of (occupied pairs vector, virtual pairs vector) for valid
 * double excitations
 */
template <typename WfnType, typename ConType>
auto generate_constraint_double_excitations(WfnType det, ConType constraint) {
  using constraint_traits = typename ConType::constraint_traits;
  using constraint_type = typename ConType::constraint_type;
  const auto C = constraint.C();
  const auto B = constraint.B();
  // Occ/Vir pairs to generate excitations
  std::vector<constraint_type> O, V;

  if (constraint.overlap(det) == 0) return std::make_tuple(O, V);

  auto o = constraint.symmetric_difference(det);
  auto v = constraint.b_mask_union(~det);

  auto o_and_c = o & C;
  auto o_and_c_count = constraint_traits::count(o_and_c);
  if (o_and_c_count >= 3) return std::make_tuple(O, V);

  // Generate Virtual Pairs
  if (o_and_c_count == 2) {
    v = o_and_c;
    o ^= v;
    // Regenerate since o changed
    // XXX: This apparently is not needed, but leaving because <shrug>
    o_and_c = o & C;
    o_and_c_count = constraint_traits::count(o_and_c);
  }

  const auto virt_ind = bits_to_indices(v);
  switch (o_and_c_count) {
    case 1:
      for (auto a : virt_ind) {
        V.emplace_back(constraint_traits::create_no_check(o_and_c, a));
      }
      o ^= o_and_c;
      break;
    default:
      generate_pairs(virt_ind, V);
      break;
  }

  // Generate Occupied Pairs
  const auto o_and_not_b = o & ~B;
  const auto o_and_not_b_count = constraint_traits::count(o_and_not_b);
  if (o_and_not_b_count > 2) return std::make_tuple(O, V);

  switch (o_and_not_b_count) {
    case 1:
      for (auto i : bits_to_indices(o & B)) {
        O.emplace_back(constraint_traits::create_no_check(o_and_not_b, i));
      }
      break;
    default:
      if (o_and_not_b_count == 2) o = o_and_not_b;
      generate_pairs(bits_to_indices(o), O);
      break;
  }

  return std::make_tuple(O, V);
}

/**
 * @brief Generate all valid single excitations from a determinant within
 * constraint bounds
 *
 * This function creates all possible single excitations from the given
 * determinant that satisfy the provided constraint. The resulting excited
 * determinants are stored in the output vector.
 *
 * @tparam WfnType Wavefunction/determinant type
 * @tparam ConType Constraint type
 * @param[in] det Source quantum determinant for excitations
 * @param[in] constraint Constraint defining allowed excitation patterns
 * @param[out] t_singles Vector to store the generated single excitations
 */
template <typename WfnType, typename ConType>
void generate_constraint_singles(WfnType det, ConType constraint,
                                 std::vector<WfnType>& t_singles) {
  using constraint_traits = typename ConType::constraint_traits;
  auto [o, v] = generate_constraint_single_excitations(det, constraint);
  const auto oc = constraint_traits::count(o);
  const auto vc = constraint_traits::count(v);
  if (!oc or !vc) return;

  t_singles.clear();
  t_singles.reserve(oc * vc);
  const auto occ = constraint_traits::state_to_occ(o);
  const auto vir = constraint_traits::state_to_occ(v);
  for (auto i : occ) {
    auto temp = constraint_traits::create_no_check(det, i);
    for (auto a : vir)
      t_singles.emplace_back(constraint_traits::create_no_check(temp, a));
  }
}

/**
 * @brief Count the number of valid single excitations from a determinant within
 * constraint bounds
 *
 * This function efficiently counts how many single excitations can be generated
 * from the given determinant while satisfying the provided constraint, without
 * actually generating the excitations.
 *
 * @tparam WfnType Wavefunction/determinant type
 * @tparam ConType Constraint type
 * @param[in] det Source quantum determinant
 * @param[in] constraint Constraint defining allowed excitation patterns
 * @return Number of valid single excitations
 */
template <typename WfnType, typename ConType>
unsigned count_constraint_singles(WfnType det, ConType constraint) {
  using constraint_traits = typename ConType::constraint_traits;
  auto [o, v] = generate_constraint_single_excitations(det, constraint);
  const auto oc = constraint_traits::count(o);
  const auto vc = constraint_traits::count(v);
  return oc * vc;
}

/**
 * @brief Generate all valid double excitations from a determinant within
 * constraint bounds
 *
 * This function creates all possible double excitations from the given
 * determinant that satisfy the provided constraint. The resulting excited
 * determinants are stored in the output vector.
 *
 * @tparam WfnType Wavefunction/determinant type
 * @tparam ConType Constraint type
 * @param[in] det Source quantum determinant for excitations
 * @param[in] constraint Constraint defining allowed excitation patterns
 * @param[out] t_doubles Vector to store the generated double excitations
 */
template <typename WfnType, typename ConType>
void generate_constraint_doubles(WfnType det, ConType constraint,
                                 std::vector<WfnType>& t_doubles) {
  auto [O, V] = generate_constraint_double_excitations(det, constraint);

  t_doubles.clear();
  for (auto ij : O) {
    const auto temp = det ^ ij;
    for (auto ab : V) {
      t_doubles.emplace_back(temp | ab);
    }
  }
}

/**
 * @brief Count the number of valid double excitations from a determinant within
 * constraint bounds
 *
 * This function efficiently counts how many double excitations can be generated
 * from the given determinant while satisfying the provided constraint, without
 * actually generating the excitations.
 *
 * @tparam WfnType Wavefunction/determinant type
 * @tparam ConType Constraint type
 * @param[in] det Source quantum determinant
 * @param[in] constraint Constraint defining allowed excitation patterns
 * @return Number of valid double excitations
 */
template <typename WfnType, typename ConType>
unsigned count_constraint_doubles(WfnType det, ConType constraint) {
  using constraint_traits = typename ConType::constraint_traits;
  const auto C = constraint.C();
  const auto B = constraint.B();
  if (constraint.overlap(det) == 0) return 0;

  auto o = constraint.symmetric_difference(det);
  auto v = constraint.b_mask_union(~det);

  auto o_and_c = o & C;
  auto o_and_c_count = constraint_traits::count(o_and_c);
  if (o_and_c_count >= 3) return 0;

  // Generate Virtual Pairs
  if (o_and_c_count == 2) {
    v = o_and_c;
    o ^= v;
    // Regenerate since o changed
    // XXX: This apparently is not needed, but leaving because <shrug>
    o_and_c = o & C;
    o_and_c_count = constraint_traits::count(o_and_c);
  }

  unsigned nv_pairs = constraint_traits::count(v);
  switch (o_and_c_count) {
    case 1:
      o ^= o_and_c;
      break;
    default:
      nv_pairs = (nv_pairs * (nv_pairs - 1)) / 2;
      break;
  }

  // Generate Occupied Pairs
  const auto o_and_not_b = o & ~B;
  const auto o_and_not_b_count = constraint_traits::count(o_and_not_b);
  if (o_and_not_b_count > 2) return 0;

  unsigned no_pairs = 0;
  switch (o_and_not_b_count) {
    case 1:
      no_pairs = constraint_traits::count(o & B);
      break;
    default:
      if (o_and_not_b_count == 2) o = o_and_not_b;
      no_pairs = constraint_traits::count(o);
      no_pairs = (no_pairs * (no_pairs - 1)) / 2;
      break;
  }

  return no_pairs * nv_pairs;
}

/**
 * @brief Calculate the total number of determinants generated by
 * constraint-based excitations
 *
 * This function computes a histogram count of all possible excitations (single
 * and double, same-spin and opposite-spin) that can be generated from a given
 * determinant while satisfying the provided constraint. This is useful for
 * memory allocation and load balancing in parallel ASCI calculations.
 *
 * @tparam WfnType Wavefunction/determinant type
 * @tparam ConType Constraint type
 * @param[in] det Source quantum determinant
 * @param[in] n_os_singles Number of opposite-spin single excitations
 * @param[in] n_os_doubles Number of opposite-spin double excitations
 * @param[in] constraint Constraint defining allowed excitation patterns
 * @return Total number of determinants that can be generated
 */
template <typename WfnType, typename ConType>
size_t constraint_histogram(WfnType det, size_t n_os_singles,
                            size_t n_os_doubles, ConType constraint) {
  auto ns = count_constraint_singles(det, constraint);
  auto nd = count_constraint_doubles(det, constraint);

  size_t ndet = 0;
  ndet += ns;                 // AA
  ndet += nd;                 // AAAA
  ndet += ns * n_os_singles;  // AABB
  if (satisfies_constraint(det, constraint)) {
    ndet += n_os_singles + n_os_doubles + 1;  // BB + BBBB + No Excitations
  }

  return ndet;
}

/**
 * @brief Generate same-spin single excitation contributions using
 * constraint-based approach
 *
 * This function generates ASCI contributions for same-spin single excitations
 * from a determinant that satisfies the given constraint. It computes matrix
 * elements and stores the contributions for later selection in the ASCI
 * algorithm.
 *
 * @tparam WfnType Wavefunction/determinant type
 * @tparam ConType Constraint type
 * @param[in] coeff Coefficient of the reference determinant
 * @param[in] det Source quantum determinant
 * @param[in] constraint Constraint defining allowed excitation patterns
 * @param[in] occ_same Occupied orbital indices for the same spin
 * @param[in] occ_othr Occupied orbital indices for the opposite spin
 * @param[in] eps Orbital energies for the same spin
 * @param[in] T_pq One-electron integral matrix
 * @param[in] LDT Leading dimension of T_pq matrix
 * @param[in] G_kpq Same-spin two-electron integral tensor
 * @param[in] LDG Leading dimension of G_kpq tensor
 * @param[in] V_kpq Opposite-spin two-electron integral tensor
 * @param[in] LDV Leading dimension of V_kpq tensor
 * @param[in] h_el_tol Threshold for matrix element magnitude
 * @param[in] root_diag Root diagonal correction term
 * @param[in] E0 Reference energy
 * @param[in] ham_gen Hamiltonian generator for fast diagonal evaluation
 * @param[in,out] asci_contributions Container to store the computed
 * contributions
 */
template <typename WfnType, typename ConType>
void generate_constraint_singles_contributions_ss(
    double coeff, WfnType det, ConType constraint,
    const std::vector<uint32_t>& occ_same,
    const std::vector<uint32_t>& occ_othr, const double* eps,
    const double* T_pq, const size_t LDT, const double* G_kpq, const size_t LDG,
    const double* V_kpq, const size_t LDV, double h_el_tol, double root_diag,
    double E0, HamiltonianGeneratorBase<double>& ham_gen,
    asci_contrib_container<WfnType>& asci_contributions) {
  using wfn_traits = wavefunction_traits<WfnType>;
  using constraint_traits = typename ConType::constraint_traits;
  auto [o, v] = generate_constraint_single_excitations(
      wfn_traits::alpha_string(det), constraint);
  const auto no = constraint_traits::count(o);
  const auto nv = constraint_traits::count(v);
  if (!no or !nv) return;

  const size_t LDG2 = LDG * LDG;
  const size_t LDV2 = LDV * LDV;
  for (int ii = 0; ii < no; ++ii) {
    const auto i = fls(o);
    o = constraint_traits::create_no_check(o, i);  // o.flip(i)
    auto v_cpy = v;
    for (int aa = 0; aa < nv; ++aa) {
      const auto a = fls(v_cpy);
      v_cpy = constraint_traits::create_no_check(v_cpy, a);  // v_cpy.flip(a)

      double h_el = T_pq[a + i * LDT];
      const double* G_ov = G_kpq + a * LDG + i * LDG2;
      const double* V_ov = V_kpq + a * LDV + i * LDV2;
      for (auto p : occ_same) h_el += G_ov[p];
      for (auto p : occ_othr) h_el += V_ov[p];

      // Early Exit
      if (std::abs(coeff * h_el) < h_el_tol) continue;

      // Calculate Excited Determinant
      auto ex_det =
          wfn_traits::template single_excitation_no_check<Spin::Alpha>(det, i,
                                                                       a);

      // Compute Sign in a Canonical Way
      auto sign = single_excitation_sign(det, a, i);
      h_el *= sign;

      // Compute Fast Diagonal Matrix Element
      auto h_diag = ham_gen.fast_diag_single(eps[i], eps[a], i, a, root_diag);
      // h_el /= (E0 - h_diag);

      asci_contributions.push_back({ex_det, coeff * h_el, E0 - h_diag});
    }
  }
}

/**
 * @brief Generate same-spin double excitation contributions using
 * constraint-based approach
 *
 * This function generates ASCI contributions for same-spin double excitations
 * from a determinant that satisfies the given constraint. It computes matrix
 * elements for two-electron excitations within the same spin manifold and
 * stores the contributions for later selection in the ASCI algorithm.
 *
 * @tparam WfnType Wavefunction/determinant type
 * @tparam ConType Constraint type
 * @param[in] coeff Coefficient of the reference determinant
 * @param[in] det Source quantum determinant
 * @param[in] constraint Constraint defining allowed excitation patterns
 * @param[in] occ_same Occupied orbital indices for the same spin
 * @param[in] occ_othr Occupied orbital indices for the opposite spin
 * @param[in] eps Orbital energies for the same spin
 * @param[in] G Same-spin two-electron integral tensor
 * @param[in] LDG Leading dimension of G tensor
 * @param[in] h_el_tol Threshold for matrix element magnitude
 * @param[in] root_diag Root diagonal correction term
 * @param[in] E0 Reference energy
 * @param[in] ham_gen Hamiltonian generator for fast diagonal evaluation
 * @param[in,out] asci_contributions Container to store the computed
 * contributions
 */
template <typename WfnType, typename ConType>
void generate_constraint_doubles_contributions_ss(
    double coeff, WfnType det, ConType constraint,
    const std::vector<uint32_t>& occ_same,
    const std::vector<uint32_t>& occ_othr, const double* eps, const double* G,
    const size_t LDG, double h_el_tol, double root_diag, double E0,
    HamiltonianGeneratorBase<double>& ham_gen,
    asci_contrib_container<WfnType>& asci_contributions) {
  using wfn_traits = wavefunction_traits<WfnType>;
  using spin_wfn_traits = wavefunction_traits<spin_wfn_t<WfnType>>;
  auto [O, V] = generate_constraint_double_excitations(
      wfn_traits::alpha_string(det), constraint);
  const auto no_pairs = O.size();
  const auto nv_pairs = V.size();
  if (!no_pairs or !nv_pairs) return;

  const size_t LDG2 = LDG * LDG;
  for (int _ij = 0; _ij < no_pairs; ++_ij) {
    const auto ij = O[_ij];
    const auto i = ffs(ij) - 1;
    const auto j = fls(ij);
    const auto G_ij = G + (j + i * LDG2) * LDG;
    const auto ex_ij =
        wfn_traits::template single_excitation_no_check<Spin::Alpha>(
            det, i, j);  // det ^ ij;
    for (int _ab = 0; _ab < nv_pairs; ++_ab) {
      const auto ab = V[_ab];
      const auto a = ffs(ab) - 1;
      const auto b = fls(ab);

      const auto G_aibj = G_ij[b + a * LDG2];

      // Early Exit
      if (std::abs(coeff * G_aibj) < h_el_tol) continue;

      // Calculate Excited Determinant (spin)
      const auto full_ex_spin =
          spin_wfn_traits::template single_excitation_no_check<Spin::Alpha>(
              ij, a, b);  // ij | ab;
      const auto ex_det_spin =
          wfn_traits::template single_excitation_no_check<Spin::Alpha>(
              ex_ij, a, b);  // ex_ij | ab;

      // Compute Sign in a Canonical Way
      auto sign =
          doubles_sign(wfn_traits::alpha_string(det),
                       wfn_traits::alpha_string(ex_det_spin), full_ex_spin);

      // Calculate Full Excited Determinant
      const auto full_ex = ex_det_spin;  // | os_det;

      // Update Sign of Matrix Element
      auto h_el = sign * G_aibj;

      // Evaluate fast diagonal matrix element
      auto h_diag = ham_gen.fast_diag_ss_double(eps[i], eps[j], eps[a], eps[b],
                                                i, j, a, b, root_diag);
      // h_el /= (E0 - h_diag);

      asci_contributions.push_back({full_ex, coeff * h_el, E0 - h_diag});
    }
  }
}

/**
 * @brief Generate opposite-spin double excitation contributions using
 * constraint-based approach
 *
 * This function generates ASCI contributions for opposite-spin (alpha-beta)
 * double excitations from a determinant that satisfies the given constraint. It
 * computes matrix elements for mixed-spin two-electron excitations and stores
 * the contributions for later selection in the ASCI algorithm.
 *
 * @tparam WfnType Wavefunction/determinant type
 * @tparam ConType Constraint type
 * @param[in] coeff Coefficient of the reference determinant
 * @param[in] det Source quantum determinant
 * @param[in] constraint Constraint defining allowed excitation patterns
 * @param[in] occ_same Occupied orbital indices for the same spin (alpha)
 * @param[in] occ_othr Occupied orbital indices for the opposite spin (beta)
 * @param[in] vir_othr Virtual orbital indices for the opposite spin (beta)
 * @param[in] eps_same Orbital energies for the same spin (alpha)
 * @param[in] eps_othr Orbital energies for the opposite spin (beta)
 * @param[in] V Opposite-spin two-electron integral tensor
 * @param[in] LDV Leading dimension of V tensor
 * @param[in] h_el_tol Threshold for matrix element magnitude
 * @param[in] root_diag Root diagonal correction term
 * @param[in] E0 Reference energy
 * @param[in] ham_gen Hamiltonian generator for fast diagonal evaluation
 * @param[in,out] asci_contributions Container to store the computed
 * contributions
 */
template <typename WfnType, typename ConType>
void generate_constraint_doubles_contributions_os(
    double coeff, WfnType det, ConType constraint,
    const std::vector<uint32_t>& occ_same,
    const std::vector<uint32_t>& occ_othr,
    const std::vector<uint32_t>& vir_othr, const double* eps_same,
    const double* eps_othr, const double* V, const size_t LDV, double h_el_tol,
    double root_diag, double E0, HamiltonianGeneratorBase<double>& ham_gen,
    asci_contrib_container<WfnType>& asci_contributions) {
  using wfn_traits = wavefunction_traits<WfnType>;
  using constraint_traits = typename ConType::constraint_traits;
  // Generate Single Excitations that Satisfy the Constraint
  auto [o, v] = generate_constraint_single_excitations(
      wfn_traits::alpha_string(det), constraint);
  const auto no = constraint_traits::count(o);
  const auto nv = constraint_traits::count(v);
  if (!no or !nv) return;

  const size_t LDV2 = LDV * LDV;
  for (int ii = 0; ii < no; ++ii) {
    const auto i = fls(o);
    o = constraint_traits::create_no_check(o, i);  // o.flip(i)
    auto v_cpy = v;
    for (int aa = 0; aa < nv; ++aa) {
      const auto a = fls(v_cpy);
      v_cpy = constraint_traits::create_no_check(v_cpy, a);  // v_cpy.flip(a)

      const auto* V_ai = V + a + i * LDV;
      double sign_same = single_excitation_sign(det, a, i);

      for (auto j : occ_othr)
        for (auto b : vir_othr) {
          const auto jb = b + j * LDV;
          const auto V_aibj = V_ai[jb * LDV2];

          // Early Exist
          if (std::abs(coeff * V_aibj) < h_el_tol) continue;

          // double sign_othr = single_excitation_sign( os_det >> (N/2),  b, j
          // );
          double sign_othr =
              single_excitation_sign(wfn_traits::beta_string(det), b, j);
          double sign = sign_same * sign_othr;

          // Compute Excited Determinant
          // auto ex_det = det | os_det;
          // ex_det.flip(i).flip(a).flip(j + N / 2).flip(b + N / 2);
          auto ex_det =
              wfn_traits::template single_excitation_no_check<Spin::Alpha>(
                  det, i, a);
          ex_det = wfn_traits::template single_excitation_no_check<Spin::Beta>(
              ex_det, j, b);

          // Finalize Matrix Element
          auto h_el = sign * V_aibj;

          auto h_diag =
              ham_gen.fast_diag_os_double(eps_same[i], eps_othr[j], eps_same[a],
                                          eps_othr[b], i, j, a, b, root_diag);
          // h_el /= (E0 - h_diag);

          asci_contributions.push_back({ex_det, coeff * h_el, E0 - h_diag});
        }  // BJ

    }  // A
  }  // I
}

/**
 * @brief Create a triplet constraint for alpha orbital indices
 *
 * This function constructs a triplet constraint object that restricts
 * excitations to involve at most three specific alpha orbital indices. Triplet
 * constraints are used in ASCI to systematically generate excitation spaces
 * while maintaining computational tractability.
 *
 * @tparam N Size of the wavefunction bitset
 * @param[in] i First alpha orbital index in the triplet
 * @param[in] j Second alpha orbital index in the triplet
 * @param[in] k Third alpha orbital index in the triplet
 * @return Alpha constraint object representing the triplet constraint
 */
template <size_t N>
auto make_triplet(unsigned i, unsigned j, unsigned k) {
  using wfn_type = wfn_t<N>;
  using wfn_traits = wavefunction_traits<wfn_type>;
  using constraint_type = alpha_constraint<wfn_traits>;
  using string_type = typename constraint_type::constraint_type;

  string_type C = 0;
  C.flip(i).flip(j).flip(k);
  string_type B = 1;
  static_assert(B.size() <= 64, "ULLONG NOT POSSIBLE HERE");
  B <<= k;
  B = B.to_ullong() - 1;

  return constraint_type(C, B, k);
}

/**
 * @brief Generate general constraints for systematic excitation space
 * exploration
 *
 * This function creates a comprehensive set of constraints for ASCI
 * calculations that systematically explore the excitation space. It generates
 * constraints at multiple levels (triplets, quadruplets, etc.) and estimates
 * the computational cost for load balancing in parallel environments.
 *
 * @tparam WfnType Wavefunction/determinant type
 * @tparam ContainerType Container type for unique alpha strings
 * @param[in] nlevels Maximum constraint level to generate (0=triplets,
 * 1=quadruplets, etc.)
 * @param[in] norb Number of molecular orbitals
 * @param[in] ns_othr Number of opposite-spin single excitations
 * @param[in] nd_othr Number of opposite-spin double excitations
 * @param[in] unique_alpha Container of unique alpha strings from the
 * wavefunction
 * @param[in] world_size Number of MPI processes for load balancing
 * @param[in] nlevel_min Minimum constraint level to include (default: 0)
 * @param[in] nrec_min Minimum number of records for constraint inclusion
 * (default: -1)
 * @return Vector of constraint objects with associated computational cost
 * estimates
 */
template <typename WfnType, typename ContainerType>
auto gen_constraints_general(size_t nlevels, size_t norb, size_t ns_othr,
                             size_t nd_othr, const ContainerType& unique_alpha,
                             int world_size, size_t nlevel_min = 0,
                             int64_t nrec_min = -1) {
  using wfn_traits = wavefunction_traits<WfnType>;
  using constraint_type = alpha_constraint<wfn_traits>;
  using string_type = typename constraint_type::constraint_type;

  constexpr bool flat_container =
      std::is_same_v<std::decay_t<WfnType>,
                     std::decay_t<typename ContainerType::value_type>>;

  // Generate triplets + heuristic
  std::vector<std::pair<constraint_type, size_t>> constraint_sizes;
  constraint_sizes.reserve(norb * norb * norb);

  // Generate all the triplets
  for (int t_i = 0; t_i < norb; ++t_i)
    for (int t_j = 0; t_j < t_i; ++t_j)
      for (int t_k = 0; t_k < t_j; ++t_k) {
        auto constraint = constraint_type::make_triplet(t_i, t_j, t_k);
        constraint_sizes.emplace_back(constraint, 0ul);
      }

  // Build up higher-order constraints as base if requested
  if (nrec_min < 0 or
      nrec_min >= constraint_sizes.size())  // nrec_min < 0 implies that you
                                            // want all the constraints upfront
    for (size_t ilevel = 0; ilevel < nlevel_min; ++ilevel) {
      decltype(constraint_sizes) cur_constraints;
      cur_constraints.reserve(constraint_sizes.size() * norb);
      for (auto [c, nw] : constraint_sizes) {
        const auto C_min = c.C_min();
        for (auto q_l = 0; q_l < C_min; ++q_l) {
          // Generate masks / counts
          string_type cn_C = c.C();
          cn_C.flip(q_l);
          string_type cn_B = c.B() >> (C_min - q_l);
          constraint_type c_next(cn_C, cn_B, q_l);
          cur_constraints.emplace_back(c_next, 0ul);
        }
      }
      constraint_sizes = std::move(cur_constraints);
    }

  struct atomic_wrapper {
    std::atomic<size_t> value;
    atomic_wrapper(size_t i = 0) : value(i) {};
    atomic_wrapper(const atomic_wrapper& other)
        : atomic_wrapper(other.value.load()) {};
    atomic_wrapper& operator=(const atomic_wrapper& other) {
      value.store(other.value.load());
      return *this;
    }
  };

  // Compute histogram
  const auto ntrip_full = constraint_sizes.size();
  std::vector<atomic_wrapper> constraint_work(ntrip_full, 0ul);
  {
#ifdef MACIS_ENABLE_MPI
    global_atomic<size_t> nxtval(MPI_COMM_WORLD);
#else
    std::atomic<size_t> nxtval(0);
#endif /* MACIS_ENABLE_MPI */
#pragma omp parallel
    {
      size_t i_trip = 0;
      while (i_trip < ntrip_full) {
        i_trip = nxtval.fetch_add(1);
        if (i_trip >= ntrip_full) break;
        // if(!(i_trip%1000)) printf("cgen %lu / %lu\n", i_trip, ntrip_full);
        auto& [constraint, __nw] = constraint_sizes[i_trip];
        auto& c_nw = constraint_work[i_trip];
        size_t nw = 0;
        for (const auto& alpha : unique_alpha) {
          if constexpr (flat_container)
            nw += constraint_histogram(wfn_traits::alpha_string(alpha), ns_othr,
                                       nd_othr, constraint);
          else
            nw += alpha.second * constraint_histogram(alpha.first, ns_othr,
                                                      nd_othr, constraint);
        }
        if (nw) c_nw.value.fetch_add(nw);
      }
    }
  }  // Scope nxtval

  std::vector<size_t> constraint_work_bare(ntrip_full);
  for (auto i_trip = 0; i_trip < ntrip_full; ++i_trip) {
    constraint_work_bare[i_trip] = constraint_work[i_trip].value.load();
  }
#ifdef MACIS_ENABLE_MPI
  allreduce(constraint_work_bare.data(), ntrip_full, MPI_SUM, MPI_COMM_WORLD);
#endif /* MACIS_ENABLE_MPI */

  // Copy over constraint work
  for (auto i_trip = 0; i_trip < ntrip_full; ++i_trip) {
    constraint_sizes[i_trip].second = constraint_work_bare[i_trip];
  }

  // Remove zeros
  {
    auto it = std::partition(constraint_sizes.begin(), constraint_sizes.end(),
                             [](const auto& p) { return p.second > 0; });
    constraint_sizes.erase(it, constraint_sizes.end());
  }

  // Compute average
  size_t total_work =
      std::accumulate(constraint_sizes.begin(), constraint_sizes.end(), 0ul,
                      [](auto s, const auto& p) { return s + p.second; });
  size_t local_average = total_work / world_size;

  // Manual refinement of top configurations
  if (nrec_min > 0 and nrec_min < constraint_sizes.size()) {
    const size_t nleave = constraint_sizes.size() - nrec_min;
    std::vector<std::pair<constraint_type, size_t>> constraint_to_refine,
        constraint_to_leave;
    constraint_to_refine.reserve(nrec_min);
    constraint_to_refine.reserve(nleave);

    std::copy_n(constraint_sizes.begin(), nrec_min,
                std::back_inserter(constraint_to_refine));
    std::copy_n(constraint_sizes.begin() + nrec_min, nleave,
                std::back_inserter(constraint_to_leave));

    // Deallocate original array
    decltype(constraint_sizes)().swap(constraint_sizes);

    // Generate refined constraints
    for (size_t ilevel = 0; ilevel < nlevel_min; ++ilevel) {
      decltype(constraint_sizes) cur_constraints;
      cur_constraints.reserve(constraint_to_refine.size() * norb);
      for (auto [c, nw] : constraint_to_refine) {
        const auto C_min = c.C_min();
        for (auto q_l = 0; q_l < C_min; ++q_l) {
          // Generate masks / counts
          string_type cn_C = c.C();
          cn_C.flip(q_l);
          string_type cn_B = c.B() >> (C_min - q_l);
          constraint_type c_next(cn_C, cn_B, q_l);
          cur_constraints.emplace_back(c_next, 0ul);
        }
      }
      constraint_to_refine = std::move(cur_constraints);
    }

    const size_t nrefine = constraint_to_refine.size();

#ifdef MACIS_ENABLE_MPI
    global_atomic<size_t> nxtval(MPI_COMM_WORLD);
#else
    std::atomic<size_t> nxtval(0);
#endif /* MACIS_ENABLE_MPI */
    std::vector<atomic_wrapper>().swap(constraint_work);
    std::vector<size_t>().swap(constraint_work_bare);
    constraint_work.resize(nrefine, 0ul);
#pragma omp parallel
    {
      size_t i_ref = 0;
      while (i_ref < nrefine) {
        i_ref = nxtval.fetch_add(1);
        if (i_ref >= nrefine) break;
        // if(!(i_ref%1000)) printf("cgen %lu / %lu\n", i_ref, nrefine);
        auto& [constraint, __nw] = constraint_to_refine[i_ref];
        auto& c_nw = constraint_work[i_ref];
        size_t nw = 0;
        for (const auto& alpha : unique_alpha) {
          if constexpr (flat_container)
            nw += constraint_histogram(wfn_traits::alpha_string(alpha), ns_othr,
                                       nd_othr, constraint);
          else
            nw += alpha.second * constraint_histogram(alpha.first, ns_othr,
                                                      nd_othr, constraint);
        }
        if (nw) c_nw.value.fetch_add(nw);
      }  // constraint "loop"
    }  // OpenMP Context

    constraint_work_bare.resize(nrefine);
    for (auto i_ref = 0; i_ref < nrefine; ++i_ref) {
      constraint_work_bare[i_ref] = constraint_work[i_ref].value.load();
    }
#ifdef MACIS_ENABLE_MPI
    allreduce(constraint_work_bare.data(), nrefine, MPI_SUM, MPI_COMM_WORLD);
#endif /* MACIS_ENABLE_MPI */

    // Copy over constraint work
    for (auto i_ref = 0; i_ref < nrefine; ++i_ref) {
      constraint_to_refine[i_ref].second = constraint_work_bare[i_ref];
    }

    // Remove zeros
    {
      auto it = std::partition(constraint_to_refine.begin(),
                               constraint_to_refine.end(),
                               [](const auto& p) { return p.second > 0; });
      constraint_to_refine.erase(it, constraint_to_refine.end());
    }

    // Concatenate the arrays
    constraint_sizes.reserve(nrefine + nleave);
    std::copy_n(constraint_to_refine.begin(), nrefine,
                std::back_inserter(constraint_sizes));
    std::copy_n(constraint_to_leave.begin(), nleave,
                std::back_inserter(constraint_sizes));

    size_t tmp =
        std::accumulate(constraint_sizes.begin(), constraint_sizes.end(), 0ul,
                        [](auto s, const auto& p) { return s + p.second; });
    if (tmp != total_work) throw std::runtime_error("Incorrect Refinement");
  }  // Selective refinement logic

  for (size_t ilevel = 0; ilevel < nlevels; ++ilevel) {
    // Select constraints larger than average to be broken apart
    std::vector<std::pair<constraint_type, size_t>> tps_to_next;
    {
      auto it = std::partition(
          constraint_sizes.begin(), constraint_sizes.end(),
          [=](const auto& a) { return a.second <= local_average; });

      // Remove constraints from full list
      tps_to_next = decltype(tps_to_next)(it, constraint_sizes.end());
      constraint_sizes.erase(it, constraint_sizes.end());
      for (auto [t, s] : tps_to_next) total_work -= s;
    }

    if (!tps_to_next.size()) break;

    // Break apart constraints
    for (auto [c, nw_trip] : tps_to_next) {
      const auto C_min = c.C_min();

      // Loop over possible constraints with one more element
      for (auto q_l = 0; q_l < C_min; ++q_l) {
        // Generate masks / counts
        string_type cn_C = c.C();
        cn_C.flip(q_l);
        string_type cn_B = c.B() >> (C_min - q_l);
        constraint_type c_next(cn_C, cn_B, q_l);

        size_t nw = 0;

        for (const auto& alpha : unique_alpha) {
          if constexpr (flat_container)
            nw += constraint_histogram(wfn_traits::alpha_string(alpha), ns_othr,
                                       nd_othr, c_next);
          else
            nw += alpha.second *
                  constraint_histogram(alpha.first, ns_othr, nd_othr, c_next);
        }
        if (nw) constraint_sizes.emplace_back(c_next, nw);
        total_work += nw;
      }
    }
  }  // Recurse into constraints

  // Sort to get optimal bucket partitioning
  std::sort(constraint_sizes.begin(), constraint_sizes.end(),
            [](const auto& a, const auto& b) { return a.second > b.second; });

  return constraint_sizes;
}

#ifdef MACIS_ENABLE_MPI
/**
 * @brief Distribute constraint generation across MPI processes for parallel
 * execution
 *
 * This function distributes the constraint generation workload across multiple
 * MPI processes using load balancing based on estimated computational costs. It
 * assigns constraints to different ranks to achieve roughly equal work
 * distribution.
 *
 * @tparam WfnType Wavefunction/determinant type
 * @tparam ContainerType Container type for unique alpha strings
 * @param[in] nlevels Maximum constraint level to generate
 * @param[in] norb Number of molecular orbitals
 * @param[in] ns_othr Number of opposite-spin single excitations
 * @param[in] nd_othr Number of opposite-spin double excitations
 * @param[in] unique_alpha Container of unique alpha strings from the
 * wavefunction
 * @param[in] comm MPI communicator for process coordination
 * @return Vector of constraint objects assigned to the calling MPI rank
 */
template <typename WfnType, typename ContainerType>
auto dist_constraint_general(size_t nlevels, size_t norb, size_t ns_othr,
                             size_t nd_othr, const ContainerType& unique_alpha,
                             MPI_Comm comm) {
  using wfn_traits = wavefunction_traits<WfnType>;
  using constraint_type = alpha_constraint<wfn_traits>;

  auto world_rank = comm_rank(comm);
  auto world_size = comm_size(comm);

  // Generate constraints subject to expected workload
  auto constraint_sizes = gen_constraints_general<WfnType>(
      nlevels, norb, ns_othr, nd_othr, unique_alpha, world_size);

  // Global workloads
  std::vector<size_t> workloads(world_size, 0);

  // Assign work
  std::vector<constraint_type> constraints;
  constraints.reserve(constraint_sizes.size() / world_size);

  for (auto [c, nw] : constraint_sizes) {
    // Get rank with least amount of work
    auto min_rank_it = std::min_element(workloads.begin(), workloads.end());
    int min_rank = std::distance(workloads.begin(), min_rank_it);

    // Assign constraint
    *min_rank_it += nw;
    if (world_rank == min_rank) {
      constraints.emplace_back(c);
    }
  }

  // if(world_rank == 0)
  // printf("[rank %2d] AFTER LOCAL WORK = %lu TOTAL WORK = %lu\n", world_rank,
  //   workloads[world_rank], total_work);

  return constraints;
}
#endif /* MACIS_ENABLE_MPI */

}  // namespace macis
