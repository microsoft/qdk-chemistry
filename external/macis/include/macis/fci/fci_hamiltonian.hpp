/*
 * MACIS Copyright (c) 2023, The Regents of the University of California,
 * through Lawrence Berkeley National Laboratory (subject to receipt of
 * any required approvals from the U.S. Dept. of Energy). All rights reserved.
 * Portions Copyright (c) Microsoft Corporation.
 *
 * See LICENSE.txt for details
 */

#pragma once
#include <macis/fci/string_space.hpp>
#include <memory>
#include <vector>

namespace macis {

/// Instruction-set levels of the hand-vectorized FCI kernels: portable C++
/// (any architecture) and x86-64 AVX2 / AVX-512, selected at run time; all
/// give the same results up to rounding
enum class FCISimd { generic = 0, avx2 = 1, avx512 = 2 };

/// Widest kernel level supported by this CPU and compiler
FCISimd fci_simd_supported();

/// Precision of the Hamiltonian applications in fci_ground_state
enum class FCIPrecision { automatic, mixed, double_precision };

/**
 * @brief Direct (string-driven) full CI Hamiltonian for real, spin-restricted
 * integrals.
 *
 * Applies H to CI vectors without forming it. Vectors are alpha-major,
 * C[Ia * Nb + Ib], with Ia/Ib addressed by FCIStringSpace; this matches the
 * determinant order of generate_hilbert_space.
 *
 * Integrals use chemists' notation and must have the 8-fold permutational
 * symmetry of real orbitals; T and V are column-major with leading dimensions
 * LDT and LDV (V(p,q,r,s) = V[p + q*LDV + r*LDV^2 + s*LDV^3]).
 *
 * @note The alpha-beta term is one DGEMM per output alpha string whose inner
 * dimension spans only that string's single excitations, a restriction of the
 * contractions of Knowles & Handy (Chem. Phys. Lett. 111, 315 (1984)) and
 * Olsen et al. (J. Chem. Phys. 89, 2185 (1988)). The same-spin terms use the
 * (n-2)-electron projection of Harrison & Zarrabian (Chem. Phys. Lett. 158,
 * 393 (1989)) within one spin: k-electron strings couple through
 * (k-2)-electron intermediates K, whose strings K + {p, r} form dense blocks
 * of the antisymmetrized pair integrals (the one-body part is folded in using
 * the electron count).
 */
class FCIHamiltonian {
 public:
  FCIHamiltonian(size_t norb, size_t nalpha, size_t nbeta, const double* T,
                 size_t LDT, const double* V, size_t LDV);

  const FCIStringSpace& alpha_strings() const { return *alpha_; }
  const FCIStringSpace& beta_strings() const { return *beta_; }

  /// Number of determinants
  size_t size() const { return alpha_->size() * beta_->size(); }

  /// Kernel level used by sigma, by default fci_simd_supported()
  FCISimd simd() const { return simd_; }

  /// Selects the kernel level (for testing and benchmarking)
  /// @throws std::invalid_argument if this CPU or compiler does not support it
  void set_simd(FCISimd level);

  /// Diagonal of H (size() entries)
  void diagonal(double* D) const;

  /**
   * @brief S = alpha * H * C + beta * S (S is not read when beta == 0).
   *
   * With equal alpha and beta electron counts, CI vectors of total spin S
   * satisfy C^T = (-1)^S C as Na x Na matrices C(Ia, Ib). A non-zero
   * spin_flip restricts the contraction to that sector: about half the work,
   * with C^T = spin_flip * C required (and S^T = spin_flip * S when
   * beta != 0); the result lies exactly in the sector.
   *
   * @param[in] spin_flip 0 for any C, +1 for C^T = C (even S), -1 for
   *                      C^T = -C (odd S)
   * @throws std::invalid_argument if spin_flip is not -1, 0 or 1, or is
   *                               non-zero for unequal electron counts
   */
  void sigma(const double* C, double* S, double alpha = 1.0, double beta = 0.0,
             int spin_flip = 0) const;

  /**
   * @brief Single-precision S = alpha * H * C + beta * S, same contract as
   * the double-precision overload (integrals rounded to float).
   *
   * Relative accuracy is that of float arithmetic (about 1e-6); the
   * mixed-precision ground-state solver uses it for correction vectors and
   * checks its result in double precision.
   */
  void sigma(const float* C, float* S, float alpha = 1.0f, float beta = 0.0f,
             int spin_flip = 0) const;

  /// Operator contract of macis::davidson: AV = alpha * H * V + beta * AV for
  /// m vectors
  void operator_action(size_t m, double alpha, const double* V, size_t LDV,
                       double beta, double* AV, size_t LDAV) const;

  /// Dense H restricted to determinants `dets` (indices into the CI vector):
  /// H_sub[i + j * dets.size()] = H(dets[i], dets[j])
  void submatrix(const std::vector<size_t>& dets, double* H_sub) const;

 private:
  /// Same-spin Hamiltonian H^ss of one string space as dense "cliques": for
  /// k >= 2 electrons, clique K lists the strings K + {p, r} (p < r) of a
  /// (k-2)-electron string K, and H^ss(I, J) is the sum over the cliques
  /// holding both of sign_I * sign_J * W(pair_I, pair_J). For k == 1 a single
  /// clique holds every string with W = h; for k == 0 there is none.
  struct SameSpinOperator {
    size_t clique_size = 0;
    size_t pair_dim = 0;  ///< W is pair_dim x pair_dim
    std::vector<double> W;
    std::vector<float> W_f;        ///< W rounded to float
    std::vector<uint32_t> string;  ///< num_cliques x clique_size addresses
    std::vector<uint32_t> pair;    ///< row of W per member, bit 31 = sign
    std::vector<double> diagonal;  ///< H^ss(I, I)
    size_t num_cliques() const {
      return clique_size ? string.size() / clique_size : 0;
    }
  };

  static std::shared_ptr<const SameSpinOperator> make_same_spin_operator(
      const FCIStringSpace& space, const double* T, size_t LDT, const double* V,
      size_t LDV);

  /// H^ss(I, J) for strings I, J of `space`
  static double same_spin_element(const SameSpinOperator& op,
                                  const FCIStringSpace& space, size_t I,
                                  size_t J);

  /// Per-thread work arrays of sigma, kept between calls: fresh ones cost up
  /// to a quarter of a sigma in page faults and unmapping (7e/14o). A call
  /// that finds them in use by a concurrent call allocates its own.
  struct Workspace;
  class Scratch;

  /// sigma for double or float vectors
  template <typename T>
  void sigma_impl(const T* C, T* S, T alpha, T beta, int spin_flip) const;

  /// S += alpha * (H^ss applied to the strings of tiles of C): along columns
  /// of the alpha-major CI matrix for alpha, along rows for beta
  template <typename T>
  void same_spin_pass(bool alpha_strings, const T* C, T* S, T alpha,
                      Scratch& scratch) const;

  size_t norb_;
  size_t npair_;
  std::shared_ptr<const FCIStringSpace> alpha_;
  std::shared_ptr<const FCIStringSpace> beta_;
  std::vector<double> vpair_;    ///< npair x npair, (pq|rs) by packed pairs
  std::vector<float> vpair_f_;   ///< vpair_ rounded to float
  std::vector<double> coulomb_;  ///< norb x norb, (pp|qq)
  std::shared_ptr<const SameSpinOperator> h_alpha_;
  std::shared_ptr<const SameSpinOperator> h_beta_;
  /// Off-diagonal excitations of the beta strings for the alpha-beta pass,
  /// those of positive sign first (beta_link_npos_ of them): target address
  /// and fci_pair_index
  std::vector<uint32_t> beta_link_target_;
  std::vector<uint32_t> beta_link_pair_;
  std::vector<uint32_t> beta_link_npos_;
  /// fci_pair_index(q, q) of the occupied orbitals q of every beta string
  std::vector<uint32_t> beta_diag_pair_;
  FCISimd simd_ = fci_simd_supported();
  std::shared_ptr<Workspace> workspace_;
};

/**
 * @brief Lowest eigenpair of a direct FCI Hamiltonian (restarted Davidson).
 *
 * The solve runs on at most omp_get_max_threads() threads, fewer for small
 * spaces: one team sized for the work of a sigma, which keeps them from
 * paying for more threads than their work can use.
 *
 * @param[in] H            FCI Hamiltonian
 * @param[in] max_iter     Maximum number of Davidson iterations
 * @param[in] res_tol      Convergence tolerance on the residual norm
 * @param[in,out] C        Used as the initial guess (projected onto the
 *                         spin-flip sector) when it has H.size() elements
 *                         and is non-zero there, otherwise the lowest
 *                         eigenvector of H within the (up to) 400
 *                         determinants (or sector functions) of lowest
 *                         diagonal energy is; the normalized eigenvector on
 *                         output
 * @param[in] spin_flip    Sector of FCIHamiltonian::sigma: 0 for no
 *                         restriction, +1 (-1) for the lowest state of even
 *                         (odd) total spin
 * @param[in] max_subspace Davidson search-space dimension (>= 2). Memory per
 *                         determinant is 8 * (2 * max_subspace + 3) bytes in
 *                         double precision and 8 * max_subspace + 28 in
 *                         mixed precision (single-precision corrections); in
 *                         a spin-flip sector, whose vectors have about half
 *                         as many coordinates, 8 * max_subspace + 28 and
 *                         4 * max_subspace + 30.
 *                         Hydrogen chains took as many iterations with 3 to
 *                         16 vectors, but strongly correlated systems need
 *                         fewer with more: [2Fe-2S] (30e, 20o) took 135
 *                         iterations with 3, 115 with 6 and 100 with 12
 * @param[in] precision    Precision of the Hamiltonian applications:
 *                         FCIPrecision::mixed applies H in single precision
 *                         where that cannot affect the result (a single-
 *                         precision phase down to a residual of about 1e-4,
 *                         then an in-place anchor: a double-precision sigma
 *                         of the Ritz vector, while later correction vectors
 *                         keep single precision, their sigma errors entering
 *                         scaled by their small coefficients) and accepts the
 *                         result only on its double-precision residual
 *                         (res_tol), so the contract matches
 *                         FCIPrecision::double_precision. The default
 *                         FCIPrecision::automatic is mixed for spaces of at
 *                         least 2^21 determinants, where it was measured
 *                         faster, and double precision below.
 * @return Lowest eigenvalue
 * @throws std::invalid_argument for an invalid or empty spin-flip sector
 * @throws std::runtime_error if Davidson does not converge
 */
double fci_ground_state(const FCIHamiltonian& H, size_t max_iter,
                        double res_tol, std::vector<double>& C,
                        int spin_flip = 0, size_t max_subspace = 10,
                        FCIPrecision precision = FCIPrecision::automatic);

/**
 * @brief fci_ground_state on caller-provided storage.
 *
 * Same as the std::vector overload, with the eigenvector written to the
 * H.size() elements at C. Unless `guess` is true their initial contents are
 * ignored, so C can be uninitialized memory: the solver writes every
 * element with the threads that later process it, without the serial
 * zero-fill of a std::vector (0.4 s for 1.7e8 determinants).
 *
 * @param[in] guess  Whether C holds the initial guess (then used as by the
 *                   std::vector overload; a zero guess is ignored)
 */
double fci_ground_state(const FCIHamiltonian& H, size_t max_iter,
                        double res_tol, double* C, bool guess,
                        int spin_flip = 0, size_t max_subspace = 10,
                        FCIPrecision precision = FCIPrecision::automatic);

/**
 * @brief Lowest eigenpair of H within the span of a determinant subset, by
 * dense diagonalization.
 *
 * With a non-zero spin_flip the subset is completed with the spin-flipped
 * partners (Ia, Ib) -> (Ib, Ia) of its determinants and H is diagonalized
 * within the matching sector (see FCIHamiltonian::sigma).
 *
 * @param[in] H         FCI Hamiltonian
 * @param[in] dets      Determinants (indices into the CI vector)
 * @param[out] C        Normalized eigenvector, H.size() elements (zero
 *                      outside the subset)
 * @param[in] spin_flip 0, +1 (C^T = C) or -1 (C^T = -C)
 * @return Lowest eigenvalue
 * @throws std::invalid_argument for an invalid spin_flip or if the subset
 *                               has no component in the sector
 */
double fci_subspace_ground_state(const FCIHamiltonian& H,
                                 const std::vector<size_t>& dets,
                                 std::vector<double>& C, int spin_flip = 0);

}  // namespace macis
