/*
 * MACIS Copyright (c) 2023, The Regents of the University of California,
 * through Lawrence Berkeley National Laboratory (subject to receipt of
 * any required approvals from the U.S. Dept. of Energy). All rights reserved.
 * Portions Copyright (c) Microsoft Corporation.
 *
 * See LICENSE.txt for details
 */

#include <blas.hh>
#include <iomanip>
#include <iostream>
#include <macis/mcscf/orbital_hessian.hpp>
#include <vector>

namespace macis {

void approx_diag_hessian(NumInactive _ni, NumActive _na, NumVirtual _nv,
                         const double* Fi, size_t LDFi, const double* Fa,
                         size_t LDFa, const double* A1RDM, size_t LDD,
                         const double* F, size_t LDF, double* H_vi,
                         double* H_va, double* H_ai) {
  const auto ni = _ni.get();
  const auto na = _na.get();
  const auto nv = _nv.get();

#define TWO_IDX(A, i, j, LDA) A[i + j * LDA]
#define FI(i, j) TWO_IDX(Fi, i, j, LDFi)
#define FA(i, j) TWO_IDX(Fa, i, j, LDFa)
#define FF(i, j) TWO_IDX(F, i, j, LDF)
#define ORDM(i, j) TWO_IDX(A1RDM, i, j, LDD)

  // Virtual - Inactive Block
  for (size_t i = 0; i < ni; ++i) {
    // Cache inactive-inactive term
    const auto ii_diff = FI(i, i) + FA(i, i);
    for (size_t a = 0; a < nv; ++a) {
      const auto a_off = a + ni + na;
      H_vi[a + i * nv] = 4. * (FI(a_off, a_off) + FA(a_off, a_off) - ii_diff);
    }
  }

  // Virtual - Active Block
  for (size_t i = 0; i < na; ++i)
    for (size_t a = 0; a < nv; ++a) {
      const auto i_off = i + ni;
      const auto a_off = a + ni + na;
      H_va[a + i * nv] =
          2. * ORDM(i, i) * (FI(a_off, a_off) + FA(a_off, a_off)) -
          2. * FF(i_off, i_off);
    }

  // Active - Inactive Block
  for (size_t i = 0; i < ni; ++i)
    for (size_t a = 0; a < na; ++a) {
      const auto a_off = a + ni;
      H_ai[a + i * na] =
          2. * ORDM(a, a) * (FI(i, i) + FA(i, i)) +
          4. * (FI(a_off, a_off) + FA(a_off, a_off) - FI(i, i) - FA(i, i)) -
          2. * FF(a_off, a_off);
    }

#undef TWO_IDX
#undef FI
#undef FA
#undef FF
#undef ORDM
}

void one_index_transformed_hamiltonian(NumOrbital norb, const double* T,
                                       size_t LDT, const double* V, size_t LDV,
                                       const double* X, size_t LDX, double* Tx,
                                       size_t LDTx, double* Vx, size_t LDVx) {
  const size_t no = norb.get();

  for (size_t p = 0; p < no; ++p)
    for (size_t q = 0; q < no; ++q) {
      double tmp = 0.0;
      for (size_t o = 0; o < no; ++o) {
        tmp +=
            X[p + o * LDX] * T[o + q * LDT] + X[q + o * LDX] * T[p + o * LDX];
      }
      Tx[p + q * LDTx] = tmp;
    }

  for (size_t p = 0; p < no; ++p)
    for (size_t q = 0; q < no; ++q)
      for (size_t r = 0; r < no; ++r)
        for (size_t s = 0; s < no; ++s) {
          double tmp = 0.0;
          for (size_t o = 0; o < no; ++o) {
            tmp += X[p + o * LDX] *
                       V[o + q * LDV + r * LDV * LDV + s * LDV * LDV * LDV] +
                   X[q + o * LDX] *
                       V[p + o * LDV + r * LDV * LDV + s * LDV * LDV * LDV] +
                   X[r + o * LDX] *
                       V[p + q * LDV + o * LDV * LDV + s * LDV * LDV * LDV] +
                   X[s + o * LDX] *
                       V[p + q * LDV + r * LDV * LDV + o * LDV * LDV * LDV];
          }
          Vx[p + q * LDVx + r * LDVx * LDVx + s * LDVx * LDVx * LDVx] = tmp;
        }
}

void orb_orb_hessian_contract(NumOrbital norb, NumInactive ninact,
                              NumActive nact, NumVirtual num_virtual_orbitals,
                              const double* T, size_t LDT, const double* V,
                              size_t LDV, const double* A1RDM, size_t LDD1,
                              const double* A2RDM, size_t LDD2,
                              const double* OG, const double* K_lin,
                              double* HK_lin) {
  const size_t no = norb.get();
  const size_t na = nact.get();
  const size_t no2 = no * no;
  const size_t no4 = no2 * no2;
  const size_t orb_rot_sz =
      num_virtual_orbitals.get() * (nact.get() + ninact.get()) +
      nact.get() * ninact.get();

  // Expand to full antisymmetric K
  std::vector<double> K_full(no2);
  linear_orb_rot_to_matrix(ninact, nact, num_virtual_orbitals, K_lin,
                           K_full.data(), no);

  // Compute one-index transformed hamiltonian
  std::vector<double> Tk(no2), Vk(no4);
  one_index_transformed_hamiltonian(norb, T, LDT, V, LDV, K_full.data(), no,
                                    Tk.data(), no, Vk.data(), no);

  // Compute gradient-like term with transformed Hamiltonian
  std::vector<double> Fk(no2);
  generalized_fock_matrix_comp_mat2(norb, ninact, nact, Tk.data(), no,
                                    Vk.data(), no, A1RDM, LDD1, A2RDM, LDD2,
                                    Fk.data(), no);
  // Store in HK to initialize the memory
  fock_to_linear_orb_grad(ninact, nact, num_virtual_orbitals, Fk.data(), no,
                          HK_lin);

  // Expand OG into full antisymmetric matrix (reuse Tk)
  std::fill(Tk.begin(), Tk.end(), 0);
  linear_orb_rot_to_matrix(ninact, nact, num_virtual_orbitals, OG, Tk.data(),
                           no);

  // Compute G*K in Fk
  blas::gemm(blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans, no,
             no, no, 1., Tk.data(), no, K_full.data(), no, 0., Fk.data(), no);

  // Replace Tk with [G,K]
  for (size_t p = 0; p < no; ++p)
    for (size_t q = 0; q < no; ++q) {
      Tk[p + q * no] = Fk[p + q * no] - Fk[q + p * no];
    }

  // Take out orbital rotation piece into a temporary buffer
  std::vector<double> tmp_hk(orb_rot_sz);
  matrix_to_linear_orb_rot(ninact, nact, num_virtual_orbitals, Tk.data(), no,
                           tmp_hk.data());

  // Add to final result
  for (size_t i = 0; i < orb_rot_sz; ++i) HK_lin[i] += tmp_hk[i];
}

}  // namespace macis
