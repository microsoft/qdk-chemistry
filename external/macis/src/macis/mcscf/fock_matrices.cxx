/*
 * MACIS Copyright (c) 2023, The Regents of the University of California,
 * through Lawrence Berkeley National Laboratory (subject to receipt of
 * any required approvals from the U.S. Dept. of Energy). All rights reserved.
 * Portions Copyright (c) Microsoft Corporation.
 *
 * See LICENSE.txt for details
 */

#include <blas.hh>
#include <macis/mcscf/fock_matrices.hpp>
#include <vector>

namespace macis {

double inactive_energy(NumInactive _ninact, const double* T, size_t LDT,
                       const double* Fi, size_t LDF) {
  const auto ninact = _ninact.get();

  double E = 0.0;
  for (size_t i = 0; i < ninact; ++i) E += T[i * (LDT + 1)] + Fi[i * (LDF + 1)];
  return E;
}

void inactive_fock_matrix(NumOrbital _norb, NumInactive _ninact,
                          const double* T, size_t LDT, const double* V,
                          size_t LDV, double* Fi, size_t LDF) {
  const auto norb = _norb.get();
  const auto ninact = _ninact.get();

  const size_t LDV2 = LDV * LDV;
  const size_t LDV3 = LDV2 * LDV;

  for (size_t p = 0; p < norb; ++p)
    for (size_t q = 0; q < norb; ++q) {
      double tmp = 0.0;
      for (size_t i = 0; i < ninact; ++i) {
        tmp += 2 * V[p + q * LDV + i * (LDV2 + LDV3)] -
               V[p + q * LDV3 + i * (LDV + LDV2)];
      }
      Fi[p + q * LDF] = T[p + q * LDT] + tmp;
    }
}

void active_submatrix_1body(NumActive _nact, NumInactive _ninact,
                            const double* A_full, size_t LDAF, double* A_sub,
                            size_t LDAS) {
  const auto ninact = _ninact.get();
  const auto nact = _nact.get();

  for (size_t x = 0; x < nact; ++x)
    for (size_t y = 0; y < nact; ++y) {
      const size_t x_off = x + ninact;
      const size_t y_off = y + ninact;
      A_sub[x + y * LDAS] = A_full[x_off + y_off * LDAF];
    }
}

void active_subtensor_2body(NumActive _nact, NumInactive _ninact,
                            const double* A_full, size_t LDAF, double* A_sub,
                            size_t LDAS) {
  const auto ninact = _ninact.get();
  const auto nact = _nact.get();

  const size_t LDAF2 = LDAF * LDAF;
  const size_t LDAF3 = LDAF2 * LDAF;
  const size_t LDAS2 = LDAS * LDAS;
  const size_t LDAS3 = LDAS2 * LDAS;

  for (size_t x = 0; x < nact; ++x)
    for (size_t y = 0; y < nact; ++y)
      for (size_t z = 0; z < nact; ++z)
        for (size_t w = 0; w < nact; ++w) {
          const size_t x_off = x + ninact;
          const size_t y_off = y + ninact;
          const size_t z_off = z + ninact;
          const size_t w_off = w + ninact;

          A_sub[x + y * LDAS + z * LDAS2 + w * LDAS3] =
              A_full[x_off + y_off * LDAF + z_off * LDAF2 + w_off * LDAF3];
        }
}

void active_hamiltonian(NumOrbital norb, NumActive nact, NumInactive ninact,
                        const double* T_full, size_t LDTF, const double* V_full,
                        size_t LDVF, double* Fi, size_t LDFi, double* T_active,
                        size_t LDTA, double* V_active, size_t LDVA) {
  // Extact all-active subblock of V
  active_subtensor_2body(nact, ninact, V_full, LDVF, V_active, LDVA);

  // Compute inactive Fock in full MO space
  inactive_fock_matrix(norb, ninact, T_full, LDTF, V_full, LDVF, Fi, LDFi);

  // Set T_active as the active-active block of inactive Fock
  active_submatrix_1body(nact, ninact, Fi, LDFi, T_active, LDTA);
}

void active_fock_matrix(NumOrbital _norb, NumInactive _ninact, NumActive _nact,
                        const double* V, size_t LDV, const double* A1RDM,
                        size_t LDD, double* Fa, size_t LDF) {
  const auto norb = _norb.get();
  const auto ninact = _ninact.get();
  const auto nact = _nact.get();

  const size_t LDV2 = LDV * LDV;
  const size_t LDV3 = LDV2 * LDV;

  for (size_t p = 0; p < norb; ++p)
    for (size_t q = 0; q < norb; ++q) {
      double tmp = 0.0;
      for (size_t v = 0; v < nact; ++v)
        for (size_t w = 0; w < nact; ++w) {
          const size_t v_off = v + ninact;
          const size_t w_off = w + ninact;
          tmp += A1RDM[v + w * LDD] *
                 (V[p + q * LDV + v_off * LDV2 + w_off * LDV3] -
                  0.5 * V[p + q * LDV3 + w_off * LDV + v_off * LDV2]);
        }
      Fa[p + q * LDF] = tmp;
    }
}

void aux_q_matrix(NumActive _nact, NumOrbital _norb, NumInactive _ninact,
                  const double* V, size_t LDV, const double* A2RDM, size_t LDD,
                  double* Q, size_t LDQ) {
  const auto norb = _norb.get();
  const auto ninact = _ninact.get();
  const auto nact = _nact.get();

  const size_t LDV2 = LDV * LDV;
  const size_t LDV3 = LDV2 * LDV;
  const size_t LDD2 = LDD * LDD;
  const size_t LDD3 = LDD2 * LDD;

  for (size_t v = 0; v < nact; ++v)
    for (size_t p = 0; p < norb; ++p) {
      double tmp = 0.0;
      for (size_t w = 0; w < nact; ++w)
        for (size_t x = 0; x < nact; ++x)
          for (size_t y = 0; y < nact; ++y) {
            const size_t w_off = w + ninact;
            const size_t x_off = x + ninact;
            const size_t y_off = y + ninact;

            tmp += A2RDM[v + w * LDD + x * LDD2 + y * LDD3] *
                   V[p + w_off * LDV + x_off * LDV2 + y_off * LDV3];
          }
      Q[v + p * LDQ] = 2. * tmp;
    }
}

void generalized_fock_matrix(NumOrbital _norb, NumInactive _ninact,
                             NumActive _nact, const double* Fi, size_t LDFi,
                             const double* Fa, size_t LDFa, const double* A1RDM,
                             size_t LDD, const double* Q, size_t LDQ, double* F,
                             size_t LDF) {
  const auto norb = _norb.get();
  const auto ninact = _ninact.get();
  const auto nact = _nact.get();

  // Inactive - General
  // F(i,p) = 2*( Fi(p,i) + Fa(p,i) )
  for (size_t i = 0; i < ninact; ++i)
    for (size_t p = 0; p < norb; ++p) {
      F[i + p * LDF] = 2. * (Fi[p + i * LDFi] + Fa[p + i * LDFa]);
    }

  // Compute X(p,x) = Fi(p,y) * A1RDM(y,x)
  std::vector<double> X(norb * nact);
  blas::gemm(blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans, norb,
             nact, nact, 1.0, Fi + ninact * LDFi, LDFi, A1RDM, LDD, 0.0,
             X.data(), norb);

  // Active - General
  for (size_t v = 0; v < nact; ++v)
    for (size_t p = 0; p < norb; ++p) {
      const size_t v_off = v + ninact;
      F[v_off + p * LDF] = X[p + v * norb] + Q[v + p * LDQ];
    }
}

void generalized_fock_matrix_comp_mat1(NumOrbital _norb, NumInactive _ninact,
                                       NumActive _nact, const double* Fi,
                                       size_t LDFi, const double* V_full,
                                       size_t LDV, const double* A1RDM,
                                       size_t LDD1, const double* A2RDM,
                                       size_t LDD2, double* F, size_t LDF) {
  const auto norb = _norb.get();
  const auto ninact = _ninact.get();
  const auto nact = _nact.get();

  const size_t norb2 = norb * norb;

  // Compute Active Fock Matrix
  std::vector<double> F_active(norb2);
  active_fock_matrix(_norb, _ninact, _nact, V_full, LDV, A1RDM, LDD1,
                     F_active.data(), norb);

  // Compute Q
  std::vector<double> Q(nact * norb);
  aux_q_matrix(_nact, _norb, _ninact, V_full, LDV, A2RDM, LDD2, Q.data(), nact);

  // Compute Generalized Fock Matrix
  generalized_fock_matrix(_norb, _ninact, _nact, Fi, LDFi, F_active.data(),
                          norb, A1RDM, LDD1, Q.data(), nact, F, LDF);
}

void generalized_fock_matrix_comp_mat2(NumOrbital _norb, NumInactive _ninact,
                                       NumActive _nact, const double* T,
                                       size_t LDT, const double* V_full,
                                       size_t LDV, const double* A1RDM,
                                       size_t LDD1, const double* A2RDM,
                                       size_t LDD2, double* F, size_t LDF) {
  const auto norb = _norb.get();
  const auto ninact = _ninact.get();
  const auto nact = _nact.get();

  std::vector<double> Fi(norb * norb);
  inactive_fock_matrix(_norb, _ninact, T, LDT, V_full, LDV, Fi.data(), norb);

  generalized_fock_matrix_comp_mat1(_norb, _ninact, _nact, Fi.data(), norb,
                                    V_full, LDV, A1RDM, LDD1, A2RDM, LDD2, F,
                                    LDF);
}

double energy_from_generalized_fock(NumInactive _ninact, NumActive _nact,
                                    const double* T, size_t LDT,
                                    const double* A1RDM, size_t LDD,
                                    const double* F, size_t LDF) {
  const auto ninact = _ninact.get();
  const auto nact = _nact.get();

  double E = 0;
  // Inactive-Inactve E <- 2*T(i,i)
  for (size_t i = 0; i < ninact; ++i) {
    E += 2. * T[i * (LDT + 1)];
  }

  // Active-Active
  // E <- A1RDM(x,y) * T(x,y)
  for (size_t x = 0; x < nact; ++x)
    for (size_t y = 0; y < nact; ++y) {
      const size_t x_off = x + ninact;
      const size_t y_off = y + ninact;
      E += A1RDM[x + y * LDD] * T[x_off + y_off * LDT];
    }

  // Fock piece E <- F(i,i) + F(x,x)
  for (size_t p = 0; p < (nact + ninact); ++p) {
    E += F[p * (LDF + 1)];
  }

  return 0.5 * E;
}

}  // namespace macis
