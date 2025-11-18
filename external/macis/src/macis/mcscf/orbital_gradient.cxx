/*
 * MACIS Copyright (c) 2023, The Regents of the University of California,
 * through Lawrence Berkeley National Laboratory (subject to receipt of
 * any required approvals from the U.S. Dept. of Energy). All rights reserved.
 * Portions Copyright (c) Microsoft Corporation.
 *
 * See LICENSE.txt for details
 */

#include <Eigen/Core>
#include <macis/mcscf/fock_matrices.hpp>
#include <macis/mcscf/orbital_gradient.hpp>
#include <macis/util/transform.hpp>
#include <unsupported/Eigen/MatrixFunctions>

namespace macis {

void compute_orbital_rotation(NumOrbital _norb, double alpha, const double* K,
                              size_t LDK, double* U, size_t LDU) {
  const size_t norb = _norb.get();
  using mat_t = Eigen::MatrixXd;
  using map_t = Eigen::Map<mat_t>;
  using cmap_t = Eigen::Map<const mat_t>;

  cmap_t K_map(K, LDK, norb);
  map_t U_map(U, LDU, norb);

  // U = EXP[-K] (Pink book 10.1.8)
  U_map.block(0, 0, norb, norb) =
      (-alpha * K_map.block(0, 0, norb, norb)).exp();
}

void fock_to_gradient(NumOrbital _norb, NumInactive _ninact, NumActive _nact,
                      NumVirtual _num_virtual_orbitals, const double* F,
                      size_t LDF, double* OG, size_t LDOG) {
  const auto norb = _norb.get();
  const auto nact = _nact.get();
  const auto ninact = _ninact.get();
  const auto num_virtual_orbitals = _num_virtual_orbitals.get();

#define FMAT(i, j) F[i + j * LDF]
#define GMAT(i, j) OG[i + j * LDOG]

  // Virtual - Inactive Block
  for (size_t i = 0; i < ninact; ++i)
    for (size_t a = 0; a < num_virtual_orbitals; ++a) {
      const size_t a_off = a + ninact + nact;
      GMAT(a_off, i) = 2 * (FMAT(a_off, i) - FMAT(i, a_off));
      GMAT(i, a_off) = -GMAT(a_off, i);
    }

  // Virtual - Active Block
  for (size_t x = 0; x < nact; ++x)
    for (size_t a = 0; a < num_virtual_orbitals; ++a) {
      const size_t x_off = x + ninact;
      const size_t a_off = a + ninact + nact;
      GMAT(a_off, x_off) = 2 * (FMAT(a_off, x_off) - FMAT(x_off, a_off));
      GMAT(x_off, a_off) = -GMAT(a_off, x_off);
    }

  // Active - Inactive Block
  for (size_t i = 0; i < ninact; ++i)
    for (size_t x = 0; x < nact; ++x) {
      const size_t x_off = x + ninact;
      GMAT(x_off, i) = 2 * (FMAT(x_off, i) - FMAT(i, x_off));
      GMAT(i, x_off) = -GMAT(x_off, i);
    }
}

void orbital_rotated_generalized_fock(
    NumOrbital _norb, NumInactive _ninact, NumActive _nact, const double* T,
    size_t LDT, const double* V, size_t LDV, const double* A1RDM, size_t LDD1,
    const double* A2RDM, size_t LDD2, const double* U, size_t LDU,
    double* T_trans, size_t LDTT, double* V_trans, size_t LDVT, double* F,
    size_t LDF) {
  const auto norb = _norb.get();
  const auto nact = _nact.get();
  const auto ninact = _ninact.get();

  // Transform Integrals
  two_index_transform(norb, norb, T, LDT, U, LDU, T_trans, LDTT);
  four_index_transform(norb, norb, V, LDV, U, LDU, V_trans, LDVT);

  // Compute Fock Matrix
  generalized_fock_matrix_comp_mat2(_norb, _ninact, _nact, T_trans, LDTT,
                                    V_trans, LDVT, A1RDM, LDD1, A2RDM, LDD2, F,
                                    LDF);
}

double orbital_rotated_energy(NumOrbital norb, NumInactive ninact,
                              NumActive nact, const double* T, size_t LDT,
                              const double* V, size_t LDV, const double* A1RDM,
                              size_t LDD1, const double* A2RDM, size_t LDD2,
                              const double* U, size_t LDU, double* T_trans,
                              size_t LDTT, double* V_trans, size_t LDVT,
                              double* F, size_t LDF) {
  orbital_rotated_generalized_fock(norb, ninact, nact, T, LDT, V, LDV, A1RDM,
                                   LDD1, A2RDM, LDD2, U, LDU, T_trans, LDTT,
                                   V_trans, LDVT, F, LDF);

  // Compute energy
  return energy_from_generalized_fock(ninact, nact, T_trans, LDTT, A1RDM, LDD1,
                                      F, LDF);
}

double orbital_rotated_energy(NumOrbital _norb, NumInactive ninact,
                              NumActive nact, const double* T, size_t LDT,
                              const double* V, size_t LDV, const double* A1RDM,
                              size_t LDD1, const double* A2RDM, size_t LDD2,
                              const double* U, size_t LDU) {
  const auto norb = _norb.get();
  size_t norb2 = norb * norb;
  size_t norb4 = norb2 * norb2;
  std::vector<double> T_trans(norb2), V_trans(norb4), F(norb2);

  return orbital_rotated_energy(_norb, ninact, nact, T, LDT, V, LDV, A1RDM,
                                LDD1, A2RDM, LDD2, U, LDU, T_trans.data(), norb,
                                V_trans.data(), norb, F.data(), norb);
}

void numerical_orbital_gradient(NumOrbital _norb, NumInactive ninact,
                                NumActive nact, const double* T, size_t LDT,
                                const double* V, size_t LDV,
                                const double* A1RDM, size_t LDD1,
                                const double* A2RDM, size_t LDD2, double* OG,
                                size_t LDOG) {
  const auto norb = _norb.get();

  const size_t norb2 = norb * norb;
  const size_t norb4 = norb2 * norb2;
  std::vector<double> F(norb2), T_trans(norb2), V_trans(norb4), K(norb2),
      U(norb2);

  auto energy = [&]() {
    return orbital_rotated_energy(
        _norb, ninact, nact, T, LDT, V, LDV, A1RDM, LDD1, A2RDM, LDD2, U.data(),
        norb, T_trans.data(), norb, V_trans.data(), norb, F.data(), norb);
  };

  const double dk = 0.001;
  for (size_t i = 0; i < norb; ++i)
    for (size_t a = i + 1; a < norb; ++a) {
      std::fill(K.begin(), K.end(), 0);
      K[a + i * norb] = dk;
      K[i + a * norb] = -dk;

      compute_orbital_rotation(_norb, 2.0, K.data(), norb, U.data(), norb);
      auto E_p2 = energy();

      compute_orbital_rotation(_norb, 1.0, K.data(), norb, U.data(), norb);
      auto E_p1 = energy();

      // U_map = (K_map).exp();
      compute_orbital_rotation(_norb, -1.0, K.data(), norb, U.data(), norb);
      auto E_m1 = energy();

      compute_orbital_rotation(_norb, -2.0, K.data(), norb, U.data(), norb);
      auto E_m2 = energy();

      OG[a + i * norb] = (E_m2 - E_p2 + 8 * (E_p1 - E_m1)) / (12 * dk);
    }
}

void numerical_orbital_hessian(NumOrbital _norb, NumInactive ninact,
                               NumActive nact, const double* T, size_t LDT,
                               const double* V, size_t LDV, const double* A1RDM,
                               size_t LDD1, const double* A2RDM, size_t LDD2,
                               double* OH, size_t LDOH) {
  const auto norb = _norb.get();

  const size_t norb2 = norb * norb;
  const size_t norb4 = norb2 * norb2;
  std::vector<double> F(norb2), T_trans(norb2), V_trans(norb4), K(norb2),
      Kx(norb2), Ky(norb2), U(norb2);

  auto energy = [&]() {
    return orbital_rotated_energy(
        _norb, ninact, nact, T, LDT, V, LDV, A1RDM, LDD1, A2RDM, LDD2, U.data(),
        norb, T_trans.data(), norb, V_trans.data(), norb, F.data(), norb);
  };

  const double dk = 0.001;
  for (size_t i = 0; i < norb; ++i)
    for (size_t a = i + 1; a < norb; ++a)
      for (size_t j = 0; j < norb; ++j)
        for (size_t b = i + 1; b < norb; ++b) {
          std::fill(Kx.begin(), Kx.end(), 0);
          std::fill(Ky.begin(), Ky.end(), 0);
          Kx[a + i * norb] = dk;
          Kx[i + a * norb] = -dk;
          Ky[b + j * norb] = dk;
          Ky[j + b * norb] = -dk;

          // E(x+h,y+h)
          std::fill(K.begin(), K.end(), 0);
          for (size_t p = 0; p < norb2; ++p) K[p] = Kx[p] + Ky[p];
          compute_orbital_rotation(_norb, 1.0, K.data(), norb, U.data(), norb);
          auto E_xp1_yp1 = energy();

          // E(x+2h,y+h)
          std::fill(K.begin(), K.end(), 0);
          for (size_t p = 0; p < norb2; ++p) K[p] = 2 * Kx[p] + Ky[p];
          compute_orbital_rotation(_norb, 1.0, K.data(), norb, U.data(), norb);
          auto E_xp2_yp1 = energy();

          // E(x+h,y+2h)
          std::fill(K.begin(), K.end(), 0);
          for (size_t p = 0; p < norb2; ++p) K[p] = Kx[p] + 2 * Ky[p];
          compute_orbital_rotation(_norb, 1.0, K.data(), norb, U.data(), norb);
          auto E_xp1_yp2 = energy();

          // E(x+2h,y+2h)
          std::fill(K.begin(), K.end(), 0);
          for (size_t p = 0; p < norb2; ++p) K[p] = 2 * Kx[p] + 2 * Ky[p];
          compute_orbital_rotation(_norb, 1.0, K.data(), norb, U.data(), norb);
          auto E_xp2_yp2 = energy();

          // E(x+h,y-h)
          std::fill(K.begin(), K.end(), 0);
          for (size_t p = 0; p < norb2; ++p) K[p] = Kx[p] - Ky[p];
          compute_orbital_rotation(_norb, 1.0, K.data(), norb, U.data(), norb);
          auto E_xp1_ym1 = energy();

          // E(x+2h,y-h)
          std::fill(K.begin(), K.end(), 0);
          for (size_t p = 0; p < norb2; ++p) K[p] = 2 * Kx[p] - Ky[p];
          compute_orbital_rotation(_norb, 1.0, K.data(), norb, U.data(), norb);
          auto E_xp2_ym1 = energy();

          // E(x+h,y-2h)
          std::fill(K.begin(), K.end(), 0);
          for (size_t p = 0; p < norb2; ++p) K[p] = Kx[p] - 2 * Ky[p];
          compute_orbital_rotation(_norb, 1.0, K.data(), norb, U.data(), norb);
          auto E_xp1_ym2 = energy();

          // E(x+2h,y-2h)
          std::fill(K.begin(), K.end(), 0);
          for (size_t p = 0; p < norb2; ++p) K[p] = 2 * Kx[p] - 2 * Ky[p];
          compute_orbital_rotation(_norb, 1.0, K.data(), norb, U.data(), norb);
          auto E_xp2_ym2 = energy();

          // E(x-h,y+h)
          std::fill(K.begin(), K.end(), 0);
          for (size_t p = 0; p < norb2; ++p) K[p] = -Kx[p] + Ky[p];
          compute_orbital_rotation(_norb, 1.0, K.data(), norb, U.data(), norb);
          auto E_xm1_yp1 = energy();

          // E(x-2h,y+h)
          std::fill(K.begin(), K.end(), 0);
          for (size_t p = 0; p < norb2; ++p) K[p] = -2 * Kx[p] + Ky[p];
          compute_orbital_rotation(_norb, 1.0, K.data(), norb, U.data(), norb);
          auto E_xm2_yp1 = energy();

          // E(x-h,y+2h)
          std::fill(K.begin(), K.end(), 0);
          for (size_t p = 0; p < norb2; ++p) K[p] = -Kx[p] + 2 * Ky[p];
          compute_orbital_rotation(_norb, 1.0, K.data(), norb, U.data(), norb);
          auto E_xm1_yp2 = energy();

          // E(x-2h,y+2h)
          std::fill(K.begin(), K.end(), 0);
          for (size_t p = 0; p < norb2; ++p) K[p] = -2 * Kx[p] + 2 * Ky[p];
          compute_orbital_rotation(_norb, 1.0, K.data(), norb, U.data(), norb);
          auto E_xm2_yp2 = energy();

          // E(x-h,y-h)
          std::fill(K.begin(), K.end(), 0);
          for (size_t p = 0; p < norb2; ++p) K[p] = -Kx[p] - Ky[p];
          compute_orbital_rotation(_norb, 1.0, K.data(), norb, U.data(), norb);
          auto E_xm1_ym1 = energy();

          // E(x-2h,y-h)
          std::fill(K.begin(), K.end(), 0);
          for (size_t p = 0; p < norb2; ++p) K[p] = -2 * Kx[p] - Ky[p];
          compute_orbital_rotation(_norb, 1.0, K.data(), norb, U.data(), norb);
          auto E_xm2_ym1 = energy();

          // E(x-h,y-2h)
          std::fill(K.begin(), K.end(), 0);
          for (size_t p = 0; p < norb2; ++p) K[p] = -Kx[p] - 2 * Ky[p];
          compute_orbital_rotation(_norb, 1.0, K.data(), norb, U.data(), norb);
          auto E_xm1_ym2 = energy();

          // E(x-2h,y-2h)
          std::fill(K.begin(), K.end(), 0);
          for (size_t p = 0; p < norb2; ++p) K[p] = -2 * Kx[p] - 2 * Ky[p];
          compute_orbital_rotation(_norb, 1.0, K.data(), norb, U.data(), norb);
          auto E_xm2_ym2 = energy();

          OH[a + i * LDOH + b * LDOH * LDOH + j * LDOH * LDOH * LDOH] =
              (E_xp1_yp1 + E_xm1_ym1 - E_xp1_ym1 - E_xm1_yp1) / (4 * dk * dk);
        }
}

}  // namespace macis
