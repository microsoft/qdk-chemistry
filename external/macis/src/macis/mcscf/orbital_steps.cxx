/*
 * MACIS Copyright (c) 2023, The Regents of the University of California,
 * through Lawrence Berkeley National Laboratory (subject to receipt of
 * any required approvals from the U.S. Dept. of Energy). All rights reserved.
 *
 * See LICENSE.txt for details
 */

#include <macis/mcscf/orbital_hessian.hpp>
#include <macis/mcscf/orbital_steps.hpp>

namespace macis {

void precond_cg_orbital_step(NumOrbital norb, NumInactive ninact,
                             NumActive nact, NumVirtual nvirt, const double* Fi,
                             size_t LDFi, const double* Fa, size_t LDFa,
                             const double* F, size_t LDF, const double* A1RDM,
                             size_t LDD, const double* OG, double* K_lin) {
  const size_t no = norb.get(), ni = ninact.get(), na = nact.get(),
               nv = nvirt.get(), orb_rot_sz = nv * (na + ni) + na * ni;
  std::vector<double> DH(orb_rot_sz);

  // Compute approximate diagonal hessian
  approx_diag_hessian(ninact, nact, nvirt, Fi, LDFi, Fa, LDFa, A1RDM, LDD, F,
                      LDF, DH.data());

  // Precondition the gradient
  for (size_t p = 0; p < orb_rot_sz; ++p) {
    K_lin[p] = -OG[p] / DH[p];
  }
}

}  // namespace macis

#if 0
/******* THIS IS OLD CODE TO BE REVISITED *******/

namespace acsi {

struct bfgs_mcscf_functor {
    using argument_type = Eigen::VectorXd;
    using arg_type = argument_type;
    using return_type   = double;

    NumOrbital  norb;
    NumInactive ninact;
    NumActive   nact;
    NumVirtual  nvirt;

    const double grad_tol;
    const double E_core;
    const double *T, *V, *A1RDM, *A2RDM;

    bfgs_mcscf_functor() = delete;

    bfgs_mcscf_functor(NumOrbital no, NumInactive ni, NumActive na,
      NumVirtual nv, double ec, const double* t, const double* v,
      const double* d1, const double* d2, double tol) :
      grad_tol(tol),
      norb(no), ninact(ni), nact(na), nvirt(nv), E_core(ec),
      T(t), V(v), A1RDM(d1), A2RDM(d2) {}

    bool converged(const arg_type& X, const arg_type& G) {
      return G.norm() < grad_tol;
    }

    return_type eval(const arg_type& K) {
      // Expand linear rotation vector into full antisymmetric
      // matrix
      std::vector<double> K_expand(norb.get() * norb.get());
      linear_orb_rot_to_matrix(ninact, nact, nvirt, K.data(),
        K_expand.data(), norb.get());

      // Compute U = EXP[-K]
      std::vector<double> U(norb.get() * norb.get());
      compute_orbital_rotation(norb, 1.0, K_expand.data(), norb.get(),
        U.data(), norb.get() );

      // Compute energy evaluated at rotated orbitals
      return orbital_rotated_energy(norb, ninact, nact, T, norb.get(),
        V, norb.get(), A1RDM, nact.get(), A2RDM, nact.get(), U.data(),
        norb.get()) + E_core;

    }

    arg_type grad(const arg_type& K) {

      // Expand linear rotation vector into full antisymmetric
      // matrix
      std::vector<double> K_expand(norb.get() * norb.get());
      linear_orb_rot_to_matrix(ninact, nact, nvirt, K.data(),
        K_expand.data(), norb.get());

      // Compute U = EXP[-K]
      std::vector<double> U(norb.get() * norb.get());
      compute_orbital_rotation(norb, 1.0, K_expand.data(), norb.get(),
        U.data(), norb.get() );


      // Compute Rotated Fock Matrix
      const size_t no = norb.get();
      std::vector<double> Tt(no*no), Vt(no*no*no*no);
      std::vector<double> F(no*no);
      macis::orbital_rotated_generalized_fock(norb, ninact, nact,
        T, no, V, no, A1RDM, nact.get(), A2RDM, nact.get(), U.data(),
        no, Tt.data(), no, Vt.data(), no, F.data(), no);

      arg_type G(K.size());
      fock_to_linear_orb_grad(ninact, nact, nvirt, F.data(), no,
        G.data());

      return G;
    }

    return_type operator()(const arg_type& x, arg_type& g) {
        g = grad(x);
        return eval(x);
    }

    static return_type dot( const arg_type& x, const arg_type& y ) {
        return x.dot(y);
    }

    static void axpy( return_type alpha, const arg_type& x, arg_type& y ) {
        y += alpha * x;
    }

    static return_type norm( const arg_type& x ) {
      return x.norm();
    }


    static void scal( return_type alpha, arg_type& x ) {
      x *= alpha;
    }
    static arg_type subtract( const arg_type& x, const arg_type& y ){
        return x - y;
    }

};

}

namespace macis {

struct AugHessianOperator {

  NumOrbital  norb;
  NumInactive ninact;
  NumActive   nact;
  NumVirtual  nvirt;
  const double *m_T, *m_V, *m_OG, *m_A1RDM, *m_A2RDM;

  void operator_action(size_t m, double alpha, const double* V, size_t /* */,
    double beta, double* AV, size_t /* */) const {

    const size_t no = norb.get();
    const size_t ni = ninact.get();
    const size_t na = nact.get();
    const size_t nv = nvirt.get();
    const size_t orb_rot_sz = nv*(na + ni) + na*ni;

    // [AV0] = [0 G**T ] [ V0 ]
    // [AK ]   [G H    ] [ K  ]
    const double* K  = V  + 1;
    double* AK = AV + 1;

    // AV0 = beta*AV0 + alpha * G**T * K
    *AV = beta*(*AV) + alpha * blas::dot(orb_rot_sz, m_OG, 1, K, 1);

    // HK = H*K
    std::vector<double> HK(orb_rot_sz);
    orb_orb_hessian_contract(norb, ninact, nact, nvirt, m_T, no,
      m_V, no, m_A1RDM, na, m_A2RDM, na, m_OG, K, HK.data());

    // AK = beta*AK + alpha*(H*K + V0*G)
    blas::scal(orb_rot_sz, beta, AK, 1);
    blas::axpy(orb_rot_sz, alpha, HK.data(), 1, AK, 1);
    blas::axpy(orb_rot_sz, alpha*(*V), m_OG, 1, AK, 1);

  }

};



void optimize_orbitals(MCSCFSettings settings, NumOrbital norb,
  NumInactive ninact, NumActive nact, NumVirtual nvirt, double E_core,
  const double* T, size_t LDT, const double* V, size_t LDV,
  const double* A1RDM, size_t LDD1, const double* A2RDM, size_t LDD2,
  double* OG, double *K, size_t LDK) {

  const size_t no = norb.get(), ni = ninact.get(), na = nact.get(), nv = nvirt.get();
  const size_t orb_rot_sz = nv*(na+ni) + na*ni;
  std::vector<double> DH(orb_rot_sz);

#if 1

  // Compute inactive Fock
  std::vector<double> Fi(no*no);
  inactive_fock_matrix(norb, ninact, T, LDT, V, LDV, Fi.data(), no);

  // Compute active fock
  std::vector<double> Fa(no*no);
  active_fock_matrix(norb, ninact, nact, V, LDV, A1RDM, LDD1, Fa.data(), no);

  // Compute Q matrix
  std::vector<double> Q(na*no);
  aux_q_matrix(nact, norb, ninact, V, LDV, A2RDM, LDD2, Q.data(), na);

  // Compute generalized Fock
  std::vector<double> F(no*no);
  generalized_fock_matrix(norb, ninact, nact, Fi.data(), no, Fa.data(), no,
    A1RDM, LDD1, Q.data(), na, F.data(), no);

  // Compute approximate diagonal hessian
  approx_diag_hessian(ninact, nact, nvirt, Fi.data(), no, Fa.data(), no,
    A1RDM, LDD1, F.data(), no, DH.data() );

  // Compute Gradient
  ///std::vector<double> OG(orb_rot_sz);
  fock_to_linear_orb_grad(ninact, nact, nvirt, F.data(), no, OG);

#if 1
  // Precondition the gradient
  std::vector<double> step(orb_rot_sz);
  for(size_t p = 0; p < orb_rot_sz; ++p) {
    // #TODO: DH[p] can be very small or zero for near-singular Hessians; add safety check or damping.
    step[p] = OG[p] / DH[p];
  }
#else
  std::vector<double> AH_diag(DH.size()+1);
  AH_diag[0] = 1.;
  std::copy(DH.begin(), DH.end(), AH_diag.begin()+1);

  AugHessianOperator op{norb, ninact, nact, nvirt, T, V, OG.data(),
    A1RDM, A2RDM};

  std::vector<double> X(AH_diag.size(), 0);
  auto D_min = std::min_element(DH.begin(), DH.end());
  auto min_idx = std::distance( DH.begin(), D_min );
  // #TODO: Replace raw std::cout with the logger.
  std::cout << "DMIN = " << *D_min << std::endl;
  X[min_idx+1] = 1.;

  auto mu = davidson(AH_diag.size(), 100, op, AH_diag.data(), 1e-8, X.data() );

  for(size_t p = 0; p < orb_rot_sz; ++p) {
    OG[p] = X[p+1] / X[0];
  }
#endif /* 1 */

  // Get NRM of step
  auto step_nrm = blas::nrm2(orb_rot_sz,step.data(),1);

  // Get ABSMAX of step
  auto step_max = *std::max_element(step.begin(), step.end(),
    [](auto a, auto b){return std::abs(a) < std::abs(b);});
  step_max = std::abs(step_max);

  auto logger = spdlog::get("casscf");
  logger->debug("{:12}step_nrm = {:.4e}, step_amax = {:.4e}", "",
    step_nrm, step_max);

  if( step_max > 0.5 ) {
    logger->info("  * decresing step from {:.2f} to {:.2f}", step_max, 0.5);
    blas::scal(orb_rot_sz, -0.5 / step_max, step.data(), 1);
  } else {
    blas::scal(orb_rot_sz, -1.0, step.data(), 1);
  }

  // Expand into full matrix
  linear_orb_rot_to_matrix(ninact, nact, nvirt, step.data(), K, LDK);

#else

  // Compute Diagonal Hessian Approximation
  approx_diag_hessian(norb, ninact, nact, nvirt, T, LDT, V, LDV, A1RDM, LDD1,
    A2RDM, LDD2, DH.data() );

  // Create BFGS Functor
  bfgs_mcscf_functor op(norb, ninact, nact, nvirt, E_core, T, V, A1RDM, A2RDM,
    settings.orb_grad_tol_bfgs);

  // Initial diagonal hessian
  bfgs::DiagInitializedBFGSHessian<bfgs_mcscf_functor> H0(DH.size(), DH.data());

  // Initial guess of K = 0
  Eigen::VectorXd K0(nv*(na+ni) + na*ni);
  K0.fill(0.);

  // Optimize Orbitals
  K0 = bfgs::bfgs(op, K0, H0, bfgs::BFGSSettings{settings.max_bfgs_iter});

  // Expand into full matrix
  linear_orb_rot_to_matrix(ninact, nact, nvirt, K0.data(), K, LDK);

#endif /* 1 */

}
}
#endif /* 0 */
