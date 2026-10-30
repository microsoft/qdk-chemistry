/*
 * MACIS Copyright (c) 2023, The Regents of the University of California,
 * through Lawrence Berkeley National Laboratory (subject to receipt of
 * any required approvals from the U.S. Dept. of Energy). All rights reserved.
 * Portions Copyright (c) Microsoft Corporation.
 *
 * See LICENSE.txt for details
 */

#include <macis/fci/fci_hamiltonian.hpp>
#include <macis/fci/fci_rdms.hpp>
#include <macis/mcscf/cas.hpp>

namespace macis {

double compute_casci_rdms(
    MCSCFSettings settings, NumOrbital norb, size_t nalpha, size_t nbeta,
    double* T, double* V, double* ORDM, double* TRDM,
    std::vector<double>& C MACIS_MPI_CODE(, MPI_Comm comm)) {
  const size_t no = norb.get();
  const size_t no2 = no * no;
  const size_t no4 = no2 * no2;

  FCIHamiltonian H(no, nalpha, nbeta, T, no, V, no);
  double E0 =
      fci_ground_state(H, settings.ci_max_subspace, settings.ci_res_tol, C);

  if (ORDM and TRDM) {
    std::vector<double> aa(no2), bb(no2), aaaa(no4), bbbb(no4), aabb(no4);
    rank4_span<double> aabb_span(aabb.data(), no, no, no, no);
    fci_rdms_spin_dep(H.alpha_strings(), H.beta_strings(), C.data(),
                      matrix_span<double>(aa.data(), no, no),
                      matrix_span<double>(bb.data(), no, no),
                      rank4_span<double>(aaaa.data(), no, no, no, no),
                      rank4_span<double>(bbbb.data(), no, no, no, no),
                      aabb_span);

    for (size_t i = 0; i < no2; ++i) ORDM[i] = aa[i] + bb[i];
    // Spin trace: both same-spin blocks and both orderings of the
    // opposite-spin block
    rank4_span<double> trdm(TRDM, no, no, no, no);
    for (size_t s = 0; s < no; ++s)
      for (size_t r = 0; r < no; ++r)
        for (size_t q = 0; q < no; ++q)
          for (size_t p = 0; p < no; ++p) {
            const size_t i = p + no * (q + no * (r + no * s));
            trdm(p, q, r, s) = aaaa[i] + bbbb[i] + aabb_span(p, q, r, s) +
                               aabb_span(r, s, p, q);
          }
  }

#ifdef MACIS_ENABLE_MPI
  bcast(&E0, 1, 0, comm);
  bcast(C.data(), C.size(), 0, comm);
  if (ORDM and TRDM) {
    bcast(ORDM, no2, 0, comm);
    bcast(TRDM, no4, 0, comm);
  }
#endif /* MACIS_ENABLE_MPI */

  return E0;
}

}  // namespace macis
