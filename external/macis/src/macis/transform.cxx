/*
 * MACIS Copyright (c) 2023, The Regents of the University of California,
 * through Lawrence Berkeley National Laboratory (subject to receipt of
 * any required approvals from the U.S. Dept. of Energy). All rights reserved.
 *
 * See LICENSE.txt for details
 */

#include <blas.hh>
#include <macis/util/transform.hpp>

#define FOUR_IDX(arr, i, j, k, l, LDA1, LDA2, LDA3) \
  arr[i + j * LDA1 + k * LDA1 * LDA2 + l * LDA1 * LDA2 * LDA3]

#define FOUR_IDX_SAME(arr, i, j, k, l, LDA) \
  FOUR_IDX(arr, i, j, k, l, LDA, LDA, LDA)

#define TWO_IDX(arr, i, j, LDA) arr[i + j * LDA]

namespace macis {

void two_index_transform(size_t norb_old, size_t norb_new, const double* X,
                         size_t LDX, const double* C, size_t LDC, double* Y,
                         size_t LDY) {
#if 1
  // #TODO: The else branch is dead code, need to be removed.
  // TMP(i,q) = X(i,j) * C(j,q)
  std::vector<double> TMP(norb_old * norb_new);
  blas::gemm(blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,
             norb_old, norb_new, norb_old, 1.0, X, LDX, C, LDC, 0.0, TMP.data(),
             norb_old);

  // Y(p,q) = C(i,p) * TMP(i,q)
  blas::gemm(blas::Layout::ColMajor, blas::Op::Trans, blas::Op::NoTrans,
             norb_new, norb_new, norb_old, 1.0, C, LDC, TMP.data(), norb_old,
             0.0, Y, LDY);
#else
  for (size_t p = 0; p < norb_new; ++p)
    for (size_t q = 0; q < norb_new; ++q) {
      TWO_IDX(Y, p, q, LDY) = 0.0;
      for (size_t i = 0; i < norb_old; ++i)
        for (size_t j = 0; j < norb_old; ++j) {
          TWO_IDX(Y, p, q, LDY) += TWO_IDX(C, i, p, LDC) *
                                   TWO_IDX(C, j, q, LDC) *
                                   TWO_IDX(X, i, j, LDX);
        }
    }
#endif /* 1 */
}

void four_index_transform(size_t norb_old, size_t norb_new, const double* X,
                          size_t LDX, const double* C, size_t LDC, double* Y,
                          size_t LDY) {
#if 1
  // #TODO: Same comment as above, need to trim dead code.
  size_t norb_new2 = norb_new * norb_new;
  size_t norb_new3 = norb_new2 * norb_new;
  size_t norb_old2 = norb_old * norb_old;
  size_t norb_old3 = norb_old2 * norb_old;

  std::vector<double> tmp1(
      std::max(norb_new * norb_old3, norb_old * norb_new3)),
      tmp2(norb_new2 * norb_old2);
  // #TODO: This may be a longer term work item: the temporaries scale as
  // O(n^4) doubles; may need a blocking strategy or explicity memory gurard
  // for large norb.

  // 1st Quarter
  // TMP1(p,j,k,l) = C(i,p) * X(i,j,k,l)
  // TMP1(p, jkl)  = C(i,p) * X(i,jkl)
  blas::gemm(blas::Layout::ColMajor, blas::Op::Trans, blas::Op::NoTrans,
             norb_new, norb_old3, norb_old, 1.0, C, LDC, X, LDX, 0.0,
             tmp1.data(), norb_new);

  // 2nd Quarter
  // TMP2(p,q,k,l) = C(j,q) * TMP1(p,j,k,l)
  // TMP2_kl(p,q)  = TMP1_kl(p,j) * C(j,q)
  for (size_t kl = 0; kl < norb_old2; ++kl) {
    auto TMP1_kl = tmp1.data() + kl * norb_old * norb_new;
    auto TMP2_kl = tmp2.data() + kl * norb_new2;
    blas::gemm(blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,
               norb_new, norb_new, norb_old, 1.0, TMP1_kl, norb_new, C, LDC,
               0.0, TMP2_kl, norb_new);
  }

  // 3rd Quarter
  // TMP1(p,q,r,l) = C(k,r) * TMP2(p,q,k,l)
  // TMP1_l(pq,r)  = TMP2_l(pq,k) * C(k,r)
  for (size_t l = 0; l < norb_old; ++l) {
    auto TMP2_l = tmp2.data() + l * norb_new2 * norb_old;
    auto TMP1_l = tmp1.data() + l * norb_new3;
    blas::gemm(blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,
               norb_new2, norb_new, norb_old, 1.0, TMP2_l, norb_new2, C, LDC,
               0.0, TMP1_l, norb_new2);
  }

  // 4th Quarter
  // Y(p,q,r,s) = C(l,s) * TMP1(p,q,r,l)
  // Y(pqr,s) = V(pqr,l) * C(l,s)
  blas::gemm(blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,
             norb_new3, norb_new, norb_old, 1.0, tmp1.data(), norb_new3, C, LDC,
             0.0, Y, norb_new3);

#else

  for (size_t p = 0; p < norb_new; ++p)
    for (size_t q = 0; q < norb_new; ++q)
      for (size_t r = 0; r < norb_new; ++r)
        for (size_t s = 0; s < norb_new; ++s) {
          FOUR_IDX_SAME(Y, p, q, r, s, LDY) = 0.0;
          for (size_t i = 0; i < norb_old; ++i)
            for (size_t j = 0; j < norb_old; ++j)
              for (size_t k = 0; k < norb_old; ++k)
                for (size_t l = 0; l < norb_old; ++l) {
                  FOUR_IDX_SAME(Y, p, q, r, s, LDY) +=
                      TWO_IDX(C, i, p, LDC) * TWO_IDX(C, j, q, LDC) *
                      TWO_IDX(C, k, r, LDC) * TWO_IDX(C, l, s, LDC) *
                      FOUR_IDX_SAME(X, i, j, k, l, LDX);
                }
        }

#endif /* 1 */
}

}  // namespace macis
