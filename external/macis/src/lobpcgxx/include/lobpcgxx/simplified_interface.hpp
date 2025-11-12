#pragma once

#include "convergence.hpp"
#include "driver.hpp"

namespace lobpcgxx {

template <typename T>
void lobpcg(const lobpcg_settings& settings, int64_t N, int64_t K, int64_t NR,
            const lobpcg_operator<T> op_functor, detail::real_t<T>* LAMR, T* V,
            int64_t LDV, detail::real_t<T>* res, T* WORK, int64_t& LWORK,
            lobpcg_convergence<T>& conv) {
  lobpcg_convergence_check<T> check = lobpcg_relres_convergence_check<T>;
  lobpcg(settings, N, K, NR, op_functor, LAMR, V, LDV, res, WORK, LWORK, check,
         conv);
}

template <typename T>
void lobpcg(const lobpcg_settings& settings, int64_t N, int64_t K, int64_t NR,
            const lobpcg_operator<T> op_functor, detail::real_t<T>* LAMR, T* V,
            int64_t LDV, detail::real_t<T>* res, lobpcg_convergence<T>& conv) {
  auto LWORK = lobpcg_lwork(N, K);
  std::vector<T> work(LWORK);

  lobpcg(settings, N, K, NR, op_functor, LAMR, V, LDV, res, work.data(), LWORK,
         conv);
}

template <typename T>
void lobpcg(const lobpcg_settings& settings, int64_t N, int64_t K, int64_t NR,
            const lobpcg_operator<T> op_functor, detail::real_t<T>* LAMR, T* V,
            int64_t LDV, detail::real_t<T>* res, T* WORK, int64_t& LWORK) {
  lobpcg_convergence<T> conv;
  lobpcg(settings, N, K, NR, op_functor, LAMR, V, LDV, res, WORK, LWORK, conv);
}

template <typename T>
void lobpcg(const lobpcg_settings& settings, int64_t N, int64_t K, int64_t NR,
            const lobpcg_operator<T> op_functor, detail::real_t<T>* LAMR, T* V,
            int64_t LDV, detail::real_t<T>* res) {
  auto LWORK = lobpcg_lwork(N, K);
  std::vector<T> work(LWORK);

  lobpcg(settings, N, K, NR, op_functor, LAMR, V, LDV, res, work.data(), LWORK);
}

}  // namespace lobpcgxx
