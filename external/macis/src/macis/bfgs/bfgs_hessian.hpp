/*
 * MACIS Copyright (c) 2023, The Regents of the University of California,
 * through Lawrence Berkeley National Laboratory (subject to receipt of
 * any required approvals from the U.S. Dept. of Energy). All rights reserved.
 *
 * See LICENSE.txt for details
 */

#pragma once
#include <iostream>
#include <memory>

#include "bfgs_traits.hpp"

namespace bfgs {

template <typename Functor>
struct BFGSHessian {
  using arg_type = detail::arg_type_t<Functor>;

  std::vector<arg_type> sk, yk;
  std::vector<double> rhok;

  virtual ~BFGSHessian() noexcept = default;

  virtual void update(const arg_type& s, const arg_type& y) {
    const auto ys = Functor::dot(y, s);
    rhok.emplace_back(1. / ys);

    sk.emplace_back(s);
    yk.emplace_back(y);
  }

  virtual void apply_H0(arg_type& x) {}  // Null call

  arg_type apply(const arg_type& x) {
    arg_type q = x;
    const int64_t nk = sk.size();

    std::vector<double> alpha(nk);
    for (int64_t i = nk - 1; i >= 0; i--) {
      alpha[i] = Functor::dot(sk[i], q) * rhok[i];
      Functor::axpy(-alpha[i], yk[i], q);
    }
    apply_H0(q);
    for (int64_t i = 0; i < nk; ++i) {
      const auto beta = Functor::dot(yk[i], q) * rhok[i];
      Functor::axpy(alpha[i] - beta, sk[i], q);
    }
    return q;
  }
};

template <typename Functor>
std::unique_ptr<BFGSHessian<Functor>> make_identity_hessian() {
  return std::make_unique<BFGSHessian<Functor>>();
}

template <typename Functor>
struct UpdatedScaledBFGSHessian : public BFGSHessian<Functor> {
  using base_type = BFGSHessian<Functor>;
  using arg_type = typename base_type::arg_type;

  static constexpr double inf = std::numeric_limits<double>::infinity();
  double gamma_k = 1.0;

  // Scaling parameter ala doi:10.1007/BF01589116
  void update(const arg_type& s, const arg_type& y) override final {
    base_type::update(s, y);
    const auto y_nrm = Functor::norm(y);
    gamma_k = (y_nrm * y_nrm) * this->rhok.back();
  }

  // Scaling H_0 ala doi:10.1007/BF01589116
  void apply_H0(arg_type& x) override final { Functor::scal(1. / gamma_k, x); }
};

template <typename Functor>
std::unique_ptr<BFGSHessian<Functor>> make_updated_scaled_hessian() {
  return std::make_unique<UpdatedScaledBFGSHessian<Functor>>();
}

template <typename Functor>
struct StaticScaledBFGSHessian : public BFGSHessian<Functor> {
  using base_type = BFGSHessian<Functor>;
  using arg_type = typename base_type::arg_type;

  std::optional<double> gamma_0;

  // Scaling parameter ala doi:10.1007/BF01589116
  void update(const arg_type& s, const arg_type& y) override final {
    base_type::update(s, y);
    if (!gamma_0.has_value()) {
      const auto y_nrm = Functor::norm(y);
      gamma_0 = (y_nrm * y_nrm) * this->rhok.back();
    }
  }

  // Scaling H_0 ala doi:10.1007/BF01589116
  void apply_H0(arg_type& x) override final {
    if (gamma_0.has_value()) Functor::scal(1. / gamma_0.value(), x);
  }
};

template <typename Functor>
std::unique_ptr<BFGSHessian<Functor>> make_static_scaled_hessian() {
  return std::make_unique<StaticScaledBFGSHessian<Functor>>();
}

template <typename Functor>
struct RuntimeInitializedBFGSHessian : public BFGSHessian<Functor> {
  using base_type = BFGSHessian<Functor>;
  using arg_type = typename base_type::arg_type;
  using op_type = std::function<void(arg_type&)>;

  op_type m_H0;

  RuntimeInitializedBFGSHessian() = delete;
  RuntimeInitializedBFGSHessian(const op_type& op) : m_H0(op) {}

  void apply_H0(arg_type& x) override final { m_H0(x); }
};

template <typename Functor>
struct DiagInitializedBFGSHessian : public BFGSHessian<Functor> {
  using base_type = BFGSHessian<Functor>;
  using arg_type = typename base_type::arg_type;
  using op_type = std::function<void(arg_type&)>;
  std::vector<double> inv_diag;

  DiagInitializedBFGSHessian() = delete;
  DiagInitializedBFGSHessian(size_t n, double* D) {
    inv_diag.resize(n);
    std::transform(D, D + n, inv_diag.begin(), [](auto x) { return 1. / x; });
  }

  void apply_H0(arg_type& x) override final {
    for (size_t i = 0; i < inv_diag.size(); ++i) {
      x[i] *= inv_diag[i];
    }
  }
};

}  // namespace bfgs
