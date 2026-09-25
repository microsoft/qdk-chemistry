// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include "util/libint2_engine.h"

#include <libint2/engine.h>
#include <qdk/chemistry/scf/util/libint2_util.h>

#include <qdk/chemistry/utils/logger.hpp>
#include <stdexcept>

namespace qdk::chemistry::scf::libint2_util {

::libint2::Shell convert_to_libint_shell(const Shell& o, bool pure) {
  QDK_LOG_TRACE_ENTERING();
  ::libint2::Shell sh;
  sh.O = o.O;
  sh.contr.resize(1);
  sh.contr[0].l = o.angular_momentum;
  sh.contr[0].pure = (pure && o.angular_momentum >= 2);
  sh.alpha.reserve(o.contraction);
  sh.contr[0].coeff.reserve(o.contraction);

  for (uint64_t i = 0; i < o.contraction; i++) {
    sh.alpha.push_back(o.exponents[i]);
    sh.contr[0].coeff.push_back(o.coefficients[i]);
  }
  return ::libint2::Shell(std::move(sh.alpha), std::move(sh.contr), sh.O,
                          false);
}

::libint2::BasisSet convert_to_libint_basisset(const BasisSet& o) {
  QDK_LOG_TRACE_ENTERING();
  std::vector<::libint2::Shell> shells;
  shells.reserve(o.shells.size());
  for (auto& sh : o.shells) {
    shells.push_back(convert_to_libint_shell(sh, o.pure));
  }
  return ::libint2::BasisSet(std::move(shells));
}

std::vector<size_t> shell_to_basis_function(const BasisSet& obs) {
  return convert_to_libint_basisset(obs).shell2bf();
}

BasisView::BasisView(const ::libint2::BasisSet& basis)
    : basis_(&basis),
      shells_(basis.shells()),
      shell2bf_(&basis.shell2bf()),
      nbf_(basis.nbf()),
      max_nprim_(basis.max_nprim()),
      max_l_(basis.max_l()) {}

Basis::Basis(const qdk::chemistry::scf::BasisSet& basis)
    : Basis(std::make_shared<const ::libint2::BasisSet>(
          convert_to_libint_basisset(basis))) {}

Basis::Basis(std::vector<::libint2::Shell> shells)
    : Basis(std::make_shared<const ::libint2::BasisSet>(std::move(shells))) {}

Basis::Basis(std::shared_ptr<const ::libint2::BasisSet> basis)
    : BasisView(*basis), storage_(std::move(basis)) {}

namespace {
constexpr ::libint2::Operator native_operator(Operator op) {
  switch (op) {
    case Operator::overlap:
      return ::libint2::Operator::overlap;
    case Operator::kinetic:
      return ::libint2::Operator::kinetic;
    case Operator::nuclear:
      return ::libint2::Operator::nuclear;
    case Operator::emultipole1:
      return ::libint2::Operator::emultipole1;
    case Operator::emultipole2:
      return ::libint2::Operator::emultipole2;
    case Operator::coulomb:
      return ::libint2::Operator::coulomb;
    case Operator::erf_coulomb:
      return ::libint2::Operator::erf_coulomb;
  }
  throw std::invalid_argument("Unsupported Libint2 integral operator");
}
}  // namespace

Engine::Engine() : engine_(std::make_unique<::libint2::Engine>()) {
  update_results();
}

Engine::Engine(Operator op, size_t max_nprim, int max_l, int deriv_order,
               double precision)
    : engine_(std::make_unique<::libint2::Engine>(
          native_operator(op), max_nprim, max_l, deriv_order, precision)) {
  update_results();
}

Engine::Engine(const Engine& other) { *this = other; }

Engine::Engine(Engine&& other) noexcept = default;

Engine& Engine::operator=(const Engine& other) {
  if (this != &other) {
    if (!other.engine_ ||
        other.engine_->oper() == ::libint2::Operator::invalid) {
      throw ::libint2::Engine::using_default_initialized();
    }
    auto engine = std::move(engine_);
    results_ = {};
    if (!engine) engine = std::make_unique<::libint2::Engine>();
    *engine = *other.engine_;
    engine_ = std::move(engine);
    update_results();
  }
  return *this;
}

Engine& Engine::operator=(Engine&& other) noexcept = default;

Engine::~Engine() = default;

std::vector<Engine> Engine::make_pool(size_t count, Engine&& prototype) {
  std::vector<Engine> engines;
  if (count == 0) return engines;
  if (!prototype.engine_ ||
      prototype.engine_->oper() == ::libint2::Operator::invalid) {
    throw ::libint2::Engine::using_default_initialized();
  }
  engines.reserve(count);
  engines.emplace_back(std::move(prototype));
  for (size_t i = 1; i < count; ++i) {
    engines.emplace_back(engines.front());
  }
  return engines;
}

Engine& Engine::set(::libint2::BraKet braket) {
  engine_->set(braket);
  update_results();
  return *this;
}

Engine& Engine::set(::libint2::ScreeningMethod screening) {
  engine_->set(screening);
  return *this;
}

Engine& Engine::set_precision(double precision) {
  engine_->set_precision(precision);
  return *this;
}

Engine& Engine::set_params(double omega) {
  engine_->set_params(omega);
  update_results();
  return *this;
}

Engine& Engine::set_params(const std::array<double, 3>& origin) {
  engine_->set_params(origin);
  update_results();
  return *this;
}

Engine& Engine::set_params(
    const std::vector<std::pair<double, std::array<double, 3>>>& charges) {
  engine_->set_params(charges);
  update_results();
  return *this;
}

const Engine::Results& Engine::update_results() {
  const auto& results = engine_->results();
  results_ = {results.data(), results.size()};
  return results_;
}

const Engine::Results& Engine::compute1(const ::libint2::Shell& bra,
                                        const ::libint2::Shell& ket) {
  engine_->compute1(bra, ket);
  return update_results();
}

template <Operator op, ::libint2::BraKet braket, size_t deriv_order>
const Engine::Results& Engine::compute2(const ::libint2::Shell& bra1,
                                        const ::libint2::Shell& bra2,
                                        const ::libint2::Shell& ket1,
                                        const ::libint2::Shell& ket2,
                                        const ::libint2::ShellPair* spbra,
                                        const ::libint2::ShellPair* spket) {
  engine_->compute2<native_operator(op), braket, deriv_order>(
      bra1, bra2, ket1, ket2, spbra, spket);
  return update_results();
}

// Keep upstream's header-only mode: its non-inline mode also instantiates the
// generic all-operator dispatch table. Only these shell kernels are needed by
// the private first-party boundary; none of Libint2's definitions are altered.
#define QDK_INSTANTIATE_LIBINT2_COMPUTE2(op, braket, deriv)         \
  template const Engine::Results&                                   \
  Engine::compute2<Operator::op, ::libint2::BraKet::braket, deriv>( \
      const ::libint2::Shell&, const ::libint2::Shell&,             \
      const ::libint2::Shell&, const ::libint2::Shell&,             \
      const ::libint2::ShellPair*, const ::libint2::ShellPair*);

QDK_INSTANTIATE_LIBINT2_COMPUTE2(coulomb, xx_xx, 0)
QDK_INSTANTIATE_LIBINT2_COMPUTE2(coulomb, xx_xx, 1)
QDK_INSTANTIATE_LIBINT2_COMPUTE2(erf_coulomb, xx_xx, 0)
QDK_INSTANTIATE_LIBINT2_COMPUTE2(erf_coulomb, xx_xx, 1)
QDK_INSTANTIATE_LIBINT2_COMPUTE2(coulomb, xs_xx, 0)
QDK_INSTANTIATE_LIBINT2_COMPUTE2(coulomb, xs_xx, 1)
QDK_INSTANTIATE_LIBINT2_COMPUTE2(coulomb, xs_xs, 0)
QDK_INSTANTIATE_LIBINT2_COMPUTE2(coulomb, xs_xs, 1)

#undef QDK_INSTANTIATE_LIBINT2_COMPUTE2

}  // namespace qdk::chemistry::scf::libint2_util
