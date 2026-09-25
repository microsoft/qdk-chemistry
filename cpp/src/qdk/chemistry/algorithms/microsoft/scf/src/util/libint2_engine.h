// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once

#include <libint2/braket.h>
#include <libint2/shell.h>
#include <qdk/chemistry/scf/fwd.h>
#include <qdk/chemistry/scf/util/libint2_fwd.h>

#include <array>
#include <cstddef>
#include <limits>
#include <memory>
#include <span>
#include <utility>
#include <vector>

namespace qdk::chemistry::scf::libint2_util {

// A non-owning, immutable view avoids basis.h, which includes engine.impl.h.
// The Libint2 basis must outlive the view and remain unmodified.
class BasisView {
 public:
  explicit BasisView(const ::libint2::BasisSet& basis);

  const ::libint2::Shell& operator[](size_t i) const { return shells_[i]; }
  size_t size() const { return shells_.size(); }
  long nbf() const { return nbf_; }
  size_t max_nprim() const { return max_nprim_; }
  long max_l() const { return max_l_; }
  const std::vector<size_t>& shell2bf() const { return *shell2bf_; }
  const ::libint2::BasisSet& get() const { return *basis_; }

 protected:
  BasisView() = default;

 private:
  const ::libint2::BasisSet* basis_ = nullptr;
  std::span<const ::libint2::Shell> shells_;
  const std::vector<size_t>* shell2bf_ = nullptr;
  long nbf_ = -1;
  size_t max_nprim_ = 0;
  long max_l_ = -1;
};

// Own the immutable basis without exposing Libint2's engine to shell loops.
class Basis : public BasisView {
 public:
  Basis() = default;
  explicit Basis(const qdk::chemistry::scf::BasisSet& basis);
  explicit Basis(std::vector<::libint2::Shell> shells);
  explicit Basis(std::shared_ptr<const ::libint2::BasisSet> basis);

  const std::shared_ptr<const ::libint2::BasisSet>& storage() const {
    return storage_;
  }

 private:
  std::shared_ptr<const ::libint2::BasisSet> storage_;
};

// These are the operators used by the first-party integral implementations.
// The public Libint2 API and its header-only configuration are unchanged.
enum class Operator {
  overlap,
  kinetic,
  nuclear,
  emultipole1,
  emultipole2,
  coulomb,
  erf_coulomb
};

// Each thread owns its engine and buffers. Only shell-level calls cross this
// boundary; Libint2's primitive loops retain their normal optimized
// definitions.
class Engine {
 public:
  using Results = std::span<const double* const>;

  Engine();
  Engine(Operator op, size_t max_nprim, int max_l, int deriv_order = 0,
         double precision = std::numeric_limits<double>::epsilon());
  /**
   * @brief Copy an initialized engine into independent scratch storage.
   * @param other Engine whose configuration is copied
   * @throws std::logic_error if other is uninitialized or moved from
   */
  Engine(const Engine& other);
  Engine(Engine&& other) noexcept;
  /**
   * @brief Copy configuration, reusing the destination's native storage.
   * @param other Initialized source engine; self-assignment is a no-op
   * @return Reference to this engine
   * @throws std::logic_error if other is uninitialized or moved from
   * @note An unusable source leaves the destination unchanged. If native
   * assignment or allocation fails, the destination has empty results and can
   * be assigned again. Assignment into a moved-from destination is supported.
   */
  Engine& operator=(const Engine& other);
  Engine& operator=(Engine&& other) noexcept;
  ~Engine();

  /**
   * @brief Construct independently buffered engines for a worker pool.
   * @param count Number of engines to create
   * @param prototype Initialized engine moved into the first entry
   * @return Engines with identical configuration and independent scratch space
   * @throws std::logic_error if a nonempty pool receives an unusable prototype
   * @note Remaining entries are deep copies of the first. A zero count returns
   * an empty pool without consuming the prototype.
   */
  static std::vector<Engine> make_pool(size_t count, Engine&& prototype);

  Engine& set(::libint2::BraKet braket);
  Engine& set(::libint2::ScreeningMethod screening);
  Engine& set_precision(double precision);
  Engine& set_params(double omega);
  Engine& set_params(const std::array<double, 3>& origin);
  Engine& set_params(
      const std::vector<std::pair<double, std::array<double, 3>>>& charges);

  // A retained reference observes buffer changes after compute/set calls, just
  // like libint2::Engine::results(). Its pointers are valid until the next
  // call.
  const Results& results() const { return results_; }
  const Results& compute1(const ::libint2::Shell& bra,
                          const ::libint2::Shell& ket);

  template <Operator op, ::libint2::BraKet braket, size_t deriv_order>
  const Results& compute2(const ::libint2::Shell& bra1,
                          const ::libint2::Shell& bra2,
                          const ::libint2::Shell& ket1,
                          const ::libint2::Shell& ket2,
                          const ::libint2::ShellPair* spbra = nullptr,
                          const ::libint2::ShellPair* spket = nullptr);

 private:
  const Results& update_results();

  std::unique_ptr<::libint2::Engine> engine_;
  Results results_;
};

}  // namespace qdk::chemistry::scf::libint2_util
