// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once
#include <memory>
#include <qdk/chemistry/algorithms/effective_hamiltonian.hpp>
#include <qdk/chemistry/data/settings.hpp>
#include <string>
#include <vector>

namespace qdk::chemistry::algorithms::microsoft {

/**
 * @brief Settings for the second-order Schrieffer-Wolff downfold.
 *
 * `regularizer` selects flow (default), shift, or bare inverse denominators;
 * the corresponding numeric setting controls the selected scheme. The current
 * denominator operator is a semicanonical, spin-free generalized Fock.
 */
class SchriefferWolffPT2Settings : public qdk::chemistry::data::Settings {
 public:
  SchriefferWolffPT2Settings();
  ~SchriefferWolffPT2Settings() override = default;
};

class SchriefferWolffPT2Diagnostics final
    : public qdk::chemistry::algorithms::EffectiveHamiltonianDiagnostics {
 public:
  SchriefferWolffPT2Diagnostics(std::string regularizer,
                                double denominator_floor,
                                double denominator_shift, double flow_parameter,
                                double min_denominator,
                                double max_raw_amplitude,
                                double higher_body_norm,
                                bool semicanonical_rotation_applied)
      : regularizer_(std::move(regularizer)),
        denominator_floor_(denominator_floor),
        denominator_shift_(denominator_shift),
        flow_parameter_(flow_parameter),
        min_denominator_(min_denominator),
        max_raw_amplitude_(max_raw_amplitude),
        higher_body_norm_(higher_body_norm),
        semicanonical_rotation_applied_(semicanonical_rotation_applied) {}

  std::string method() const override { return "swpt2"; }
  const std::string& regularizer() const { return regularizer_; }
  double denominator_floor() const { return denominator_floor_; }
  double denominator_shift() const { return denominator_shift_; }
  double flow_parameter() const { return flow_parameter_; }
  double min_denominator() const { return min_denominator_; }
  double max_raw_amplitude() const { return max_raw_amplitude_; }
  double higher_body_norm() const { return higher_body_norm_; }
  bool semicanonical_rotation_applied() const {
    return semicanonical_rotation_applied_;
  }

 private:
  std::string regularizer_;
  double denominator_floor_;
  double denominator_shift_;
  double flow_parameter_;
  double min_denominator_;
  double max_raw_amplitude_;
  double higher_body_norm_;
  bool semicanonical_rotation_applied_;
};

/**
 * @brief Second-order Schrieffer-Wolff (Van Vleck) effective-Hamiltonian
 * downfold with semicanonical generalized-Fock orbital-energy denominators.
 *
 * Computes `H_eff = H_BD + 1/2 [S, H_OD]`, truncated to <= 2-body, folding the
 * external space Q of the window onto the reference active space P. With bare
 * denominators, S solves `[F0, S] = H_OD` for a diagonal generalized-Fock F0;
 * flow and shift settings replace `1/Delta` by a regularized inverse and hence
 * define regularized variants of that generator.
 *
 * The implementation assumes a common restricted MO basis, supporting RHF,
 * ROHF, and spin-adapted CAS references. Every singly occupied ROHF orbital
 * must be active. Noncanonical orbitals are semicanonicalized independently
 * within inactive, active, and virtual blocks. The kernel computes intruder
 * and discarded-body diagnostics and returns them with the effective
 * Hamiltonian. A large-amplitude warning is also logged.
 * See `swpt2_kernel.hpp` for the operator and tensor conventions.
 */
class SchriefferWolffPT2Constructor
    : public qdk::chemistry::algorithms::EffectiveHamiltonianConstructor {
 public:
  SchriefferWolffPT2Constructor() {
    _settings = std::make_unique<SchriefferWolffPT2Settings>();
  }
  ~SchriefferWolffPT2Constructor() override = default;

  std::string name() const final { return "qdk_swpt2"; }
  std::vector<std::string> aliases() const override {
    return {"qdk_swpt2", "swpt2", "sw", "schrieffer_wolff"};
  }

 protected:
  qdk::chemistry::algorithms::EffectiveHamiltonianResult _run_impl(
      std::shared_ptr<data::Wavefunction> reference,
      std::shared_ptr<data::Hamiltonian> hamiltonian) const override;
};

}  // namespace qdk::chemistry::algorithms::microsoft
