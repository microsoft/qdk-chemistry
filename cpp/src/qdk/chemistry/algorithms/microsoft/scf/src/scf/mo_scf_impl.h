// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once

#include "scf/scf_impl.h"

namespace qdk::chemistry::scf {

/**
 * @brief Integral-input adapter for the shared Hartree-Fock iteration engine.
 *
 * The working basis is orthonormal and common to both spins. There is no
 * molecular geometry, atomic basis, or nuclear-property calculation.
 */
class MoSCFImpl final : public SCFImpl {
 public:
  MoSCFImpl(const RowMajorMatrix& one_body_integrals, std::shared_ptr<ERI> eri,
            int nalpha, int nbeta, double core_energy, const SCFConfig& cfg);

 protected:
  void build_one_electron_integrals_() override;
  double calc_nuclear_repulsion_energy_() override { return core_energy_; }
  void properties_() override {}

 private:
  RowMajorMatrix one_body_integrals_;
  double core_energy_;
};

}  // namespace qdk::chemistry::scf
