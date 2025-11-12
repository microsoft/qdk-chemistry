// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include "gauxc.h"

#include <qdk/chemistry/scf/exc/gauxc_impl.h>
#include <qdk/chemistry/scf/util/gauxc_registry.h>

namespace qdk::chemistry::scf {
GAUXC::GAUXC(std::shared_ptr<BasisSet> basis_set, const SCFConfig& cfg)
    : EXC(basis_set, cfg) {
  // Store the GAUXCInput and other parameters needed for registry lookup
  gauxc_input_ = cfg.xc_input;
  unrestricted_ = cfg.unrestricted;
  xc_name_ = cfg.exc.xc_name;

  // Get or create the GAUXC implementation via the registry
  impl::GAUXC* gauxc_impl = util::GAUXCRegistry::get_or_create(
      *basis_set, gauxc_input_, unrestricted_, xc_name_);

  // Store exchange-correlation parameters
  x_alpha_ = gauxc_impl->x_alpha;
  x_beta_ = gauxc_impl->x_beta;
  x_omega_ = gauxc_impl->x_omega;
}

GAUXC::~GAUXC() noexcept = default;

void GAUXC::build_XC(const double* D, double* XC, double* xc_energy) {
  // Get the GAUXC implementation from the registry
  impl::GAUXC* gauxc_impl = util::GAUXCRegistry::find(gauxc_input_);
  if (gauxc_impl) {
    gauxc_impl->build_XC(D, XC, xc_energy);
  } else {
    throw std::runtime_error("GAUXC implementation not found in registry");
  }
}

void GAUXC::get_gradients(const double* D, double* dXC) {
  // Get the GAUXC implementation from the registry
  impl::GAUXC* gauxc_impl = util::GAUXCRegistry::find(gauxc_input_);
  if (gauxc_impl) {
    gauxc_impl->get_gradients(D, dXC);
  } else {
    throw std::runtime_error("GAUXC implementation not found in registry");
  }
}
void GAUXC::eval_fxc_contraction(const double* D, const double* tD,
                                 double* Fxc) {
  // Get the GAUXC implementation from the registry
  impl::GAUXC* gauxc_impl = util::GAUXCRegistry::find(gauxc_input_);
  if (gauxc_impl) {
    gauxc_impl->eval_fxc_contraction(D, tD, Fxc);
  } else {
    throw std::runtime_error("GAUXC implementation not found in registry");
  }
}
}  // namespace qdk::chemistry::scf
