// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <qdk/chemistry/scf/config.h>

#include <algorithm>
#include <nlohmann/json.hpp>
#include <string>

#include "exc/GauXC/gauxc.h"
#ifdef ENABLE_NVTX3
#include "nvtx3/nvtx3.hpp"
#endif
#include <qdk/chemistry/scf/core/scf.h>
#include <spdlog/spdlog.h>

#include "util/macros.h"

namespace qdk::chemistry::scf {
EXC::EXC(std::shared_ptr<BasisSet> basis_set, const SCFConfig& cfg)
    : basis_set_(basis_set),
      cfg_(cfg.exc),
      mpi_(cfg.mpi),
      x_alpha_(0.0),
      x_beta_(0.0),
      x_omega_(0.0) {}

std::shared_ptr<EXC> EXC::create(std::shared_ptr<BasisSet> basis_set,
                                 const SCFConfig& cfg) {
#ifdef ENABLE_NVTX3
  nvtx3::scoped_range r{nvtx3::rgb{255, 0, 255}, "EXC::create"};
#endif
  switch (cfg.exc.method) {
    case EXCMethod::GauXC:
      return std::make_shared<GAUXC>(basis_set, cfg);
    default:
      return nullptr;
  }
}
}  // namespace qdk::chemistry::scf
