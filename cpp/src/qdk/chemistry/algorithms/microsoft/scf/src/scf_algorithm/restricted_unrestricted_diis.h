// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once

#include "diis_base.h"

namespace qdk::chemistry::scf {

class RestrictedUnrestrictedDIIS : public DIISBase {
 public:
  explicit RestrictedUnrestrictedDIIS(const SCFContext& ctx,
                                      size_t subspace_size = 8);
  ~RestrictedUnrestrictedDIIS() noexcept override = default;

 protected:
  const RowMajorMatrix& get_active_fock(const SCFImpl& scf_impl) const override;
  RowMajorMatrix& active_density(SCFImpl& scf_impl) override;
};

}  // namespace qdk::chemistry::scf
