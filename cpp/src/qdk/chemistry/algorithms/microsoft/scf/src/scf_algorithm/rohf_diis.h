// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once

#include "diis_base.h"

namespace qdk::chemistry::scf {

class ROHFDIIS : public DIISBase {
 public:
  explicit ROHFDIIS(const SCFContext& ctx, size_t subspace_size = 8);
  ~ROHFDIIS() noexcept override = default;
  void build_rohf_f_p_matrix(const RowMajorMatrix& F, const RowMajorMatrix& C,
                             const RowMajorMatrix& P, int nelec_alpha,
                             int nelec_beta);
  const RowMajorMatrix& get_fock_matrix() const;
  const RowMajorMatrix& get_density_matrix() const;
  RowMajorMatrix& density_matrix();

 protected:
  void before_diis_iteration(SCFImpl& scf_impl) override;
  const RowMajorMatrix& get_active_fock(const SCFImpl& scf_impl) const override;
  RowMajorMatrix& active_density(SCFImpl& scf_impl) override;
  void update_density_matrix(RowMajorMatrix& P, const RowMajorMatrix& C,
                             bool unrestricted, int nelec_alpha,
                             int nelec_beta) override;

  RowMajorMatrix effective_F_;
  RowMajorMatrix total_P_;
};

}  // namespace qdk::chemistry::scf
