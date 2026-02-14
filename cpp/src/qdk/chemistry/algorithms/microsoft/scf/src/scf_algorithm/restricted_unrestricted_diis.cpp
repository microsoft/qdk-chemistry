// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include "restricted_unrestricted_diis.h"

#include <qdk/chemistry/utils/logger.hpp>

#include "../scf/scf_impl.h"
#include "util/macros.h"

namespace qdk::chemistry::scf {

RestrictedUnrestrictedDIIS::RestrictedUnrestrictedDIIS(const SCFContext& ctx,
                                                       size_t subspace_size)
    : DIISBase(ctx, subspace_size) {}

const RowMajorMatrix& RestrictedUnrestrictedDIIS::get_active_fock(
    const SCFImpl& scf_impl) const {
  QDK_LOG_TRACE_ENTERING();
  return scf_impl.get_fock_matrix();
}

RowMajorMatrix& RestrictedUnrestrictedDIIS::active_density(SCFImpl& scf_impl) {
  QDK_LOG_TRACE_ENTERING();
  return scf_impl.density_matrix();
}

}  // namespace qdk::chemistry::scf
