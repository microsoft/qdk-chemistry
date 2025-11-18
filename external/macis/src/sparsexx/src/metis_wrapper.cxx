/*
 * MACIS Copyright (c) 2023, The Regents of the University of California,
 * through Lawrence Berkeley National Laboratory (subject to receipt of
 * any required approvals from the U.S. Dept. of Energy). All rights reserved.
 *
 * See LICENSE.txt for details
 */

#include <sparsexx/sparsexx_config.hpp>
#include <sparsexx/util/graph.hpp>

#ifdef SPARSEXX_ENABLE_METIS
#include <metis.h>
#endif /* SPARSEXX_ENABLE_METIS */

namespace sparsexx::detail {

template <typename IndexType>
void metis_kway_partitioning(int64_t _nvert, int64_t _npart, IndexType* _xadj,
                             IndexType* _adjncy,
                             std::vector<IndexType>& _part) {
#ifdef SPARSEXX_ENABLE_METIS
  idx_t nvert = _nvert;
  idx_t nweights = 1;
  idx_t nparts = _npart;
  idx_t obj;
  idx_t options[METIS_NOPTIONS];

  METIS_SetDefaultOptions(options);

  std::vector<idx_t> xadj_local, adjncy_local, part_local;
  idx_t *xadj, *adjncy, *part;
  if constexpr (std::is_same_v<IndexType, idx_t>) {
    xadj = _xadj;
    adjncy = _adjncy;
    part = _part.data();
  } else {
    throw std::runtime_error("METIS + Index Mismatch NYI");
  }

  METIS_PartGraphKway(&nvert, &nweights, xadj, adjncy, NULL, NULL, NULL,
                      &nparts, NULL, NULL, options, &obj, part);
#else
  throw std::runtime_error("METIS Not Enabled");
#endif /* SPARSEXX_ENABLE_METIS */
}

#define IMPL(T)                                                         \
  template void metis_kway_partitioning(int64_t _nvert, int64_t _npart, \
                                        T* _xadj, T* _adjncy,           \
                                        std::vector<T>& _part);

IMPL(int32_t);
IMPL(int64_t);

}  // namespace sparsexx::detail
