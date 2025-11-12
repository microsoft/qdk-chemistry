/*
 * MACIS Copyright (c) 2023, The Regents of the University of California,
 * through Lawrence Berkeley National Laboratory (subject to receipt of
 * any required approvals from the U.S. Dept. of Energy). All rights reserved.
 *
 * See LICENSE.txt for details
 */

#pragma once

#include <map>
#include <memory>
#include <sparsexx/util/hash.hpp>
#include <sparsexx/util/mpi.hpp>

#include "type_traits.hpp"

namespace sparsexx {

class dist_pmap {
 protected:
  MPI_Comm comm_;

 public:
  dist_pmap(MPI_Comm c) noexcept : comm_(c) {}
  dist_pmap() noexcept : dist_pmap(MPI_COMM_WORLD) {}

  virtual size_t owner(size_t i, size_t j) const = 0;
  virtual ~dist_pmap() noexcept = default;

  bool i_own(size_t i, size_t j) const {
    auto my_rank = detail::get_mpi_rank(comm_);
    return my_rank == owner(i, j);
  }

  MPI_Comm comm() const { return comm_; }
};

class cyclic_pmap : public dist_pmap {
 protected:
  size_t np_;

 public:
  template <typename... Args>
  cyclic_pmap(Args&&... args) : dist_pmap(std::forward<Args>(args)...) {
    np_ = detail::get_mpi_size(this->comm_);
  }

  virtual ~cyclic_pmap() noexcept = default;
};

struct row_cyclic_pmap : public cyclic_pmap {
  template <typename... Args>
  row_cyclic_pmap(Args&&... args) : cyclic_pmap(std::forward<Args>(args)...) {}

  virtual size_t owner(size_t i, size_t j) const {
    (void)(j);
    return i % this->np_;
  };
};

struct col_cyclic_pmap : public cyclic_pmap {
  template <typename... Args>
  col_cyclic_pmap(Args&&... args) : cyclic_pmap(std::forward<Args>(args)...) {}

  virtual size_t owner(size_t i, size_t j) const {
    (void)(i);
    return j % this->np_;
  };
};

template <typename... Args>
static inline std::unique_ptr<dist_pmap> make_row_cyclic_pmap(Args&&... args) {
  return std::make_unique<row_cyclic_pmap>(std::forward<Args>(args)...);
}

template <typename... Args>
static inline std::unique_ptr<dist_pmap> make_col_cyclic_pmap(Args&&... args) {
  return std::make_unique<col_cyclic_pmap>(std::forward<Args>(args)...);
}

template <typename... Args>
static inline std::unique_ptr<dist_pmap> make_default_pmap(Args&&... args) {
  return make_row_cyclic_pmap(std::forward<Args>(args)...);
}

template <typename SpMatType>
class dist_sparse_matrix {
 public:
  using value_type = detail::value_type_t<SpMatType>;
  using index_type = detail::index_type_t<SpMatType>;
  using size_type = detail::size_type_t<SpMatType>;
  using tile_type = SpMatType;

  struct sparse_tile {
    using index_extent_t = std::pair<index_type, index_type>;

    index_extent_t global_row_extent;
    index_extent_t global_col_extent;
    SpMatType local_matrix;
  };

 protected:
  MPI_Comm comm_;

  size_type global_m_;
  size_type global_n_;

  std::vector<index_type> row_tiling_;
  std::vector<index_type> col_tiling_;

  std::unique_ptr<dist_pmap> pmap_;

  using tile_index_t = std::pair<index_type, index_type>;
  std::unordered_map<tile_index_t, sparse_tile, detail::pair_hasher>
      local_tiles_;

  static decltype(local_tiles_) populate_local_tiles(
      const dist_pmap& pmap, const decltype(row_tiling_)& rt,
      const decltype(col_tiling_)& ct) {
    decltype(local_tiles_) lt;

    for (index_type it = 0; it < (index_type)rt.size() - 1; ++it)
      for (index_type jt = 0; jt < (index_type)ct.size() - 1; ++jt)
        if (pmap.i_own(it, jt)) {
          sparse_tile tile;
          tile.global_row_extent = {rt[it], rt[it + 1]};
          tile.global_col_extent = {ct[jt], ct[jt + 1]};
          lt[tile_index_t{it, jt}] = std::move(tile);
        }

    return lt;
  }

 public:
  constexpr dist_sparse_matrix() noexcept = default;

  dist_sparse_matrix(MPI_Comm c, size_t M, size_t N,
                     const std::vector<index_type>& rt,
                     const std::vector<index_type>& ct)
      : comm_(c),
        global_m_(M),
        global_n_(N),
        row_tiling_(rt),
        col_tiling_(ct),
        pmap_(make_default_pmap(c)) {
    local_tiles_ = std::move(populate_local_tiles(*pmap_, rt, ct));
  }

  auto begin() { return local_tiles_.begin(); }
  auto end() { return local_tiles_.end(); }

  const auto begin() const { return local_tiles_.begin(); }
  const auto end() const { return local_tiles_.end(); }

  const auto cbegin() const { return local_tiles_.cbegin(); }
  const auto cend() const { return local_tiles_.cend(); }

  auto comm() const { return comm_; }

  auto global_m() const { return global_m_; }
  auto global_n() const { return global_n_; }

  const auto& row_tiling() const { return row_tiling_; }
  const auto& col_tiling() const { return col_tiling_; }
  auto& row_tiling() { return row_tiling_; }
  auto& col_tiling() { return col_tiling_; }
};

template <typename... Args>
using dist_csr_matrix = dist_sparse_matrix<csr_matrix<Args...>>;
template <typename... Args>
using dist_coo_matrix = dist_sparse_matrix<coo_matrix<Args...>>;

}  // namespace sparsexx
