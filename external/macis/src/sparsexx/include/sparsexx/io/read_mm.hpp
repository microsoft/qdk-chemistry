/*
 * MACIS Copyright (c) 2023, The Regents of the University of California,
 * through Lawrence Berkeley National Laboratory (subject to receipt of
 * any required approvals from the U.S. Dept. of Energy). All rights reserved.
 *
 * See LICENSE.txt for details
 */

#pragma once

#include <cassert>
#include <fstream>
#include <iostream>
#include <sparsexx/matrix_types/type_traits.hpp>
#include <sparsexx/util/string.hpp>
#include <stdexcept>
#include <string>

namespace sparsexx {

#if 0
template <
  typename T,
  typename index_t = int64_t,
  typename Alloc   = std::allocator<T>
>
csr_matrix<T,index_t,Alloc> read_mm( std::string fname ) {



  std::ifstream f_in(fname);

  std::string line;

  int64_t m, n, nnz;
  bool is_sym = false;
  {
    std::getline( f_in, line );
    auto tokens = tokenize( line );

    // Check if this is actually a MM file...

    if( tokens[0].compare("%%MatrixMarket") or tokens.size() != 5)
      throw std::runtime_error(fname + " is not a MM file");

    is_sym = !tokens[4].compare("symmetric");

    while(std::getline( f_in, line )) {
      if( line[0] != '%' ) break;
    }

    //std::getline( f_in, line );
    tokens = tokenize( line );
    if( tokens.size() != 3 )
      throw std::runtime_error(fname + " contains an invalid spec for problem dimension");

    m   = std::stoll(tokens[0]);
    n   = std::stoll(tokens[1]);
    nnz = std::stoll(tokens[2]);

    if( is_sym and m != n )
      throw std::runtime_error( fname + " symmetric not compatible with M!=N" );

    if( is_sym ) nnz = 2*nnz - n;
  }

#if 0
  std::vector< std::tuple< int64_t, int64_t, T > > coo;
  coo.reserve( nnz );

  while( std::getline( f_in, line ) ) {

    auto tokens = tokenize( line );
    int64_t i = std::stoll( tokens[0] );
    int64_t j = std::stoll( tokens[1] );
    T       v = std::stod(  tokens[2] );

    coo.push_back({i, j, v});
    if( i != j and is_sym )
      coo.push_back({j, i, v});

  }

  assert( coo.size() == (size_t)nnz );



  // Sort based on row
  std::stable_sort( coo.begin(), coo.end(),
    []( auto a, auto b ){ return std::get<0>(a) < std::get<0>(b); }
  );

  // Determine if we're zero based
  bool zero_based = std::any_of( coo.begin(), coo.end(),
    [](auto x){ return std::get<0>(x)==0 or std::get<1>(x)==0; }
  );

  // Allocate matrix
  csr_matrix<T,index_t,Alloc> A(m,n,nnz, !zero_based);
  // Init rowptr accordingly
  A.rowptr()[0] = !zero_based;



  auto begin_row = coo.begin();
  auto* colind = A.colind().data();
  auto* nzval = A.nzval().data();
  for( int64_t i = 0; i < m; ++i ) {

    // Get start of next row
    auto next_row = std::find_if( begin_row, coo.end(),
      [&](auto x){ return std::get<0>(x) == i + 1 + !zero_based; });

    // Sort within row
    std::stable_sort( begin_row, next_row,
      []( auto a, auto b ){ return std::get<1>(a) < std::get<1>(b); }
    );

    // Calculate row pointer for subsequent iteration
    A.rowptr()[i + 1] = A.rowptr()[i] + std::distance( begin_row, next_row );

    for( auto it = begin_row; it != next_row; ++it ) {
      *(colind++) = std::get<1>(*it);
      *(nzval++)  = std::get<2>(*it);
    }

    begin_row = next_row;
  }
  assert( std::distance( A.colind().data(), colind ) == nnz );
  assert( std::distance( A.nzval().data(), nzval ) == nnz );
#else


  std::vector<index_t> rowind(nnz), colind(nnz), nzval(nnz);

  size_t nnz_idx = 0;
  while( std::getline( f_in, line ) ) {

    auto tokens = tokenize( line );
    int64_t i = std::stoll( tokens[0] );
    int64_t j = std::stoll( tokens[1] );
    T       v = std::stod(  tokens[2] );

    rowind[nnz_idx] = i;
    colind[nnz_idx] = j;
    nzval[nnz_idx]  = v;
    nnz_idx++;

    if( is_sym and i != j ) {
      rowind[nnz_idx] = j;
      colind[nnz_idx] = i;
      nzval[nnz_idx]  = v;
      nnz_idx++;
    }

  }


  assert( nnz == nnz_idx );

  ranges::sort( ranges::views::zip( rowind, colind, nzval ),
    []( const std::tuple<index_t, index_t, T>& el1,
        const std::tuple<index_t, index_t, T>& el2) {
      return std::get<0>(el1) < std::get<0>(el2);
    });

  assert( std::is_sorted( rowind.begin(), rowind.end() ) );

  auto eq_zero = [](const auto x){ return x == 0; };
  bool zero_based = std::any_of( rowind.begin(), rowind.end(), eq_zero ) or
                    std::any_of( colind.begin(), colind.end(), eq_zero );


  // Allocate matrix
  csr_matrix<T,index_t,Alloc> A(m,n,nnz, !zero_based);
  // Init rowptr accordingly
  A.rowptr()[0] = !zero_based;


  auto begin_row = rowind.begin();
  auto* colind_csr = A.colind().data();
  auto* nzval_csr  = A.nzval().data();
  for( int64_t i = 0; i < m; ++i ) {

    // Get start of next row
    auto next_row = std::find( begin_row, rowind.end(), i+1+!zero_based );

    auto begin_idx = std::distance( rowind.begin(), begin_row );
    auto end_idx   = std::distance( rowind.begin(), next_row  );

    // Sort the row internally
    ranges::sort( ranges::views::zip( rowind, colind, nzval ) |
                  ranges::views::slice(begin_idx, end_idx) ,
      []( const std::tuple<index_t, index_t, T>& el1,
          const std::tuple<index_t, index_t, T>& el2) {
        return std::get<1>(el1) < std::get<1>(el2);
      });

    assert( std::is_sorted( colind.begin() + begin_idx, colind.begin() + end_idx ) );


    // Calculate row pointer for subsequent iteration
    A.rowptr()[i + 1] = A.rowptr()[i] + end_idx - begin_idx;

    for( int64_t j = begin_idx; j < end_idx; ++j ) {
      *(colind_csr++) = colind[j];
      *(nzval_csr++)  = nzval[j];
    }

    begin_row = next_row;

  }
  assert( std::distance( A.colind().data(), colind_csr ) == nnz );
  assert( std::distance( A.nzval().data(), nzval_csr ) == nnz );
#endif /* 0 */


  return A;
} // read_mm (CSR)

#else

namespace detail {
enum class coo_sort_scheme { no_sort, sort_by_row, sort_by_col };

template <typename MatType>
struct get_default_coo_sort_scheme;
template <typename... Args>
struct get_default_coo_sort_scheme<coo_matrix<Args...> > {
  static constexpr auto value = coo_sort_scheme::no_sort;
};
template <typename... Args>
struct get_default_coo_sort_scheme<csr_matrix<Args...> > {
  static constexpr auto value = coo_sort_scheme::sort_by_row;
};
template <typename... Args>
struct get_default_coo_sort_scheme<csc_matrix<Args...> > {
  static constexpr auto value = coo_sort_scheme::sort_by_col;
};

template <typename MatType>
inline constexpr auto default_coo_sort_scheme_v =
    get_default_coo_sort_scheme<MatType>::value;
}  // namespace detail

template <typename T, typename index_t, typename Alloc>
coo_matrix<T, index_t, Alloc> read_mm_as_coo(
    std::string fname, detail::coo_sort_scheme sort_scheme) {
  std::ifstream f_in(fname);

  std::string line;

  int64_t m, n, nnz_min;
  bool is_sym = false;
  {
    std::getline(f_in, line);
    auto tokens = tokenize(line);

    // Check if this is actually a MM file...

    if (tokens[0].compare("%%MatrixMarket") or tokens.size() != 5)
      throw std::runtime_error(fname + " is not a MM file");

    is_sym = !tokens[4].compare("symmetric");

    while (std::getline(f_in, line)) {
      if (line[0] != '%') break;
    }

    // std::getline( f_in, line );
    tokens = tokenize(line);
    if (tokens.size() != 3)
      throw std::runtime_error(
          fname + " contains an invalid spec for problem dimension");

    m = std::stoll(tokens[0]);
    n = std::stoll(tokens[1]);
    nnz_min = std::stoll(tokens[2]);

    if (is_sym and m != n)
      throw std::runtime_error(fname + " symmetric not compatible with M!=N");

    if (is_sym) nnz_min *= 2;
  }

#if 0
  coo_matrix<T,index_t,Alloc> A(m, n, nnz_min);
  auto& rowind = A.rowind();
  auto& colind = A.colind();
  auto& nzval  = A.nzval();

  size_t nnz_idx = 0;
  while( std::getline( f_in, line ) ) {

    auto tokens = tokenize( line );
    int64_t i = std::stoll( tokens[0] );
    int64_t j = std::stoll( tokens[1] );
    T       v = std::stod(  tokens[2] );

    rowind[nnz_idx] = i;
    colind[nnz_idx] = j;
    nzval[nnz_idx]  = v;
    nnz_idx++;

    if( is_sym and i != j ) {
      rowind[nnz_idx] = j;
      colind[nnz_idx] = i;
      nzval[nnz_idx]  = v;
      nnz_idx++;
    }

  }
#else

  std::vector<index_t> rowind, colind;
  std::vector<T> nzval;
  rowind.reserve(nnz_min);
  colind.reserve(nnz_min);
  nzval.reserve(nnz_min);

  size_t nnz_true = 0;
  while (std::getline(f_in, line)) {
    auto tokens = tokenize(line);
    int64_t i = std::stoll(tokens[0]);
    int64_t j = std::stoll(tokens[1]);
    T v = (tokens.size() == 3) ? std::stod(tokens[2]) : 1.;

    rowind.push_back(i);
    colind.push_back(j);
    nzval.push_back(v);
    nnz_true++;

    if (is_sym and i != j) {
      rowind.push_back(j);
      colind.push_back(i);
      nzval.push_back(v);
      nnz_true++;
    }
  }

  coo_matrix<T, index_t, Alloc> A(m, n, std::move(colind), std::move(rowind),
                                  std::move(nzval));
#endif /* 0 */

  A.determine_indexing_from_adj();
  if (sort_scheme == detail::coo_sort_scheme::sort_by_row) {
    A.sort_by_row_index();
    assert(A.is_sorted_by_row_index());
  }
  if (sort_scheme == detail::coo_sort_scheme::sort_by_col) {
    A.sort_by_col_index();
    assert(A.is_sorted_by_col_index());
  }

  return A;
}

template <typename SpMatType>
SpMatType read_mm(std::string fname) {
  using value_t = detail::value_type_t<SpMatType>;
  using index_t = detail::index_type_t<SpMatType>;
  using allocator_t = detail::allocator_type_t<SpMatType>;

  if constexpr (detail::is_coo_matrix_v<SpMatType>)
    return read_mm_as_coo<value_t, index_t, allocator_t>(
        fname, detail::coo_sort_scheme::no_sort);
  else
    return SpMatType(read_mm_as_coo<value_t, index_t, allocator_t>(
        fname, detail::default_coo_sort_scheme_v<SpMatType>));
  abort();
}

#endif /* 0 */

}  // namespace sparsexx
