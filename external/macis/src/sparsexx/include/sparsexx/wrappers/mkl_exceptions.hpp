/*
 * MACIS Copyright (c) 2023, The Regents of the University of California,
 * through Lawrence Berkeley National Laboratory (subject to receipt of
 * any required approvals from the U.S. Dept. of Energy). All rights reserved.
 *
 * See LICENSE.txt for details
 */

#pragma once
#include <sparsexx/sparsexx_config.hpp>
#if SPARSEXX_ENABLE_MKL

#include <cassert>
#include <stdexcept>

#include "mkl_types.hpp"

namespace sparsexx::detail::mkl {

static inline const char* get_mkl_sparse_error_string(sparse_status_t s) {
  switch (s) {
    case SPARSE_STATUS_SUCCESS:
      return "The operation was successful.";

    case SPARSE_STATUS_NOT_INITIALIZED:
      return "The routine encountered an empty handle or matrix array.";

    case SPARSE_STATUS_ALLOC_FAILED:
      return "Internal memory allocation failed.";

    case SPARSE_STATUS_INVALID_VALUE:
      return "The input parameters contain an invalid value.";

    case SPARSE_STATUS_EXECUTION_FAILED:
      return "Execution failed.";

    case SPARSE_STATUS_INTERNAL_ERROR:
      return "An error in algorithm implementation occurred.";

    case SPARSE_STATUS_NOT_SUPPORTED:
      return "NOT SUPPORTED";

    default:
      return "UNKNOWN";
  }
}

static inline const char* get_mkl_dss_error_string(int s) {
  switch (s) {
    case MKL_DSS_SUCCESS:
      return "Success!";

    case MKL_DSS_ZERO_PIVOT:
      return "Zero Pivot";

    case MKL_DSS_OUT_OF_MEMORY:
      return "Out of Memory";

    case MKL_DSS_FAILURE:
      return "Failure";

    case MKL_DSS_ROW_ERR:
      return "Row Error";

    case MKL_DSS_COL_ERR:
      return "Col Error";

    case MKL_DSS_TOO_FEW_VALUES:
      return "Too Few Values";

    case MKL_DSS_TOO_MANY_VALUES:
      return "Too Many Values";

    case MKL_DSS_NOT_SQUARE:
      return "Not Square";

    case MKL_DSS_STATE_ERR:
      return "State Error";

    case MKL_DSS_INVALID_OPTION:
      return "Invalid Option";

    case MKL_DSS_OPTION_CONFLICT:
      return "Option Conflict";

    case MKL_DSS_MSG_LVL_ERR:
      return "Message Level Error";

    case MKL_DSS_TERM_LVL_ERR:
      return "Termination Level Error";

    case MKL_DSS_STRUCTURE_ERR:
      return "Structure Error";

    case MKL_DSS_REORDER_ERR:
      return "Reorder Error";

    case MKL_DSS_VALUES_ERR:
      return "Values Error";

    case MKL_DSS_STATISTICS_INVALID_MATRIX:
      return "Statistics: Invalid Matrix";

    case MKL_DSS_STATISTICS_INVALID_STATE:
      return "Statistics Invalid State";

    case MKL_DSS_STATISTICS_INVALID_STRING:
      return "Statistics: Invalid String";

    case MKL_DSS_REORDER1_ERR:
      return "Reorder Error (1)";

    case MKL_DSS_PREORDER_ERR:
      return "Preorder Error";

    case MKL_DSS_DIAG_ERR:
      return "Diag Error";

    case MKL_DSS_I32BIT_ERR:
      return "I32 Bit Error";

    case MKL_DSS_OOC_MEM_ERR:
      return "Out of Core: Memory Error";

    case MKL_DSS_OOC_OC_ERR:
      return "Out of Core: OC Error";

    case MKL_DSS_OOC_RW_ERR:
      return "Out of Core: Read-Write Error";

    default:
      return "Unrecognized Error";
  }
}

class mkl_sparse_exception : public std::exception {
  sparse_status_t stat_;

  const char* what() const throw() {
    return get_mkl_sparse_error_string(stat_);
  }

 public:
  mkl_sparse_exception() = delete;
  mkl_sparse_exception(sparse_status_t s) : stat_(s) {}
};

class mkl_dss_exception : public std::exception {
  int stat_;

  const char* what() const throw() { return get_mkl_dss_error_string(stat_); }

 public:
  mkl_dss_exception() = delete;
  mkl_dss_exception(int s) : stat_(s) {}
};

}  // namespace sparsexx::detail::mkl
#endif /* SPARSEXX_ENABLE_MKL */
