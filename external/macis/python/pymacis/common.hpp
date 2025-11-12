// Copyright (c) Microsoft Corporation.

#pragma once

// pybind11 headers
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

// macis headers - common headers used across multiple files
#include <macis/sd_operations.hpp>
#include <macis/wfn/raw_bitset.hpp>

// standard library headers
#include <algorithm>   // for std::copy
#include <filesystem>  // for std::filesystem::exists
#include <fstream>     // for std::ifstream
#include <stdexcept>   // for std::runtime_error
#include <string>      // for std::string
#include <vector>      // for std::vector

namespace py = pybind11;

// Type aliases
using np_double_array =
    py::array_t<double, py::array::c_style | py::array::forcecast>;

/**
 * @brief Utility function to check if a file exists and throw an exception if
 * not
 * @param filename Path to the file to check
 * @throws std::runtime_error if the file does not exist
 */
void throw_if_file_not_found(const std::string &filename);

/**
 * @brief Convert a NumPy array to a std::vector
 * @tparam T Element type
 * @param array Input NumPy array
 * @return std::vector containing the array data
 */
template <typename T>
std::vector<T> array_to_vector(const py::array_t<T> &array);

/**
 * @brief Convert a std::vector to a NumPy array
 * @tparam T Element type
 * @param vec Input vector
 * @return NumPy array containing the vector data
 */
template <typename T>
py::array_t<T> vector_to_array(const std::vector<T> &vec);

/**
 * @brief Template dispatcher based on number of orbitals
 * @tparam Func Function struct with templated impl method
 * @tparam Args Argument types
 * @param norb Number of orbitals to determine bit size
 * @param args Arguments to forward to the function
 * @return Result of the dispatched function call
 */
template <typename Func, typename... Args>
auto dispatch_by_norb(size_t norb, Args &&...args);

/**
 * @brief Convert Python list of strings to vector of wavefunction determinants
 * @tparam N Number of bits for wavefunction representation
 * @param det_strings Python list containing determinant strings
 * @return Vector of wavefunction determinants
 */
template <size_t N>
std::vector<macis::wfn_t<N>> strings_to_wfn_vector(const py::list &det_strings);

/**
 * @brief Convert vector of wavefunction determinants to Python list of strings
 * @tparam N Number of bits for wavefunction representation
 * @param dets Vector of wavefunction determinants
 * @param norb Number of orbitals (for string truncation)
 * @return Python list containing determinant strings
 */
template <size_t N>
py::list wfn_vector_to_strings(const std::vector<macis::wfn_t<N>> &dets,
                               size_t norb);

/**
 * @brief Validate electron count parameters
 * @param nalpha Number of alpha electrons
 * @param nbeta Number of beta electrons
 * @param norbital Number of orbitals
 * @throws std::runtime_error if parameters are invalid
 */
void validate_electron_counts(size_t nalpha, size_t nbeta, size_t norbital);

// Include template implementations
#include "common.inl"
