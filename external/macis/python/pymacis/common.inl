// Copyright (c) Microsoft Corporation.

// Template implementations for common.hpp

#pragma once

// Template implementation for array_to_vector
template <typename T>
std::vector<T> array_to_vector(const py::array_t<T> &array) {
  py::buffer_info buffer = array.request();
  T *ptr = static_cast<T *>(buffer.ptr);
  return std::vector<T>(ptr, ptr + buffer.size);
}

// Template implementation for vector_to_array
template <typename T>
py::array_t<T> vector_to_array(const std::vector<T> &vec) {
  py::array_t<T> result(vec.size());
  py::buffer_info buf = result.request();
  T *ptr = static_cast<T *>(buf.ptr);
  std::copy(vec.begin(), vec.end(), ptr);
  return result;
}

// Template implementation for dispatch_by_norb
template <typename Func, typename... Args>
auto dispatch_by_norb(size_t norb, Args &&...args) {
  if (norb < 32) {
    return Func::template impl<64>(std::forward<Args>(args)...);
  } else if (norb < 64) {
    return Func::template impl<128>(std::forward<Args>(args)...);
  } else if (norb < 128) {
    return Func::template impl<256>(std::forward<Args>(args)...);
  } else {
    throw std::runtime_error(
        "Function not implemented for more than 128 orbitals");
    return Func::template impl<64>(std::forward<Args>(args)...);
  }
}

// Template implementation for strings_to_wfn_vector
template <size_t N>
std::vector<macis::wfn_t<N>> strings_to_wfn_vector(
    const py::list &det_strings) {
  std::vector<macis::wfn_t<N>> dets;
  dets.reserve(det_strings.size());
  for (const auto &det_str : det_strings) {
    std::string s = det_str.cast<std::string>();
    dets.emplace_back(macis::from_canonical_string<macis::wfn_t<N>>(s));
  }
  return dets;
}

// Template implementation for wfn_vector_to_strings
template <size_t N>
py::list wfn_vector_to_strings(const std::vector<macis::wfn_t<N>> &dets,
                               size_t norb) {
  py::list dets_py;
  for (const auto &det : dets) {
    dets_py.append(macis::to_canonical_string(det).substr(0, norb));
  }
  return dets_py;
}
