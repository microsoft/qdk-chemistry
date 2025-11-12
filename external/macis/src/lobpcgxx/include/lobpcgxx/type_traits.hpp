#pragma once
#include <complex>

namespace lobpcgxx {

namespace detail {

template <typename T>
struct real {
  using type = T;
};

template <typename T>
struct real<std::complex<T>> {
  using type = T;
};

template <typename T>
using real_t = typename real<T>::type;

}  // namespace detail

}  // namespace lobpcgxx
