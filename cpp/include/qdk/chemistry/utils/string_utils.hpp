// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once

#include <array>
#include <complex>
#include <cstddef>
#include <string>
#include <string_view>
#include <type_traits>

namespace qdk::chemistry::utils {

/**
 * @brief Convert a PascalCase/camelCase string to snake_case at runtime
 *
 * This function inserts an underscore before each uppercase letter
 * (except at position 0) and converts all letters to lowercase.
 *
 * @param input Input string in PascalCase or camelCase
 * @return std::string containing the snake_case version
 *
 * Examples:
 * - "Ansatz" -> "ansatz"
 * - "ConfigurationSet" -> "configuration_set"
 * - "StabilityResult" -> "stability_result"
 */
inline std::string to_snake_case(const char* input) {
  std::string result;
  for (std::size_t i = 0; input[i] != '\0'; ++i) {
    char c = input[i];
    if (c >= 'A' && c <= 'Z') {
      if (i > 0) {
        result += '_';
      }
      result += static_cast<char>(c + 32);
    } else {
      result += c;
    }
  }
  return result;
}

/// \cond

template <std::size_t N>
struct FixedString {
  char data[N]{};

  constexpr FixedString() = default;

  constexpr FixedString(const char (&value)[N]) {
    for (std::size_t i = 0; i < N; ++i) {
      data[i] = value[i];
    }
  }

  static constexpr std::size_t size() { return N - 1; }

  constexpr const char* c_str() const { return data; }

  constexpr std::string_view view() const { return {data, N - 1}; }
};

template <std::size_t N>
FixedString(const char (&)[N]) -> FixedString<N>;

constexpr std::size_t snake_case_size(std::string_view value) {
  std::size_t size = value.size();
  for (std::size_t i = 1; i < value.size(); ++i) {
    if (value[i] >= 'A' && value[i] <= 'Z') {
      ++size;
    }
  }
  return size;
}

template <FixedString Token>
inline constexpr auto snake_case_v = [] {
  FixedString<snake_case_size(Token.view()) + 1> result{};
  std::size_t output_index = 0;
  for (std::size_t i = 0; i < Token.size(); ++i) {
    char value = Token.data[i];
    if (value >= 'A' && value <= 'Z') {
      if (i > 0) {
        result.data[output_index++] = '_';
      }
      value = static_cast<char>(value + ('a' - 'A'));
    }
    result.data[output_index++] = value;
  }
  result.data[output_index] = '\0';
  return result;
}();

namespace detail {

template <class...>
inline constexpr bool always_false = false;

template <auto Value>
consteval auto decimal() {
  using ValueType = std::remove_cv_t<decltype(Value)>;
  if constexpr (!std::is_integral_v<ValueType> ||
                std::is_same_v<ValueType, bool>) {
    static_assert(always_false<ValueType>,
                  "wire non-type parameters must be integral values");
    return FixedString{"?"};
  } else {
    using Unsigned = std::make_unsigned_t<ValueType>;
    constexpr bool is_negative = std::is_signed_v<ValueType> && Value < 0;
    constexpr Unsigned magnitude =
        is_negative ? Unsigned{0} - static_cast<Unsigned>(Value)
                    : static_cast<Unsigned>(Value);
    constexpr std::size_t digits = [] {
      std::size_t count = 1;
      auto remaining = magnitude;
      while (remaining >= 10) {
        remaining /= 10;
        ++count;
      }
      return count;
    }();

    FixedString<digits + (is_negative ? 1 : 0) + 1> result{};
    auto remaining = magnitude;
    std::size_t output_index = digits + (is_negative ? 1 : 0);
    result.data[output_index] = '\0';
    do {
      result.data[--output_index] = static_cast<char>('0' + (remaining % 10));
      remaining /= 10;
    } while (remaining != 0);
    if constexpr (is_negative) {
      result.data[0] = 'm';
    }
    return result;
  }
}

}  // namespace detail

template <class T>
struct WireTag {
  static_assert(detail::always_false<T>,
                "no wire spelling registered; add "
                "QDK_REGISTER_WIRE_TAG(\"tag\", YourType)");
  static constexpr FixedString value{"?"};
};

template <class T>
inline constexpr auto wire_tag_v = WireTag<T>::value;

template <auto Value>
struct NTTPTag {};

template <auto Value>
struct WireTag<NTTPTag<Value>> {
  static constexpr auto value = detail::decimal<Value>();
};

template <class...>
struct ParamList {};

template <class T>
struct RecoveredParams {
  using type = ParamList<>;
  static constexpr bool recovered = false;
};

template <template <class...> class T, class... Args>
struct RecoveredParams<T<Args...>> {
  using type = ParamList<Args...>;
  static constexpr bool recovered = true;
};

template <template <auto, class...> class T, auto Value, class... Args>
struct RecoveredParams<T<Value, Args...>> {
  using type = ParamList<NTTPTag<Value>, Args...>;
  static constexpr bool recovered = true;
};

namespace detail {

template <class Declared, class Recovered>
struct ParamsMatch : std::false_type {};

template <>
struct ParamsMatch<ParamList<>, ParamList<>> : std::true_type {};

template <class DeclaredHead, class... DeclaredTail, class RecoveredHead,
          class... RecoveredTail>
struct ParamsMatch<ParamList<DeclaredHead, DeclaredTail...>,
                   ParamList<RecoveredHead, RecoveredTail...>>
    : std::bool_constant<wire_tag_v<DeclaredHead>.view() ==
                             wire_tag_v<RecoveredHead>.view() &&
                         ParamsMatch<ParamList<DeclaredTail...>,
                                     ParamList<RecoveredTail...>>::value> {};

template <class Declared, class Recovered>
inline constexpr bool params_match_v = ParamsMatch<Declared, Recovered>::value;

template <class Enclosing, class... Declared>
constexpr void verify_params() {
  using Recovered = RecoveredParams<Enclosing>;
  if constexpr (Recovered::recovered) {
    static_assert(
        params_match_v<ParamList<Declared...>, typename Recovered::type>,
        "DATACLASS_TO_SNAKE_CASE arguments must match the enclosing class "
        "template parameters in declaration order");
  }
}

template <FixedString Token, class... Params>
inline constexpr std::size_t wire_name_size =
    snake_case_v<Token>.size() + ((1 + wire_tag_v<Params>.size()) + ... + 0);

}  // namespace detail

template <FixedString Token, class... Params>
inline constexpr FixedString<detail::wire_name_size<Token, Params...> + 1>
    wire_name_v = [] {
      FixedString<detail::wire_name_size<Token, Params...> + 1> result{};
      std::size_t output_index = 0;
      auto append = [&](std::string_view value) {
        for (char character : value) {
          result.data[output_index++] = character;
        }
      };
      append(snake_case_v<Token>.view());
      ((append("_"), append(wire_tag_v<Params>.view())), ...);
      result.data[output_index] = '\0';
      return result;
    }();

}  // namespace qdk::chemistry::utils

#define QDK_REGISTER_WIRE_TAG(Tag, ...)                               \
  namespace qdk::chemistry::utils {                                   \
  template <>                                                         \
  struct WireTag<__VA_ARGS__> {                                       \
    static constexpr ::qdk::chemistry::utils::FixedString value{Tag}; \
  };                                                                  \
  }                                                                   \
  static_assert(true, "")

QDK_REGISTER_WIRE_TAG("real", double);
QDK_REGISTER_WIRE_TAG("real32", float);
QDK_REGISTER_WIRE_TAG("complex", std::complex<double>);
QDK_REGISTER_WIRE_TAG("complex32", std::complex<float>);
QDK_REGISTER_WIRE_TAG("uint", std::size_t);

#define DATACLASS_TO_SNAKE_CASE(ClassName, ...) \
  (::qdk::chemistry::utils::detail::verify_params<                      \
       ClassName __VA_OPT__(, ) __VA_ARGS__>(),                         \
   ::qdk::chemistry::utils::wire_name_v<                                \
       ::qdk::chemistry::utils::FixedString(#ClassName)                 \
           __VA_OPT__(, ) __VA_ARGS__>                                  \
       .c_str())

/// \endcond
