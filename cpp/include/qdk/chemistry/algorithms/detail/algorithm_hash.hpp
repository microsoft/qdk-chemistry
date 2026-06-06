// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <type_traits>
#include <variant>
#include <vector>

#include "qdk/chemistry/data/data_class.hpp"
#include "qdk/chemistry/utils/hash_context.hpp"

namespace qdk::chemistry::algorithms::detail {

inline void hash_algorithm_arg(utils::HashContext& ctx,
                               const data::DataClass& value) {
  ctx.update(uint8_t(0));
  ctx.update(value.content_hash());
}

inline void hash_algorithm_arg(utils::HashContext& ctx,
                               const std::string& value) {
  ctx.update(uint8_t(1));
  ctx.update(value);
}

inline void hash_algorithm_arg(utils::HashContext& ctx, const char* value) {
  ctx.update(uint8_t(1));
  ctx.update(std::string(value));
}

template <typename T>
std::enable_if_t<std::is_arithmetic_v<T>, void> hash_algorithm_arg(
    utils::HashContext& ctx, T value) {
  if constexpr (std::is_same_v<T, bool>) {
    ctx.update(uint8_t(6));
    ctx.update(value);
  } else if constexpr (std::is_floating_point_v<T>) {
    ctx.update(uint8_t(7));
    ctx.update(static_cast<double>(value));
  } else if constexpr (std::is_signed_v<T>) {
    ctx.update(uint8_t(8));
    ctx.update(static_cast<int64_t>(value));
  } else {
    ctx.update(uint8_t(9));
    ctx.update(static_cast<uint64_t>(value));
  }
}

template <typename T>
void hash_algorithm_arg(utils::HashContext& ctx,
                        const std::shared_ptr<T>& value) {
  ctx.update(uint8_t(2));
  if (value) {
    ctx.update(true);
    hash_algorithm_arg(ctx, *value);
  } else {
    ctx.update(false);
  }
}

template <typename T>
void hash_algorithm_arg(utils::HashContext& ctx,
                        const std::optional<T>& value) {
  ctx.update(uint8_t(3));
  if (value) {
    ctx.update(true);
    hash_algorithm_arg(ctx, *value);
  } else {
    ctx.update(false);
  }
}

template <typename T>
void hash_algorithm_arg(utils::HashContext& ctx,
                        const std::vector<T>& values) {
  ctx.update(uint8_t(4));
  ctx.update(static_cast<uint64_t>(values.size()));
  for (const auto& value : values) {
    hash_algorithm_arg(ctx, value);
  }
}

template <typename... Ts>
void hash_algorithm_arg(utils::HashContext& ctx,
                        const std::variant<Ts...>& value) {
  ctx.update(uint8_t(5));
  ctx.update(static_cast<uint64_t>(value.index()));
  std::visit([&ctx](const auto& v) { hash_algorithm_arg(ctx, v); }, value);
}

}  // namespace qdk::chemistry::algorithms::detail