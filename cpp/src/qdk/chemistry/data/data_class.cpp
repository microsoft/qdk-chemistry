// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <qdk/chemistry/data/data_class.hpp>
#include <qdk/chemistry/utils/hash_context.hpp>

namespace qdk::chemistry::data {

std::string DataClass::content_hash(size_t truncate_chars) const {
  qdk::chemistry::utils::HashContext ctx;
  hash_update(ctx);
  return ctx.hexdigest(truncate_chars);
}

void hash_value(qdk::chemistry::utils::HashContext& ctx,
                const DataClass& value) {
  qdk::chemistry::utils::hash_value(ctx, value.content_hash());
}

}  // namespace qdk::chemistry::data
