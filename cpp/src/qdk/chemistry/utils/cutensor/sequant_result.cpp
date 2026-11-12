// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <SeQuant/core/eval/backends/btas/result.hpp>
#include <algorithm>
#include <qdk/chemistry/utils/cutensor/sequant_result.hpp>
#include <stdexcept>
#include <vector>

namespace qdk::chemistry::utils::cutensor {

namespace {

using sequant::eval_result;
using sequant::Result;
using sequant::ResultPtr;
using annot_t = ResultTensorCUTENSOR::annot_t;
using HostTensor = btas::Tensor<double>;
using HostResult = sequant::ResultTensorBTAS<HostTensor>;
using Scalar = sequant::ResultScalar<double>;

// Relabels SeQuant's 64-bit index hashes as consecutive cuTENSOR modes in order
// of first appearance, so structurally equal operations share a cached plan.
class ModeMap {
 public:
  Modes operator()(const annot_t& annot) {
    Modes modes;
    modes.reserve(annot.size());
    for (const auto label : annot) {
      const auto found = std::find(labels_.begin(), labels_.end(), label);
      modes.push_back(static_cast<int32_t>(found - labels_.begin()));
      if (found == labels_.end()) labels_.push_back(label);
    }
    return modes;
  }

 private:
  std::vector<long> labels_;
};

bool contains(const annot_t& annot, long label) {
  return std::find(annot.begin(), annot.end(), label) != annot.end();
}

bool has_repeats(const annot_t& annot) {
  for (auto it = annot.begin(); it != annot.end(); ++it)
    if (std::find(std::next(it), annot.end(), *it) != annot.end()) return true;
  return false;
}

// A label of one operand that is neither in the other operand nor the result
// is summed within that operand, which a cuTENSOR contraction cannot express.
bool has_single_operand_sum(const annot_t& own, const annot_t& other,
                            const annot_t& result) {
  return std::any_of(own.begin(), own.end(), [&](long label) {
    return !contains(other, label) && !contains(result, label);
  });
}

const DeviceTensor& tensor_of(const Result& result) {
  return result.get<DeviceTensor>();
}

ResultPtr device_result(DeviceTensor tensor) {
  return eval_result<ResultTensorCUTENSOR>(std::move(tensor));
}

Extents extents_of(const annot_t& target, const annot_t& left_annot,
                   const DeviceTensor& left,
                   const annot_t* right_annot = nullptr,
                   const DeviceTensor* right = nullptr) {
  Extents extents;
  extents.reserve(target.size());
  for (const auto label : target) {
    if (const auto found =
            std::find(left_annot.begin(), left_annot.end(), label);
        found != left_annot.end()) {
      extents.push_back(left.extents()[found - left_annot.begin()]);
      continue;
    }
    if (!right_annot || !right)
      throw std::logic_error(
          "cuTENSOR backend: result label missing from the operands");
    const auto found =
        std::find(right_annot->begin(), right_annot->end(), label);
    if (found == right_annot->end())
      throw std::logic_error(
          "cuTENSOR backend: result label missing from the operands");
    extents.push_back(right->extents()[found - right_annot->begin()]);
  }
  return extents;
}

ResultPtr host_copy(const Result& result) {
  return eval_result<HostResult>(to_host(tensor_of(result)));
}

ResultPtr upload_result(const ResultPtr& host) {
  return host->is<HostResult>() ? to_device(host->get<HostTensor>()) : host;
}

}  // namespace

ResultPtr ResultTensorCUTENSOR::sum(
    Result const& other, std::array<std::any, 3> const& annot) const {
  if (!other.is<ResultTensorCUTENSOR>())
    throw std::invalid_argument(
        "cuTENSOR backend: a sum needs two device tensors");
  const sequant::Annot<annot_t> a{annot};
  if (has_repeats(a.lannot) || has_repeats(a.rannot))
    return upload_result(host_copy(*this)->sum(*host_copy(other), annot));
  auto& engine = Engine::current();
  const auto& left = tensor_of(*this);
  ModeMap map;
  const auto modes_left = map(a.lannot);
  const auto modes_right = map(a.rannot);
  const auto modes_result = map(a.this_annot);
  DeviceTensor result(extents_of(a.this_annot, a.lannot, left), engine);
  engine.permute(1.0, tensor_of(other), modes_right, result, modes_result);
  engine.add(1.0, left, modes_left, 1.0, result, modes_result);
  return device_result(std::move(result));
}

ResultPtr ResultTensorCUTENSOR::prod(Result const& other,
                                     std::array<std::any, 3> const& annot,
                                     sequant::DeNest flag) const {
  const sequant::Annot<annot_t> a{annot};
  auto& engine = Engine::current();
  const auto& left = tensor_of(*this);

  if (other.is<Scalar>()) {
    if (has_repeats(a.lannot))
      return upload_result(host_copy(*this)->prod(other, annot, flag));
    ModeMap map;
    const auto modes_left = map(a.lannot);
    const auto modes_result = map(a.this_annot);
    DeviceTensor result(extents_of(a.this_annot, a.lannot, left), engine);
    engine.permute(other.as<Scalar>().value(), left, modes_left, result,
                   modes_result);
    return device_result(std::move(result));
  }

  if (!other.is<ResultTensorCUTENSOR>())
    throw std::invalid_argument(
        "cuTENSOR backend: a product needs a device tensor or a scalar");
  if (has_repeats(a.lannot) || has_repeats(a.rannot) ||
      has_repeats(a.this_annot) ||
      has_single_operand_sum(a.lannot, a.rannot, a.this_annot) ||
      has_single_operand_sum(a.rannot, a.lannot, a.this_annot))
    return upload_result(
        host_copy(*this)->prod(*host_copy(other), annot, flag));

  const auto& right = tensor_of(other);
  ModeMap map;
  const auto modes_left = map(a.lannot);
  const auto modes_right = map(a.rannot);
  if (a.this_annot.empty()) {
    if (modes_left == modes_right)
      return eval_result<Scalar>(engine.dot(left, right));
    DeviceTensor aligned(left.extents(), engine);
    engine.permute(1.0, right, modes_right, aligned, modes_left);
    return eval_result<Scalar>(engine.dot(left, aligned));
  }
  const auto modes_result = map(a.this_annot);
  DeviceTensor result(
      extents_of(a.this_annot, a.lannot, left, &a.rannot, &right), engine);
  engine.contract(1.0, left, modes_left, right, modes_right, result,
                  modes_result);
  return device_result(std::move(result));
}

ResultPtr ResultTensorCUTENSOR::permute(
    std::array<std::any, 2> const& annot) const {
  const auto& before = std::any_cast<const annot_t&>(annot[0]);
  const auto& after = std::any_cast<const annot_t&>(annot[1]);
  if (has_repeats(before))
    return upload_result(host_copy(*this)->permute(annot));
  auto& engine = Engine::current();
  const auto& source = tensor_of(*this);
  ModeMap map;
  const auto modes_before = map(before);
  const auto modes_after = map(after);
  DeviceTensor result(extents_of(after, before, source), engine);
  engine.permute(1.0, source, modes_before, result, modes_after);
  return device_result(std::move(result));
}

ResultPtr ResultTensorCUTENSOR::adjoint(
    std::array<std::any, 2> const& annot) const {
  // Real data: the adjoint is the bra/ket-swapping permutation.
  return permute(annot);
}

void ResultTensorCUTENSOR::add_inplace(Result const& other) {
  auto& target = get<DeviceTensor>();
  const auto& source = tensor_of(other);
  if (target.extents() != source.extents())
    throw std::invalid_argument(
        "cuTENSOR backend: add_inplace needs equal extents");
  Engine::current().axpy(1.0, source, target);
}

double ResultTensorCUTENSOR::norm2() const {
  return Engine::current().norm2(tensor_of(*this));
}

ResultPtr ResultTensorCUTENSOR::clone() const {
  auto& engine = Engine::current();
  const auto& source = tensor_of(*this);
  DeviceTensor result(source.extents(), engine);
  engine.copy(source, result);
  return device_result(std::move(result));
}

ResultPtr ResultTensorCUTENSOR::symmetrize() const {
  return upload_result(host_copy(*this)->symmetrize());
}

ResultPtr ResultTensorCUTENSOR::antisymmetrize(std::size_t bra_rank) const {
  return upload_result(host_copy(*this)->antisymmetrize(bra_rank));
}

ResultPtr ResultTensorCUTENSOR::mult_by_phase(std::int8_t factor) const {
  auto& engine = Engine::current();
  const auto& source = tensor_of(*this);
  Modes identity(source.rank());
  for (std::size_t i = 0; i < identity.size(); ++i)
    identity[i] = static_cast<int32_t>(i);
  DeviceTensor result(source.extents(), engine);
  engine.permute(static_cast<double>(factor), source, identity, result,
                 identity);
  return device_result(std::move(result));
}

std::size_t ResultTensorCUTENSOR::size_in_bytes() const {
  return tensor_of(*this).volume() * sizeof(double);
}

ResultPtr to_device(const HostTensor& host) {
  Extents extents(host.rank());
  for (std::size_t i = 0; i < extents.size(); ++i)
    extents[i] = static_cast<int64_t>(host.extent(i));
  return to_device(host.data(), std::move(extents));
}

ResultPtr to_device(const double* host, Extents extents) {
  auto& engine = Engine::current();
  DeviceTensor tensor(std::move(extents), engine);
  engine.upload(host, tensor);
  return device_result(std::move(tensor));
}

HostTensor to_host(const DeviceTensor& tensor) {
  std::vector<std::size_t> extents(tensor.extents().begin(),
                                   tensor.extents().end());
  HostTensor host{btas::Range{extents}};
  Engine::current().download(tensor, host.data());
  return host;
}

}  // namespace qdk::chemistry::utils::cutensor
