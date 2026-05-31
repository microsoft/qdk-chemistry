// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <cstring>
#include <iomanip>
#include <qdk/chemistry/utils/hash_context.hpp>
#include <sstream>

namespace qdk::chemistry::utils {

// SHA-256 constants (first 32 bits of the fractional parts of the cube roots
// of the first 64 primes)
static constexpr uint32_t K[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1,
    0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
    0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786,
    0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147,
    0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
    0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
    0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a,
    0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
    0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};

static inline uint32_t rotr(uint32_t x, int n) {
  return (x >> n) | (x << (32 - n));
}

static inline uint32_t ch(uint32_t x, uint32_t y, uint32_t z) {
  return (x & y) ^ (~x & z);
}

static inline uint32_t maj(uint32_t x, uint32_t y, uint32_t z) {
  return (x & y) ^ (x & z) ^ (y & z);
}

static inline uint32_t sigma0(uint32_t x) {
  return rotr(x, 2) ^ rotr(x, 13) ^ rotr(x, 22);
}

static inline uint32_t sigma1(uint32_t x) {
  return rotr(x, 6) ^ rotr(x, 11) ^ rotr(x, 25);
}

static inline uint32_t gamma0(uint32_t x) {
  return rotr(x, 7) ^ rotr(x, 18) ^ (x >> 3);
}

static inline uint32_t gamma1(uint32_t x) {
  return rotr(x, 17) ^ rotr(x, 19) ^ (x >> 10);
}

HashContext::HashContext()
    : _state{0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
             0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19},
      _buffer{},
      _buffer_len(0),
      _total_len(0) {}

void HashContext::_process_block(const uint8_t block[64]) {
  uint32_t w[64];
  for (int i = 0; i < 16; ++i) {
    w[i] = (uint32_t(block[i * 4]) << 24) | (uint32_t(block[i * 4 + 1]) << 16) |
           (uint32_t(block[i * 4 + 2]) << 8) | uint32_t(block[i * 4 + 3]);
  }
  for (int i = 16; i < 64; ++i) {
    w[i] = gamma1(w[i - 2]) + w[i - 7] + gamma0(w[i - 15]) + w[i - 16];
  }

  uint32_t a = _state[0], b = _state[1], c = _state[2], d = _state[3];
  uint32_t e = _state[4], f = _state[5], g = _state[6], h = _state[7];

  for (int i = 0; i < 64; ++i) {
    uint32_t t1 = h + sigma1(e) + ch(e, f, g) + K[i] + w[i];
    uint32_t t2 = sigma0(a) + maj(a, b, c);
    h = g;
    g = f;
    f = e;
    e = d + t1;
    d = c;
    c = b;
    b = a;
    a = t1 + t2;
  }

  _state[0] += a;
  _state[1] += b;
  _state[2] += c;
  _state[3] += d;
  _state[4] += e;
  _state[5] += f;
  _state[6] += g;
  _state[7] += h;
}

void HashContext::update(const void* data, size_t len) {
  const auto* bytes = static_cast<const uint8_t*>(data);
  _total_len += len;

  // Fill buffer first
  while (len > 0) {
    size_t space = 64 - _buffer_len;
    size_t to_copy = (len < space) ? len : space;
    std::memcpy(_buffer.data() + _buffer_len, bytes, to_copy);
    _buffer_len += to_copy;
    bytes += to_copy;
    len -= to_copy;

    if (_buffer_len == 64) {
      _process_block(_buffer.data());
      _buffer_len = 0;
    }
  }
}

void HashContext::update(double val) {
  // Use memcpy to avoid strict aliasing issues
  uint8_t buf[8];
  std::memcpy(buf, &val, 8);
  update(buf, 8);
}

void HashContext::update(int64_t val) {
  uint8_t buf[8];
  std::memcpy(buf, &val, 8);
  update(buf, 8);
}

void HashContext::update(uint64_t val) {
  uint8_t buf[8];
  std::memcpy(buf, &val, 8);
  update(buf, 8);
}

void HashContext::update(uint8_t val) { update(&val, 1); }

void HashContext::update(bool val) {
  uint8_t b = val ? 1 : 0;
  update(&b, 1);
}

void HashContext::update(const std::string& s) {
  // Length-prefix to prevent "ab"+"c" == "a"+"bc" collisions
  update(static_cast<uint64_t>(s.size()));
  if (!s.empty()) {
    update(s.data(), s.size());
  }
}

void HashContext::update(const Eigen::MatrixXd& m) {
  // Hash dimensions for disambiguation
  update(static_cast<int64_t>(m.rows()));
  update(static_cast<int64_t>(m.cols()));
  if (m.size() > 0) {
    update(m.data(), static_cast<size_t>(m.size()) * sizeof(double));
  }
}

void HashContext::update(const Eigen::VectorXd& v) {
  update(static_cast<int64_t>(v.size()));
  if (v.size() > 0) {
    update(v.data(), static_cast<size_t>(v.size()) * sizeof(double));
  }
}

void HashContext::update(const Eigen::VectorXi& v) {
  update(static_cast<int64_t>(v.size()));
  if (v.size() > 0) {
    update(v.data(), static_cast<size_t>(v.size()) * sizeof(int));
  }
}

void HashContext::update(const Eigen::MatrixXcd& m) {
  update(static_cast<int64_t>(m.rows()));
  update(static_cast<int64_t>(m.cols()));
  if (m.size() > 0) {
    // complex<double> is 16 bytes (2 doubles)
    update(m.data(),
           static_cast<size_t>(m.size()) * sizeof(std::complex<double>));
  }
}

void HashContext::update(const Eigen::VectorXcd& v) {
  update(static_cast<int64_t>(v.size()));
  if (v.size() > 0) {
    update(v.data(),
           static_cast<size_t>(v.size()) * sizeof(std::complex<double>));
  }
}

void HashContext::update(const Eigen::SparseMatrix<double>& m) {
  update(static_cast<int64_t>(m.rows()));
  update(static_cast<int64_t>(m.cols()));
  update(static_cast<int64_t>(m.nonZeros()));
  // Iterate in column-major order (Eigen's default storage order)
  for (int k = 0; k < m.outerSize(); ++k) {
    for (Eigen::SparseMatrix<double>::InnerIterator it(m, k); it; ++it) {
      update(static_cast<int64_t>(it.row()));
      update(static_cast<int64_t>(it.col()));
      update(it.value());
    }
  }
}

void HashContext::update(const VectorVariant& v) {
  std::visit(
      [this](const auto& vec) {
        using T = std::decay_t<decltype(vec)>;
        if constexpr (std::is_same_v<T, Eigen::VectorXd>) {
          update(uint8_t(0));  // tag: real
          update(vec);
        } else {
          update(uint8_t(1));  // tag: complex
          update(vec);
        }
      },
      v);
}

void HashContext::update(const MatrixVariant& m) {
  std::visit(
      [this](const auto& mat) {
        using T = std::decay_t<decltype(mat)>;
        if constexpr (std::is_same_v<T, Eigen::MatrixXd>) {
          update(uint8_t(0));
          update(mat);
        } else {
          update(uint8_t(1));
          update(mat);
        }
      },
      m);
}

void HashContext::update(const std::vector<size_t>& v) {
  update(static_cast<uint64_t>(v.size()));
  for (auto val : v) {
    update(static_cast<uint64_t>(val));
  }
}

std::string HashContext::hexdigest(size_t truncate_chars) const {
  // Create a copy to finalize without mutating the original
  HashContext copy = *this;

  // Padding: append 1 bit, then zeros, then 64-bit big-endian length
  uint64_t total_bits = copy._total_len * 8;

  // Append 0x80 byte
  uint8_t pad_byte = 0x80;
  copy.update(&pad_byte, 1);

  // Pad with zeros until 56 bytes mod 64
  while (copy._buffer_len != 56) {
    uint8_t zero = 0;
    copy.update(&zero, 1);
  }

  // Append total length in bits as big-endian 64-bit
  uint8_t len_bytes[8];
  for (int i = 7; i >= 0; --i) {
    len_bytes[7 - i] = static_cast<uint8_t>(total_bits >> (i * 8));
  }
  // Directly process to avoid recursion through update() which would change
  // _total_len
  std::memcpy(copy._buffer.data() + copy._buffer_len, len_bytes, 8);
  copy._buffer_len += 8;
  copy._process_block(copy._buffer.data());

  // Convert state to hex string
  std::ostringstream oss;
  oss << std::hex << std::setfill('0');
  for (int i = 0; i < 8; ++i) {
    oss << std::setw(8) << copy._state[i];
  }

  std::string full_hash = oss.str();
  if (truncate_chars > 0 && truncate_chars < full_hash.size()) {
    return full_hash.substr(0, truncate_chars);
  }
  return full_hash;
}

}  // namespace qdk::chemistry::utils
