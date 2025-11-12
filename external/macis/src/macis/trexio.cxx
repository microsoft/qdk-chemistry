/*
 * MACIS Copyright (c) 2023, The Regents of the University of California,
 * through Lawrence Berkeley National Laboratory (subject to receipt of
 * any required approvals from the U.S. Dept. of Energy). All rights reserved.
 *
 * See LICENSE.txt for details
 */

#include <array>
#include <iostream>
#include <macis/util/trexio.hpp>
#include <vector>

namespace macis {

trexio_exception::trexio_exception(std::string func_name, trexio_exit_code rc)
    : msg_("TREXIO Error: " + func_name +
           "\n  Error Message: " + std::string(trexio_string_of_error(rc))) {}

const char* trexio_exception::what() const noexcept { return msg_.c_str(); }

#define TREXIO_EXCEPTION(rc) throw trexio_exception(__PRETTY_FUNCTION__, rc)

TREXIOFile::TREXIOFile(std::string name, char mode, int backend) {
  trexio_exit_code rc;
  file_handle_ = trexio_open(name.c_str(), mode, backend, &rc);
  if (rc != TREXIO_SUCCESS) TREXIO_EXCEPTION(rc);
}

TREXIOFile::~TREXIOFile() noexcept {
  if (file_handle_) trexio_close(file_handle_);
}

TREXIOFile::TREXIOFile(TREXIOFile&& other) noexcept {
  this->file_handle_ = other.file_handle_;
  other.file_handle_ = nullptr;
}

int64_t TREXIOFile::read_mo_num() const {
  int64_t nmo;
  auto rc = trexio_read_mo_num_64(file_handle_, &nmo);
  if (rc != TREXIO_SUCCESS) TREXIO_EXCEPTION(rc);
  return nmo;
}

int64_t TREXIOFile::read_mo_2e_int_eri_size() const {
  int64_t neri;
  auto rc = trexio_read_mo_2e_int_eri_size(file_handle_, &neri);
  if (rc != TREXIO_SUCCESS) TREXIO_EXCEPTION(rc);
  return neri;
}

double TREXIOFile::read_nucleus_repulsion() const {
  double E;
  auto rc = trexio_read_nucleus_repulsion(file_handle_, &E);
  if (rc != TREXIO_SUCCESS) TREXIO_EXCEPTION(rc);
  return E;
}

void TREXIOFile::read_mo_1e_int_core_hamiltonian(double* h) const {
  auto rc = trexio_read_mo_1e_int_core_hamiltonian(file_handle_, h);
  if (rc != TREXIO_SUCCESS) TREXIO_EXCEPTION(rc);
}

void TREXIOFile::read_mo_2e_int_eri(double* V) const {
  const auto norb = read_mo_num();
  const auto neri = read_mo_2e_int_eri_size();
  const size_t nbatch = 100000;
  // #TODO: Promote nbatch to a configurable knob; this magic constant may not
  // suit different memory budgets.
  std::vector<std::array<int32_t, 4>> idx_batch(nbatch);
  std::vector<double> V_batch(nbatch);

  size_t ioff = 0;
  int64_t icount = nbatch;
  while (icount == nbatch) {
    if (ioff < neri) {
      // #TODO: Check the return code from trexio_read_mo_2e_int_eri so we fail
      // fast instead of silently continuing.
      trexio_read_mo_2e_int_eri(file_handle_, ioff, &icount,
                                idx_batch[0].data(), V_batch.data());
    } else {
      icount = 0;
    }

    for (size_t ii = 0; ii < icount; ++ii) {
      const auto [i, j, k, l] = idx_batch[ii];
      const auto v = V_batch[ii];

      // #TODO: Because TREXIO only returns nonzero quartets, we rely on callers
      // having zeroed the full tensor first; otherwise reused buffers leave
      // stale values in untouched slots that masquerade as integrals. Document
      // or enforce that zero-initialization contract.
      V[i + j * norb + k * norb * norb + l * norb * norb * norb] = v;
      V[i + j * norb + l * norb * norb + k * norb * norb * norb] = v;
      V[j + i * norb + k * norb * norb + l * norb * norb * norb] = v;
      V[j + i * norb + l * norb * norb + k * norb * norb * norb] = v;
      V[k + l * norb + i * norb * norb + j * norb * norb * norb] = v;
      V[k + l * norb + j * norb * norb + i * norb * norb * norb] = v;
      V[l + k * norb + i * norb * norb + j * norb * norb * norb] = v;
      V[l + k * norb + j * norb * norb + i * norb * norb * norb] = v;
    }
    ioff += icount;
  }
}

int64_t TREXIOFile::read_determinant_num() const {
  int64_t ndet;
  auto rc = trexio_read_determinant_num_64(file_handle_, &ndet);
  if (rc != TREXIO_SUCCESS) TREXIO_EXCEPTION(rc);
  return ndet;
}

int32_t TREXIOFile::get_determinant_int64_num() const {
  int32_t n64;
  auto rc = trexio_get_int64_num(file_handle_, &n64);
  if (rc != TREXIO_SUCCESS) TREXIO_EXCEPTION(rc);
  return n64;
}

void TREXIOFile::read_determinant_list(int64_t ndet, int64_t* dets,
                                       int64_t ioff) const {
  int64_t icount = ndet;
  auto rc = trexio_read_determinant_list(file_handle_, ioff, &icount, dets);
  if (rc != TREXIO_SUCCESS) TREXIO_EXCEPTION(rc);
}

void TREXIOFile::write_mo_num(int64_t nmo) {
  auto rc = trexio_write_mo_num_64(file_handle_, nmo);
  if (rc != TREXIO_SUCCESS) TREXIO_EXCEPTION(rc);
}

void TREXIOFile::write_nucleus_repulsion(double E) {
  auto rc = trexio_write_nucleus_repulsion(file_handle_, E);
  if (rc != TREXIO_SUCCESS) TREXIO_EXCEPTION(rc);
}

void TREXIOFile::write_mo_1e_int_core_hamiltonian(const double* h) {
  auto rc = trexio_write_mo_1e_int_core_hamiltonian(file_handle_, h);
  if (rc != TREXIO_SUCCESS) TREXIO_EXCEPTION(rc);
}

void TREXIOFile::write_mo_2e_int_eri(const double* V) {
  const auto norb = read_mo_num();
  const auto npair = (norb * (norb + 1)) / 2;
  const auto nquad = (npair * (npair + 1)) / 2;
  std::vector<double> V_compress(nquad);
  std::vector<std::array<int32_t, 4>> idx(nquad);

  size_t ijkl = 0;
  for (size_t i = 0, ij = 0; i < norb; ++i)
    for (size_t j = 0; j <= i; ++j, ++ij) {
      for (size_t k = 0, kl = 0; k < norb; ++k)
        for (size_t l = 0; l <= k; ++l, ++kl) {
          if (kl > ij) continue;
          V_compress.at(ijkl) =
              V[i + j * norb + k * norb * norb + l * norb * norb * norb];
          idx.at(ijkl) = {i, j, k, l};
          ijkl++;
        }
    }

  auto rc = trexio_write_mo_2e_int_eri(file_handle_, 0, nquad, idx[0].data(),
                                       V_compress.data());
  if (rc != TREXIO_SUCCESS) TREXIO_EXCEPTION(rc);
}

void TREXIOFile::write_determinant_list(int64_t ndet, const int64_t* dets,
                                        int64_t ioff) {
  int64_t icount = ndet;
  auto rc = trexio_write_determinant_list(file_handle_, ioff, icount, dets);
  if (rc != TREXIO_SUCCESS) TREXIO_EXCEPTION(rc);
}

}  // namespace macis
