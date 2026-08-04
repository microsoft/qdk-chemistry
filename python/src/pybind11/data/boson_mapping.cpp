// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <pybind11/complex.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstddef>
#include <nlohmann/json.hpp>
#include <qdk/chemistry/data/boson_mapping.hpp>
#include <qdk/chemistry/data/bosonic_modes.hpp>
#include <qdk/chemistry/data/hamiltonian.hpp>
#include <qdk/chemistry/data/pauli_operator.hpp>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "path_utils.hpp"

namespace py = pybind11;

namespace {

using qdk::chemistry::data::BosonPauliTerms;

// The C++ primitives return ``(coefficient, word)`` pairs; Python callers
// consistently see ``(word, coefficient)`` so that the shape matches the
// ``(words, coefficients)`` tuple returned by ``boson_map_hamiltonian``.
py::list terms_to_python(const BosonPauliTerms& terms) {
  py::list out;
  for (const auto& [coefficient, word] : terms) {
    out.append(py::make_tuple(py::cast(word), py::cast(coefficient)));
  }
  return out;
}

}  // namespace

void bind_boson_mapping(pybind11::module& data) {
  using namespace qdk::chemistry::data;

  py::enum_<BosonEncoding>(data, "BosonEncoding",
                           R"(
Boson-to-qubit encoding family.

Every encoding maps the truncated local Fock space of one mode onto
``nq = log2(d)`` qubits. ``StandardBinary``, the default, uses
``codeword(n) = n``; ``GrayCode`` uses ``codeword(n) = n XOR (n >> 1)``, so
adjacent occupations differ in a single bit. The two use exactly the same
number of Pauli terms and differ only in which computational basis state
represents which occupation number.

``Custom`` is not an encoding rule but a tag: it marks a mapping built by
:meth:`BosonMapping.from_codeword_table`, whose codeword table is arbitrary and
is carried explicitly by the object. It is never inferred, so a table that
happens to equal the standard-binary one still reports ``Custom``, and it cannot
be passed to :meth:`BosonMapping.for_encoding`.
)")
      .value("StandardBinary", BosonEncoding::StandardBinary)
      .value("GrayCode", BosonEncoding::GrayCode)
      .value("Custom", BosonEncoding::Custom);

  data.def("boson_encoding_from_string", &boson_encoding_from_string,
           py::arg("name"),
           R"(
Parse a boson encoding name.

Args:
    name (str): Case-insensitive encoding name; ``"standard-binary"`` (aliases ``"standard_binary"``, ``"binary"``, ``"sb"``), ``"gray-code"`` (aliases ``"gray_code"``, ``"gray"``, ``"gc"``) or ``"custom"``.

Returns:
    BosonEncoding: The parsed encoding. ``"custom"`` yields ``BosonEncoding.Custom``, which is a tag rather than a rule and cannot be given to :meth:`BosonMapping.for_encoding`.

Raises:
    ValueError: If the name is not a recognised encoding.
)");

  py::class_<BosonMapping, DataClass, py::smart_holder> mapping(data,
                                                                "BosonMapping",
                                                                R"(
Boson-to-qubit encoding for a truncated bosonic Fock space.

Each bosonic mode is truncated to a local dimension ``d = mode_dimension(i)``
and encoded on ``nq = log2(d)`` qubits via an isometry that sends occupation
``n`` to the computational basis state ``codeword(i, n)``. Every cutoff must be
a power of two: the code is then surjective, the encoded subspace is the whole
Hilbert space, and there is no leakage.

The cutoff is owned by the ``BosonicModes`` basis and is attributed per mode, so
every accessor here takes a mode index. The uniform factories
(``standard_binary``, ``gray_code``, ``for_encoding``) simply give every mode the
same dimension; ``for_basis`` carries whatever the basis states.

The encoding set is open. An encoding *is* its codeword table, so
``from_codeword_table`` accepts any per-mode permutation of ``range(d)`` and the
named families are conveniences on top of it. ``codeword_table(i)`` is the exact
inverse of that factory.

Qubit layout follows the library convention that qubit 0 is the **rightmost**
character of a Pauli label. Mode ``i`` owns the contiguous qubit block starting
at ``sum(qubits_per_mode(j) for j > i)``, so mode 0 occupies the most
significant (leftmost) block and the encoded basis index of an occupation
tuple ``(n_0, ..., n_{L-1})`` is row-major in that tuple.

Examples:
    >>> from qdk_chemistry.data import BosonMapping, BosonEncoding
    >>> mapping = BosonMapping.standard_binary(num_modes=2, mode_dimension=4)
    >>> mapping.num_qubits()
    4
    >>> gray = BosonMapping.gray_code(num_modes=2, mode_dimension=4)
    >>> gray.codeword_table(0)
    [0, 1, 3, 2]
)");

  mapping
      .def_static("standard_binary", &BosonMapping::standard_binary,
                  py::arg("num_modes"), py::arg("mode_dimension"),
                  R"(
Create a standard-binary boson-to-qubit mapping.

Args:
    num_modes (int): Number of bosonic modes.
    mode_dimension (int): Local Fock-space dimension; must be a power of two of at least 2.

Returns:
    BosonMapping: The mapping.

Raises:
    ValueError: If ``mode_dimension`` is not a power of two of at least 2.
)")
      .def_static("gray_code", &BosonMapping::gray_code, py::arg("num_modes"),
                  py::arg("mode_dimension"),
                  R"(
Create a Gray-code boson-to-qubit mapping.

Args:
    num_modes (int): Number of bosonic modes.
    mode_dimension (int): Local Fock-space dimension; must be a power of two of at least 2.

Returns:
    BosonMapping: The mapping.

Raises:
    ValueError: If ``mode_dimension`` is not a power of two of at least 2.
)")
      .def_static("for_encoding", &BosonMapping::for_encoding,
                  py::arg("num_modes"), py::arg("mode_dimension"),
                  py::arg("encoding"),
                  R"(
Create a mapping for an explicitly chosen encoding.

Args:
    num_modes (int): Number of bosonic modes.
    mode_dimension (int): Local Fock-space dimension; must be a power of two of at least 2.
    encoding (BosonEncoding): The encoding family to use.

Returns:
    BosonMapping: The mapping.

Raises:
    ValueError: If ``mode_dimension`` is not a power of two of at least 2.
)")
      .def_static("for_basis", &BosonMapping::for_basis, py::arg("modes"),
                  py::arg("encoding") = BosonEncoding::StandardBinary,
                  R"(
Create a mapping that matches a bosonic basis.

The cutoff is read from ``modes`` per mode; the mapping never owns or duplicates
it. This is the recommended entry point.

Args:
    modes (BosonicModes): The bosonic single-particle basis.
    encoding (BosonEncoding, optional): The encoding family; defaults to standard binary.

Returns:
    BosonMapping: A mapping whose cutoffs match ``modes``.

Raises:
    ValueError: If any mode dimension of the basis is not a power of two.
)")
      .def_static("from_codeword_table", &BosonMapping::from_codeword_table,
                  py::arg("per_mode_codewords"), py::arg("name") = "",
                  R"(
Create a mapping from an explicit codeword table.

This is the open end of the encoding set: any injective assignment of
occupation numbers to computational basis states can be used, not only the
named families. ``per_mode_codewords[i][n]`` is the codeword representing
occupation ``n`` of mode ``i``, and ``len(per_mode_codewords[i])`` is that
mode's local dimension ``d``.

Because every cutoff is a power of two and the code must be surjective on
``nq = log2(d)`` qubits, a valid table is exactly a permutation of
``range(d)``. ``standard_binary`` is the identity permutation and ``gray_code``
is ``n ^ (n >> 1)``; passing either reproduces the corresponding named mapping
operator for operator.

The resulting mapping always reports ``BosonEncoding.Custom`` -- the table is
never matched against the named families -- and the table, not the enum, is
what is written to and read back from serialized documents.

Args:
    per_mode_codewords (list[list[int]]): One list per mode; entry ``n`` is the codeword for occupation ``n``.
    name (str, optional): Label reported by ``name``; defaults to ``"custom"``.

Returns:
    BosonMapping: A mapping using exactly the supplied codewords.

Raises:
    ValueError: If the table is empty, if any mode has fewer than 2 levels, if any mode's level count is not a power of two, or if a mode's codewords are not a permutation of ``range(d)`` (out of range or repeated).

Examples:
    >>> from qdk_chemistry.data import BosonMapping, BosonEncoding
    >>> mapping = BosonMapping.from_codeword_table([[0, 1, 3, 2]] * 2)
    >>> mapping.encoding == BosonEncoding.Custom
    True
    >>> mapping.codeword_table(0) == BosonMapping.gray_code(num_modes=2, mode_dimension=4).codeword_table(0)
    True
)")
      .def("num_modes", &BosonMapping::num_modes,
           R"(
Number of bosonic modes.

Returns:
    int: The number of modes.
)")
      .def("mode_dimension", &BosonMapping::mode_dimension, py::arg("mode"),
           R"(
Local Fock-space dimension of a single mode.

Args:
    mode (int): Mode index.

Returns:
    int: That mode's dimension ``d = n_max + 1``.

Raises:
    IndexError: If ``mode`` is not a valid mode index.
)")
      .def("mode_dimensions", &BosonMapping::mode_dimensions,
           R"(
All local Fock-space dimensions, indexed by mode.

Returns:
    list[int]: One local Fock-space dimension per mode.
)")
      .def("uniform_dimension", &BosonMapping::uniform_dimension,
           R"(
Common local dimension, when every mode shares one.

Returns:
    int | None: The uniform dimension, or ``None`` if it varies by mode.
)")
      .def("max_occupation", &BosonMapping::max_occupation, py::arg("mode"),
           R"(
Largest occupation number representable on a mode.

Args:
    mode (int): Mode index.

Returns:
    int: ``n_max = d - 1`` for that mode.

Raises:
    IndexError: If ``mode`` is not a valid mode index.
)")
      .def("qubits_per_mode", &BosonMapping::qubits_per_mode, py::arg("mode"),
           R"(
Number of qubits used to encode a single mode.

Args:
    mode (int): Mode index.

Returns:
    int: ``nq = log2(d)`` for that mode.

Raises:
    IndexError: If ``mode`` is not a valid mode index.
)")
      .def("num_qubits", &BosonMapping::num_qubits,
           R"(
Total number of qubits used by the mapping.

Returns:
    int: The sum of ``qubits_per_mode(i)`` over every mode.
)")
      .def_property_readonly("encoding", &BosonMapping::encoding,
                             R"(
BosonEncoding: The encoding family used by this mapping. Mappings built by ``from_codeword_table`` always report ``BosonEncoding.Custom``.
)")
      .def_property_readonly("name", &BosonMapping::name,
                             R"(
str: Canonical name of the encoding, e.g. ``"standard-binary"``, or the caller-supplied label of a custom table (``"custom"`` by default).
)")
      .def("codeword", &BosonMapping::codeword, py::arg("mode"),
           py::arg("level"),
           R"(
Computational basis state representing an occupation number.

Args:
    mode (int): Mode index.
    level (int): Occupation number ``n`` in ``[0, d)`` for that mode.

Returns:
    int: The codeword bit pattern for that occupation.

Raises:
    IndexError: If ``mode`` or ``level`` is out of range.
)")
      .def("level", &BosonMapping::level, py::arg("mode"), py::arg("codeword"),
           R"(
Occupation number represented by a computational basis state.

Args:
    mode (int): Mode index.
    codeword (int): Codeword bit pattern in ``[0, d)`` for that mode.

Returns:
    int: The occupation number it encodes.

Raises:
    IndexError: If ``mode`` or ``codeword`` is out of range.
)")
      .def("codeword_table", &BosonMapping::codeword_table, py::arg("mode"),
           R"(
Codewords of every occupation number of a mode, in occupation order.

Args:
    mode (int): Mode index.

Returns:
    list[int]: ``[codeword(mode, 0), ..., codeword(mode, d - 1)]``. Feeding one such list per mode to :meth:`from_codeword_table` reproduces this mapping exactly.

Raises:
    IndexError: If ``mode`` is not a valid mode index.
)")
      .def(
          "isometry",
          [](const BosonMapping& self, std::size_t mode) {
            const auto flat = self.isometry(mode);
            const auto d = self.mode_dimension(mode);
            py::array_t<double> out({d, d});
            std::copy(flat.begin(), flat.end(), out.mutable_data());
            return out;
          },
          py::arg("mode"),
          R"(
Encoding isometry of a single mode as a dense matrix.

Row ``c``, column ``n`` is 1 when ``codeword(mode, n) == c`` and 0 otherwise.
For a power-of-two cutoff this is a permutation matrix, so the encoded subspace
is the full ``2**nq``-dimensional Hilbert space and there is no leakage.

Args:
    mode (int): Mode index.

Returns:
    numpy.ndarray: A ``(d, d)`` matrix.

Raises:
    IndexError: If ``mode`` is not a valid mode index.
)")
      .def("mode_qubits", &BosonMapping::mode_qubits, py::arg("mode"),
           R"(
Global qubit indices owned by a mode, least significant first.

Args:
    mode (int): Mode index.

Returns:
    list[int]: The ``nq`` global qubit indices of that mode.

Raises:
    IndexError: If ``mode`` is not a valid mode index.
)")
      .def("validate_basis", &BosonMapping::validate_basis, py::arg("modes"),
           R"(
Check that a bosonic basis agrees with this mapping.

Args:
    modes (BosonicModes): The basis to check.

Raises:
    ValueError: If the mode count or any mode dimension disagrees with the mapping.
)")
      .def(
          "diagonal",
          [](const BosonMapping& self, const std::vector<double>& values,
             std::size_t mode, double threshold) {
            return terms_to_python(self.diagonal(values, mode, threshold));
          },
          py::arg("values"), py::arg("mode"), py::arg("threshold") = 1e-14,
          R"(
Exact Pauli decomposition of an arbitrary diagonal single-mode operator.

``values[n]`` is the eigenvalue on occupation ``n``. The decomposition uses a
Walsh-Hadamard transform and is exact: one routine covers the number operator,
its powers, and occupation penalties.

Args:
    values (list[float]): One eigenvalue per occupation number; length must equal ``mode_dimension(mode)``.
    mode (int): Mode the operator acts on.
    threshold (float, optional): Drop terms with magnitude below this value.

Returns:
    list[tuple]: ``(word, coefficient)`` pairs of Pauli words and complex coefficients.

Raises:
    ValueError: If ``values`` has the wrong length.
    IndexError: If ``mode`` is not a valid mode index.
)")
      .def(
          "number",
          [](const BosonMapping& self, std::size_t mode, double threshold) {
            return terms_to_python(self.number(mode, threshold));
          },
          py::arg("mode"), py::arg("threshold") = 1e-14,
          R"(
Exact Pauli decomposition of the number operator ``n_hat``.

Args:
    mode (int): Mode the operator acts on.
    threshold (float, optional): Drop terms with magnitude below this value.

Returns:
    list[tuple]: ``(word, coefficient)`` pairs; at most ``qubits_per_mode(mode) + 1`` terms, all of weight at most 1 for the standard-binary encoding.

Raises:
    IndexError: If ``mode`` is not a valid mode index.
)")
      .def(
          "number_squared",
          [](const BosonMapping& self, std::size_t mode, double threshold) {
            return terms_to_python(self.number_squared(mode, threshold));
          },
          py::arg("mode"), py::arg("threshold") = 1e-14,
          R"(
Exact Pauli decomposition of ``n_hat**2``.

Args:
    mode (int): Mode the operator acts on.
    threshold (float, optional): Drop terms with magnitude below this value.

Returns:
    list[tuple]: ``(word, coefficient)`` pairs.

Raises:
    IndexError: If ``mode`` is not a valid mode index.
)")
      .def(
          "number_times_number_minus_one",
          [](const BosonMapping& self, std::size_t mode, double threshold) {
            return terms_to_python(
                self.number_times_number_minus_one(mode, threshold));
          },
          py::arg("mode"), py::arg("threshold") = 1e-14,
          R"(
Exact Pauli decomposition of ``n_hat * (n_hat - 1)``.

This is the on-site interaction of the Bose-Hubbard model. It vanishes
identically for ``mode_dimension(mode) == 2`` (hard-core bosons).

Args:
    mode (int): Mode the operator acts on.
    threshold (float, optional): Drop terms with magnitude below this value.

Returns:
    list[tuple]: ``(word, coefficient)`` pairs; at most ``1 + nq * (nq + 1) / 2`` terms of weight at most 2 for the standard-binary encoding.

Raises:
    IndexError: If ``mode`` is not a valid mode index.
)")
      .def(
          "annihilation",
          [](const BosonMapping& self, std::size_t mode, double threshold) {
            return terms_to_python(self.annihilation(mode, threshold));
          },
          py::arg("mode"), py::arg("threshold") = 1e-14,
          R"(
Exact Pauli decomposition of the truncated annihilation operator ``b``.

``b = sum_{n=1}^{d-1} sqrt(n) |n-1><n|``. Note that truncation makes ``b`` and
``b_dag`` satisfy ``[b, b_dag] = I - d |d-1><d-1|`` rather than the exact
canonical commutation relation.

Args:
    mode (int): Mode the operator acts on.
    threshold (float, optional): Drop terms with magnitude below this value.

Returns:
    list[tuple]: ``(word, coefficient)`` pairs; ``nq * 2**nq`` terms before thresholding.

Raises:
    IndexError: If ``mode`` is not a valid mode index.
)")
      .def(
          "creation",
          [](const BosonMapping& self, std::size_t mode, double threshold) {
            return terms_to_python(self.creation(mode, threshold));
          },
          py::arg("mode"), py::arg("threshold") = 1e-14,
          R"(
Exact Pauli decomposition of the truncated creation operator ``b_dag``.

Args:
    mode (int): Mode the operator acts on.
    threshold (float, optional): Drop terms with magnitude below this value.

Returns:
    list[tuple]: ``(word, coefficient)`` pairs.

Raises:
    IndexError: If ``mode`` is not a valid mode index.
)")
      .def(
          "ladder_product",
          [](const BosonMapping& self,
             const std::vector<std::pair<std::size_t, bool>>& factors,
             double threshold) {
            return terms_to_python(self.ladder_product(factors, threshold));
          },
          py::arg("factors"), py::arg("threshold") = 1e-14,
          R"(
Exact Pauli decomposition of an ordered product of ladder operators.

Args:
    factors (list[tuple[int, bool]]): Ordered ``(mode, is_creation)`` factors, applied left to right; ``True`` selects ``b_dag``, ``False`` selects ``b``.
    threshold (float, optional): Drop terms with magnitude below this value.

Returns:
    list[tuple]: ``(word, coefficient)`` pairs.

Raises:
    IndexError: If any mode index is invalid.
)")
      .def(
          "to_json",
          [](const BosonMapping& self) { return self.to_json().dump(); },
          R"(
Serialize the mapping to a JSON string.

Returns:
    str: The JSON representation.
)")
      .def_static(
          "from_json",
          [](const std::string& json_str) {
            return BosonMapping::from_json(nlohmann::json::parse(json_str));
          },
          py::arg("json_str"),
          R"(
Load a mapping from a JSON string (static method).

Args:
    json_str (str): JSON string produced by ``to_json()``.

Returns:
    BosonMapping: The deserialized mapping.

Raises:
    RuntimeError: If the JSON string is malformed.
)")
      .def(
          "to_json_file",
          [](const BosonMapping& self, const py::object& filename) {
            self.to_json_file(
                qdk::chemistry::python::utils::to_string_path(filename));
          },
          py::arg("filename"),
          R"(
Write the mapping to a JSON file.

Args:
    filename (str | os.PathLike): Destination path.
)")
      .def_static(
          "from_json_file",
          [](const py::object& filename) {
            return BosonMapping::from_json_file(
                qdk::chemistry::python::utils::to_string_path(filename));
          },
          py::arg("filename"),
          R"(
Load a mapping from a JSON file (static method).

Args:
    filename (str | os.PathLike): Source path.

Returns:
    BosonMapping: The deserialized mapping.
)")
      .def(
          "to_hdf5_file",
          [](const BosonMapping& self, const py::object& filename) {
            self.to_hdf5_file(
                qdk::chemistry::python::utils::to_string_path(filename));
          },
          py::arg("filename"),
          R"(
Write the mapping to an HDF5 file.

Args:
    filename (str | os.PathLike): Destination path.
)")
      .def_static(
          "from_hdf5_file",
          [](const py::object& filename) {
            return BosonMapping::from_hdf5_file(
                qdk::chemistry::python::utils::to_string_path(filename));
          },
          py::arg("filename"),
          R"(
Load a mapping from an HDF5 file (static method).

Args:
    filename (str | os.PathLike): Source path.

Returns:
    BosonMapping: The deserialized mapping.
)")
      .def("__repr__",
           [](const BosonMapping& self) { return self.get_summary(); })
      .def("__str__",
           [](const BosonMapping& self) { return self.get_summary(); })
      .def(py::pickle(
          [](const BosonMapping& self) -> std::string {
            return self.to_json().dump();
          },
          [](const std::string& json_str) -> BosonMapping {
            return BosonMapping::from_json(nlohmann::json::parse(json_str));
          }));

  data.def(
      "boson_map_hamiltonian",
      [](const BosonMapping& mapping, const Hamiltonian& hamiltonian,
         double threshold, double integral_threshold) -> py::tuple {
        const BosonMapResult result = boson_map_hamiltonian(
            mapping, hamiltonian, threshold, integral_threshold);
        return py::make_tuple(py::cast(result.words),
                              py::cast(result.coefficients));
      },
      py::arg("mapping"), py::arg("hamiltonian"), py::arg("threshold") = 1e-12,
      py::arg("integral_threshold") = 1e-14,
      R"(
Map a bosonic Hamiltonian to qubit Pauli terms.

Assembles ``H = sum_pq h_pq b_p_dag b_q + 0.5 * sum_pqrs (pq|rs) b_p_dag
b_r_dag b_s b_q`` from the Hamiltonian's stored integrals and encodes it with
``mapping``. This is exactly the chemist-notation storage convention already
used for fermionic Hamiltonians, so no new container is required: with
``h_ii = -mu``, ``h_ij = -t`` on bonds and ``(ii|ii) = U`` the two-body
contraction collapses to ``(U / 2) * sum_i n_i (n_i - 1)``.

The Hamiltonian's constant energy shift is intentionally **not** included,
matching ``majorana_map_hamiltonian``.

Args:
    mapping (BosonMapping): The boson-to-qubit encoding.
    hamiltonian (Hamiltonian): The bosonic Hamiltonian.
    threshold (float, optional): Drop Pauli terms with magnitude below this value.
    integral_threshold (float, optional): Skip integrals with magnitude below this value.

Returns:
    tuple: ``(words, coefficients)`` where ``words`` is a list of sparse Pauli words and ``coefficients`` is a list of complex coefficients.

Raises:
    ValueError: If the Hamiltonian's mode count disagrees with the mapping, or its basis is a ``BosonicModes`` whose cutoff disagrees with the mapping.
)");
}
