// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once
#include <H5Cpp.h>

#include <complex>
#include <cstdint>
#include <nlohmann/json_fwd.hpp>
#include <optional>
#include <qdk/chemistry/data/data_class.hpp>
#include <qdk/chemistry/data/pauli_operator.hpp>
#include <string>
#include <utility>
#include <vector>

namespace qdk::chemistry::data {

class BosonicModes;

class Hamiltonian;

/**
 * @brief A list of (coefficient, Pauli word) pairs.
 *
 * Used for the Pauli images of the bosonic operator primitives. Words carry
 * global qubit indices and are sorted by qubit index; the list itself is
 * sorted deterministically so results are reproducible run to run.
 */
using BosonPauliTerms =
    std::vector<std::pair<std::complex<double>, SparsePauliWord>>;

/**
 * @brief Boson-to-qubit encodings supported by @ref BosonMapping.
 *
 * @ref BosonEncoding::StandardBinary and @ref BosonEncoding::GrayCode are named
 * conveniences: each fixes one particular codeword table. @ref
 * BosonEncoding::Custom is what an arbitrary table supplied to
 * @ref BosonMapping::from_codeword_table reports, and it is *not* a single
 * encoding — the table, not this tag, identifies such a mapping.
 */
enum class BosonEncoding : std::uint8_t {
  /// Level @f$n@f$ maps to the binary representation of @f$n@f$.
  StandardBinary = 0,
  /// Level @f$n@f$ maps to @f$n \oplus (n \gg 1)@f$.
  GrayCode = 1,
  /// An explicit codeword table supplied by the caller; see
  /// @ref BosonMapping::from_codeword_table.
  Custom = 2,
};

/**
 * @brief Canonical lowercase name of a boson encoding.
 * @param encoding The encoding.
 * @return ``"standard-binary"``, ``"gray-code"`` or ``"custom"``.
 */
std::string to_string(BosonEncoding encoding);

/**
 * @brief Parse a boson encoding name.
 *
 * Accepts ``"standard-binary"``/``"standard_binary"``/``"sb"``/``"binary"``,
 * ``"gray-code"``/``"gray_code"``/``"gray"``/``"gc"`` and ``"custom"``,
 * case-insensitively.
 *
 * @note ``"custom"`` round-trips the tag only. It names no particular table, so
 *       it cannot be handed to @ref BosonMapping::for_encoding; build a custom
 *       mapping with @ref BosonMapping::from_codeword_table instead.
 *
 * @param name Encoding name.
 * @return The corresponding encoding.
 * @throws std::invalid_argument If the name is not recognised.
 */
BosonEncoding boson_encoding_from_string(const std::string& name);

/**
 * @class BosonMapping
 * @brief Data class describing a boson-to-qubit encoding.
 *
 * Each of the @f$L@f$ bosonic modes is truncated to @f$d_i@f$ levels and stored
 * in its own register of @f$n_q(i) = \log_2 d_i@f$ qubits via an injective
 * *codeword* map @f$\mathrm{cw}: \{0,\ldots,d_i-1\} \to \{0,1\}^{n_q(i)}@f$.
 * The associated isometry is @f$V = \sum_n |\mathrm{cw}(n)\rangle\langle n|@f$.
 *
 * @par The encoding set is open
 * An encoding *is* a codeword table, and this class stores exactly that. The
 * named factories (@ref standard_binary, @ref gray_code, @ref for_encoding,
 * @ref for_basis) are conveniences that fill the table for you;
 * @ref from_codeword_table accepts any table directly, so callers are not
 * limited to the encodings this header happens to name. This mirrors
 * @ref MajoranaMapping, where @c from_table is the general entry point and the
 * named transforms sit on top of it.
 *
 * @par Where the cutoff lives
 * The cutoff is owned by the @ref BosonicModes basis and is attributed per
 * mode. This class mirrors that: it stores one dimension per mode and every
 * accessor and internal loop is per-mode. The named factories construct
 * uniformly; @ref for_basis carries whatever the basis states, and
 * @ref from_codeword_table takes one dimension per mode from the table it is
 * given.
 *
 * @par Power-of-two truncation
 * Only power-of-two @f$d_i@f$ is accepted. The codeword map is then a
 * bijection onto the whole register, so @f$V@f$ is unitary, the unphysical
 * subspace is empty, leakage is identically zero, and the "canonical"
 * (zero-extended) and "native" (code-equivalent) Pauli images coincide
 * uniquely. Padding a requested cutoff up to a power of two is free in
 * Pauli-term count — @f$d=3@f$ and @f$d=4@f$ both cost 32 hopping terms — and
 * only lowers the truncation error. Padding is never applied implicitly; opt
 * in with @ref BosonicModes::padded_to_power_of_two.
 *
 * @par Register layout
 * Mode @f$i@f$ owns the contiguous global qubit block starting at
 * @f$\sum_{j>i} n_q(j)@f$, so mode 0 occupies the most significant block.
 * Within a block, local qubit 0 (the encoding's least significant bit) is the
 * least significant global qubit. The global basis index of
 * @f$|n_0 n_1 \ldots n_{L-1}\rangle@f$ is therefore
 * @f$\sum_i \mathrm{cw}(n_i)\, 2^{\sum_{j>i} n_q(j)}@f$, matching row-major
 * occupation ordering.
 *
 * @par Pauli conventions
 * Words are returned as @ref SparsePauliWord with global qubit indices, which
 * is endianness-free. Rendering to a label via ::sparse_pauli_word_to_label
 * uses the ``QubitOperator`` convention (qubit 0 is the rightmost character).
 */
class BosonMapping : public DataClass {
 public:
  /**
   * @brief Standard-binary encoding.
   * @param num_modes Number of bosonic modes.
   * @param mode_dimension Local Fock-space dimension; must be a power of two.
   * @return The mapping.
   * @throws std::invalid_argument If the arguments are invalid.
   */
  static BosonMapping standard_binary(std::size_t num_modes,
                                      std::size_t mode_dimension);

  /**
   * @brief Gray-code encoding.
   *
   * Adjacent occupation levels differ in exactly one qubit. The Pauli-term
   * count of the hopping operator is identical to standard binary; only the
   * circuit depth of the diagonal terms differs.
   *
   * @param num_modes Number of bosonic modes.
   * @param mode_dimension Local Fock-space dimension; must be a power of two.
   * @return The mapping.
   * @throws std::invalid_argument If the arguments are invalid.
   */
  static BosonMapping gray_code(std::size_t num_modes,
                                std::size_t mode_dimension);

  /**
   * @brief Build a mapping for an explicit encoding choice.
   * @param num_modes Number of bosonic modes.
   * @param mode_dimension Local Fock-space dimension; must be a power of two.
   * @param encoding Which encoding to use.
   * @return The mapping.
   * @throws std::invalid_argument If the arguments are invalid.
   */
  static BosonMapping for_encoding(std::size_t num_modes,
                                   std::size_t mode_dimension,
                                   BosonEncoding encoding);

  /**
   * @brief Build a mapping that matches a bosonic mode basis.
   *
   * The cutoff is read from @p modes per mode; the mapping never owns or
   * duplicates it. This is the recommended entry point, and the only one that
   * carries a per-mode cutoff.
   *
   * @param modes The bosonic mode basis.
   * @param encoding Which encoding to use.
   * @return The mapping.
   * @throws std::invalid_argument If any mode dimension is not a power of two.
   */
  static BosonMapping for_basis(
      const BosonicModes& modes,
      BosonEncoding encoding = BosonEncoding::StandardBinary);

  /**
   * @brief Build a mapping from an explicit codeword table.
   *
   * This is the general constructor, the boson analogue of
   * @c MajoranaMapping::from_table: an encoding *is* the injective map
   * @f$\mathrm{cw}: \{0,\ldots,d-1\} \to \{0,1\}^{n_q}@f$, and this factory
   * takes that map directly. @ref standard_binary and @ref gray_code are
   * conveniences that build particular tables.
   *
   * One inner vector per mode; its length is that mode's local dimension
   * @f$d@f$, and entry @c level is the codeword of @f$|level\rangle@f$. The
   * qubit count of each mode is derived from the table exactly as the named
   * factories derive it, @f$n_q = \log_2 d@f$.
   *
   * @par What counts as a valid table
   * Every mode's table must be a permutation of @f$\{0,\ldots,d-1\}@f$: it must
   * be injective, @f$d@f$ must be a power of two, and every codeword must fit
   * in @f$n_q@f$ qubits. That is not an implementation restriction but the same
   * surjectivity condition the named encodings satisfy — it is what makes the
   * isometry unitary, the unphysical subspace empty and leakage identically
   * zero. Standard binary is the identity permutation and Gray code is the
   * reflected-binary permutation.
   *
   * @par How such a mapping identifies itself
   * @ref encoding always reports @ref BosonEncoding::Custom for a mapping built
   * here, even when the supplied table happens to coincide with a named
   * encoding: the tag records how the mapping was built, and no table
   * recognition is attempted. The table is the identity of the mapping — it is
   * what is hashed, what is written to disk, and what is restored on read.
   * @ref name carries @p name, or ``"custom"`` when none is given.
   *
   * @param per_mode_codewords One codeword list per mode; entry @c level holds
   *        the codeword of occupation @c level.
   * @param name Optional label reported by @ref name; defaults to
   *        ``"custom"``.
   * @return The mapping.
   * @throws std::invalid_argument If the table is empty, or if any mode's
   *         codeword list has fewer than 2 entries, has a length that is not a
   *         power of two, holds a codeword that does not fit in that mode's
   *         qubits, or repeats a codeword.
   */
  static BosonMapping from_codeword_table(
      std::vector<std::vector<std::uint64_t>> per_mode_codewords,
      std::string name = "");

  /// Number of bosonic modes.
  std::size_t num_modes() const { return mode_dimensions_.size(); }

  /**
   * @brief Local Fock-space dimension @f$d = n_{\max}+1@f$ of one mode.
   *
   * The authoritative accessor for the cutoff in force on a mode; the cutoff
   * itself is owned by the @ref BosonicModes basis this mapping was built for.
   *
   * @param mode Mode index.
   * @return That mode's local dimension.
   * @throws std::out_of_range If @p mode is not a valid mode index.
   */
  std::size_t mode_dimension(std::size_t mode) const;

  /// All local Fock-space dimensions, indexed by mode.
  const std::vector<std::size_t>& mode_dimensions() const {
    return mode_dimensions_;
  }

  /**
   * @brief Common local dimension, when every mode shares one.
   * @return The uniform dimension, or @c std::nullopt if it varies by mode.
   */
  std::optional<std::size_t> uniform_dimension() const;

  /**
   * @brief Largest representable occupation @f$n_{\max} = d - 1@f$ of a mode.
   * @param mode Mode index.
   * @return That mode's largest representable occupation.
   * @throws std::out_of_range If @p mode is not a valid mode index.
   */
  std::size_t max_occupation(std::size_t mode) const;

  /**
   * @brief Number of qubits used by one mode, @f$\log_2 d@f$.
   * @param mode Mode index.
   * @return That mode's register width.
   * @throws std::out_of_range If @p mode is not a valid mode index.
   */
  std::size_t qubits_per_mode(std::size_t mode) const;

  /// Total number of qubits, the sum of qubits_per_mode(i) over all modes.
  std::size_t num_qubits() const { return num_qubits_; }

  /**
   * @brief Which encoding this mapping implements.
   *
   * @ref BosonEncoding::Custom for any mapping built by
   * @ref from_codeword_table, whose identity is its @ref codeword_table rather
   * than this tag.
   *
   * @return The encoding tag.
   */
  BosonEncoding encoding() const { return encoding_; }

  /**
   * @brief Human-readable encoding name.
   *
   * ``"standard-binary"`` or ``"gray-code"`` for the named encodings; for a
   * mapping built by @ref from_codeword_table, the label supplied there, or
   * ``"custom"`` if none was given. This is the string a plugin backend reads
   * to pick a transform, and the string recorded on a mapped
   * @ref QubitOperator.
   *
   * @note @ref MajoranaMapping deliberately carries two names — @c name and
   *       @c base_encoding — because tapering renames a mapping while leaving
   *       it built on an underlying transform, so a backend must be told which
   *       untapered transform to use. @ref BosonMapping has no such
   *       transformation, so the two would be the same string; it therefore
   *       offers this one accessor only. If boson tapering is ever added, a
   *       @c base_encoding accessor should be added *with* it, so that it is
   *       introduced already carrying a meaning.
   *
   * @return The name.
   */
  const std::string& name() const { return name_; }

  /**
   * @brief Codeword (register bit pattern) of an occupation level on a mode.
   * @param mode Mode index.
   * @param level Occupation number in [0, d) for that mode.
   * @return The bit pattern, with bit @c k the value of local qubit @c k.
   * @throws std::out_of_range If @p mode or @p level is out of range.
   */
  std::uint64_t codeword(std::size_t mode, std::size_t level) const;

  /**
   * @brief Occupation level of a codeword (inverse of @ref codeword).
   * @param mode Mode index.
   * @param codeword Register bit pattern in [0, d) for that mode.
   * @return The occupation level.
   * @throws std::out_of_range If @p mode or @p codeword is out of range.
   */
  std::size_t level(std::size_t mode, std::uint64_t codeword) const;

  /**
   * @brief The full codeword table of a mode, indexed by occupation level.
   *
   * The exact inverse of @ref from_codeword_table: feeding one table per mode
   * back to that factory reproduces this mapping's operators.
   *
   * @param mode Mode index.
   * @return That mode's codeword table.
   * @throws std::out_of_range If @p mode is not a valid mode index.
   */
  const std::vector<std::uint64_t>& codeword_table(std::size_t mode) const;

  /**
   * @brief Dense isometry @f$V = \sum_n |\mathrm{cw}(n)\rangle\langle n|@f$
   *        of one mode.
   *
   * Row-major with @f$2^{n_q}@f$ rows and @f$d@f$ columns. Since @f$d@f$ is a
   * power of two the matrix is square and is a permutation matrix.
   *
   * @param mode Mode index.
   * @return The isometry, flattened row-major.
   * @throws std::out_of_range If @p mode is not a valid mode index.
   * @throws std::overflow_error If the matrix would exceed 2^22 entries.
   */
  std::vector<double> isometry(std::size_t mode) const;

  /**
   * @brief Global qubit indices owned by a mode, least significant first.
   *
   * Mode 0 owns the most significant block, matching the row-major
   * @f$(n_0, \ldots, n_{L-1})@f$ occupation ordering, so the block of mode
   * @f$i@f$ starts at @f$\sum_{j>i} n_q(j)@f$.
   *
   * @param mode Mode index.
   * @return @c qubits_per_mode(mode) global indices.
   * @throws std::out_of_range If @p mode is not a valid mode index.
   */
  std::vector<std::uint64_t> mode_qubits(std::size_t mode) const;

  /**
   * @brief Check that a bosonic basis is compatible with this mapping.
   * @param modes The basis to check.
   * @throws std::invalid_argument If the mode count or any mode dimension
   *         disagrees with this mapping.
   */
  void validate_basis(const BosonicModes& modes) const;

  /**
   * @brief Pauli image of an arbitrary diagonal function of the occupation.
   *
   * Computes @f$\sum_{n} f(n)\, |\mathrm{cw}(n)\rangle\langle
   * \mathrm{cw}(n)|@f$ exactly, via a fast Walsh-Hadamard transform: every
   * diagonal operator is a real combination of the @f$2^{n_q}@f$ products of
   * @f$Z@f$ on the mode's qubits, with @f$c_S = 2^{-n_q}\sum_n
   * f(n)\,(-1)^{|\mathrm{cw}(n)\wedge S|}@f$. One routine therefore covers
   * @f$\hat n@f$, @f$\hat n^2@f$,
   * @f$\hat n(\hat n - 1)@f$ and occupation penalties alike.
   *
   * @param values Function values @f$f(0), \ldots, f(d-1)@f$ for that
   *        mode's dimension @f$d@f$.
   * @param mode Mode the operator acts on.
   * @param threshold Terms with @f$|c_S| <@f$ this are dropped.
   * @return Pauli terms with global qubit indices.
   * @throws std::invalid_argument If @p values does not have
   *         @c mode_dimension(mode) entries.
   * @throws std::out_of_range If @p mode is not a valid mode index.
   */
  BosonPauliTerms diagonal(const std::vector<double>& values, std::size_t mode,
                           double threshold = 1e-14) const;

  /**
   * @brief Pauli image of the number operator @f$\hat n@f$ on a mode.
   * @param mode Mode index.
   * @param threshold Terms with modulus below this are dropped.
   * @return Pauli terms with global qubit indices.
   */
  BosonPauliTerms number(std::size_t mode, double threshold = 1e-14) const;

  /**
   * @brief Pauli image of @f$\hat n^2@f$ on a mode.
   * @param mode Mode index.
   * @param threshold Terms with modulus below this are dropped.
   * @return Pauli terms with global qubit indices.
   */
  BosonPauliTerms number_squared(std::size_t mode,
                                 double threshold = 1e-14) const;

  /**
   * @brief Pauli image of @f$\hat n(\hat n - 1)@f$ on a mode.
   *
   * This is the on-site interaction of the Bose-Hubbard model. It vanishes
   * identically for @f$d = 2@f$ (hard-core bosons).
   *
   * @param mode Mode index.
   * @param threshold Terms with modulus below this are dropped.
   * @return Pauli terms with global qubit indices.
   */
  BosonPauliTerms number_times_number_minus_one(std::size_t mode,
                                                double threshold = 1e-14) const;

  /**
   * @brief Pauli image of the annihilation operator @f$b@f$ on a mode.
   *
   * @f$b = \sum_{n=1}^{d-1}\sqrt{n}\,|n-1\rangle\langle n|@f$, expanded
   * exactly. The result has @f$n_q\,2^{n_q}@f$ terms.
   *
   * @param mode Mode index.
   * @param threshold Terms with modulus below this are dropped.
   * @return Pauli terms with global qubit indices.
   */
  BosonPauliTerms annihilation(std::size_t mode,
                               double threshold = 1e-14) const;

  /**
   * @brief Pauli image of the creation operator @f$b^\dagger@f$ on a mode.
   * @param mode Mode index.
   * @param threshold Terms with modulus below this are dropped.
   * @return Pauli terms with global qubit indices.
   */
  BosonPauliTerms creation(std::size_t mode, double threshold = 1e-14) const;

  /**
   * @brief Pauli image of an ordered product of ladder operators.
   *
   * Bosonic operators on different modes commute, so the factors are grouped
   * by mode (preserving their relative order within each mode), each group is
   * contracted into a dense @f$d \times d@f$ matrix, and the exact Pauli
   * decomposition of every group is tensored together. Groups whose matrix is
   * diagonal take the Walsh-Hadamard path.
   *
   * @param factors Ordered factors as (mode, is_creation) pairs, leftmost
   *        first.
   * @param threshold Terms with modulus below this are dropped.
   * @return Pauli terms with global qubit indices.
   * @throws std::out_of_range If any mode index is invalid.
   */
  BosonPauliTerms ladder_product(
      const std::vector<std::pair<std::size_t, bool>>& factors,
      double threshold = 1e-14) const;

  /// @brief Get the data type name for serialization.
  std::string get_data_type_name() const override { return "boson_mapping"; }

  /// @brief Get a human-readable summary of the mapping.
  std::string get_summary() const override;

  /**
   * @brief Save to file in the specified format.
   * @param filename Path to the output file.
   * @param type Format type ("json", "hdf5", or "h5").
   */
  void to_file(const std::string& filename,
               const std::string& type) const override;

  /**
   * @brief Serialize to JSON.
   *
   * Always writes @c version, @c num_modes, @c mode_dimensions and
   * @c encoding. A mapping built by @ref from_codeword_table additionally
   * writes @c codewords (one array of codewords per mode) and @c name, because
   * for such a mapping the table — not the @c encoding tag — is what identifies
   * it. Both extra fields are omitted for the named encodings, whose payload is
   * therefore unchanged, and both are optional on read.
   *
   * @return The JSON representation.
   */
  nlohmann::json to_json() const override;

  /**
   * @brief Deserialize from JSON.
   *
   * When @c codewords is present it is the source of truth and the mapping is
   * rebuilt from the table; @c encoding is then only a tag. When it is absent
   * the named encoding in @c encoding regenerates the table, which is how every
   * document written before custom tables existed is read.
   *
   * @param data JSON object produced by @ref to_json.
   * @return The reconstructed mapping.
   * @throws std::invalid_argument If the document is malformed, or claims the
   *         ``"custom"`` encoding without carrying a @c codewords table.
   */
  static BosonMapping from_json(const nlohmann::json& data);

  /// @brief Save to a JSON file.
  void to_json_file(const std::string& filename) const override;

  /**
   * @brief Load from a JSON file.
   * @param filename Path to the input file.
   * @return The reconstructed mapping.
   */
  static BosonMapping from_json_file(const std::string& filename);

  /// @brief Save to an HDF5 group.
  void to_hdf5(H5::Group& group) const override;

  /**
   * @brief Load from an HDF5 group.
   * @param group HDF5 group produced by @ref to_hdf5.
   * @return The reconstructed mapping.
   */
  static BosonMapping from_hdf5(H5::Group& group);

  /// @brief Save to an HDF5 file.
  void to_hdf5_file(const std::string& filename) const override;

  /**
   * @brief Load from an HDF5 file.
   * @param filename Path to the input file.
   * @return The reconstructed mapping.
   */
  static BosonMapping from_hdf5_file(const std::string& filename);

  /**
   * @brief Load from file in the specified format.
   * @param filename Path to the input file.
   * @param type Format type ("json", "hdf5", or "h5").
   * @return The reconstructed mapping.
   */
  static BosonMapping from_file(const std::string& filename,
                                const std::string& type);

 private:
  BosonMapping(std::vector<std::size_t> mode_dimensions,
               BosonEncoding encoding);

  /// General constructor: the codeword table is the encoding. Everything else
  /// — the per-mode dimensions, the qubit counts, the register layout and the
  /// inverse tables — is derived from it.
  BosonMapping(std::vector<std::vector<std::uint64_t>> per_mode_codewords,
               BosonEncoding encoding, std::string name);

  void hash_update(qdk::chemistry::utils::HashContext& ctx) const override;

  /// Throw unless d >= 2 and d is a power of two, naming the offending mode.
  static void validate_mode_dimension(std::size_t mode,
                                      std::size_t mode_dimension);

  /// Throw unless every mode's codeword list is a permutation of [0, d),
  /// naming the offending mode, level(s) and codeword.
  static void validate_codeword_table(
      const std::vector<std::vector<std::uint64_t>>& per_mode_codewords);

  /// Codeword table of a named encoding: standard binary is the identity
  /// permutation of [0, d), Gray code the reflected-binary one. Validates each
  /// dimension on the way through.
  static std::vector<std::vector<std::uint64_t>> builtin_codeword_table(
      const std::vector<std::size_t>& mode_dimensions, BosonEncoding encoding);

  /// Throw unless @p mode is a valid mode index.
  void check_mode(std::size_t mode) const;

  /// Global qubit index of local qubit @p local of mode @p mode.
  std::uint64_t global_qubit(std::size_t mode, std::size_t local) const;

  /// Local Fock-space dimension of every mode; one entry per mode.
  std::vector<std::size_t> mode_dimensions_;

  /// log2(mode_dimensions_[i]); one entry per mode.
  std::vector<std::size_t> mode_qubit_counts_;

  /// Global index of the least significant qubit of each mode's block.
  std::vector<std::size_t> mode_qubit_offsets_;

  /// Total register width, the sum of mode_qubit_counts_.
  std::size_t num_qubits_;

  /// Which encoding is implemented.
  BosonEncoding encoding_;

  /// Human-readable encoding name.
  std::string name_;

  /// Codeword table of each mode, indexed by occupation level.
  std::vector<std::vector<std::uint64_t>> codewords_;

  /// Inverse codeword table of each mode, indexed by register bit pattern.
  std::vector<std::vector<std::size_t>> levels_;

  /// Serialization schema version.
  static constexpr const char* SERIALIZATION_VERSION = "0.1.0";
};

/**
 * @brief Result of a boson-to-qubit mapping.
 *
 * Parallel arrays of Pauli words and their complex coefficients.
 */
struct BosonMapResult {
  /// Pauli words (one per non-zero term).
  std::vector<SparsePauliWord> words;
  /// Complex coefficients (parallel to ``words``).
  std::vector<std::complex<double>> coefficients;
};

/**
 * @brief Map a bosonic Hamiltonian in chemist notation to Pauli terms.
 *
 * Assembles
 * @f[
 *   H = E_\mathrm{core}
 *     + \sum_{pq} h_{pq}\, b_p^\dagger b_q
 *     + \tfrac12 \sum_{pqrs} (pq|rs)\, b_p^\dagger b_r^\dagger b_s b_q ,
 * @f]
 * i.e. exactly the storage convention already used for fermionic
 * Hamiltonians. With @f$h_{ii} = -\mu@f$, @f$h_{ij} = -t@f$ on bonds and
 * @f$(ii|ii) = U@f$ this reproduces the Bose-Hubbard model, the two-body
 * contraction collapsing to @f$\tfrac{U}{2}\sum_i n_i(n_i-1)@f$.
 *
 * @param mapping The boson-to-qubit encoding.
 * @param core_energy Constant energy shift added to the identity term.
 * @param one_body Row-major one-body integrals @f$h_{pq}@f$ (L x L).
 * @param two_body_indices Flattened (p,q,r,s) indices, 4 ints per entry.
 * @param two_body_values Integral values, one per entry.
 * @param num_entries Number of stored non-zero two-body integrals.
 * @param num_modes Number of bosonic modes L.
 * @param threshold Pauli terms with |coeff| < threshold are dropped.
 * @param integral_threshold Integrals with |value| < this are skipped.
 * @return BosonMapResult with Pauli words and coefficients.
 * @throws std::invalid_argument If @p num_modes disagrees with @p mapping or
 *         any index is outside [0, num_modes).
 */
BosonMapResult boson_map_hamiltonian(const BosonMapping& mapping,
                                     double core_energy, const double* one_body,
                                     const int* two_body_indices,
                                     const double* two_body_values,
                                     std::size_t num_entries,
                                     std::size_t num_modes, double threshold,
                                     double integral_threshold);

/**
 * @brief Map a bosonic Hamiltonian to Pauli terms.
 *
 * Reads the one- and two-body integrals from @p hamiltonian and applies
 * ::boson_map_hamiltonian. The constant energy shift is excluded
 * (``core_energy = 0``), matching the fermionic qubit-mapper behaviour.
 *
 * @param mapping The boson-to-qubit encoding.
 * @param hamiltonian The bosonic Hamiltonian.
 * @param threshold Pauli terms with |coeff| < threshold are dropped.
 * @param integral_threshold Integrals with |value| < this are skipped.
 * @return BosonMapResult with Pauli words and coefficients.
 * @throws std::invalid_argument If the Hamiltonian's orbital count disagrees
 *         with the mapping, or if its basis is a @ref BosonicModes whose
 *         cutoff disagrees with the mapping.
 */
BosonMapResult boson_map_hamiltonian(const BosonMapping& mapping,
                                     const Hamiltonian& hamiltonian,
                                     double threshold,
                                     double integral_threshold);

}  // namespace qdk::chemistry::data
