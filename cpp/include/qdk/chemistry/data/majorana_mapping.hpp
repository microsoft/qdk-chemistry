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
#include <qdk/chemistry/data/tapering.hpp>
#include <string>
#include <utility>
#include <vector>

namespace qdk::chemistry::data {

class Hamiltonian;

class LatticeGraph;

/**
 * @brief Data class describing a fermion-to-qubit encoding.
 *
 * Majorana-atomic mappings store a 2N-entry Pauli table for individual
 * gamma_k; bilinear-only mappings (via from_bilinears) store the bilinear
 * images directly. bilinear(j, k) is available on both forms.
 */
class MajoranaMapping : public DataClass {
 public:
  /**
   * @brief Construct a Majorana-atomic mapping from a 2N-entry table.
   *
   * @param table 2N SparsePauliWord entries (gamma_0, ..., gamma_{2N-1}).
   * @param name  Optional encoding label.
   * @throws std::invalid_argument If the table is empty or its size is odd.
   */
  static MajoranaMapping from_table(std::vector<SparsePauliWord> table,
                                    std::string name = "");

  /**
   * @brief Construct a bilinear-only mapping from pre-computed bilinears.
   *
   * The upper_triangle vector stores (coeff, word) for each pair (j, k) with
   * j < k, in row-major order: (0,1), (0,2), ..., (0,M-1), (1,2), ...,
   * (M-2,M-1) where M = 2*num_modes.  Size must be M*(M-1)/2.
   *
   * @param num_modes Number of fermionic modes (N).
   * @param upper_triangle Bilinear entries for all j < k.
   * @param name Optional encoding label.
   * @throws std::invalid_argument If sizes are inconsistent or num_modes == 0.
   */
  static MajoranaMapping from_bilinears(
      std::size_t num_modes,
      std::vector<std::pair<std::complex<double>, SparsePauliWord>>
          upper_triangle,
      std::string name = "");

  /// Number of fermionic modes.
  std::size_t num_modes() const { return num_modes_; }

  /// Number of qubits in the encoding table.
  std::size_t num_qubits() const { return num_qubits_; }

  /**
   * @brief Pauli word for Majorana operator gamma_k.
   * @throws std::out_of_range if k >= 2N.
   * @throws std::logic_error if the mapping is not Majorana-atomic.
   */
  const SparsePauliWord& operator()(std::size_t k) const;

  /// Named alias for operator()(k).
  const SparsePauliWord& majorana(std::size_t k) const { return (*this)(k); }

  /**
   * @brief Pauli image of the bilinear i*gamma_j*gamma_k.
   *
   * O(1) lookup. For Majorana-atomic mappings the coefficient is real (+/-1);
   * for bilinear-only mappings it is whatever was provided at construction.
   *
   * @throws std::out_of_range if j or k >= 2N.
   * @throws std::invalid_argument if j == k.
   */
  std::pair<std::complex<double>, const SparsePauliWord&> bilinear(
      std::size_t j, std::size_t k) const;

  /// Whether individual Majoranas have a Pauli image.
  bool is_majorana_atomic() const { return majorana_atomic_; }

  /// The full Majorana-to-Pauli table (empty for bilinear-only mappings).
  const std::vector<SparsePauliWord>& table() const { return table_; }

  /// Encoding name (may be empty for custom encodings).
  const std::string& name() const { return name_; }

  /// Encoding name used by third-party plugin backends to select their own
  /// transform.
  const std::string& base_encoding() const { return base_encoding_; }

  /// Optional post-mapping tapering specification.
  const std::optional<TaperingSpecification>& tapering() const {
    return tapering_;
  }

  /**
   * @brief Stabilizer terms carried by this mapping.
   * @return Reference to the vector of stabilizers.
   */
  const std::vector<std::pair<std::complex<double>, SparsePauliWord>>&
  stabilizers() const {
    return stabilizers_;
  }

  /**
   * @brief Return a copy with tapering removed and the base encoding name
   *        restored.
   *
   * The returned mapping has the same Pauli table and bilinears as the
   * original, but ``tapering()`` is ``std::nullopt`` and ``name()`` equals
   * ``base_encoding()``.
   *
   * @return An untapered copy of this mapping.
   */
  MajoranaMapping without_tapering() const;

  /// @brief Get the static data type name for serialization.
  /// @return "majorana_mapping"
  static std::string data_type_name() {
    return DATACLASS_TO_SNAKE_CASE(MajoranaMapping);
  }

  /// @brief Get the data type name for this instance.
  /// @return "majorana_mapping"
  std::string get_data_type_name() const override { return data_type_name(); }

  /// @brief Get a human-readable summary of the mapping.
  std::string get_summary() const override;

  /**
   * @brief Save to file in the specified format.
   * @param filename Path to the output file.
   * @param type Format type ("json", "hdf5", or "h5").
   */
  void to_file(const std::string& filename,
               const std::string& type) const override;

  /// @brief Serialize to JSON.
  nlohmann::json to_json() const override;

  /// @brief Deserialize from JSON.
  static MajoranaMapping from_json(const nlohmann::json& data);

  /// @brief Save to a JSON file.
  void to_json_file(const std::string& filename) const override;

  /// @brief Load from a JSON file.
  static MajoranaMapping from_json_file(const std::string& filename);

  /// @brief Save to an HDF5 group.
  void to_hdf5(H5::Group& group) const override;

  /// @brief Load from an HDF5 group.
  static MajoranaMapping from_hdf5(H5::Group& group);

  /// @brief Save to an HDF5 file.
  void to_hdf5_file(const std::string& filename) const override;

  /// @brief Load from an HDF5 file.
  static MajoranaMapping from_hdf5_file(const std::string& filename);

  /**
   * @brief Load from file in the specified format.
   * @param filename Path to the input file.
   * @param type Format type ("json", "hdf5", or "h5").
   */
  static MajoranaMapping from_file(const std::string& filename,
                                   const std::string& type);

  // --- Factory methods for standard encodings ---

  /**
   * @brief Jordan-Wigner encoding.
   *
   * Maps fermionic modes to qubits using the Jordan-Wigner transform.
   * Each mode maps to one qubit; the Pauli-Z string encodes parity.
   *
   * @param num_modes Number of fermionic modes (= number of qubits).
   * @return MajoranaMapping with name ``"jordan-wigner"``.
   * @throws std::invalid_argument If num_modes == 0.
   */
  static MajoranaMapping jordan_wigner(std::size_t num_modes);

  /**
   * @brief Bravyi-Kitaev (Fenwick-tree) encoding.
   *
   * Uses the Fenwick-tree construction to balance parity and occupation
   * information across qubits.
   *
   * @param num_modes Number of fermionic modes (= number of qubits).
   * @return MajoranaMapping with name ``"bravyi-kitaev"``.
   * @throws std::invalid_argument If num_modes == 0.
   */
  static MajoranaMapping bravyi_kitaev(std::size_t num_modes);

  /**
   * @brief Balanced binary-tree Bravyi-Kitaev encoding.
   *
   * Recursive balanced binary-tree construction.  Produces shorter
   * Pauli strings than the Fenwick variant for non-power-of-two mode
   * counts.
   *
   * @param num_modes Number of fermionic modes (= number of qubits).
   * @return MajoranaMapping with name ``"bravyi-kitaev-tree"``.
   * @throws std::invalid_argument If num_modes == 0.
   */
  static MajoranaMapping bravyi_kitaev_tree(std::size_t num_modes);

  /**
   * @brief Parity encoding.
   *
   * Each qubit stores the parity (cumulative occupation) up to its
   * corresponding mode, rather than the occupation itself.
   *
   * @param num_modes Number of fermionic modes (= number of qubits).
   * @return MajoranaMapping with name ``"parity"``.
   * @throws std::invalid_argument If num_modes == 0.
   */
  static MajoranaMapping parity(std::size_t num_modes);

  /**
   * @brief Parity encoding with two-qubit reduction.
   *
   * Attaches TaperingSpecification metadata for post-mapping removal
   * of two symmetry qubits (alpha-parity and total-parity).
   *
   * @param num_modes Number of fermionic modes.
   * @param n_alpha   Number of alpha electrons.
   * @param n_beta    Number of beta electrons.
   * @return MajoranaMapping with name ``"parity-2q-reduced"`` and tapering.
   * @throws std::invalid_argument If num_modes is odd, < 4, or electron
   *         counts exceed spatial orbitals.
   */
  static MajoranaMapping parity(std::size_t num_modes, std::size_t n_alpha,
                                std::size_t n_beta);

  /**
   * @brief Symmetry-conserving Bravyi-Kitaev (SCBK) encoding.
   *
   * Combines a balanced BK-tree base mapping with tapering metadata
   * that removes two symmetry qubits.  Post-mapping tapering is applied
   * by the qubit mapper backends.
   *
   * @param num_modes Number of fermionic modes.
   * @param n_alpha   Number of alpha electrons.
   * @param n_beta    Number of beta electrons.
   * @return MajoranaMapping with name ``"symmetry-conserving-bravyi-kitaev"``
   *         and tapering.
   * @throws std::invalid_argument If num_modes is odd, < 4, or electron
   *         counts exceed spatial orbitals.
   */
  static MajoranaMapping symmetry_conserving_bravyi_kitaev(
      std::size_t num_modes, std::size_t n_alpha, std::size_t n_beta);

  /**
   * @brief Verstraete-Cirac encoding.
   *
   * Maps fermionic modes on a `LatticeGraph` to qubits.
   *
   * @param lattice The `LatticeGraph` connectivity.
   * @return MajoranaMapping with name ``"verstraete-cirac"``.
   */
  static MajoranaMapping verstraete_cirac(const LatticeGraph& lattice);

 private:
  MajoranaMapping(
      std::vector<SparsePauliWord> table,
      std::vector<std::pair<std::complex<double>, SparsePauliWord>> bilinears,
      std::string name, std::size_t num_modes, std::size_t num_qubits,
      std::string base_encoding,
      std::optional<TaperingSpecification> tapering = std::nullopt,
      std::vector<std::pair<std::complex<double>, SparsePauliWord>>
          stabilizers = {});

  /// Majorana-to-Pauli table (empty for bilinear-only mappings).
  std::vector<SparsePauliWord> table_;

  /// Cached bilinear table, upper triangle row-major. Always populated.
  std::vector<std::pair<std::complex<double>, SparsePauliWord>> bilinears_;

  /// Human-readable encoding name.
  std::string name_;

  /// Base encoding name associated with the pre-taper Pauli table.
  std::string base_encoding_;

  /// Number of fermionic modes.
  std::size_t num_modes_;

  /// Cached qubit count (max qubit index + 1).
  std::size_t num_qubits_;

  /// True for table-constructed mappings, false for bilinear-only.
  bool majorana_atomic_;

  /// Optional tapering metadata for post-mapping qubit reduction.
  std::optional<TaperingSpecification> tapering_;

  /// Feed the mapping's identifying data into a content hash.
  void hash_update(qdk::chemistry::utils::HashContext& ctx) const override;

  /// Optional stabilizer terms.
  std::vector<std::pair<std::complex<double>, SparsePauliWord>> stabilizers_;

  /// Upper-triangle index: (j, k) with j < k, M = 2*num_modes.
  std::size_t bilinear_index(std::size_t j, std::size_t k) const {
    const std::size_t M = 2 * num_modes_;
    return j * (2 * M - j - 1) / 2 + (k - j - 1);
  }

  static std::size_t compute_num_qubits(
      const std::vector<SparsePauliWord>& table);

  /// Serialization schema version.
  static constexpr const char* SERIALIZATION_VERSION = "0.1.0";
};

/**
 * @brief Result of a Majorana-loop fermion-to-qubit mapping.
 *
 * Parallel arrays of Pauli words and their complex coefficients.
 */
struct MajoranaMapResult {
  /// Pauli words (one per non-zero term).
  std::vector<SparsePauliWord> words;
  /// Complex coefficients (parallel to ``words``).
  std::vector<std::complex<double>> coefficients;
};

/**
 * @brief Map a fermionic Hamiltonian to qubit Pauli terms via Majorana loops.
 *
 * Decomposes each fermionic operator into Majorana products, looks up each
 * gamma_k in the mapping, and accumulates the resulting Pauli words.
 *
 * @param mapping The Majorana-to-Pauli encoding.
 * @param core_energy Core (nuclear repulsion + frozen core) energy.
 * @param h1_alpha One-body integrals, alpha spin (n_spatial x n_spatial).
 * @param h1_beta One-body integrals, beta spin (n_spatial x n_spatial).
 * @param eri_aaaa Flattened two-body integrals (aa|aa), chemist notation.
 * @param eri_aabb Flattened two-body integrals (aa|bb), chemist notation.
 * @param eri_bbbb Flattened two-body integrals (bb|bb), chemist notation.
 * @param n_spatial Number of spatial orbitals.
 * @param spin_symmetric True for restricted input: the caller guarantees
 *        h_alpha == h_beta and eri_aaaa == eri_bbbb == eri_aabb (not
 *        validated).  8-fold integrals use spin-summed excitations; 4-fold
 *        integrals, (pq|rs) = (qp|sr) = (rs|pq), use spin-summed symmetric
 *        and antisymmetric excitation pairs.  False for unrestricted input:
 *        the excitation-pair split is used when aa and bb are 4-fold and ab
 *        satisfies (pq|rs) = (qp|sr), and every spin channel is looped over
 *        in full otherwise.  Either split falls back to the full loop when
 *        integral screening would break the symmetry it needs.
 * @param threshold Pauli terms with |coeff| < threshold are dropped.
 * @param integral_threshold Integrals with |value| < this are skipped.
 * @return MajoranaMapResult with Pauli words and coefficients.
 * @throws std::invalid_argument If spin_symmetric is true and the two-body
 *         integrals define a non-Hermitian operator, or are Hermitian but not
 *         stored with (pq|rs) = (rs|pq); in the latter case, map their
 *         bra-ket average instead.
 */
MajoranaMapResult majorana_map_hamiltonian(
    const MajoranaMapping& mapping, double core_energy, const double* h1_alpha,
    const double* h1_beta, const double* eri_aaaa, const double* eri_aabb,
    const double* eri_bbbb, std::size_t n_spatial, bool spin_symmetric,
    double threshold, double integral_threshold);

/**
 * @brief Map a fermionic Hamiltonian to qubit Pauli terms.
 *
 * Dispatches to the dense, Cholesky, or sparse engine entry point based on
 * the Hamiltonian's container type, without reconstructing a dense N^4
 * two-body tensor when a specialized container is present.  One-body
 * integrals are flattened to row-major layout internally.
 *
 * The constant energy shift (nuclear repulsion / frozen core) is excluded
 * from the mapped operator (`core_energy = 0`), matching the buffer-based
 * overload as invoked by ``QdkQubitMapper``.
 *
 * @param mapping The Majorana-to-Pauli encoding.
 * @param hamiltonian The fermionic Hamiltonian.
 * @param spin_symmetric True for restricted Hamiltonians; see the
 *        buffer-based overload for how integral symmetry selects the path.
 * @param threshold Pauli terms with |coeff| < threshold are dropped.
 * @param integral_threshold Integrals with |value| < this are skipped.
 * @return MajoranaMapResult with Pauli words and coefficients.
 * @throws std::invalid_argument As documented for the container's entry
 *         point.
 */
MajoranaMapResult majorana_map_hamiltonian(const MajoranaMapping& mapping,
                                           const Hamiltonian& hamiltonian,
                                           bool spin_symmetric,
                                           double threshold,
                                           double integral_threshold);

/**
 * @brief Map a fermionic Hamiltonian to qubit Pauli terms directly from
 *        three-center (Cholesky/density-fitted) two-body factors.
 *
 * Equivalent to ::majorana_map_hamiltonian but consumes the low-rank
 * factors instead of a dense N^4 tensor.  The auxiliary index is
 * contracted in integral space one (pq|.) row at a time — a vectorized
 * matrix-vector product per orbital pair — so the dense four-center
 * tensor is never reconstructed.
 *
 * Factors symmetric in their orbital pair give 8-fold integrals.  Otherwise
 * their pair-symmetric and pair-antisymmetric parts S and A give 4-fold
 * integrals exactly when S A^T = 0, which is checked through thin QR
 * factors of S and A (scratch O(N^2 naux + naux^2) for naux <= N^2).
 *
 * @param mapping The Majorana-to-Pauli encoding.
 * @param core_energy Core (nuclear repulsion + frozen core) energy.
 * @param h1_alpha One-body integrals, alpha spin (n_spatial x n_spatial),
 *        row-major.
 * @param h1_beta One-body integrals, beta spin (row-major).
 * @param three_center_aa Alpha three-center factors, column-major
 *        [n_spatial^2 x naux] with the orbital pair index in row-major order.
 * @param three_center_bb Beta three-center factors (same layout). For
 *        restricted inputs this may alias @p three_center_aa.
 * @param n_spatial Number of spatial orbitals.
 * @param naux Number of auxiliary (Cholesky) vectors.
 * @param spin_symmetric As for ::majorana_map_hamiltonian, with the integral
 *        symmetry determined from the factors.
 * @param threshold Pauli terms with |coeff| < threshold are dropped.
 * @param integral_threshold Integrals with |value| < this are skipped.
 * @return MajoranaMapResult with Pauli words and coefficients.
 * @throws std::invalid_argument If spin_symmetric is true and S A^T != 0, so
 *         the reconstructed two-body integrals are non-Hermitian.
 */
MajoranaMapResult majorana_map_hamiltonian_cholesky(
    const MajoranaMapping& mapping, double core_energy, const double* h1_alpha,
    const double* h1_beta, const double* three_center_aa,
    const double* three_center_bb, std::size_t n_spatial, std::size_t naux,
    bool spin_symmetric, double threshold, double integral_threshold);

/**
 * @brief Map a fermionic Hamiltonian to qubit Pauli terms directly from a
 *        sparse two-body integral list.
 *
 * Equivalent to ::majorana_map_hamiltonian but consumes the stored
 * (p,q,r,s) integrals without reconstructing a dense N^4 tensor, iterating
 * nonzero pair products rather than the full N^4 index space.
 *
 * Entries are read literally: positions that are not stored are zero, as in
 * the dense view of SparseHamiltonianContainer, and the resulting tensor is
 * classified like dense restricted integrals.  A complete 4-fold suborbit
 * therefore needs no explicit zeros for the rest of its 8-fold orbit.
 *
 * As a deprecated shorthand, a nonzero entry that is the only stored member
 * of a larger 8-fold orbit is expanded over that orbit, and a warning is
 * logged. Read literally, such an entry is never 4-fold symmetric and would
 * be rejected, so the expansion cannot change the meaning of an accepted
 * tensor.
 *
 * Duplicate entries for the same position must agree exactly; conflicting
 * duplicates are rejected. The result is independent of entry order.
 *
 * @param mapping The Majorana-to-Pauli encoding.
 * @param core_energy Core (nuclear repulsion + frozen core) energy.
 * @param h1_alpha One-body integrals, alpha spin (row-major).
 * @param h1_beta One-body integrals, beta spin (row-major).
 * @param two_body_indices Flattened (p,q,r,s) indices, 4 ints per entry;
 *        each index must lie in [0, n_spatial).
 * @param two_body_values Integral values, one per entry.
 * @param num_entries Number of stored two-body entries, including explicit
 *        zeros.
 * @param n_spatial Number of spatial orbitals.
 * @param spin_symmetric Must be true: the sparse fast path is spin-summed
 *        (SparseHamiltonianContainer is restricted-only).  Unrestricted
 *        Hamiltonians must use ::majorana_map_hamiltonian.
 * @param threshold Pauli terms with |coeff| < threshold are dropped.
 * @param integral_threshold Integrals with |value| < this are skipped.
 * @return MajoranaMapResult with Pauli words and coefficients.
 * @throws std::invalid_argument If spin_symmetric is false, any two-body
 *         index is outside [0, n_spatial), a value is not finite, duplicate
 *         entries for the same position carry conflicting values, or the
 *         literal tensor defines a non-Hermitian operator or is not stored
 *         with (pq|rs) = (rs|pq).
 */
MajoranaMapResult majorana_map_hamiltonian_sparse(
    const MajoranaMapping& mapping, double core_energy, const double* h1_alpha,
    const double* h1_beta, const int* two_body_indices,
    const double* two_body_values, std::size_t num_entries,
    std::size_t n_spatial, bool spin_symmetric, double threshold,
    double integral_threshold);

}  // namespace qdk::chemistry::data
