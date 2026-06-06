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
   * @brief Codespace stabilizers for redundant (qubit-overcomplete) encodings.
   *
   * Each entry is a Pauli operator ``coeff * word`` that commutes with the
   * physical Hamiltonian and equals ``+1`` on the physical subspace.  Standard
   * (non-redundant) encodings such as Jordan-Wigner return an empty vector.
   * The mapper engine uses these generically to append an energy penalty
   * ``lambda * (I - S)`` per stabilizer, lifting non-codespace states out of
   * the low-energy spectrum without changing codespace eigenvalues.
   *
   * @return Reference to the (possibly empty) stabilizer list.
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

  /// @brief Get the data type name for serialization.
  std::string get_data_type_name() const override { return "majorana_mapping"; }

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
   * @brief Verstraete-Cirac (auxiliary-qubit) encoding for a 2D lattice.
   *
   * Builds a locality-preserving fermion-to-qubit encoding directly from the
   * edges of a rectangular 2D ``LatticeGraph``.  Each lattice site is paired
   * with one auxiliary qubit; nearest-neighbour hopping terms map to
   * constant-weight Pauli operators independent of system size.  The encoding
   * acts on the physical subspace defined by the returned ``stabilizers()``.
   *
   * The lattice describes a single spin species (``n_sites`` sites); the
   * factory produces a mapping with ``num_modes == 2 * n_sites`` (one
   * Verstraete-Cirac block per spin sector), so it is consumed by
   * ``QubitMapper`` exactly like ``jordan_wigner(num_modes=2*n_sites)`` and
   * uses ``2 * num_modes`` qubits.
   *
   * @param lattice A single connected rectangular 2D lattice (e.g. built by
   *        ``LatticeGraph::square`` with nearest-neighbour edges only).
   * @return MajoranaMapping with name ``"verstraete-cirac"`` and stabilizers.
   * @throws std::invalid_argument If the lattice is empty or its connectivity
   *         is not a single rectangular nearest-neighbour grid.
   *
   * @see F. Verstraete and J. I. Cirac, J. Stat. Mech. (2005) P09012.
   * @see J. D. Whitfield, V. Havlicek, M. Troyer, Phys. Rev. A 94, 030301(R).
   * @see V. Havlicek, M. Troyer, J. D. Whitfield, Phys. Rev. A 95, 032332.
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

  /// Codespace stabilizers for redundant encodings (empty otherwise).
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
 * @param spin_symmetric If true, use the spin-summed fast path. This assumes
 *        identical integrals across all spin channels (h_alpha == h_beta,
 *        eri_aaaa == eri_bbbb == eri_aabb), as produced by restricted orbitals.
 *        For unrestricted orbital sets, pass false — the engine handles each
 *        spin channel independently.
 * @param threshold Pauli terms with |coeff| < threshold are dropped.
 * @param integral_threshold Integrals with |value| < this are skipped.
 * @return MajoranaMapResult with Pauli words and coefficients.
 */
MajoranaMapResult majorana_map_hamiltonian(
    const MajoranaMapping& mapping, double core_energy, const double* h1_alpha,
    const double* h1_beta, const double* eri_aaaa, const double* eri_aabb,
    const double* eri_bbbb, std::size_t n_spatial, bool spin_symmetric,
    double threshold, double integral_threshold);

}  // namespace qdk::chemistry::data
