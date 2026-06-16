// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

/// QROM-based state preparation for block encoding PREPARE oracles.
///
/// Implements state preparation using Quantum Read-Only Memory (QROM):
///   |0⟩ → Σ_ℓ √(p_ℓ) |ℓ⟩
///
/// This approach stores discretized rotation angles in a QROM table and
/// applies multiplexed Ry rotations to prepare the target state.
/// Suitable for resource estimation (accurate Toffoli/T-gate counts).
///
/// Two modes:
///   1. Multiplexed rotation (SBM-style): n layers of QROM-loaded Ry rotations
///   2. Direct amplitude loading: QROM loads angle bits, phase gradient synthesizes
///
/// Reference: arXiv:2502.15882v1, Section C (SBM decomposition).
namespace QDKChemistry.Utils.QROMStatePrep {

    import Std.Canon.ApplyControlledOnInt;
    import Std.Convert.IntAsDouble;
    import Std.Core.Length;
    import Std.Math.AbsD;
    import Std.Math.ArcCos;
    import Std.Math.Ceiling;
    import Std.Math.Lg;
    import Std.Math.PI;
    import Std.Math.Sqrt;

    /// Parameters for QROM-based state preparation.
    struct QROMStatePrepParams {
        /// Target amplitudes (real, need not be normalized), length L.
        amplitudes : Double[],
        /// Number of bits for rotation angle precision.
        rotationBitPrecision : Int,
        /// Number of qubits for the state register: ⌈log₂ L⌉.
        numStateQubits : Int,
    }

    /// QROM state preparation using Shende-Bullock-Markov (SBM) decomposition.
    ///
    /// Prepares: |0⟩^n → Σ_j c_j |j⟩ using n layers of multiplexed Ry rotations.
    ///
    /// Each layer l applies 2^l controlled-Ry rotations on qubit l,
    /// where angles are determined by the ratio of sub-tree norms.
    ///
    /// For resource estimation, the multiplexed Ry would be implemented via:
    ///   1. QROM to load angle bits (controlled on address qubits[0..l-1])
    ///   2. Phase gradient rotation (adds angle register into gradient register)
    ///   3. QROM uncompute (measurement-based for adjoint)
    ///
    /// For simulation correctness, uses ApplyControlledOnInt with Ry.
    operation QROMStatePrepare(
        params : QROMStatePrepParams,
        qs : Qubit[],
    ) : Unit is Adj + Ctl {
        let n = params.numStateQubits;
        let amplitudes = params.amplitudes;

        // Compute SBM angles for each layer
        let angles = ComputeSBMAngles(amplitudes, n);

        // Apply n layers of multiplexed Ry rotations (MSB-first)
        for l in 0..n - 1 {
            let controlQubits = qs[n - 1..-1..n - l]; // empty for l=0
            let targetQubit = qs[n - 1 - l];
            let layerAngles = angles[l];

            ApplyMultiplexedRy(layerAngles, controlQubits, targetQubit);
        }
    }

    /// Apply multiplexed Ry: for each basis state |j⟩ of controls,
    /// rotate target by Ry(2·angle[j]).
    ///
    /// For resource estimation, this would use QROM + phase gradient.
    /// For simulation, uses controlled-on-int with Ry.
    operation ApplyMultiplexedRy(
        angles : Double[],
        controlQubits : Qubit[],
        targetQubit : Qubit,
    ) : Unit is Adj + Ctl {
        let numAddresses = Length(angles);
        if numAddresses == 1 {
            // Layer 0: no control, just apply Ry
            Ry(2.0 * angles[0], targetQubit);
        } else {
            for j in 0..numAddresses - 1 {
                if AbsD(angles[j]) > 1e-15 {
                    ApplyControlledOnInt(j, q => Ry(2.0 * angles[j], q), controlQubits, targetQubit);
                }
            }
        }
    }

    /// Compute SBM decomposition angles for each layer.
    ///
    /// For n qubits preparing state Σ c_j |j⟩:
    ///   Layer l has 2^l angles, one per block of 2^(n-l) amplitudes.
    ///   angle[k] = arccos(norm_left / norm_total) for block k,
    ///   where left = first half of block, total = whole block.
    ///
    /// Returns: Double[n][2^l] array of angles per layer.
    internal function ComputeSBMAngles(amplitudes : Double[], n : Int) : Double[][] {
        let nPadded = 1 <<< n;

        // Pad amplitudes to power of 2
        mutable padded : Double[] = [];
        for i in 0..nPadded - 1 {
            if i < Length(amplitudes) {
                set padded += [amplitudes[i]];
            } else {
                set padded += [0.0];
            }
        }

        // Compute angles layer by layer (MSB first)
        mutable allAngles : Double[][] = [];
        for l in n - 1..-1..0 {
            let blockSize = 1 <<< (n - l);
            let halfBlock = blockSize / 2;
            let numBlocks = nPadded / blockSize;
            mutable layerAngles : Double[] = [];
            for k in 0..numBlocks - 1 {
                mutable normLeft = 0.0;
                mutable normTotal = 0.0;
                for i in 0..halfBlock - 1 {
                    let idx = k * blockSize + i;
                    set normLeft += padded[idx] * padded[idx];
                }
                for i in 0..blockSize - 1 {
                    let idx = k * blockSize + i;
                    set normTotal += padded[idx] * padded[idx];
                }
                let angle = if normTotal > 1e-30 {
                    ArcCos(Sqrt(normLeft / normTotal))
                } else {
                    0.0
                };
                set layerAngles += [angle];
            }
            set allAngles += [layerAngles];
        }
        // Reverse to get layer 0 first
        return Reversed(allAngles);
    }

    /// Create a QROM state preparation callable.
    function MakeQROMStatePrepOp(params : QROMStatePrepParams) : Qubit[] => Unit is Adj + Ctl {
        QROMStatePrepare(params, _)
    }

    /// Circuit entry point for QROM state preparation (allocates qubits).
    operation MakeQROMStatePrepCircuit(
        amplitudes : Double[],
        rotationBitPrecision : Int,
        numStateQubits : Int,
    ) : Unit {
        let params = new QROMStatePrepParams {
            amplitudes = amplitudes,
            rotationBitPrecision = rotationBitPrecision,
            numStateQubits = numStateQubits,
        };
        use qs = Qubit[numStateQubits];
        QROMStatePrepare(params, qs);
    }

    /// Internal import for Reversed
    import Std.Arrays.Reversed;
}
