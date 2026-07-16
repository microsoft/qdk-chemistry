// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

/// QROM-based state preparation.
///
/// Uses SBM (Shende-Bullock-Markov) decomposition: n layers of multiplexed
/// Ry rotations. Each layer's rotation angles are loaded via SelectSwap QROM
/// and applied via phase gradient addition (RyViaPhaseGradient).
///
/// References:
///   - Low et al. (arXiv:2502.15882v1), Section C SBM decomposition
///   - Low, Kliuchnikov, Schaeffer (arXiv:1812.00954): QROM
///   - Sanders et al. (arXiv:2007.07391): Phase gradient rotation
namespace QDKChemistry.Utils.QROMStatePrep {

    import Std.Arrays.Any;
    import Std.Arrays.Reversed;
    import Std.Canon.ApplyPauliFromBitString;
    import Std.Convert.IntAsBoolArray;
    import Std.Convert.IntAsDouble;
    import Std.Math.ArcCos;
    import Std.Math.MinD;
    import Std.Math.MinI;
    import Std.Math.PI;
    import Std.Math.Round;
    import Std.Math.Sqrt;
    import QDKChemistry.Utils.PhaseGradient.PreparePhaseGradientState;
    import QDKChemistry.Utils.PhaseGradient.RyViaPhaseGradient;
    import QDKChemistry.Utils.SelectSwap.SelectSwap;

    /// Parameters for QROM-based state preparation.
    struct QROMStatePrepParams {
        /// Target amplitudes (real, need not be normalized), length L.
        amplitudes : Double[],
        /// Number of bits for rotation angle precision.
        rotationBitPrecision : Int,
        /// Number of qubits for the state register: ⌈log₂ L⌉.
        numStateQubits : Int,
    }

    /// QROM state preparation using SBM decomposition.
    ///
    /// Prepares: |0⟩^n → Σ_j c_j |j⟩ using n layers of multiplexed Ry rotations.
    ///
    /// Allocates phase gradient and angle ancilla registers internally.
    operation QROMStatePrepare(
        params : QROMStatePrepParams,
        target : Qubit[],
    ) : Unit is Adj + Ctl {
        let bRot = params.rotationBitPrecision;
        use phaseGradient = Qubit[bRot];
        use angleReg = Qubit[bRot];
        within {
            PreparePhaseGradientState(phaseGradient);
        } apply {
            QROMStatePrepareCore(params, target, phaseGradient, angleReg);
        }
    }

    /// Core QROM state preparation.
    ///
    /// Each layer l targets qubit target[l] and uses Reversed(target[0..l-1])
    /// as the LE address register for SelectSwap QROM lookup.
    ///
    /// Qubit ordering: target[0] = MSB (first-allocated = highest bit of state index).
    ///
    /// # Input
    /// ## params
    /// State preparation parameters.
    /// ## target
    /// State register (n qubits), initialized to |0...0⟩.
    /// ## phaseGradient
    /// Phase gradient ancilla (bRot qubits), pre-initialized.
    /// ## angleReg
    /// Angle scratch register (bRot qubits), initialized to |0...0⟩.
    internal operation QROMStatePrepareCore(
        params : QROMStatePrepParams,
        target : Qubit[],
        phaseGradient : Qubit[],
        angleReg : Qubit[],
    ) : Unit is Adj + Ctl {
        let n = params.numStateQubits;
        let bRot = params.rotationBitPrecision;
        let angleTree = ComputeSBMAngles(params.amplitudes, n, bRot);

        // Iterate MSB-first: layer l targets target[l].
        for level in 0..n - 1 {
            let targetQubit = target[level];

            if level == 0 {
                // Root level: single unconditional rotation Ry(2θ_root).
                let angleBits = IntAsBoolArray(angleTree[1], bRot);
                within {
                    ApplyPauliFromBitString(PauliX, true, angleBits, angleReg);
                } apply {
                    RyViaPhaseGradient(targetQubit, angleReg, phaseGradient);
                }
            } else {
                // Address = previously prepared qubits in LE order.
                // target[0] = MSB, so Reversed gives LE for Select.
                let address = Reversed(target[0..level - 1]);

                let startIdx = 1 <<< level;
                let numAngles = 1 <<< level;
                let data = ComputeQROMData(angleTree, startIdx, numAngles, bRot);

                within {
                    SelectSwap(-1, data, address, angleReg);
                } apply {
                    RyViaPhaseGradient(targetQubit, angleReg, phaseGradient);
                }
            }
        }

        // Sign correction: Ry rotations produce |α_j| (positive amplitudes).
        // For negative coefficients, flip the phase via QROM-loaded Z.
        let signData = ComputeSignBits(params.amplitudes, n);
        if Any(row -> row[0], signData) {
            use signBit = Qubit[1];
            // Reversed(target) gives LE address so index j matches coefficient j.
            within {
                SelectSwap(-1, signData, Reversed(target), signBit);
            } apply {
                Z(signBit[0]);
            }
        }
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

    // --- helper functions ---

    /// Compute the SBM rotation angles as quantized integers in a binary heap.
    ///
    /// Builds a probability tree p_i = Σ_{leaves(i)} |α_j|² and converts
    /// each node to a quantized Ry angle:
    ///   θ_i = arccos(√(p_left / p_i))
    ///   quantized_i = Round(2^b · θ / (2π)) mod 2^b
    ///
    /// RyViaPhaseGradient applies Ry(4π·x/2^b), so to get Ry(2θ) we need
    /// x = 2^b · θ / (2π).
    ///
    /// # Output
    /// Int array of length 2^nQubits, indexed as a binary heap (root at index 1).
    internal function ComputeSBMAngles(
        coefficients : Double[],
        nQubits : Int,
        bRot : Int,
    ) : Int[] {
        let nCoeffs = 1 <<< nQubits;

        // Pad coefficients to 2^n.
        mutable coeffs = Repeated(0.0, nCoeffs);
        for i in 0..MinI(Length(coefficients), nCoeffs) - 1 {
            set coeffs w/= i <- coefficients[i];
        }

        // Build probability tree (binary heap, 1-indexed).
        // Leaves at [nCoeffs..2*nCoeffs-1] store |α_j|².
        mutable tree = Repeated(0.0, 2 * nCoeffs);
        for i in 0..nCoeffs - 1 {
            set tree w/= (nCoeffs + i) <- coeffs[i] * coeffs[i];
        }
        // Internal nodes = sum of children.
        for i in (nCoeffs - 1)..-1..1 {
            set tree w/= i <- tree[2 * i] + tree[2 * i + 1];
        }

        // Compute quantized angles.
        let scale = IntAsDouble(1 <<< bRot);
        mutable angles = Repeated(0, nCoeffs);

        for level in 0..nQubits - 1 {
            let startIdx = 1 <<< level;
            let endIdx = (1 <<< (level + 1)) - 1;
            for node in startIdx..endIdx {
                let pTotal = tree[node];
                let pLeft = tree[2 * node];

                mutable angle = 0.0;
                if pTotal > 1e-15 {
                    let cosVal = MinD(1.0, Sqrt(pLeft / pTotal));
                    set angle = ArcCos(cosVal);
                }

                // Quantize: x = 2^b · θ / (2π), so Ry(4π·x/2^b) = Ry(2θ).
                let xInt = Round(scale * angle / (2.0 * PI())) % (1 <<< bRot);
                set angles w/= node <- xInt;
            }
        }

        return angles;
    }

    /// Converts a slice of the angle tree into QROM-compatible Bool[][] format.
    internal function ComputeQROMData(
        angleTree : Int[],
        startIdx : Int,
        count : Int,
        bRot : Int,
    ) : Bool[][] {
        mutable data : Bool[][] = [];
        for i in 0..count - 1 {
            set data += [IntAsBoolArray(angleTree[startIdx + i], bRot)];
        }
        return data;
    }

    /// Computes the sign correction table for real-valued state preparation.
    ///
    /// Returns Bool[2^n][1] where entry j is [true] if coefficients[j] < 0.
    internal function ComputeSignBits(
        coefficients : Double[],
        nQubits : Int,
    ) : Bool[][] {
        let nCoeffs = 1 <<< nQubits;
        mutable signTable : Bool[][] = [];
        for i in 0..nCoeffs - 1 {
            if i < Length(coefficients) {
                set signTable += [[coefficients[i] < 0.0]];
            } else {
                set signTable += [[false]];
            }
        }
        return signTable;
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Test wrappers
    // ═══════════════════════════════════════════════════════════════════════════

    /// Test wrapper: run QROM state preparation and leave state for dump_machine.
    operation RunQROMStatePrep(
        amplitudes : Double[],
        rotationBitPrecision : Int,
        numStateQubits : Int,
    ) : Unit {
        let qs = QIR.Runtime.AllocateQubitArray(numStateQubits);
        let params = new QROMStatePrepParams {
            amplitudes = amplitudes,
            rotationBitPrecision = rotationBitPrecision,
            numStateQubits = numStateQubits,
        };
        QROMStatePrepare(params, qs);
    }
}
