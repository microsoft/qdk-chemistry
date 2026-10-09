// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

/// Arithmetic on quantum registers with classical constants.
namespace QDKChemistry.Utils.Arithmetic {

    import Std.Convert.IntAsBoolArray;

    /// Applies |x⟩ → |x + c mod 2^n⟩ to the little-endian `target` with the ripple-carry
    /// adder of Sanders et al. (PRX Quantum 1, 020312, 2020), Fig. 18.
    ///
    /// Costs n − 2 AND gates (4(n − 2) T gates) and n − 1 clean ancillas; the AND gates are
    /// uncomputed by measurement.
    operation AddConstant(c : Int, target : Qubit[]) : Unit {
        let n = Length(target);
        let cMod = ((c % (1 <<< n)) + (1 <<< n)) % (1 <<< n);
        let cBits = IntAsBoolArray(cMod, n); // LE: cBits[0] = LSB

        if n == 1 {
            if cBits[0] {
                X(target[0]);
            }
        } elif n == 2 {
            // 2-bit: no Toffolis needed. Use Clifford-only circuit.
            if cBits[1] {
                if cBits[0] {
                    // +3 = -1 mod 4: decrement
                    X(target[0]);
                    CNOT(target[0], target[1]);
                } else {
                    // +2: flip MSB
                    X(target[1]);
                }
            } else {
                if cBits[0] {
                    // +1: increment
                    CNOT(target[0], target[1]);
                    X(target[0]);
                }
                // +0: identity
            }
        } else {
            use ancillas = Qubit[n - 1];

            // --- Forward pass: compute carry bits ---
            if cBits[0] {
                CNOT(target[0], ancillas[0]);
            }

            for i in 1..n - 2 {
                let j = i - 1;
                CNOT(ancillas[j], target[i]);
                if cBits[i] {
                    X(ancillas[j]);
                }
                AND(ancillas[j], target[i], ancillas[i]);
                if cBits[i] {
                    X(ancillas[j]);
                }
                CNOT(ancillas[j], ancillas[i]);
            }

            // --- MSB: XOR final carry ---
            CNOT(ancillas[n - 2], target[n - 1]);

            // --- Reverse pass: uncompute ancillas ---
            for i in n - 2..-1..1 {
                let j = i - 1;
                CNOT(ancillas[j], ancillas[i]);
                if cBits[i] {
                    X(ancillas[j]);
                }
                Adjoint AND(ancillas[j], target[i], ancillas[i]);
                if cBits[i] {
                    X(ancillas[j]);
                }
            }

            if cBits[0] {
                CNOT(target[0], ancillas[0]);
            }

            // --- Final XOR: add constant bits ---
            for i in 0..n - 1 {
                if cBits[i] {
                    X(target[i]);
                }
            }
        }
    }
}
