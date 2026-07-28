import Std.StatePreparation.PreparePureStateD;
open Microsoft.Quantum.Intrinsic;
open Microsoft.Quantum.Canon;
open Microsoft.Quantum.Math;
open Microsoft.Quantum.Diagnostics;
open Microsoft.Quantum.Arrays;

// ========================================================================
// 1. Operation for the Power of the Block Encoding (H^k)
// ========================================================================

/// # Summary
/// Applies the k-th power of a Hamiltonian block encoding (H^k).
///
/// # Input
/// - prep: The FOQCS state preparation oracle (P_R or P_L).
/// - select: The FOQCS SELECT oracle applying operations to the system.
/// - power: The integer power 'k' to raise the Hamiltonian to.
/// - ancillaBlocks: An array of distinct qubit arrays. Must contain at least `power` blocks.
/// - system: The target physical system register.
///
/// ## Controlled specialization
/// Only the PREP / PREP† pair is controlled.
/// `select` runs unconditionally because it is sandwiched
/// between PREP/PREP†: when the outer control is |0⟩, the ancilla
/// stays |0…0⟩ so SELECT acts as identity.
operation ApplyFOQCSPower(
    prep : (Qubit[] => Unit is Adj + Ctl),
    select : ((Qubit[], Qubit[]) => Unit is Adj + Ctl),
    power : Int,
    ancillaBlocks : Qubit[][],
    system : Qubit[]
) : Unit is Adj + Ctl {
    body ... {
        for k in 0 .. power - 1 {
            let currentAncillas = ancillaBlocks[k];
            prep(currentAncillas);
        }
        for k in 0 .. power - 1 {
            let currentAncillas = ancillaBlocks[k];
            select(currentAncillas, system);
        }
        for k in 0 .. power - 1 {
            let currentAncillas = ancillaBlocks[k];
            Adjoint prep(currentAncillas);
        }
    }
    adjoint auto;
    controlled (ctls, ...) {
        for k in 0 .. power - 1 {
            let currentAncillas = ancillaBlocks[k];
            Controlled prep(ctls, currentAncillas);
        }
        for k in 0 .. power - 1 {
            let currentAncillas = ancillaBlocks[k];
            select(currentAncillas, system);
        }
        for k in 0 .. power - 1 {
            let currentAncillas = ancillaBlocks[k];
            Controlled Adjoint prep(ctls, currentAncillas);
        }
    }
    controlled adjoint auto;
}

// ========================================================================
// 2. Operation for the Matrix Polynomial (\sum c_k H^k)
// ========================================================================

/// # Summary
/// Implements the d-qubit circuit from Equation 51 of the FOQCS-LCU paper.
/// This circuit acts as the POLY_L (or POLY_R) state preparation oracle,
/// preparing the coefficients of the matrix polynomial using a unary encoding.
///
/// The controlled specialization implements Equation 56 (Theorem 4):
/// only the first gates are controlled by the external control qubits,
/// avoiding exponential Toffoli overhead.
///
/// # Input
/// - qs: The d-qubit register for the unary encoding.
/// - thetas: Array of d rotation angles for the amplitudes.
/// - phis: Array of d phase angles for the complex coefficients.
operation UnaryPolynomialPrep(qs: Qubit[], thetas: Double[], phis: Double[]) : Unit is Adj + Ctl {
    body ... {
        let d = Length(qs);
        Ry(thetas[0], qs[0]);
        R1(phis[0], qs[0]);

        for i in 1 .. d - 1 {
            Controlled Ry([qs[i-1]], (thetas[i], qs[i]));
            R1(phis[i], qs[i]);
        }
    }
    controlled (ctrls, ...) {
        let d = Length(qs);
        Controlled Ry(ctrls, (thetas[0], qs[0]));
        R1(phis[0], qs[0]);

        for i in 1 .. d - 1 {
            Controlled Ry([qs[i-1]], (thetas[i], qs[i]));
            R1(phis[i], qs[i]);
        }
    }
}

// ========================================================================
// 2. Operation for the Matrix Polynomial (\sum c_k H^k)
// ========================================================================

/// # Summary
/// Constructs a matrix polynomial \sum_{k=0}^d c_k H^k of the FOQCS block encoding.
///
/// # Input
/// - prep: The FOQCS state preparation oracle.
/// - select: The FOQCS SELECT oracle.
/// - degree: The maximum polynomial degree 'd'.
/// - coefficients: An array of `ComplexPolar` representing the weights c_k.
/// - polyAncillas: The outer polynomial LCU control register.
/// - ancillaBlocks: Distinct ancilla registers for the Hamiltonian applications.
/// - system: The target physical system register.
operation ApplyFOQCSPolynomial(
    prep : (Qubit[] => Unit is Adj + Ctl),
    select : ((Qubit[], Qubit[]) => Unit is Adj + Ctl),
    degree : Int,
    thetas : Double[],
    phis : Double[],
    polyAncillas : Qubit[],
    ancillaBlocks : Qubit[][],
    system : Qubit[]
) : Unit is Adj + Ctl {
    body ... {
        if (degree != Length(thetas) or degree != Length(phis) or degree != Length(polyAncillas) or degree != Length(ancillaBlocks)) {
            fail "Input arrays must all have length equal to the polynomial degree.";
        }
        // Apply preparation, then manually embed the complex phase difference
        // in the matrix coefficient directly inside the within block?!
        UnaryPolynomialPrep(polyAncillas, thetas, phis);

        for k in 0 .. degree - 1 {
            let singlePower = ApplyFOQCSPower(prep, select, 1, [ancillaBlocks[k]], _);
            Controlled singlePower([polyAncillas[k]], system);
        }

        let zeroPhis =[0.0, size = Length(phis)];
        Adjoint UnaryPolynomialPrep(polyAncillas, thetas, zeroPhis);


    }
    adjoint auto;
    controlled (ctls, ...) {
        Controlled UnaryPolynomialPrep(ctls, (polyAncillas, thetas, phis));

        for k in 0 .. degree - 1 {
            let singlePower = ApplyFOQCSPower(prep, select, 1, [ancillaBlocks[k]], _);
            Controlled singlePower([polyAncillas[k]], system);
        }

        let zeroPhis =[0.0, size = Length(phis)];
        Controlled Adjoint UnaryPolynomialPrep(ctls, (polyAncillas, thetas, zeroPhis));
    }
    controlled adjoint auto;
}
