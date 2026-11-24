// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.

/// MPS state preparation one site at a time, with dense site isometries (`MPSSequential`),
/// block-sparse ones exploiting symmetry (`MPSSparse`) or Householder ones (`MPSHouseholder`)
/// (Berry et al., PRX Quantum 6, 020327; Rupprecht & Wölk, arXiv:2605.28489), and charge- and
/// spin-adapted MPS from reduced tensors (`MPSSpinAdapted`, Hennefarth et al., arXiv:2611.19728).
namespace QDKChemistry.Utils.MPSStatePrep {

    import Std.Arithmetic.ApplyIfGreaterOrEqualL;
    import Std.Arithmetic.ApplyIfLessOrEqualL;
    import Std.Arithmetic.IncByI;
    import Std.Arrays.All;
    import Std.Arrays.Enumerated;
    import Std.Arrays.MappedOverRange;
    import Std.Arrays.Partitioned;
    import Std.Arrays.Sorted;
    import Std.Arrays.Subarray;
    import Std.Canon.ApplyToEachA;
    import Std.Convert.IntAsBigInt;
    import Std.Convert.IntAsDouble;
    import Std.Diagnostics.Fact;
    import Std.Math.ArcCos;
    import Std.Math.BitSizeI;
    import Std.Math.Sqrt;
    import Std.Math.TrailingZeroCountI;
    import Std.ResourceEstimation.BeginEstimateCaching;
    import Std.ResourceEstimation.EndEstimateCaching;
    import QDKChemistry.Utils.IsometrySynthesis.ApplyDenseIsometry;
    import QDKChemistry.Utils.IsometrySynthesis.ApplyHouseholderIsometry;
    import QDKChemistry.Utils.IsometrySynthesis.ApplySparseIsometry;
    import QDKChemistry.Utils.IsometrySynthesis.ColumnLoadCacheKey;
    import QDKChemistry.Utils.IsometrySynthesis.DenseIsometrySynthesis;
    import QDKChemistry.Utils.IsometrySynthesis.HouseholderIsometrySynthesis;
    import QDKChemistry.Utils.IsometrySynthesis.PrepareControlledColumn;
    import QDKChemistry.Utils.IsometrySynthesis.SparseIsometrySynthesis;
    import QDKChemistry.Utils.PhaseGradient.PreparePhaseGradientState;
    import QDKChemistry.Utils.QROMStatePrep.QROMStatePrepParams;
    import QDKChemistry.Utils.QROMStatePrep.QROMStatePrepare;
    import QDKChemistry.Utils.UnitarySynthesis.ApplyControlledMultiplexedRy;
    import QDKChemistry.Utils.UnitarySynthesis.QuantizeTrimmedRyAngles;

    /// Returns the qubit indices of `orbital` in the blocked Jordan-Wigner layout.
    internal function MPSSiteQubitIndices(numQubits : Int, numOrbitals : Int, orbital : Int) : Int[] {
        MappedOverRange(channel -> channel * numOrbitals + orbital, 0..numQubits / numOrbitals - 1)
    }

    /// Returns the qubits of `orbital`; see `MPSSiteQubitIndices`.
    internal function MPSSiteQubits(state : Qubit[], numOrbitals : Int, orbital : Int) : Qubit[] {
        Subarray(MPSSiteQubitIndices(Length(state), numOrbitals, orbital), state)
    }

    /// Converts MPS chain-order fermionic signs to the blocked Jordan-Wigner convention.
    internal operation ApplyMPSFermionicOrderSigns(siteToOrbitalOrder : Int[], state : Qubit[]) : Unit is Adj {
        let numSites = Length(siteToOrbitalOrder);
        let numQubits = Length(state);
        // Element-wise checks, because QIR generation does not support whole-array comparison.
        if numQubits == 2 * numSites and numSites >= 6 and All((index, orbital) -> index == orbital, Enumerated(siteToOrbitalOrder)) {
            within {
                // α_j holds the parity of α_j, ..., α_{N-1}.
                for j in numSites - 1..-1..2 {
                    CNOT(state[j], state[j - 1]);
                }
            } apply {
                for j in 1..numSites - 1 {
                    CZ(state[numSites + j - 1], state[j]);
                }
            }
        } else {
            for first in 0..numSites - 1 {
                let firstModes = MPSSiteQubitIndices(numQubits, numSites, siteToOrbitalOrder[first]);
                for second in first + 1..numSites - 1 {
                    // The modes of an earlier site are created before those of a later site.
                    for firstMode in firstModes {
                        for secondMode in MPSSiteQubitIndices(numQubits, numSites, siteToOrbitalOrder[second]) {
                            if firstMode > secondMode {
                                CZ(state[firstMode], state[secondMode]);
                            }
                        }
                    }
                }
            }
        }
    }

    /// Prepares an MPS one site at a time (Berry et al., PRX Quantum 6, 020327,
    /// https://doi.org/10.1103/PRXQuantum.6.020327).
    internal operation PrepareMPS<'Site>(
        initialStateVec : Double[],
        numSites : Int,
        siteToOrbitalOrder : Int[],
        siteDecompositions : 'Site[],
        applySite : ('Site, Qubit[], Qubit[], Qubit[], Qubit[]) => Unit is Adj,
        state : Qubit[],
        ancilla : Qubit[],
        phaseGradient : Qubit[]
    ) : Unit is Adj {
        Fact(Length(siteDecompositions) == numSites - 1, "MPS preparation needs one decomposition per site after the first.");
        Fact(
            Length(state) == numSites or Length(state) == 2 * numSites,
            "The state register must hold one or two qubits per MPS site."
        );
        Fact(
            Length(siteToOrbitalOrder) == numSites and All((index, orbital) -> index == orbital, Enumerated(Sorted((a, b) -> a <= b, siteToOrbitalOrder))),
            "siteToOrbitalOrder must be a permutation of the MPS sites."
        );
        let qubitsPerSite = Length(state) / numSites;
        let initialBondBits = BitSizeI(Length(initialStateVec) / (1 <<< qubitsPerSite) - 1);
        Fact(
            Length(initialStateVec) == 1 <<< (qubitsPerSite + initialBondBits) and initialBondBits <= Length(ancilla),
            "The initial state must cover the first site and a power-of-two part of the bond register."
        );
        let rotationBitPrecision = Length(phaseGradient);

        let initReg = ancilla[...initialBondBits - 1] + MPSSiteQubits(state, numSites, siteToOrbitalOrder[0]);
        QROMStatePrepare(
            new QROMStatePrepParams {
                amplitudes = initialStateVec,
                rotationBitPrecision = rotationBitPrecision,
                numStateQubits = Length(initReg),
            },
            initReg,
            phaseGradient
        );

        {
            use angleReg = Qubit[rotationBitPrecision];
            for siteIdx in 0..numSites - 2 {
                let newSite = MPSSiteQubits(state, numSites, siteToOrbitalOrder[siteIdx + 1]);
                applySite(siteDecompositions[siteIdx], ancilla, newSite, phaseGradient, angleReg);
            }
        }
        ApplyMPSFermionicOrderSigns(siteToOrbitalOrder, state);
    }

    /// Applies `prepare(state, ancilla, phaseGradient)` to the first `numStateQubits` qubits of
    /// `state`, leaving any later ones untouched, with phase gradient and bond registers allocated
    /// here; see `PrepareMPSOnRegister`.
    internal operation PrepareMPSOnState(
        numStateQubits : Int,
        numAncillaQubits : Int,
        rotationBitPrecision : Int,
        prepare : (Qubit[], Qubit[], Qubit[]) => Unit is Adj,
        state : Qubit[]
    ) : Unit is Adj {
        use phaseGradient = Qubit[rotationBitPrecision];
        within {
            PreparePhaseGradientState(phaseGradient);
        } apply {
            PrepareMPSOnRegister(numStateQubits, numAncillaQubits, rotationBitPrecision, prepare, state[...numStateQubits - 1] + phaseGradient);
        }
    }

    /// Applies `prepare(state, ancilla, phaseGradient)` to `qs` = `[state | phaseGradient]`
    /// with a bond register `ancilla` allocated here.
    ///
    /// The last `rotationBitPrecision` qubits of `qs` are a caller-owned register prepared by
    /// `PreparePhaseGradientState`. Quantized rotations return `ancilla` to |0⟩ only up to
    /// rotation quantization error, so
    /// it is reset before release in both directions. The operation and its adjoint are
    /// therefore inverse only up to that error, which is of the same order as the preparation
    /// error in `state`.
    internal operation PrepareMPSOnRegister(
        numStateQubits : Int,
        numAncillaQubits : Int,
        rotationBitPrecision : Int,
        prepare : (Qubit[], Qubit[], Qubit[]) => Unit is Adj,
        qs : Qubit[]
    ) : Unit is Adj {
        body (...) {
            Fact(
                Length(qs) == numStateQubits + rotationBitPrecision,
                "The register must hold the MPS state followed by the phase gradient."
            );
            use ancilla = Qubit[numAncillaQubits];
            prepare(qs[...numStateQubits - 1], ancilla, qs[numStateQubits...]);
            ResetAll(ancilla);
        }
        adjoint (...) {
            Fact(
                Length(qs) == numStateQubits + rotationBitPrecision,
                "The register must hold the MPS state followed by the phase gradient."
            );
            use ancilla = Qubit[numAncillaQubits];
            Adjoint prepare(qs[...numStateQubits - 1], ancilla, qs[numStateQubits...]);
            ResetAll(ancilla);
        }
    }

    /// Parameters of a composable MPS sequential state preparation with dense site isometries.
    struct MPSSequentialParams {
        initialStateVec : Double[],
        numSites : Int,
        numQubitsPerSite : Int,
        siteToOrbitalOrder : Int[],
        rotationBitPrecision : Int,
        numAncillaQubits : Int,
        siteDecompositions : DenseIsometrySynthesis[],
    }

    /// `PrepareMPS` with the site isometries applied by `ApplyDenseIsometry`.
    operation MPSSequential(params : MPSSequentialParams, state : Qubit[], ancilla : Qubit[], phaseGradient : Qubit[]) : Unit is Adj {
        PrepareMPS(params.initialStateVec, params.numSites, params.siteToOrbitalOrder, params.siteDecompositions, ApplyDenseIsometry, state, ancilla, phaseGradient);
    }

    /// Returns `MPSSequential` on the state register, with its own phase gradient; see `PrepareMPSOnState`.
    function MakeMPSSequentialOp(params : MPSSequentialParams) : Qubit[] => Unit is Adj {
        PrepareMPSOnState(params.numQubitsPerSite * params.numSites, params.numAncillaQubits, params.rotationBitPrecision, MPSSequential(params, _, _, _), _)
    }

    /// Returns `MPSSequential` on `[state | phaseGradient]`; see `PrepareMPSOnRegister`.
    function MakeMPSSequentialOpWithPhaseGradient(params : MPSSequentialParams) : Qubit[] => Unit is Adj {
        PrepareMPSOnRegister(params.numQubitsPerSite * params.numSites, params.numAncillaQubits, params.rotationBitPrecision, MPSSequential(params, _, _, _), _)
    }

    /// Circuit entry point for `MPSSequential`, leaving the state on the allocated qubits.
    operation MakeMPSSequentialCircuit(params : MPSSequentialParams) : Unit {
        use state = Qubit[params.numQubitsPerSite * params.numSites];
        MakeMPSSequentialOp(params)(state);
    }

    /// Parameters of a composable MPS state preparation with block-sparse site isometries.
    struct MPSSparseParams {
        initialStateVec : Double[],
        numSites : Int,
        numQubitsPerSite : Int,
        siteToOrbitalOrder : Int[],
        rotationBitPrecision : Int,
        numAncillaQubits : Int,
        siteDecompositions : SparseIsometrySynthesis[],
    }

    /// `PrepareMPS` with the site isometries applied by `ApplySparseIsometry`.
    operation MPSSparse(params : MPSSparseParams, state : Qubit[], ancilla : Qubit[], phaseGradient : Qubit[]) : Unit is Adj {
        PrepareMPS(params.initialStateVec, params.numSites, params.siteToOrbitalOrder, params.siteDecompositions, ApplySparseIsometry, state, ancilla, phaseGradient);
    }

    /// Returns `MPSSparse` on the state register, with its own phase gradient; see `PrepareMPSOnState`.
    function MakeMPSSparseOp(params : MPSSparseParams) : Qubit[] => Unit is Adj {
        PrepareMPSOnState(params.numQubitsPerSite * params.numSites, params.numAncillaQubits, params.rotationBitPrecision, MPSSparse(params, _, _, _), _)
    }

    /// Returns `MPSSparse` on `[state | phaseGradient]`; see `PrepareMPSOnRegister`.
    function MakeMPSSparseOpWithPhaseGradient(params : MPSSparseParams) : Qubit[] => Unit is Adj {
        PrepareMPSOnRegister(params.numQubitsPerSite * params.numSites, params.numAncillaQubits, params.rotationBitPrecision, MPSSparse(params, _, _, _), _)
    }

    /// Circuit entry point for `MPSSparse`, leaving the state on the allocated qubits.
    operation MakeMPSSparseCircuit(params : MPSSparseParams) : Unit {
        use state = Qubit[params.numQubitsPerSite * params.numSites];
        MakeMPSSparseOp(params)(state);
    }

    // Householder site isometries (Low, Kliuchnikov & Schaeffer, arXiv:1812.00954).

    /// Parameters of a composable MPS state preparation with Householder site isometries.
    struct MPSHouseholderParams {
        initialStateVec : Double[],
        numSites : Int,
        numQubitsPerSite : Int,
        siteToOrbitalOrder : Int[],
        rotationBitPrecision : Int,
        numAncillaQubits : Int,
        siteDecompositions : HouseholderIsometrySynthesis[],
    }

    /// `PrepareMPS` with the site isometries applied by `ApplyHouseholderIsometry`.
    operation MPSHouseholder(params : MPSHouseholderParams, state : Qubit[], ancilla : Qubit[], phaseGradient : Qubit[]) : Unit is Adj {
        PrepareMPS(params.initialStateVec, params.numSites, params.siteToOrbitalOrder, params.siteDecompositions, ApplyHouseholderIsometry, state, ancilla, phaseGradient);
    }

    /// Returns `MPSHouseholder` on the state register, with its own phase gradient; see `PrepareMPSOnState`.
    function MakeMPSHouseholderOp(params : MPSHouseholderParams) : Qubit[] => Unit is Adj {
        PrepareMPSOnState(params.numQubitsPerSite * params.numSites, params.numAncillaQubits, params.rotationBitPrecision, MPSHouseholder(params, _, _, _), _)
    }

    /// Returns `MPSHouseholder` on `[state | phaseGradient]`; see `PrepareMPSOnRegister`.
    function MakeMPSHouseholderOpWithPhaseGradient(params : MPSHouseholderParams) : Qubit[] => Unit is Adj {
        PrepareMPSOnRegister(params.numQubitsPerSite * params.numSites, params.numAncillaQubits, params.rotationBitPrecision, MPSHouseholder(params, _, _, _), _)
    }

    /// Circuit entry point for `MPSHouseholder`, leaving the state on the allocated qubits.
    operation MakeMPSHouseholderCircuit(params : MPSHouseholderParams) : Unit {
        use state = Qubit[params.numQubitsPerSite * params.numSites];
        MakeMPSHouseholderOp(params)(state);
    }

    // Charge- and spin-adapted (U(1) x SU(2)) MPS from reduced tensors (Hennefarth et al., arXiv:2611.19728).
    // Sites are loaded from the last to the first (Fig. 1), each as the Householder product of reflections
    // over the columns of its left-canonical tensor (Sec. 3.1). The bond register holds (alpha, n, 2S, 2m);
    // alpha doubles as the contiguizer beta of App. A.2.

    /// Reduced block B_s of one right sector s = (n_R, 2S_R) in the channel form of Eq. (C.3).
    struct SpinAdaptedSector {
        /// Particle number n_R of the right sector.
        numParticles : Int,
        /// Twice the spin S_R of the right sector.
        twoSpin : Int,
        /// Left particle number n_L of each fusion channel, ordered by ascending multiplicity (App. A.2).
        caseNumParticles : Int[],
        /// Left 2S_L of each channel.
        caseTwoSpin : Int[],
        /// Channel code Γ = k + 2l of the physical qubits (k: alpha, l: beta), Def. C.1.
        caseCodes : Int[],
        /// Multiplicity c of each channel, the length of its part of the contiguizer.
        caseMultiplicities : Int[],
        /// SBM angles 2θ of each column, indexed by column, level and address prefix.
        columnAngles : Double[][][],
        /// Sign bit of each contiguized amplitude of each column, empty for a sign-free column.
        columnSigns : Bool[][],
    }

    /// Right sectors of one MPS site.
    struct SpinAdaptedSite {
        sectors : SpinAdaptedSector[],
    }

    /// Parameters of the spin-adapted MPS preparation.
    struct MPSSpinAdaptedParams {
        numSites : Int,
        siteToOrbitalOrder : Int[],
        rotationBitPrecision : Int,
        maxSwapBits : Int,
        /// Widths of the alpha (multiplicity and contiguizer), n, 2S and 2m registers.
        registerWidths : Int[],
        /// Target (n, 2S, 2M_S) of the right boundary bond.
        target : Int[],
        sites : SpinAdaptedSite[],
    }

    /// Ry angles -2θ of the Clebsch-Gordan uncoupling G (Eqs. C.6-C.7) addressed by the
    /// little-endian 2m (two's complement, low bits) and 2S_L (high bits) registers.
    function ClebschGordanAngles(spinBits : Int, projectionBits : Int) : Double[] {
        mutable angles = [];
        for address in 0..(1 <<< (spinBits + projectionBits)) - 1 {
            // The register stores 2S_L, so 𝔰 = 2S_L + 1 of Eq. (C.7) is its value plus one.
            let s = (address >>> projectionBits) + 1;
            let raw = address &&& ((1 <<< projectionBits) - 1);
            let m = raw >= 1 <<< (projectionBits - 1) ? raw - (1 <<< projectionBits) | raw;
            let valid = m <= s and -m <= s and (s + m) % 2 == 0;
            set angles += [valid ? -2.0 * ArcCos(Sqrt(IntAsDouble(s + m) / IntAsDouble(2 * s))) | 0.0];
        }
        angles
    }

    /// Applies M·G of Def. C.2: rotates the coupled channel codes |10⟩, |01⟩ into the spin
    /// projections with Clebsch-Gordan amplitudes (Prop. C.3), then sets 2m ← 2m - k + l.
    operation ApplyClebschGordanStage(
        maxSwapBits : Int,
        spinRegister : Qubit[],
        projectionRegister : Qubit[],
        physical : Qubit[],
        phaseGradient : Qubit[],
        angleReg : Qubit[]
    ) : Unit is Adj {
        let table = QuantizeTrimmedRyAngles(ClebschGordanAngles(Length(spinRegister), Length(projectionRegister)), Length(angleReg));
        within {
            CNOT(physical[0], physical[1]);
        } apply {
            ApplyControlledMultiplexedRy(table, maxSwapBits, projectionRegister + spinRegister, [physical[1]], physical[0], phaseGradient, angleReg);
        }
        let wrap = 1 <<< Length(projectionRegister);
        Controlled IncByI([physical[0]], (wrap - 1, projectionRegister));
        Controlled IncByI([physical[1]], (1, projectionRegister));
    }

    /// One decoding stage of App. A.2: moves beta past channel t - 1 of size c by (case t-1, beta)
    /// → (case t, beta - c) when beta ≥ c.
    ///
    /// The paper leaves the flag uncomputation open. Rotation quantization leaves small amplitudes on
    /// arbitrary register values, so the stage is unitary on every state: it swaps
    /// A = {case t-1, beta ≥ c} with B = {case t, beta < 2^w - c} and fixes everything else. CNOTs make
    /// the two cases differ only in one distinguishing bit b (App. A.1), so one register match finds
    /// the pair. Complementing beta when b = 1 maps both A and B to beta' ≥ c, where the swap is
    /// beta' → c - 1 - beta' with b flipped; that keeps the flag, so it is uncomputed by its own
    /// adjoint. Column states never lie in B before the forward stage, nor in A before its adjoint.
    operation ApplyDecodeStage(fromValue : Int, toValue : Int, size : Int, reflect : Qubit, registers : Qubit[], alpha : Qubit[]) : Unit is Adj {
        let delta = fromValue ^^^ toValue;
        let d = TrailingZeroCountI(delta);
        let fromBit = (fromValue >>> d) &&& 1 == 1;
        // Case t - 1 after the CNOTs from bit d onto the other differing bits, without bit d.
        let transformed = fromValue ^^^ (fromBit ? delta | 0);
        let pattern = (transformed &&& ((1 <<< d) - 1)) + ((transformed >>> (d + 1)) <<< d);
        use (select, flag) = (Qubit(), Qubit());
        within {
            for e in 0..Length(registers) - 1 {
                if e != d and (delta >>> e) &&& 1 == 1 {
                    CNOT(registers[d], registers[e]);
                }
            }
            ApplyControlledOnInt(2 * pattern + 1, X, [reflect] + registers[...d - 1] + registers[d + 1...], select);
            if fromBit {
                X(registers[d]);
            }
            ApplyToEachA(CNOT(registers[d], _), alpha);
            // Std comparators test the constant against the register: size <= beta'.
            ApplyIfLessOrEqualL(Controlled X([select], _), IntAsBigInt(size), alpha, flag);
        } apply {
            CNOT(flag, registers[d]);
            ApplyToEachA(CNOT(flag, _), alpha);
            Controlled IncByI([flag], (size, alpha));
        }
    }
    /// `LoadReducedColumnUncached`, traced once per cost key by the resource estimator.
    operation LoadReducedColumn(
        sector : SpinAdaptedSector,
        column : Int,
        maxSwapBits : Int,
        alpha : Qubit[],
        numberRegister : Qubit[],
        spinRegister : Qubit[],
        physical : Qubit[],
        reflect : Qubit,
        phaseGradient : Qubit[],
        angleReg : Qubit[]
    ) : Unit is Adj {
        body (...) {
            if BeginEstimateCaching(ReducedColumnCacheName(sector, column, maxSwapBits, [Length(angleReg), Length(alpha), Length(numberRegister), Length(spinRegister)]), 0) {
                LoadReducedColumnUncached(sector, column, maxSwapBits, alpha, numberRegister, spinRegister, physical, reflect, phaseGradient, angleReg);
                EndEstimateCaching();
            }
        }
        adjoint (...) {
            if BeginEstimateCaching(ReducedColumnCacheName(sector, column, maxSwapBits, [Length(angleReg), Length(alpha), Length(numberRegister), Length(spinRegister)]), 1) {
                Adjoint LoadReducedColumnUncached(sector, column, maxSwapBits, alpha, numberRegister, spinRegister, physical, reflect, phaseGradient, angleReg);
                EndEstimateCaching();
            }
        }
    }

    /// Cache key of a reduced column load: the column load key and the channel multiplicities, which are the decoding constants.
    function ReducedColumnCacheName(sector : SpinAdaptedSector, column : Int, maxSwapBits : Int, widths : Int[]) : String {
        $"SpinAdaptedColumn_{ColumnLoadCacheKey(sector.columnAngles[column], sector.columnSigns[column], maxSwapBits, widths)}_c{sector.caseMultiplicities}"
    }

    /// The controlled column preparation U_{α_R, n_R, S_R} of Eq. (26) in its contiguized form
    /// (App. A.2): writes the first channel's (n_L, 2S_L, Γ), prepares the column over the
    /// contiguizer beta, then decodes beta channel by channel.
    operation LoadReducedColumnUncached(
        sector : SpinAdaptedSector,
        column : Int,
        maxSwapBits : Int,
        alpha : Qubit[],
        numberRegister : Qubit[],
        spinRegister : Qubit[],
        physical : Qubit[],
        reflect : Qubit,
        phaseGradient : Qubit[],
        angleReg : Qubit[]
    ) : Unit is Adj {
        let numberBits = Length(numberRegister);
        let spinBits = Length(spinRegister);
        let caseValue = t -> sector.caseNumParticles[t] + (sector.caseTwoSpin[t] <<< numberBits) + (sector.caseCodes[t] <<< (numberBits + spinBits));
        let registers = numberRegister + spinRegister + physical;
        let rightValue = sector.numParticles + (sector.twoSpin <<< numberBits);
        Controlled ApplyXorInPlace([reflect], (rightValue ^^^ caseValue(0), registers));

        let levels = sector.columnAngles[column];
        PrepareControlledColumn(levels, sector.columnSigns[column], maxSwapBits, reflect, alpha[...Length(levels) - 1], phaseGradient, angleReg);

        // Matching (n, 2S, phys) as well as beta keeps the loaded columns of other sectors fixed, since
        // a channel's (n_L, 2S_L, Γ) determines its right sector (Eq. C.4).
        for t in 1..Length(sector.caseCodes) - 1 {
            ApplyDecodeStage(caseValue(t - 1), caseValue(t), sector.caseMultiplicities[t - 1], reflect, registers, alpha);
        }
    }

    /// Reflects about the target states |E_{α_R, n_R, S_R, m_R}⟩ (Eq. 24) of every valid m_R,
    /// after W† has mapped them to |0⟩ on the reflection ancilla and alpha (Fig. 3).
    operation ReflectAboutColumnTarget(
        sector : SpinAdaptedSector,
        alpha : Qubit[],
        numberRegister : Qubit[],
        spinRegister : Qubit[],
        projectionRegister : Qubit[],
        physical : Qubit[],
        reflect : Qubit
    ) : Unit is Adj {
        use valid = Qubit();
        // The comparator of Fig. 3: |2m_R| ≤ 2S_R with the parity of 2S_R, after shifting by 2S_R.
        within {
            IncByI(sector.twoSpin, projectionRegister);
            ApplyIfGreaterOrEqualL(X, IntAsBigInt(2 * sector.twoSpin), projectionRegister, valid);
        } apply {
            let alphaBits = Length(alpha);
            let pattern = (sector.numParticles <<< (1 + alphaBits)) + (sector.twoSpin <<< (1 + alphaBits + Length(numberRegister)));
            ApplyControlledOnInt(pattern, Z, [reflect] + alpha + numberRegister + spinRegister + physical + [projectionRegister[0]], valid);
        }
    }

    /// Applies 1 - 2Σ_{m_R}|D⟩⟨D| for one column (Eq. 27): conjugating the target reflection by
    /// W (Eq. 23), which writes α_R on the |0⟩ branch, and by the column preparation.
    operation ApplyColumnReflection(
        sector : SpinAdaptedSector,
        column : Int,
        maxSwapBits : Int,
        alpha : Qubit[],
        numberRegister : Qubit[],
        spinRegister : Qubit[],
        projectionRegister : Qubit[],
        physical : Qubit[],
        reflect : Qubit,
        phaseGradient : Qubit[],
        angleReg : Qubit[]
    ) : Unit is Adj {
        within {
            Adjoint LoadReducedColumn(sector, column, maxSwapBits, alpha, numberRegister, spinRegister, physical, reflect, phaseGradient, angleReg);
            Adjoint ApplyControlledOnInt(0, ApplyXorInPlace(column, _), [reflect], alpha);
            Adjoint Z(reflect);
            Adjoint H(reflect);
        } apply {
            ReflectAboutColumnTarget(sector, alpha, numberRegister, spinRegister, projectionRegister, physical, reflect);
        }
    }

    /// Applies the site isometry |α_R, n_R, S_R, m_R⟩|00⟩ → |T^[i] column⟩ (Eq. 19): the
    /// Householder reflections over all columns, M·G, which turns them into the full columns
    /// (Thm. C.4), and X on the reflection ancilla.
    ///
    /// The paper conjugates the reflections of each S_R by its own Clebsch-Gordan stage (Eq. 34).
    /// Here the data-independent M·G of App. C acts once per site: on the input of a site the
    /// physical qubits are |00⟩, where M·G is the identity, so its conjugating adjoint is omitted.
    operation ApplySpinAdaptedSite(
        site : SpinAdaptedSite,
        maxSwapBits : Int,
        alpha : Qubit[],
        numberRegister : Qubit[],
        spinRegister : Qubit[],
        projectionRegister : Qubit[],
        physical : Qubit[],
        reflect : Qubit,
        phaseGradient : Qubit[],
        angleReg : Qubit[]
    ) : Unit is Adj {
        for sector in site.sectors {
            for column in 0..Length(sector.columnAngles) - 1 {
                ApplyColumnReflection(sector, column, maxSwapBits, alpha, numberRegister, spinRegister, projectionRegister, physical, reflect, phaseGradient, angleReg);
            }
        }
        ApplyClebschGordanStage(maxSwapBits, spinRegister, projectionRegister, physical, phaseGradient, angleReg);
        X(reflect);
    }

    /// Number of bond-register qubits: alpha, n, 2S, 2m and the reflection ancilla.
    function SpinAdaptedBondQubits(registerWidths : Int[]) : Int {
        registerWidths[0] + registerWidths[1] + registerWidths[2] + registerWidths[3] + 1
    }

    /// Prepares the spin-adapted MPS on `state` (layout of `MPSSiteQubits`), starting from the
    /// bond register |0, n, 2S, 2M_S⟩ of the target and ending on the vacuum (Cor. C.5).
    operation MPSSpinAdapted(params : MPSSpinAdaptedParams, state : Qubit[], ancilla : Qubit[], phaseGradient : Qubit[]) : Unit is Adj {
        Fact(Length(params.sites) == params.numSites, "Spin-adapted MPS preparation needs one site per orbital.");
        Fact(Length(state) == 2 * params.numSites, "The state register must hold two qubits per MPS site.");
        let widths = params.registerWidths;
        let registers = Partitioned(widths + [1], ancilla);
        let (alpha, numberRegister, spinRegister, projectionRegister, reflect) = (registers[0], registers[1], registers[2], registers[3], registers[4][0]);
        ApplyXorInPlace(params.target[0], numberRegister);
        ApplyXorInPlace(params.target[1], spinRegister);
        ApplyXorInPlace((params.target[2] + (1 <<< widths[3])) % (1 <<< widths[3]), projectionRegister);
        {
            use angleReg = Qubit[params.rotationBitPrecision];
            for siteIdx in params.numSites - 1..-1..0 {
                let physical = MPSSiteQubits(state, params.numSites, params.siteToOrbitalOrder[siteIdx]);
                ApplySpinAdaptedSite(params.sites[siteIdx], params.maxSwapBits, alpha, numberRegister, spinRegister, projectionRegister, physical, reflect, phaseGradient, angleReg);
            }
        }
        ApplyMPSFermionicOrderSigns(params.siteToOrbitalOrder, state);
    }

    /// Returns `MPSSpinAdapted` on the state register, with its own phase gradient; see `PrepareMPSOnState`.
    function MakeMPSSpinAdaptedOp(params : MPSSpinAdaptedParams) : Qubit[] => Unit is Adj {
        PrepareMPSOnState(2 * params.numSites, SpinAdaptedBondQubits(params.registerWidths), params.rotationBitPrecision, MPSSpinAdapted(params, _, _, _), _)
    }

    /// Returns `MPSSpinAdapted` on `[state | phaseGradient]`; see `PrepareMPSOnRegister`.
    function MakeMPSSpinAdaptedOpWithPhaseGradient(params : MPSSpinAdaptedParams) : Qubit[] => Unit is Adj {
        PrepareMPSOnRegister(2 * params.numSites, SpinAdaptedBondQubits(params.registerWidths), params.rotationBitPrecision, MPSSpinAdapted(params, _, _, _), _)
    }

    /// Circuit entry point for `MPSSpinAdapted`, leaving the state on the allocated qubits.
    operation MakeMPSSpinAdaptedCircuit(params : MPSSpinAdaptedParams) : Unit {
        use state = Qubit[2 * params.numSites];
        MakeMPSSpinAdaptedOp(params)(state);
    }
}
