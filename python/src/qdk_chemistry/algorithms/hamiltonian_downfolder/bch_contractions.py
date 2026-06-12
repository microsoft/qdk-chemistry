"""BCH commutator contractions for DUCC Hamiltonian downfolding.

Auto-generated from ExaChem's ducc-t_ccsd.hpp by generate_bch_contractions.py.
DO NOT EDIT MANUALLY — re-run the generator script instead.

All tensors are in physicist antisymmetrized spin-orbital notation:
  v2[p,q,r,s] = <pq||rs> (antisymmetrized two-electron integrals)
  f1[p,q] = Fock matrix elements
  t1[e,m] = singles amplitudes (virt, occ) with T_int zeroed
  t2[e,f,m,n] = doubles amplitudes (virt, virt, occ, occ) with T_int zeroed

Index ranges (slice objects):
  O  = all occupied spin-orbitals
  V  = all virtual spin-orbitals
  OI = active (internal) occupied spin-orbitals
  VI = active (internal) virtual spin-orbitals

Each function updates the 9 active-space tensor blocks in-place
and returns the fully-contracted scalar contribution to the energy shift.
"""

from __future__ import annotations

import numpy as np

def f_1(
    ft, vt, idx,
    f1, t1, t2,
):
    """BCH commutator F_1: Fock operator f1.

    Auto-generated from ExaChem's ducc-t_ccsd.hpp.
    All tensors use full spin-orbital arrays in physicist notation.
    Output is accumulated into active-subspace views (ft, vt).

    Args:
        ft: dict with keys 'ij','ia','ab' -> views into active-space 1e blocks.
        vt: dict with keys 'ijkl','ijka','aijb','ijab','iabc','abcd' -> active 2e blocks.
        idx: dict with index slice arrays: 'O','V','OI','VI'.
        f1: full spin-orbital Fock matrix, shape (nso, nso).
        t1: T1 amplitudes (nso, nso), T_int zeroed.
        t2: T2 amplitudes (nso, nso, nso, nso), T_int zeroed.
    """
    OI, VI = idx['OI'], idx['VI']
    scalar = 0.0

    # --- HH ---
    ft['ij'] += (np.einsum('ej,ei->ij', t1, f1))[np.ix_(OI, OI)]
    ft['ij'] += (np.einsum('je,ei->ij', f1, t1))[np.ix_(OI, OI)]

    # --- PP ---
    ft['ab'] += (-np.einsum('mb,am->ab', f1, t1))[np.ix_(VI, VI)]  # [T_int=0]
    ft['ab'] += (-np.einsum('bm,am->ab', t1, f1))[np.ix_(VI, VI)]  # [T_int=0]

    # --- HP/PH ---
    ft['ia'] += (np.einsum('me,aeim->ia', f1, t2))[np.ix_(OI, VI)]
    ft['ia'] += (-np.einsum('mi,am->ia', f1, t1))[np.ix_(OI, VI)]  # [T_int=0]
    ft['ia'] += (np.einsum('ae,ei->ia', f1, t1))[np.ix_(OI, VI)]

    # --- HHHP/HPHH ---
    vt['ijka'] += (-np.einsum('ke,aeij->ijka', f1, t2))[np.ix_(OI, OI, OI, VI)]

    # --- PPPH/PHPP ---
    vt['iabc'] += (np.einsum('ma,cbim->iabc', f1, t2))[np.ix_(OI, VI, VI, VI)]  # [T_int=0]

    # --- HHPP/PPHH ---
    vt['ijab'] += (np.einsum('be,aeij->ijab', f1, t2))[np.ix_(OI, OI, VI, VI)]
    vt['ijab'] += (-np.einsum('mj,abim->ijab', f1, t2))[np.ix_(OI, OI, VI, VI)]  # [T_int=0]
    vt['ijab'] += (np.einsum('mi,abjm->ijab', f1, t2))[np.ix_(OI, OI, VI, VI)]  # [T_int=0]
    vt['ijab'] += (-np.einsum('ae,beij->ijab', f1, t2))[np.ix_(OI, OI, VI, VI)]

    return scalar

def v_1(
    ft, vt, idx,
    v2, t1, t2,
):
    """BCH commutator V_1: bare 2e integrals v2.

    Auto-generated from ExaChem's ducc-t_ccsd.hpp.
    All tensors use full spin-orbital arrays in physicist notation.
    Output is accumulated into active-subspace views (ft, vt).

    Args:
        ft: dict with keys 'ij','ia','ab' -> views into active-space 1e blocks.
        vt: dict with keys 'ijkl','ijka','aijb','ijab','iabc','abcd' -> active 2e blocks.
        idx: dict with index slice arrays: 'O','V','OI','VI'.
        v2: full spin-orbital antisym 2e integrals, shape (nso, nso, nso, nso).
        t1: T1 amplitudes (nso, nso), T_int zeroed.
        t2: T2 amplitudes (nso, nso, nso, nso), T_int zeroed.
    """
    OI, VI = idx['OI'], idx['VI']
    scalar = 0.0

    # --- HH ---
    ft['ij'] += ((-1/2) * np.einsum('mjef,efim->ij', v2, t2))[np.ix_(OI, OI)]
    ft['ij'] += ((-1/2) * np.einsum('efmj,imef->ij', t2, v2))[np.ix_(OI, OI)]
    ft['ij'] += (-np.einsum('mjie,em->ij', v2, t1))[np.ix_(OI, OI)]
    ft['ij'] += (-np.einsum('em,mije->ij', t1, v2))[np.ix_(OI, OI)]

    # --- PP ---
    ft['ab'] += ((1/2) * np.einsum('ebmn,mnae->ab', t2, v2))[np.ix_(VI, VI)]
    ft['ab'] += ((1/2) * np.einsum('mneb,aemn->ab', v2, t2))[np.ix_(VI, VI)]
    ft['ab'] += (-np.einsum('mabe,em->ab', v2, t1))[np.ix_(VI, VI)]
    ft['ab'] += (-np.einsum('em,mbae->ab', t1, v2))[np.ix_(VI, VI)]

    # --- HP/PH ---
    ft['ia'] += ((1/2) * np.einsum('mafe,efim->ia', v2, t2))[np.ix_(OI, VI)]
    ft['ia'] += ((-1/2) * np.einsum('mnie,aemn->ia', v2, t2))[np.ix_(OI, VI)]
    ft['ia'] += (np.einsum('em,imae->ia', t1, v2))[np.ix_(OI, VI)]
    ft['ia'] += (-np.einsum('maie,em->ia', v2, t1))[np.ix_(OI, VI)]

    # --- HHHH ---
    vt['ijkl'] += ((-1/2) * np.einsum('lkef,efij->ijkl', v2, t2))[np.ix_(OI, OI, OI, OI)]
    vt['ijkl'] += ((-1/2) * np.einsum('eflk,ijef->ijkl', t2, v2))[np.ix_(OI, OI, OI, OI)]
    vt['ijkl'] += (-np.einsum('lkie,ej->ijkl', v2, t1))[np.ix_(OI, OI, OI, OI)]
    vt['ijkl'] += (np.einsum('lkje,ei->ijkl', v2, t1))[np.ix_(OI, OI, OI, OI)]
    vt['ijkl'] += (np.einsum('ek,jile->ijkl', t1, v2))[np.ix_(OI, OI, OI, OI)]
    vt['ijkl'] += (-np.einsum('el,jike->ijkl', t1, v2))[np.ix_(OI, OI, OI, OI)]

    # --- PPPP ---
    vt['abcd'] += ((-1/2) * np.einsum('mndc,abmn->abcd', v2, t2))[np.ix_(VI, VI, VI, VI)]  # [T_int=0]
    vt['abcd'] += ((-1/2) * np.einsum('dcmn,mnab->abcd', t2, v2))[np.ix_(VI, VI, VI, VI)]  # [T_int=0]
    vt['abcd'] += (np.einsum('dm,mcab->abcd', t1, v2))[np.ix_(VI, VI, VI, VI)]  # [T_int=0]
    vt['abcd'] += (-np.einsum('cm,mdab->abcd', t1, v2))[np.ix_(VI, VI, VI, VI)]  # [T_int=0]
    vt['abcd'] += (-np.einsum('mbcd,am->abcd', v2, t1))[np.ix_(VI, VI, VI, VI)]  # [T_int=0]
    vt['abcd'] += (np.einsum('macd,bm->abcd', v2, t1))[np.ix_(VI, VI, VI, VI)]  # [T_int=0]

    # --- HHHP/HPHH ---
    vt['ijka'] += (np.einsum('mkje,aeim->ijka', v2, t2))[np.ix_(OI, OI, OI, VI)]
    vt['ijka'] += (-np.einsum('mkie,aejm->ijka', v2, t2))[np.ix_(OI, OI, OI, VI)]
    vt['ijka'] += ((-1/2) * np.einsum('kafe,efij->ijka', v2, t2))[np.ix_(OI, OI, OI, VI)]
    vt['ijka'] += (np.einsum('mkij,am->ijka', v2, t1))[np.ix_(OI, OI, OI, VI)]  # [T_int=0]
    vt['ijka'] += (-np.einsum('ek,ijae->ijka', t1, v2))[np.ix_(OI, OI, OI, VI)]
    vt['ijka'] += (-np.einsum('kaje,ei->ijka', v2, t1))[np.ix_(OI, OI, OI, VI)]
    vt['ijka'] += (np.einsum('kaie,ej->ijka', v2, t1))[np.ix_(OI, OI, OI, VI)]

    # --- PPPH/PHPP ---
    vt['iabc'] += (np.einsum('mbae,ceim->iabc', v2, t2))[np.ix_(OI, VI, VI, VI)]
    vt['iabc'] += (-np.einsum('mcae,beim->iabc', v2, t2))[np.ix_(OI, VI, VI, VI)]
    vt['iabc'] += ((-1/2) * np.einsum('mnia,cbmn->iabc', v2, t2))[np.ix_(OI, VI, VI, VI)]  # [T_int=0]
    vt['iabc'] += (np.einsum('am,imcb->iabc', t1, v2))[np.ix_(OI, VI, VI, VI)]  # [T_int=0]
    vt['iabc'] += (-np.einsum('cbea,ei->iabc', v2, t1))[np.ix_(OI, VI, VI, VI)]
    vt['iabc'] += (-np.einsum('mcia,bm->iabc', v2, t1))[np.ix_(OI, VI, VI, VI)]  # [T_int=0]
    vt['iabc'] += (np.einsum('mbia,cm->iabc', v2, t1))[np.ix_(OI, VI, VI, VI)]  # [T_int=0]

    # --- HHPP/PPHH ---
    vt['ijab'] += (np.einsum('maje,beim->ijab', v2, t2))[np.ix_(OI, OI, VI, VI)]
    vt['ijab'] += ((1/2) * np.einsum('mnij,abmn->ijab', v2, t2))[np.ix_(OI, OI, VI, VI)]  # [T_int=0]
    vt['ijab'] += (np.einsum('mbie,aejm->ijab', v2, t2))[np.ix_(OI, OI, VI, VI)]
    vt['ijab'] += (-np.einsum('maie,bejm->ijab', v2, t2))[np.ix_(OI, OI, VI, VI)]
    vt['ijab'] += (-np.einsum('mbje,aeim->ijab', v2, t2))[np.ix_(OI, OI, VI, VI)]
    vt['ijab'] += ((1/2) * np.einsum('abef,efij->ijab', v2, t2))[np.ix_(OI, OI, VI, VI)]
    vt['ijab'] += (np.einsum('ieab,ej->ijab', v2, t1))[np.ix_(OI, OI, VI, VI)]
    vt['ijab'] += (-np.einsum('jima,bm->ijab', v2, t1))[np.ix_(OI, OI, VI, VI)]  # [T_int=0]
    vt['ijab'] += (np.einsum('jimb,am->ijab', v2, t1))[np.ix_(OI, OI, VI, VI)]  # [T_int=0]
    vt['ijab'] += (-np.einsum('jeab,ei->ijab', v2, t1))[np.ix_(OI, OI, VI, VI)]

    # --- PHHP ---
    vt['aijb'] += (np.einsum('mieb,aejm->aijb', v2, t2))[np.ix_(VI, OI, OI, VI)]
    vt['aijb'] += (np.einsum('ebmi,jmae->aijb', t2, v2))[np.ix_(VI, OI, OI, VI)]
    vt['aijb'] += (-np.einsum('mijb,am->aijb', v2, t1))[np.ix_(VI, OI, OI, VI)]  # [T_int=0]
    vt['aijb'] += (np.einsum('iabe,ej->aijb', v2, t1))[np.ix_(VI, OI, OI, VI)]
    vt['aijb'] += (-np.einsum('bm,mjia->aijb', t1, v2))[np.ix_(VI, OI, OI, VI)]  # [T_int=0]
    vt['aijb'] += (np.einsum('ei,jbae->aijb', t1, v2))[np.ix_(VI, OI, OI, VI)]

    return scalar

def f_2(
    ft, vt, idx,
    f1, t1, t2,
):
    """BCH commutator F_2: Fock operator f1.

    Auto-generated from ExaChem's ducc-t_ccsd.hpp.
    All tensors use full spin-orbital arrays in physicist notation.
    Output is accumulated into active-subspace views (ft, vt).

    Args:
        ft: dict with keys 'ij','ia','ab' -> views into active-space 1e blocks.
        vt: dict with keys 'ijkl','ijka','aijb','ijab','iabc','abcd' -> active 2e blocks.
        idx: dict with index slice arrays: 'O','V','OI','VI'.
        f1: full spin-orbital Fock matrix, shape (nso, nso).
        t1: T1 amplitudes (nso, nso), T_int zeroed.
        t2: T2 amplitudes (nso, nso, nso, nso), T_int zeroed.
    """
    OI, VI = idx['OI'], idx['VI']
    scalar = 0.0

    # --- FC ---
    # PARSE ERROR: (-1.0/2.0) * t2(e, f, m, n) * f1(e, g) * t2(f, g, m, n)
    scalar += (1/2) * np.einsum('me,fn,efmn', f1, t1, t2)
    scalar += -np.einsum('em,nm,en', t1, f1, t1)
    scalar += np.einsum('em,ef,fm', t1, f1, t1)
    scalar += (1/2) * np.einsum('efmn,me,fn', t2, f1, t1)
    scalar += (1/2) * np.einsum('efmn,om,efno', t2, f1, t2)

    # --- HH ---
    ft['ij'] += ((1/2) * np.einsum('efmj,nm,efin->ij', t2, f1, t2))[np.ix_(OI, OI)]
    ft['ij'] += (np.einsum('efmj,eg,fgim->ij', t2, f1, t2))[np.ix_(OI, OI)]
    ft['ij'] += ((-1/4) * np.einsum('efmn,jm,efin->ij', t2, f1, t2))[np.ix_(OI, OI)]
    ft['ij'] += ((-1/4) * np.einsum('efmj,ni,efmn->ij', t2, f1, t2))[np.ix_(OI, OI)]
    ft['ij'] += ((1/2) * np.einsum('efmj,em,fi->ij', t2, f1, t1))[np.ix_(OI, OI)]
    ft['ij'] += ((-1/2) * np.einsum('efmj,ei,fm->ij', t2, f1, t1))[np.ix_(OI, OI)]
    ft['ij'] += ((-1/2) * np.einsum('em,jf,efim->ij', t1, f1, t2))[np.ix_(OI, OI)]
    ft['ij'] += ((1/2) * np.einsum('ej,mf,efim->ij', t1, f1, t2))[np.ix_(OI, OI)]
    ft['ij'] += ((-1/2) * np.einsum('ej,mi,em->ij', t1, f1, t1))[np.ix_(OI, OI)]
    ft['ij'] += ((-1/2) * np.einsum('em,jm,ei->ij', t1, f1, t1))[np.ix_(OI, OI)]
    ft['ij'] += (np.einsum('ej,ef,fi->ij', t1, f1, t1))[np.ix_(OI, OI)]

    # --- PP ---
    ft['ab'] += ((1/2) * np.einsum('ebmn,ef,afmn->ab', t2, f1, t2))[np.ix_(VI, VI)]
    ft['ab'] += (np.einsum('ebmn,om,aeno->ab', t2, f1, t2))[np.ix_(VI, VI)]
    ft['ab'] += ((-1/4) * np.einsum('ebmn,af,efmn->ab', t2, f1, t2))[np.ix_(VI, VI)]
    ft['ab'] += ((-1/4) * np.einsum('efmn,eb,afmn->ab', t2, f1, t2))[np.ix_(VI, VI)]
    ft['ab'] += ((1/2) * np.einsum('ebmn,am,en->ab', t2, f1, t1))[np.ix_(VI, VI)]
    ft['ab'] += ((-1/2) * np.einsum('ebmn,em,an->ab', t2, f1, t1))[np.ix_(VI, VI)]  # [T_int=0]
    ft['ab'] += ((1/2) * np.einsum('em,nb,aemn->ab', t1, f1, t2))[np.ix_(VI, VI)]
    ft['ab'] += ((-1/2) * np.einsum('bm,ne,aemn->ab', t1, f1, t2))[np.ix_(VI, VI)]  # [T_int=0]
    ft['ab'] += (np.einsum('bm,nm,an->ab', t1, f1, t1))[np.ix_(VI, VI)]  # [T_int=0]
    ft['ab'] += ((-1/2) * np.einsum('bm,ae,em->ab', t1, f1, t1))[np.ix_(VI, VI)]  # [T_int=0]
    ft['ab'] += ((-1/2) * np.einsum('em,eb,am->ab', t1, f1, t1))[np.ix_(VI, VI)]  # [T_int=0]

    # --- HP/PH ---
    ft['ia'] += ((1/2) * np.einsum('efmn,em,afin->ia', t2, f1, t2))[np.ix_(OI, VI)]
    ft['ia'] += ((-1/4) * np.einsum('efmn,am,efin->ia', t2, f1, t2))[np.ix_(OI, VI)]
    ft['ia'] += ((-1/4) * np.einsum('efmn,ei,afmn->ia', t2, f1, t2))[np.ix_(OI, VI)]
    ft['ia'] += (-np.einsum('em,nm,aein->ia', t1, f1, t2))[np.ix_(OI, VI)]
    ft['ia'] += (np.einsum('em,ef,afim->ia', t1, f1, t2))[np.ix_(OI, VI)]
    ft['ia'] += ((1/2) * np.einsum('em,ni,aemn->ia', t1, f1, t2))[np.ix_(OI, VI)]
    ft['ia'] += ((-1/2) * np.einsum('em,af,efim->ia', t1, f1, t2))[np.ix_(OI, VI)]
    ft['ia'] += ((-1/2) * np.einsum('em,am,ei->ia', t1, f1, t1))[np.ix_(OI, VI)]
    ft['ia'] += (-np.einsum('me,ei,am->ia', f1, t1, t1))[np.ix_(OI, VI)]  # [T_int=0]
    ft['ia'] += ((-1/2) * np.einsum('em,ei,am->ia', t1, f1, t1))[np.ix_(OI, VI)]  # [T_int=0]

    # --- HHHH ---
    vt['ijkl'] += ((-1/4) * np.einsum('eflk,mi,efjm->ijkl', t2, f1, t2))[np.ix_(OI, OI, OI, OI)]
    vt['ijkl'] += (np.einsum('eflk,eg,fgij->ijkl', t2, f1, t2))[np.ix_(OI, OI, OI, OI)]
    vt['ijkl'] += ((1/4) * np.einsum('eflk,mj,efim->ijkl', t2, f1, t2))[np.ix_(OI, OI, OI, OI)]
    vt['ijkl'] += ((-1/4) * np.einsum('efml,km,efij->ijkl', t2, f1, t2))[np.ix_(OI, OI, OI, OI)]
    vt['ijkl'] += ((1/4) * np.einsum('efmk,lm,efij->ijkl', t2, f1, t2))[np.ix_(OI, OI, OI, OI)]
    vt['ijkl'] += ((1/2) * np.einsum('eflk,ej,fi->ijkl', t2, f1, t1))[np.ix_(OI, OI, OI, OI)]
    vt['ijkl'] += ((-1/2) * np.einsum('eflk,ei,fj->ijkl', t2, f1, t1))[np.ix_(OI, OI, OI, OI)]
    vt['ijkl'] += ((1/2) * np.einsum('ek,lf,efij->ijkl', t1, f1, t2))[np.ix_(OI, OI, OI, OI)]
    vt['ijkl'] += ((-1/2) * np.einsum('el,kf,efij->ijkl', t1, f1, t2))[np.ix_(OI, OI, OI, OI)]

    # --- PPPP ---
    vt['abcd'] += (-np.einsum('dcmn,om,abno->abcd', t2, f1, t2))[np.ix_(VI, VI, VI, VI)]  # [T_int=0]
    vt['abcd'] += ((1/4) * np.einsum('edmn,ec,abmn->abcd', t2, f1, t2))[np.ix_(VI, VI, VI, VI)]  # [T_int=0]
    vt['abcd'] += ((-1/4) * np.einsum('ecmn,ed,abmn->abcd', t2, f1, t2))[np.ix_(VI, VI, VI, VI)]  # [T_int=0]
    vt['abcd'] += ((1/4) * np.einsum('dcmn,ae,bemn->abcd', t2, f1, t2))[np.ix_(VI, VI, VI, VI)]  # [T_int=0]
    vt['abcd'] += ((-1/4) * np.einsum('dcmn,be,aemn->abcd', t2, f1, t2))[np.ix_(VI, VI, VI, VI)]  # [T_int=0]
    vt['abcd'] += ((1/2) * np.einsum('dcmn,bm,an->abcd', t2, f1, t1))[np.ix_(VI, VI, VI, VI)]  # [T_int=0]
    vt['abcd'] += ((-1/2) * np.einsum('dm,nc,abmn->abcd', t1, f1, t2))[np.ix_(VI, VI, VI, VI)]  # [T_int=0]
    vt['abcd'] += ((1/2) * np.einsum('cm,nd,abmn->abcd', t1, f1, t2))[np.ix_(VI, VI, VI, VI)]  # [T_int=0]
    vt['abcd'] += ((-1/2) * np.einsum('dcmn,am,bn->abcd', t2, f1, t1))[np.ix_(VI, VI, VI, VI)]  # [T_int=0]

    # --- HHHP/HPHH ---
    vt['ijka'] += ((-1/2) * np.einsum('efmk,em,afij->ijka', t2, f1, t2))[np.ix_(OI, OI, OI, VI)]
    vt['ijka'] += ((1/4) * np.einsum('efmk,am,efij->ijka', t2, f1, t2))[np.ix_(OI, OI, OI, VI)]
    vt['ijka'] += ((-1/2) * np.einsum('efmk,ei,afjm->ijka', t2, f1, t2))[np.ix_(OI, OI, OI, VI)]
    vt['ijka'] += ((1/2) * np.einsum('efmk,ej,afim->ijka', t2, f1, t2))[np.ix_(OI, OI, OI, VI)]
    vt['ijka'] += (-np.einsum('ek,ef,afij->ijka', t1, f1, t2))[np.ix_(OI, OI, OI, VI)]
    vt['ijka'] += ((1/2) * np.einsum('ek,mj,aeim->ijka', t1, f1, t2))[np.ix_(OI, OI, OI, VI)]
    vt['ijka'] += ((-1/2) * np.einsum('ek,mi,aejm->ijka', t1, f1, t2))[np.ix_(OI, OI, OI, VI)]
    vt['ijka'] += ((1/2) * np.einsum('em,km,aeij->ijka', t1, f1, t2))[np.ix_(OI, OI, OI, VI)]
    vt['ijka'] += ((1/2) * np.einsum('ek,af,efij->ijka', t1, f1, t2))[np.ix_(OI, OI, OI, VI)]

    # --- PPPH/PHPP ---
    vt['iabc'] += ((1/2) * np.einsum('eamn,em,cbin->iabc', t2, f1, t2))[np.ix_(OI, VI, VI, VI)]
    vt['iabc'] += ((-1/4) * np.einsum('eamn,ei,cbmn->iabc', t2, f1, t2))[np.ix_(OI, VI, VI, VI)]  # [T_int=0]
    vt['iabc'] += ((-1/2) * np.einsum('eamn,bm,cein->iabc', t2, f1, t2))[np.ix_(OI, VI, VI, VI)]
    vt['iabc'] += ((1/2) * np.einsum('eamn,cm,bein->iabc', t2, f1, t2))[np.ix_(OI, VI, VI, VI)]
    vt['iabc'] += ((1/2) * np.einsum('am,ni,cbmn->iabc', t1, f1, t2))[np.ix_(OI, VI, VI, VI)]  # [T_int=0]
    vt['iabc'] += ((1/2) * np.einsum('am,be,ceim->iabc', t1, f1, t2))[np.ix_(OI, VI, VI, VI)]  # [T_int=0]
    vt['iabc'] += ((-1/2) * np.einsum('am,ce,beim->iabc', t1, f1, t2))[np.ix_(OI, VI, VI, VI)]  # [T_int=0]
    vt['iabc'] += (-np.einsum('am,nm,cbin->iabc', t1, f1, t2))[np.ix_(OI, VI, VI, VI)]  # [T_int=0]
    vt['iabc'] += ((1/2) * np.einsum('em,ea,cbim->iabc', t1, f1, t2))[np.ix_(OI, VI, VI, VI)]  # [T_int=0]

    # --- HHPP/PPHH ---
    vt['ijab'] += ((-1/2) * np.einsum('em,bm,aeij->ijab', t1, f1, t2))[np.ix_(OI, OI, VI, VI)]
    vt['ijab'] += ((1/2) * np.einsum('em,ei,abjm->ijab', t1, f1, t2))[np.ix_(OI, OI, VI, VI)]  # [T_int=0]
    vt['ijab'] += ((-1/2) * np.einsum('em,ej,abim->ijab', t1, f1, t2))[np.ix_(OI, OI, VI, VI)]  # [T_int=0]
    vt['ijab'] += ((1/2) * np.einsum('em,am,beij->ijab', t1, f1, t2))[np.ix_(OI, OI, VI, VI)]
    vt['ijab'] += (-np.einsum('me,ej,abim->ijab', f1, t1, t2))[np.ix_(OI, OI, VI, VI)]  # [T_int=0]
    vt['ijab'] += (np.einsum('me,am,beij->ijab', f1, t1, t2))[np.ix_(OI, OI, VI, VI)]  # [T_int=0]
    vt['ijab'] += (-np.einsum('me,bm,aeij->ijab', f1, t1, t2))[np.ix_(OI, OI, VI, VI)]  # [T_int=0]
    vt['ijab'] += (np.einsum('me,ei,abjm->ijab', f1, t1, t2))[np.ix_(OI, OI, VI, VI)]  # [T_int=0]

    # --- PHHP ---
    vt['aijb'] += (np.einsum('ebmi,ef,afjm->aijb', t2, f1, t2))[np.ix_(VI, OI, OI, VI)]
    vt['aijb'] += (-np.einsum('ebmi,nm,aejn->aijb', t2, f1, t2))[np.ix_(VI, OI, OI, VI)]
    vt['aijb'] += ((1/2) * np.einsum('ebmi,nj,aemn->aijb', t2, f1, t2))[np.ix_(VI, OI, OI, VI)]
    vt['aijb'] += ((1/2) * np.einsum('ebmn,im,aejn->aijb', t2, f1, t2))[np.ix_(VI, OI, OI, VI)]
    vt['aijb'] += ((-1/2) * np.einsum('efmi,eb,afjm->aijb', t2, f1, t2))[np.ix_(VI, OI, OI, VI)]
    vt['aijb'] += ((-1/2) * np.einsum('ebmi,af,efjm->aijb', t2, f1, t2))[np.ix_(VI, OI, OI, VI)]
    vt['aijb'] += ((-1/2) * np.einsum('bm,ie,aejm->aijb', t1, f1, t2))[np.ix_(VI, OI, OI, VI)]  # [T_int=0]
    vt['aijb'] += ((-1/2) * np.einsum('ei,mb,aejm->aijb', t1, f1, t2))[np.ix_(VI, OI, OI, VI)]
    vt['aijb'] += ((-1/2) * np.einsum('ebmi,am,ej->aijb', t2, f1, t1))[np.ix_(VI, OI, OI, VI)]
    vt['aijb'] += ((-1/2) * np.einsum('ebmi,ej,am->aijb', t2, f1, t1))[np.ix_(VI, OI, OI, VI)]  # [T_int=0]

    return scalar

def v_2(
    ft, vt, idx,
    v2, t1, t2,
):
    """BCH commutator V_2: bare 2e integrals v2.

    Auto-generated from ExaChem's ducc-t_ccsd.hpp.
    All tensors use full spin-orbital arrays in physicist notation.
    Output is accumulated into active-subspace views (ft, vt).

    Args:
        ft: dict with keys 'ij','ia','ab' -> views into active-space 1e blocks.
        vt: dict with keys 'ijkl','ijka','aijb','ijab','iabc','abcd' -> active 2e blocks.
        idx: dict with index slice arrays: 'O','V','OI','VI'.
        v2: full spin-orbital antisym 2e integrals, shape (nso, nso, nso, nso).
        t1: T1 amplitudes (nso, nso), T_int zeroed.
        t2: T2 amplitudes (nso, nso, nso, nso), T_int zeroed.
    """
    OI, VI = idx['OI'], idx['VI']
    scalar = 0.0

    # --- FC ---
    # PARSE ERROR: (-1.0/2.0) * t1(e, m) * v2ijka(n, o, m, f) * t2(e, f, n, o)
    # PARSE ERROR: (-1.0/2.0) * t2(e, f, m, n) * v2ijka(n, m, o, e) * t1(f, o)
    scalar += (1/2) * np.einsum('mnef,em,fn', v2, t1, t1)
    scalar += -np.einsum('em,nemf,fn', t1, v2, t1)
    scalar += (1/2) * np.einsum('em,negf,fgmn', t1, v2, t2)
    scalar += (1/2) * np.einsum('efmn,mgef,gn', t2, v2, t1)
    scalar += -np.einsum('efmn,oemh,fhno', t2, v2, t2)
    scalar += (1/8) * np.einsum('efmn,opmn,efop', t2, v2, t2)
    scalar += (1/8) * np.einsum('efmn,efgh,ghmn', t2, v2, t2)
    scalar += (1/2) * np.einsum('em,fn,mnef', t1, t1, v2)

    # --- HH ---
    ft['ij'] += (-np.einsum('efmj,nemg,fgin->ij', t2, v2, t2))[np.ix_(OI, OI)]
    ft['ij'] += ((-1/2) * np.einsum('efmn,ojim,efno->ij', t2, v2, t2))[np.ix_(OI, OI)]
    ft['ij'] += (np.einsum('efmn,jemg,fgin->ij', t2, v2, t2))[np.ix_(OI, OI)]
    ft['ij'] += (np.einsum('efmj,neig,fgmn->ij', t2, v2, t2))[np.ix_(OI, OI)]
    ft['ij'] += ((-1/4) * np.einsum('efmn,ojmn,efio->ij', t2, v2, t2))[np.ix_(OI, OI)]
    ft['ij'] += ((-1/4) * np.einsum('efmj,noim,efno->ij', t2, v2, t2))[np.ix_(OI, OI)]
    ft['ij'] += ((-1/4) * np.einsum('efmj,efgh,ghim->ij', t2, v2, t2))[np.ix_(OI, OI)]
    ft['ij'] += ((-1/2) * np.einsum('efmn,jeig,fgmn->ij', t2, v2, t2))[np.ix_(OI, OI)]
    ft['ij'] += ((-1/2) * np.einsum('efmn,mije,fn->ij', t2, v2, t1))[np.ix_(OI, OI)]
    ft['ij'] += ((1/2) * np.einsum('efmj,mgef,gi->ij', t2, v2, t1))[np.ix_(OI, OI)]
    ft['ij'] += (np.einsum('efmj,mine,fn->ij', t2, v2, t1))[np.ix_(OI, OI)]
    ft['ij'] += (np.einsum('em,njmf,efin->ij', t1, v2, t2))[np.ix_(OI, OI)]
    ft['ij'] += ((-1/4) * np.einsum('efmn,nmje,fi->ij', t2, v2, t1))[np.ix_(OI, OI)]
    ft['ij'] += ((-1/2) * np.einsum('em,njif,efmn->ij', t1, v2, t2))[np.ix_(OI, OI)]
    ft['ij'] += ((-1/2) * np.einsum('efmj,igef,gm->ij', t2, v2, t1))[np.ix_(OI, OI)]
    ft['ij'] += ((-1/4) * np.einsum('ej,mnif,efmn->ij', t1, v2, t2))[np.ix_(OI, OI)]
    ft['ij'] += ((1/2) * np.einsum('ej,megf,fgim->ij', t1, v2, t2))[np.ix_(OI, OI)]
    ft['ij'] += ((-1/2) * np.einsum('em,jegf,fgim->ij', t1, v2, t2))[np.ix_(OI, OI)]
    ft['ij'] += (-np.einsum('mjef,ei,fm->ij', v2, t1, t1))[np.ix_(OI, OI)]
    ft['ij'] += (np.einsum('em,njim,en->ij', t1, v2, t1))[np.ix_(OI, OI)]
    ft['ij'] += (-np.einsum('em,fj,imef->ij', t1, t1, v2))[np.ix_(OI, OI)]
    ft['ij'] += (-np.einsum('em,jemf,fi->ij', t1, v2, t1))[np.ix_(OI, OI)]
    ft['ij'] += (-np.einsum('ej,meif,fm->ij', t1, v2, t1))[np.ix_(OI, OI)]
    ft['ij'] += (np.einsum('em,jeif,fm->ij', t1, v2, t1))[np.ix_(OI, OI)]

    # --- PP ---
    ft['ab'] += (np.einsum('ebmn,oemf,afno->ab', t2, v2, t2))[np.ix_(VI, VI)]
    ft['ab'] += ((1/2) * np.einsum('efmn,aegb,fgmn->ab', t2, v2, t2))[np.ix_(VI, VI)]
    ft['ab'] += (-np.einsum('ebmn,oamf,efno->ab', t2, v2, t2))[np.ix_(VI, VI)]
    ft['ab'] += (-np.einsum('efmn,oemb,afno->ab', t2, v2, t2))[np.ix_(VI, VI)]
    ft['ab'] += ((1/4) * np.einsum('ebmn,opmn,aeop->ab', t2, v2, t2))[np.ix_(VI, VI)]
    ft['ab'] += ((1/4) * np.einsum('ebmn,aefg,fgmn->ab', t2, v2, t2))[np.ix_(VI, VI)]
    ft['ab'] += ((1/4) * np.einsum('efmn,efgb,agmn->ab', t2, v2, t2))[np.ix_(VI, VI)]
    ft['ab'] += ((1/2) * np.einsum('efmn,oamb,efno->ab', t2, v2, t2))[np.ix_(VI, VI)]
    ft['ab'] += ((1/2) * np.einsum('ebmn,nmoe,ao->ab', t2, v2, t1))[np.ix_(VI, VI)]  # [T_int=0]
    ft['ab'] += ((-1/2) * np.einsum('em,nabf,efmn->ab', t1, v2, t2))[np.ix_(VI, VI)]
    ft['ab'] += (np.einsum('ebmn,mfae,fn->ab', t2, v2, t1))[np.ix_(VI, VI)]
    ft['ab'] += ((-1/4) * np.einsum('bm,nafe,efmn->ab', t1, v2, t2))[np.ix_(VI, VI)]  # [T_int=0]
    ft['ab'] += ((-1/2) * np.einsum('efmn,mbae,fn->ab', t2, v2, t1))[np.ix_(VI, VI)]
    ft['ab'] += (np.einsum('em,nebf,afmn->ab', t1, v2, t2))[np.ix_(VI, VI)]
    ft['ab'] += ((-1/2) * np.einsum('ebmn,nmoa,eo->ab', t2, v2, t1))[np.ix_(VI, VI)]
    ft['ab'] += ((-1/2) * np.einsum('em,nomb,aeno->ab', t1, v2, t2))[np.ix_(VI, VI)]
    ft['ab'] += ((1/2) * np.einsum('bm,nome,aeno->ab', t1, v2, t2))[np.ix_(VI, VI)]  # [T_int=0]
    ft['ab'] += ((-1/4) * np.einsum('efmn,mbef,an->ab', t2, v2, t1))[np.ix_(VI, VI)]  # [T_int=0]
    ft['ab'] += (np.einsum('em,bn,mnae->ab', t1, t1, v2))[np.ix_(VI, VI)]  # [T_int=0]
    ft['ab'] += (-np.einsum('em,aefb,fm->ab', t1, v2, t1))[np.ix_(VI, VI)]
    ft['ab'] += (np.einsum('mneb,am,en->ab', v2, t1, t1))[np.ix_(VI, VI)]  # [T_int=0]
    ft['ab'] += (np.einsum('bm,name,en->ab', t1, v2, t1))[np.ix_(VI, VI)]  # [T_int=0]
    ft['ab'] += (-np.einsum('em,namb,en->ab', t1, v2, t1))[np.ix_(VI, VI)]
    ft['ab'] += (np.einsum('em,nemb,an->ab', t1, v2, t1))[np.ix_(VI, VI)]  # [T_int=0]

    # --- HP/PH ---
    ft['ia'] += ((1/2) * np.einsum('efmn,mioa,efno->ia', t2, v2, t2))[np.ix_(OI, VI)]
    ft['ia'] += ((1/4) * np.einsum('efmn,nmoa,efio->ia', t2, v2, t2))[np.ix_(OI, VI)]
    ft['ia'] += (-np.einsum('efmn,mioe,afno->ia', t2, v2, t2))[np.ix_(OI, VI)]
    ft['ia'] += ((-1/2) * np.einsum('efmn,nmoe,afio->ia', t2, v2, t2))[np.ix_(OI, VI)]
    ft['ia'] += ((1/2) * np.einsum('efmn,mgef,agin->ia', t2, v2, t2))[np.ix_(OI, VI)]
    ft['ia'] += (np.einsum('efmn,mgae,fgin->ia', t2, v2, t2))[np.ix_(OI, VI)]
    ft['ia'] += ((-1/4) * np.einsum('efmn,igef,agmn->ia', t2, v2, t2))[np.ix_(OI, VI)]
    ft['ia'] += ((-1/2) * np.einsum('efmn,igae,fgmn->ia', t2, v2, t2))[np.ix_(OI, VI)]
    ft['ia'] += (np.einsum('em,namf,efin->ia', t1, v2, t2))[np.ix_(OI, VI)]
    ft['ia'] += (-np.einsum('em,nemf,afin->ia', t1, v2, t2))[np.ix_(OI, VI)]
    ft['ia'] += (np.einsum('em,neif,afmn->ia', t1, v2, t2))[np.ix_(OI, VI)]
    ft['ia'] += ((1/2) * np.einsum('em,noim,aeno->ia', t1, v2, t2))[np.ix_(OI, VI)]
    ft['ia'] += ((-1/2) * np.einsum('em,naif,efmn->ia', t1, v2, t2))[np.ix_(OI, VI)]
    ft['ia'] += ((1/2) * np.einsum('em,aefg,fgim->ia', t1, v2, t2))[np.ix_(OI, VI)]
    ft['ia'] += ((1/2) * np.einsum('efmn,imae,fn->ia', t2, v2, t1))[np.ix_(OI, VI)]
    ft['ia'] += (np.einsum('mnef,em,afin->ia', v2, t1, t2))[np.ix_(OI, VI)]
    ft['ia'] += ((1/4) * np.einsum('efmn,imef,an->ia', t2, v2, t1))[np.ix_(OI, VI)]  # [T_int=0]
    ft['ia'] += ((1/4) * np.einsum('efmn,mnae,fi->ia', t2, v2, t1))[np.ix_(OI, VI)]
    ft['ia'] += ((-1/2) * np.einsum('mnef,am,efin->ia', v2, t1, t2))[np.ix_(OI, VI)]  # [T_int=0]
    ft['ia'] += ((-1/2) * np.einsum('mnef,ei,afmn->ia', v2, t1, t2))[np.ix_(OI, VI)]
    ft['ia'] += (-np.einsum('em,mina,en->ia', t1, v2, t1))[np.ix_(OI, VI)]
    ft['ia'] += (np.einsum('mafe,ei,fm->ia', v2, t1, t1))[np.ix_(OI, VI)]
    ft['ia'] += (-np.einsum('mnie,am,en->ia', v2, t1, t1))[np.ix_(OI, VI)]  # [T_int=0]
    ft['ia'] += (-np.einsum('em,mfae,fi->ia', t1, v2, t1))[np.ix_(OI, VI)]
    ft['ia'] += (np.einsum('em,mine,an->ia', t1, v2, t1))[np.ix_(OI, VI)]  # [T_int=0]
    ft['ia'] += (np.einsum('em,ifae,fm->ia', t1, v2, t1))[np.ix_(OI, VI)]

    # --- HHHH ---
    vt['ijkl'] += (np.einsum('eflk,meig,fgjm->ijkl', t2, v2, t2))[np.ix_(OI, OI, OI, OI)]
    vt['ijkl'] += (-np.einsum('efmk,lemg,fgij->ijkl', t2, v2, t2))[np.ix_(OI, OI, OI, OI)]
    vt['ijkl'] += ((-1/4) * np.einsum('efmk,nlij,efmn->ijkl', t2, v2, t2))[np.ix_(OI, OI, OI, OI)]
    vt['ijkl'] += (np.einsum('efml,kemg,fgij->ijkl', t2, v2, t2))[np.ix_(OI, OI, OI, OI)]
    vt['ijkl'] += ((-1/4) * np.einsum('efmn,lkjm,efin->ijkl', t2, v2, t2))[np.ix_(OI, OI, OI, OI)]
    vt['ijkl'] += ((1/4) * np.einsum('efmn,lkim,efjn->ijkl', t2, v2, t2))[np.ix_(OI, OI, OI, OI)]
    vt['ijkl'] += ((1/4) * np.einsum('efml,nkij,efmn->ijkl', t2, v2, t2))[np.ix_(OI, OI, OI, OI)]
    vt['ijkl'] += (-np.einsum('eflk,mejg,fgim->ijkl', t2, v2, t2))[np.ix_(OI, OI, OI, OI)]
    vt['ijkl'] += ((-1/8) * np.einsum('eflk,mnij,efmn->ijkl', t2, v2, t2))[np.ix_(OI, OI, OI, OI)]
    vt['ijkl'] += ((-1/8) * np.einsum('efmn,lkmn,efij->ijkl', t2, v2, t2))[np.ix_(OI, OI, OI, OI)]
    vt['ijkl'] += ((-1/4) * np.einsum('eflk,efgh,ghij->ijkl', t2, v2, t2))[np.ix_(OI, OI, OI, OI)]
    vt['ijkl'] += ((-1/2) * np.einsum('efmk,nljm,efin->ijkl', t2, v2, t2))[np.ix_(OI, OI, OI, OI)]
    vt['ijkl'] += ((-1/2) * np.einsum('efml,nkim,efjn->ijkl', t2, v2, t2))[np.ix_(OI, OI, OI, OI)]
    vt['ijkl'] += ((1/2) * np.einsum('efml,nkjm,efin->ijkl', t2, v2, t2))[np.ix_(OI, OI, OI, OI)]
    vt['ijkl'] += ((1/2) * np.einsum('efmk,nlim,efjn->ijkl', t2, v2, t2))[np.ix_(OI, OI, OI, OI)]
    vt['ijkl'] += (-np.einsum('efml,kejg,fgim->ijkl', t2, v2, t2))[np.ix_(OI, OI, OI, OI)]
    vt['ijkl'] += (np.einsum('efmk,lejg,fgim->ijkl', t2, v2, t2))[np.ix_(OI, OI, OI, OI)]
    vt['ijkl'] += (-np.einsum('efmk,leig,fgjm->ijkl', t2, v2, t2))[np.ix_(OI, OI, OI, OI)]
    vt['ijkl'] += (np.einsum('efml,keig,fgjm->ijkl', t2, v2, t2))[np.ix_(OI, OI, OI, OI)]
    vt['ijkl'] += ((-1/2) * np.einsum('em,lkjf,efim->ijkl', t1, v2, t2))[np.ix_(OI, OI, OI, OI)]
    vt['ijkl'] += ((-1/2) * np.einsum('efmk,jile,fm->ijkl', t2, v2, t1))[np.ix_(OI, OI, OI, OI)]
    vt['ijkl'] += (np.einsum('em,lkmf,efij->ijkl', t1, v2, t2))[np.ix_(OI, OI, OI, OI)]
    vt['ijkl'] += ((1/2) * np.einsum('em,lkif,efjm->ijkl', t1, v2, t2))[np.ix_(OI, OI, OI, OI)]
    vt['ijkl'] += ((-1/2) * np.einsum('el,mkif,efjm->ijkl', t1, v2, t2))[np.ix_(OI, OI, OI, OI)]
    vt['ijkl'] += ((1/2) * np.einsum('ek,mlif,efjm->ijkl', t1, v2, t2))[np.ix_(OI, OI, OI, OI)]
    vt['ijkl'] += ((-1/2) * np.einsum('ek,mljf,efim->ijkl', t1, v2, t2))[np.ix_(OI, OI, OI, OI)]
    vt['ijkl'] += ((1/2) * np.einsum('efml,jike,fm->ijkl', t2, v2, t1))[np.ix_(OI, OI, OI, OI)]
    vt['ijkl'] += ((1/2) * np.einsum('el,mkjf,efim->ijkl', t1, v2, t2))[np.ix_(OI, OI, OI, OI)]
    vt['ijkl'] += (np.einsum('eflk,jime,fm->ijkl', t2, v2, t1))[np.ix_(OI, OI, OI, OI)]
    vt['ijkl'] += ((-1/2) * np.einsum('efml,mike,fj->ijkl', t2, v2, t1))[np.ix_(OI, OI, OI, OI)]
    vt['ijkl'] += ((-1/2) * np.einsum('efmk,mjle,fi->ijkl', t2, v2, t1))[np.ix_(OI, OI, OI, OI)]
    vt['ijkl'] += ((1/2) * np.einsum('efmk,mile,fj->ijkl', t2, v2, t1))[np.ix_(OI, OI, OI, OI)]
    vt['ijkl'] += ((1/2) * np.einsum('efml,mjke,fi->ijkl', t2, v2, t1))[np.ix_(OI, OI, OI, OI)]
    vt['ijkl'] += ((-1/2) * np.einsum('eflk,igef,gj->ijkl', t2, v2, t1))[np.ix_(OI, OI, OI, OI)]
    vt['ijkl'] += ((1/2) * np.einsum('eflk,jgef,gi->ijkl', t2, v2, t1))[np.ix_(OI, OI, OI, OI)]
    vt['ijkl'] += ((1/2) * np.einsum('ek,legf,fgij->ijkl', t1, v2, t2))[np.ix_(OI, OI, OI, OI)]
    vt['ijkl'] += ((-1/2) * np.einsum('el,kegf,fgij->ijkl', t1, v2, t2))[np.ix_(OI, OI, OI, OI)]
    vt['ijkl'] += ((1/2) * np.einsum('el,mkij,em->ijkl', t1, v2, t1))[np.ix_(OI, OI, OI, OI)]
    vt['ijkl'] += ((-1/2) * np.einsum('em,lkjm,ei->ijkl', t1, v2, t1))[np.ix_(OI, OI, OI, OI)]
    vt['ijkl'] += (-np.einsum('el,fk,ijef->ijkl', t1, t1, v2))[np.ix_(OI, OI, OI, OI)]
    vt['ijkl'] += (-np.einsum('lkef,ei,fj->ijkl', v2, t1, t1))[np.ix_(OI, OI, OI, OI)]
    vt['ijkl'] += ((-1/2) * np.einsum('ek,mlij,em->ijkl', t1, v2, t1))[np.ix_(OI, OI, OI, OI)]
    vt['ijkl'] += ((1/2) * np.einsum('em,lkim,ej->ijkl', t1, v2, t1))[np.ix_(OI, OI, OI, OI)]
    vt['ijkl'] += (-np.einsum('ek,leif,fj->ijkl', t1, v2, t1))[np.ix_(OI, OI, OI, OI)]
    vt['ijkl'] += (np.einsum('el,keif,fj->ijkl', t1, v2, t1))[np.ix_(OI, OI, OI, OI)]
    vt['ijkl'] += (np.einsum('ek,lejf,fi->ijkl', t1, v2, t1))[np.ix_(OI, OI, OI, OI)]
    vt['ijkl'] += (-np.einsum('el,kejf,fi->ijkl', t1, v2, t1))[np.ix_(OI, OI, OI, OI)]

    # --- PPPP ---
    vt['abcd'] += ((-1/4) * np.einsum('ecmn,abfd,efmn->abcd', t2, v2, t2))[np.ix_(VI, VI, VI, VI)]
    vt['abcd'] += ((-1/4) * np.einsum('dcmn,opmn,abop->abcd', t2, v2, t2))[np.ix_(VI, VI, VI, VI)]  # [T_int=0]
    vt['abcd'] += ((-1/8) * np.einsum('dcmn,abef,efmn->abcd', t2, v2, t2))[np.ix_(VI, VI, VI, VI)]  # [T_int=0]
    vt['abcd'] += (-np.einsum('ecmn,oemd,abno->abcd', t2, v2, t2))[np.ix_(VI, VI, VI, VI)]
    vt['abcd'] += ((-1/8) * np.einsum('efmn,efdc,abmn->abcd', t2, v2, t2))[np.ix_(VI, VI, VI, VI)]  # [T_int=0]
    vt['abcd'] += (np.einsum('edmn,oemc,abno->abcd', t2, v2, t2))[np.ix_(VI, VI, VI, VI)]
    vt['abcd'] += ((1/4) * np.einsum('edmn,abfc,efmn->abcd', t2, v2, t2))[np.ix_(VI, VI, VI, VI)]
    vt['abcd'] += ((-1/4) * np.einsum('efmn,bedc,afmn->abcd', t2, v2, t2))[np.ix_(VI, VI, VI, VI)]
    vt['abcd'] += (np.einsum('dcmn,oame,beno->abcd', t2, v2, t2))[np.ix_(VI, VI, VI, VI)]  # [T_int=0]
    vt['abcd'] += ((1/4) * np.einsum('efmn,aedc,bfmn->abcd', t2, v2, t2))[np.ix_(VI, VI, VI, VI)]
    vt['abcd'] += (-np.einsum('dcmn,obme,aeno->abcd', t2, v2, t2))[np.ix_(VI, VI, VI, VI)]  # [T_int=0]
    vt['abcd'] += (np.einsum('ecmn,obmd,aeno->abcd', t2, v2, t2))[np.ix_(VI, VI, VI, VI)]
    vt['abcd'] += ((-1/2) * np.einsum('edmn,aefc,bfmn->abcd', t2, v2, t2))[np.ix_(VI, VI, VI, VI)]
    vt['abcd'] += ((1/2) * np.einsum('ecmn,aefd,bfmn->abcd', t2, v2, t2))[np.ix_(VI, VI, VI, VI)]
    vt['abcd'] += ((-1/2) * np.einsum('ecmn,befd,afmn->abcd', t2, v2, t2))[np.ix_(VI, VI, VI, VI)]
    vt['abcd'] += (-np.einsum('edmn,obmc,aeno->abcd', t2, v2, t2))[np.ix_(VI, VI, VI, VI)]
    vt['abcd'] += ((1/2) * np.einsum('edmn,befc,afmn->abcd', t2, v2, t2))[np.ix_(VI, VI, VI, VI)]
    vt['abcd'] += (-np.einsum('ecmn,oamd,beno->abcd', t2, v2, t2))[np.ix_(VI, VI, VI, VI)]
    vt['abcd'] += (np.einsum('edmn,oamc,beno->abcd', t2, v2, t2))[np.ix_(VI, VI, VI, VI)]
    vt['abcd'] += ((1/2) * np.einsum('cm,nbde,aemn->abcd', t1, v2, t2))[np.ix_(VI, VI, VI, VI)]  # [T_int=0]
    vt['abcd'] += (-np.einsum('dcmn,meab,en->abcd', t2, v2, t1))[np.ix_(VI, VI, VI, VI)]  # [T_int=0]
    vt['abcd'] += ((1/2) * np.einsum('ecmn,mdab,en->abcd', t2, v2, t1))[np.ix_(VI, VI, VI, VI)]
    vt['abcd'] += (-np.einsum('em,necd,abmn->abcd', t1, v2, t2))[np.ix_(VI, VI, VI, VI)]  # [T_int=0]
    vt['abcd'] += ((-1/2) * np.einsum('edmn,mcbe,an->abcd', t2, v2, t1))[np.ix_(VI, VI, VI, VI)]  # [T_int=0]
    vt['abcd'] += ((-1/2) * np.einsum('edmn,mcab,en->abcd', t2, v2, t1))[np.ix_(VI, VI, VI, VI)]
    vt['abcd'] += ((1/2) * np.einsum('dm,nomc,abno->abcd', t1, v2, t2))[np.ix_(VI, VI, VI, VI)]  # [T_int=0]
    vt['abcd'] += ((-1/2) * np.einsum('cm,nomd,abno->abcd', t1, v2, t2))[np.ix_(VI, VI, VI, VI)]  # [T_int=0]
    vt['abcd'] += ((-1/2) * np.einsum('cm,nade,bemn->abcd', t1, v2, t2))[np.ix_(VI, VI, VI, VI)]  # [T_int=0]
    vt['abcd'] += ((1/2) * np.einsum('ecmn,mdbe,an->abcd', t2, v2, t1))[np.ix_(VI, VI, VI, VI)]  # [T_int=0]
    vt['abcd'] += ((1/2) * np.einsum('edmn,mcae,bn->abcd', t2, v2, t1))[np.ix_(VI, VI, VI, VI)]  # [T_int=0]
    vt['abcd'] += ((-1/2) * np.einsum('ecmn,mdae,bn->abcd', t2, v2, t1))[np.ix_(VI, VI, VI, VI)]  # [T_int=0]
    vt['abcd'] += ((1/2) * np.einsum('em,nbcd,aemn->abcd', t1, v2, t2))[np.ix_(VI, VI, VI, VI)]
    vt['abcd'] += ((-1/2) * np.einsum('em,nacd,bemn->abcd', t1, v2, t2))[np.ix_(VI, VI, VI, VI)]
    vt['abcd'] += ((-1/2) * np.einsum('dm,nbce,aemn->abcd', t1, v2, t2))[np.ix_(VI, VI, VI, VI)]  # [T_int=0]
    vt['abcd'] += ((1/2) * np.einsum('dm,nace,bemn->abcd', t1, v2, t2))[np.ix_(VI, VI, VI, VI)]  # [T_int=0]
    vt['abcd'] += ((1/2) * np.einsum('dcmn,nmoa,bo->abcd', t2, v2, t1))[np.ix_(VI, VI, VI, VI)]  # [T_int=0]
    vt['abcd'] += ((-1/2) * np.einsum('dcmn,nmob,ao->abcd', t2, v2, t1))[np.ix_(VI, VI, VI, VI)]  # [T_int=0]
    vt['abcd'] += ((-1/2) * np.einsum('cm,abed,em->abcd', t1, v2, t1))[np.ix_(VI, VI, VI, VI)]  # [T_int=0]
    vt['abcd'] += (-np.einsum('mndc,am,bn->abcd', v2, t1, t1))[np.ix_(VI, VI, VI, VI)]  # [T_int=0]
    vt['abcd'] += (-np.einsum('dm,cn,mnab->abcd', t1, t1, v2))[np.ix_(VI, VI, VI, VI)]  # [T_int=0]
    vt['abcd'] += ((1/2) * np.einsum('dm,abec,em->abcd', t1, v2, t1))[np.ix_(VI, VI, VI, VI)]  # [T_int=0]
    vt['abcd'] += ((1/2) * np.einsum('em,aedc,bm->abcd', t1, v2, t1))[np.ix_(VI, VI, VI, VI)]  # [T_int=0]
    vt['abcd'] += ((-1/2) * np.einsum('em,bedc,am->abcd', t1, v2, t1))[np.ix_(VI, VI, VI, VI)]  # [T_int=0]
    vt['abcd'] += (np.einsum('cm,nbmd,an->abcd', t1, v2, t1))[np.ix_(VI, VI, VI, VI)]  # [T_int=0]
    vt['abcd'] += (-np.einsum('cm,namd,bn->abcd', t1, v2, t1))[np.ix_(VI, VI, VI, VI)]  # [T_int=0]
    vt['abcd'] += (-np.einsum('dm,nbmc,an->abcd', t1, v2, t1))[np.ix_(VI, VI, VI, VI)]  # [T_int=0]
    vt['abcd'] += (np.einsum('dm,namc,bn->abcd', t1, v2, t1))[np.ix_(VI, VI, VI, VI)]  # [T_int=0]

    # --- HHHP/HPHH ---
    vt['ijka'] += (np.einsum('efmk,mine,afjn->ijka', t2, v2, t2))[np.ix_(OI, OI, OI, VI)]
    vt['ijka'] += ((-1/2) * np.einsum('efmk,mgef,agij->ijka', t2, v2, t2))[np.ix_(OI, OI, OI, VI)]
    vt['ijka'] += ((-1/2) * np.einsum('efmk,mina,efjn->ijka', t2, v2, t2))[np.ix_(OI, OI, OI, VI)]
    vt['ijka'] += (-np.einsum('efmk,mgae,fgij->ijka', t2, v2, t2))[np.ix_(OI, OI, OI, VI)]
    vt['ijka'] += ((1/4) * np.einsum('efmk,jina,efmn->ijka', t2, v2, t2))[np.ix_(OI, OI, OI, VI)]
    vt['ijka'] += (-np.einsum('efmk,jine,afmn->ijka', t2, v2, t2))[np.ix_(OI, OI, OI, VI)]
    vt['ijka'] += ((-1/2) * np.einsum('efmn,mike,afjn->ijka', t2, v2, t2))[np.ix_(OI, OI, OI, VI)]
    vt['ijka'] += (-np.einsum('efmk,mjne,afin->ijka', t2, v2, t2))[np.ix_(OI, OI, OI, VI)]
    vt['ijka'] += ((1/2) * np.einsum('efmn,mjke,afin->ijka', t2, v2, t2))[np.ix_(OI, OI, OI, VI)]
    vt['ijka'] += ((1/4) * np.einsum('efmn,mika,efjn->ijka', t2, v2, t2))[np.ix_(OI, OI, OI, VI)]
    vt['ijka'] += ((1/2) * np.einsum('efmk,mjna,efin->ijka', t2, v2, t2))[np.ix_(OI, OI, OI, VI)]
    vt['ijka'] += ((-1/4) * np.einsum('efmn,mjka,efin->ijka', t2, v2, t2))[np.ix_(OI, OI, OI, VI)]
    vt['ijka'] += ((1/4) * np.einsum('efmn,jike,afmn->ijka', t2, v2, t2))[np.ix_(OI, OI, OI, VI)]
    vt['ijka'] += ((1/4) * np.einsum('efmn,nmke,afij->ijka', t2, v2, t2))[np.ix_(OI, OI, OI, VI)]
    vt['ijka'] += ((-1/2) * np.einsum('efmk,igef,agjm->ijka', t2, v2, t2))[np.ix_(OI, OI, OI, VI)]
    vt['ijka'] += (-np.einsum('efmk,igae,fgjm->ijka', t2, v2, t2))[np.ix_(OI, OI, OI, VI)]
    vt['ijka'] += (np.einsum('efmk,jgae,fgim->ijka', t2, v2, t2))[np.ix_(OI, OI, OI, VI)]
    vt['ijka'] += ((-1/8) * np.einsum('efmn,nmka,efij->ijka', t2, v2, t2))[np.ix_(OI, OI, OI, VI)]
    vt['ijka'] += ((1/2) * np.einsum('efmk,jgef,agim->ijka', t2, v2, t2))[np.ix_(OI, OI, OI, VI)]
    vt['ijka'] += (-np.einsum('mkef,em,afij->ijka', v2, t1, t2))[np.ix_(OI, OI, OI, VI)]
    vt['ijka'] += (np.einsum('em,nkim,aejn->ijka', t1, v2, t2))[np.ix_(OI, OI, OI, VI)]
    vt['ijka'] += ((1/2) * np.einsum('efmk,ijae,fm->ijka', t2, v2, t1))[np.ix_(OI, OI, OI, VI)]
    vt['ijka'] += ((1/2) * np.einsum('mkef,am,efij->ijka', v2, t1, t2))[np.ix_(OI, OI, OI, VI)]  # [T_int=0]
    vt['ijka'] += (-np.einsum('em,kamf,efij->ijka', t1, v2, t2))[np.ix_(OI, OI, OI, VI)]
    vt['ijka'] += ((1/2) * np.einsum('em,kajf,efim->ijka', t1, v2, t2))[np.ix_(OI, OI, OI, VI)]
    vt['ijka'] += ((1/4) * np.einsum('efmk,ijef,am->ijka', t2, v2, t1))[np.ix_(OI, OI, OI, VI)]  # [T_int=0]
    vt['ijka'] += (-np.einsum('em,kejf,afim->ijka', t1, v2, t2))[np.ix_(OI, OI, OI, VI)]
    vt['ijka'] += (-np.einsum('em,nkjm,aein->ijka', t1, v2, t2))[np.ix_(OI, OI, OI, VI)]
    vt['ijka'] += (-np.einsum('ek,meif,afjm->ijka', t1, v2, t2))[np.ix_(OI, OI, OI, VI)]
    vt['ijka'] += (np.einsum('em,keif,afjm->ijka', t1, v2, t2))[np.ix_(OI, OI, OI, VI)]
    vt['ijka'] += ((-1/2) * np.einsum('em,nkij,aemn->ijka', t1, v2, t2))[np.ix_(OI, OI, OI, VI)]
    vt['ijka'] += ((1/2) * np.einsum('ek,maif,efjm->ijka', t1, v2, t2))[np.ix_(OI, OI, OI, VI)]
    vt['ijka'] += ((-1/2) * np.einsum('ek,majf,efim->ijka', t1, v2, t2))[np.ix_(OI, OI, OI, VI)]
    vt['ijka'] += (np.einsum('ek,mejf,afim->ijka', t1, v2, t2))[np.ix_(OI, OI, OI, VI)]
    vt['ijka'] += ((-1/2) * np.einsum('em,kaif,efjm->ijka', t1, v2, t2))[np.ix_(OI, OI, OI, VI)]
    vt['ijka'] += ((-1/4) * np.einsum('ek,mnij,aemn->ijka', t1, v2, t2))[np.ix_(OI, OI, OI, VI)]
    vt['ijka'] += (np.einsum('em,kemf,afij->ijka', t1, v2, t2))[np.ix_(OI, OI, OI, VI)]
    vt['ijka'] += ((-1/2) * np.einsum('ek,aefg,fgij->ijka', t1, v2, t2))[np.ix_(OI, OI, OI, VI)]
    vt['ijka'] += ((1/2) * np.einsum('efmk,jmae,fi->ijka', t2, v2, t1))[np.ix_(OI, OI, OI, VI)]
    vt['ijka'] += ((-1/2) * np.einsum('efmk,imae,fj->ijka', t2, v2, t1))[np.ix_(OI, OI, OI, VI)]
    vt['ijka'] += (-np.einsum('mkef,ei,afjm->ijka', v2, t1, t2))[np.ix_(OI, OI, OI, VI)]
    vt['ijka'] += (np.einsum('mkef,ej,afim->ijka', v2, t1, t2))[np.ix_(OI, OI, OI, VI)]
    vt['ijka'] += (-np.einsum('kafe,ei,fj->ijka', v2, t1, t1))[np.ix_(OI, OI, OI, VI)]
    vt['ijka'] += (-np.einsum('mkje,ei,am->ijka', v2, t1, t1))[np.ix_(OI, OI, OI, VI)]  # [T_int=0]
    vt['ijka'] += (np.einsum('mkie,am,ej->ijka', v2, t1, t1))[np.ix_(OI, OI, OI, VI)]  # [T_int=0]
    vt['ijka'] += ((1/2) * np.einsum('ek,jima,em->ijka', t1, v2, t1))[np.ix_(OI, OI, OI, VI)]
    vt['ijka'] += (-np.einsum('ek,ifae,fj->ijka', t1, v2, t1))[np.ix_(OI, OI, OI, VI)]
    vt['ijka'] += (-np.einsum('ek,jime,am->ijka', t1, v2, t1))[np.ix_(OI, OI, OI, VI)]  # [T_int=0]
    vt['ijka'] += ((1/2) * np.einsum('em,jike,am->ijka', t1, v2, t1))[np.ix_(OI, OI, OI, VI)]  # [T_int=0]
    vt['ijka'] += ((-1/2) * np.einsum('em,mjka,ei->ijka', t1, v2, t1))[np.ix_(OI, OI, OI, VI)]
    vt['ijka'] += (np.einsum('ek,jfae,fi->ijka', t1, v2, t1))[np.ix_(OI, OI, OI, VI)]
    vt['ijka'] += ((1/2) * np.einsum('em,mika,ej->ijka', t1, v2, t1))[np.ix_(OI, OI, OI, VI)]

    # --- PPPH/PHPP ---
    vt['iabc'] += ((-1/2) * np.einsum('efmn,mace,bfin->iabc', t2, v2, t2))[np.ix_(OI, VI, VI, VI)]
    vt['iabc'] += (-np.einsum('eamn,mioe,cbno->iabc', t2, v2, t2))[np.ix_(OI, VI, VI, VI)]
    vt['iabc'] += ((1/2) * np.einsum('eamn,ifbe,cfmn->iabc', t2, v2, t2))[np.ix_(OI, VI, VI, VI)]
    vt['iabc'] += (-np.einsum('eamn,mioc,beno->iabc', t2, v2, t2))[np.ix_(OI, VI, VI, VI)]
    vt['iabc'] += ((-1/2) * np.einsum('eamn,nmoc,beio->iabc', t2, v2, t2))[np.ix_(OI, VI, VI, VI)]
    vt['iabc'] += ((1/2) * np.einsum('eamn,nmob,ceio->iabc', t2, v2, t2))[np.ix_(OI, VI, VI, VI)]
    vt['iabc'] += ((1/4) * np.einsum('efmn,iace,bfmn->iabc', t2, v2, t2))[np.ix_(OI, VI, VI, VI)]
    vt['iabc'] += (-np.einsum('eamn,mfbe,cfin->iabc', t2, v2, t2))[np.ix_(OI, VI, VI, VI)]
    vt['iabc'] += ((-1/2) * np.einsum('eamn,nmoe,cbio->iabc', t2, v2, t2))[np.ix_(OI, VI, VI, VI)]
    vt['iabc'] += (-np.einsum('eamn,mfcb,efin->iabc', t2, v2, t2))[np.ix_(OI, VI, VI, VI)]
    vt['iabc'] += (np.einsum('eamn,miob,ceno->iabc', t2, v2, t2))[np.ix_(OI, VI, VI, VI)]
    vt['iabc'] += (np.einsum('eamn,mfce,bfin->iabc', t2, v2, t2))[np.ix_(OI, VI, VI, VI)]
    vt['iabc'] += ((1/4) * np.einsum('eamn,ifcb,efmn->iabc', t2, v2, t2))[np.ix_(OI, VI, VI, VI)]
    vt['iabc'] += ((1/4) * np.einsum('efmn,macb,efin->iabc', t2, v2, t2))[np.ix_(OI, VI, VI, VI)]
    vt['iabc'] += ((1/2) * np.einsum('efmn,mabe,cfin->iabc', t2, v2, t2))[np.ix_(OI, VI, VI, VI)]
    vt['iabc'] += ((-1/2) * np.einsum('eamn,ifce,bfmn->iabc', t2, v2, t2))[np.ix_(OI, VI, VI, VI)]
    vt['iabc'] += ((1/4) * np.einsum('efmn,maef,cbin->iabc', t2, v2, t2))[np.ix_(OI, VI, VI, VI)]
    vt['iabc'] += ((-1/4) * np.einsum('efmn,iabe,cfmn->iabc', t2, v2, t2))[np.ix_(OI, VI, VI, VI)]
    vt['iabc'] += ((-1/8) * np.einsum('efmn,iaef,cbmn->iabc', t2, v2, t2))[np.ix_(OI, VI, VI, VI)]  # [T_int=0]
    vt['iabc'] += ((1/2) * np.einsum('em,cbfa,efim->iabc', t1, v2, t2))[np.ix_(OI, VI, VI, VI)]
    vt['iabc'] += (np.einsum('mnea,em,cbin->iabc', v2, t1, t2))[np.ix_(OI, VI, VI, VI)]
    vt['iabc'] += ((-1/4) * np.einsum('eamn,mncb,ei->iabc', t2, v2, t1))[np.ix_(OI, VI, VI, VI)]
    vt['iabc'] += ((-1/2) * np.einsum('eamn,imcb,en->iabc', t2, v2, t1))[np.ix_(OI, VI, VI, VI)]
    vt['iabc'] += ((1/4) * np.einsum('am,cbef,efim->iabc', t1, v2, t2))[np.ix_(OI, VI, VI, VI)]  # [T_int=0]
    vt['iabc'] += ((-1/2) * np.einsum('mnea,ei,cbmn->iabc', v2, t1, t2))[np.ix_(OI, VI, VI, VI)]  # [T_int=0]
    vt['iabc'] += (np.einsum('em,neia,cbmn->iabc', t1, v2, t2))[np.ix_(OI, VI, VI, VI)]  # [T_int=0]
    vt['iabc'] += ((1/2) * np.einsum('am,nbie,cemn->iabc', t1, v2, t2))[np.ix_(OI, VI, VI, VI)]  # [T_int=0]
    vt['iabc'] += ((-1/2) * np.einsum('am,ncie,bemn->iabc', t1, v2, t2))[np.ix_(OI, VI, VI, VI)]  # [T_int=0]
    vt['iabc'] += (np.einsum('em,befa,cfim->iabc', t1, v2, t2))[np.ix_(OI, VI, VI, VI)]
    vt['iabc'] += ((1/2) * np.einsum('am,noim,cbno->iabc', t1, v2, t2))[np.ix_(OI, VI, VI, VI)]  # [T_int=0]
    vt['iabc'] += (-np.einsum('em,ncma,bein->iabc', t1, v2, t2))[np.ix_(OI, VI, VI, VI)]
    vt['iabc'] += (np.einsum('am,ncme,bein->iabc', t1, v2, t2))[np.ix_(OI, VI, VI, VI)]  # [T_int=0]
    vt['iabc'] += (-np.einsum('em,nema,cbin->iabc', t1, v2, t2))[np.ix_(OI, VI, VI, VI)]
    vt['iabc'] += (-np.einsum('am,nbme,cein->iabc', t1, v2, t2))[np.ix_(OI, VI, VI, VI)]  # [T_int=0]
    vt['iabc'] += (np.einsum('em,nbma,cein->iabc', t1, v2, t2))[np.ix_(OI, VI, VI, VI)]
    vt['iabc'] += (-np.einsum('em,cefa,bfim->iabc', t1, v2, t2))[np.ix_(OI, VI, VI, VI)]
    vt['iabc'] += ((1/2) * np.einsum('em,ncia,bemn->iabc', t1, v2, t2))[np.ix_(OI, VI, VI, VI)]
    vt['iabc'] += ((-1/2) * np.einsum('em,nbia,cemn->iabc', t1, v2, t2))[np.ix_(OI, VI, VI, VI)]
    vt['iabc'] += (np.einsum('mnea,cm,bein->iabc', v2, t1, t2))[np.ix_(OI, VI, VI, VI)]  # [T_int=0]
    vt['iabc'] += (-np.einsum('mnea,bm,cein->iabc', v2, t1, t2))[np.ix_(OI, VI, VI, VI)]  # [T_int=0]
    vt['iabc'] += ((1/2) * np.einsum('eamn,imce,bn->iabc', t2, v2, t1))[np.ix_(OI, VI, VI, VI)]  # [T_int=0]
    vt['iabc'] += ((-1/2) * np.einsum('eamn,imbe,cn->iabc', t2, v2, t1))[np.ix_(OI, VI, VI, VI)]  # [T_int=0]
    vt['iabc'] += (-np.einsum('mnia,cm,bn->iabc', v2, t1, t1))[np.ix_(OI, VI, VI, VI)]  # [T_int=0]
    vt['iabc'] += ((-1/2) * np.einsum('em,iabe,cm->iabc', t1, v2, t1))[np.ix_(OI, VI, VI, VI)]  # [T_int=0]
    vt['iabc'] += ((1/2) * np.einsum('em,iace,bm->iabc', t1, v2, t1))[np.ix_(OI, VI, VI, VI)]  # [T_int=0]
    vt['iabc'] += (-np.einsum('mbae,cm,ei->iabc', v2, t1, t1))[np.ix_(OI, VI, VI, VI)]  # [T_int=0]
    vt['iabc'] += (np.einsum('mcae,ei,bm->iabc', v2, t1, t1))[np.ix_(OI, VI, VI, VI)]  # [T_int=0]
    vt['iabc'] += (np.einsum('am,minb,cn->iabc', t1, v2, t1))[np.ix_(OI, VI, VI, VI)]  # [T_int=0]
    vt['iabc'] += (-np.einsum('am,minc,bn->iabc', t1, v2, t1))[np.ix_(OI, VI, VI, VI)]  # [T_int=0]
    vt['iabc'] += ((1/2) * np.einsum('em,macb,ei->iabc', t1, v2, t1))[np.ix_(OI, VI, VI, VI)]
    vt['iabc'] += (-np.einsum('am,mecb,ei->iabc', t1, v2, t1))[np.ix_(OI, VI, VI, VI)]  # [T_int=0]
    vt['iabc'] += ((1/2) * np.einsum('am,iecb,em->iabc', t1, v2, t1))[np.ix_(OI, VI, VI, VI)]  # [T_int=0]

    # --- HHPP/PPHH ---
    vt['ijab'] += (-np.einsum('mnef,beim,afjn->ijab', v2, t2, t2))[np.ix_(OI, OI, VI, VI)]
    vt['ijab'] += ((1/2) * np.einsum('efmn,imae,bfjn->ijab', t2, v2, t2))[np.ix_(OI, OI, VI, VI)]
    vt['ijab'] += ((1/2) * np.einsum('efmn,jmbe,afin->ijab', t2, v2, t2))[np.ix_(OI, OI, VI, VI)]
    vt['ijab'] += ((-1/2) * np.einsum('mnef,abim,efjn->ijab', v2, t2, t2))[np.ix_(OI, OI, VI, VI)]  # [T_int=0]
    vt['ijab'] += ((-1/2) * np.einsum('efmn,imbe,afjn->ijab', t2, v2, t2))[np.ix_(OI, OI, VI, VI)]
    vt['ijab'] += ((-1/2) * np.einsum('mnef,efim,abjn->ijab', v2, t2, t2))[np.ix_(OI, OI, VI, VI)]
    vt['ijab'] += ((-1/4) * np.einsum('efmn,imef,abjn->ijab', t2, v2, t2))[np.ix_(OI, OI, VI, VI)]
    vt['ijab'] += ((-1/2) * np.einsum('efmn,jmae,bfin->ijab', t2, v2, t2))[np.ix_(OI, OI, VI, VI)]
    vt['ijab'] += ((-1/4) * np.einsum('efmn,mnae,bfij->ijab', t2, v2, t2))[np.ix_(OI, OI, VI, VI)]
    vt['ijab'] += ((1/2) * np.einsum('mnef,beij,afmn->ijab', v2, t2, t2))[np.ix_(OI, OI, VI, VI)]
    vt['ijab'] += ((1/4) * np.einsum('efmn,mnbe,afij->ijab', t2, v2, t2))[np.ix_(OI, OI, VI, VI)]
    vt['ijab'] += ((-1/2) * np.einsum('mnef,aeij,bfmn->ijab', v2, t2, t2))[np.ix_(OI, OI, VI, VI)]
    vt['ijab'] += ((1/4) * np.einsum('efmn,jmef,abin->ijab', t2, v2, t2))[np.ix_(OI, OI, VI, VI)]
    vt['ijab'] += (np.einsum('mnef,aeim,bfjn->ijab', v2, t2, t2))[np.ix_(OI, OI, VI, VI)]
    vt['ijab'] += ((-1/4) * np.einsum('efmn,imab,efjn->ijab', t2, v2, t2))[np.ix_(OI, OI, VI, VI)]
    vt['ijab'] += ((1/4) * np.einsum('efmn,jmab,efin->ijab', t2, v2, t2))[np.ix_(OI, OI, VI, VI)]
    vt['ijab'] += ((-1/4) * np.einsum('efmn,ijae,bfmn->ijab', t2, v2, t2))[np.ix_(OI, OI, VI, VI)]
    vt['ijab'] += ((1/4) * np.einsum('efmn,ijbe,afmn->ijab', t2, v2, t2))[np.ix_(OI, OI, VI, VI)]
    vt['ijab'] += ((1/4) * np.einsum('mnef,efij,abmn->ijab', v2, t2, t2))[np.ix_(OI, OI, VI, VI)]  # [T_int=0]
    vt['ijab'] += ((1/8) * np.einsum('efmn,mnab,efij->ijab', t2, v2, t2))[np.ix_(OI, OI, VI, VI)]
    vt['ijab'] += ((1/8) * np.einsum('efmn,ijef,abmn->ijab', t2, v2, t2))[np.ix_(OI, OI, VI, VI)]  # [T_int=0]
    vt['ijab'] += (np.einsum('em,ifae,bfjm->ijab', t1, v2, t2))[np.ix_(OI, OI, VI, VI)]
    vt['ijab'] += (-np.einsum('em,ifbe,afjm->ijab', t1, v2, t2))[np.ix_(OI, OI, VI, VI)]
    vt['ijab'] += (-np.einsum('mnie,em,abjn->ijab', v2, t1, t2))[np.ix_(OI, OI, VI, VI)]
    vt['ijab'] += (np.einsum('mnje,em,abin->ijab', v2, t1, t2))[np.ix_(OI, OI, VI, VI)]
    vt['ijab'] += (np.einsum('mnie,bm,aejn->ijab', v2, t1, t2))[np.ix_(OI, OI, VI, VI)]  # [T_int=0]
    vt['ijab'] += (np.einsum('em,mfae,bfij->ijab', t1, v2, t2))[np.ix_(OI, OI, VI, VI)]
    vt['ijab'] += ((1/2) * np.einsum('em,jina,bemn->ijab', t1, v2, t2))[np.ix_(OI, OI, VI, VI)]
    vt['ijab'] += (np.einsum('em,mjne,abin->ijab', t1, v2, t2))[np.ix_(OI, OI, VI, VI)]
    vt['ijab'] += ((-1/2) * np.einsum('em,jinb,aemn->ijab', t1, v2, t2))[np.ix_(OI, OI, VI, VI)]
    vt['ijab'] += ((-1/2) * np.einsum('mafe,bm,efij->ijab', v2, t1, t2))[np.ix_(OI, OI, VI, VI)]  # [T_int=0]
    vt['ijab'] += (-np.einsum('mnie,am,bejn->ijab', v2, t1, t2))[np.ix_(OI, OI, VI, VI)]  # [T_int=0]
    vt['ijab'] += (-np.einsum('em,mfbe,afij->ijab', t1, v2, t2))[np.ix_(OI, OI, VI, VI)]
    vt['ijab'] += ((-1/2) * np.einsum('mnje,ei,abmn->ijab', v2, t1, t2))[np.ix_(OI, OI, VI, VI)]  # [T_int=0]
    vt['ijab'] += (-np.einsum('mbfe,ei,afjm->ijab', v2, t1, t2))[np.ix_(OI, OI, VI, VI)]
    vt['ijab'] += (np.einsum('mnje,am,bein->ijab', v2, t1, t2))[np.ix_(OI, OI, VI, VI)]  # [T_int=0]
    vt['ijab'] += ((1/2) * np.einsum('em,jfab,efim->ijab', t1, v2, t2))[np.ix_(OI, OI, VI, VI)]
    vt['ijab'] += (np.einsum('em,jine,abmn->ijab', t1, v2, t2))[np.ix_(OI, OI, VI, VI)]  # [T_int=0]
    vt['ijab'] += (-np.einsum('em,jfae,bfim->ijab', t1, v2, t2))[np.ix_(OI, OI, VI, VI)]
    vt['ijab'] += (np.einsum('em,mjna,bein->ijab', t1, v2, t2))[np.ix_(OI, OI, VI, VI)]
    vt['ijab'] += (-np.einsum('em,mfab,efij->ijab', t1, v2, t2))[np.ix_(OI, OI, VI, VI)]
    vt['ijab'] += (-np.einsum('em,mine,abjn->ijab', t1, v2, t2))[np.ix_(OI, OI, VI, VI)]
    vt['ijab'] += (-np.einsum('em,mina,bejn->ijab', t1, v2, t2))[np.ix_(OI, OI, VI, VI)]
    vt['ijab'] += ((-1/2) * np.einsum('em,ifab,efjm->ijab', t1, v2, t2))[np.ix_(OI, OI, VI, VI)]
    vt['ijab'] += ((1/2) * np.einsum('mbfe,am,efij->ijab', v2, t1, t2))[np.ix_(OI, OI, VI, VI)]  # [T_int=0]
    vt['ijab'] += (np.einsum('mafe,em,bfij->ijab', v2, t1, t2))[np.ix_(OI, OI, VI, VI)]
    vt['ijab'] += (-np.einsum('mnje,bm,aein->ijab', v2, t1, t2))[np.ix_(OI, OI, VI, VI)]  # [T_int=0]
    vt['ijab'] += ((1/2) * np.einsum('mnie,ej,abmn->ijab', v2, t1, t2))[np.ix_(OI, OI, VI, VI)]  # [T_int=0]
    vt['ijab'] += (-np.einsum('em,mjnb,aein->ijab', t1, v2, t2))[np.ix_(OI, OI, VI, VI)]
    vt['ijab'] += (np.einsum('em,minb,aejn->ijab', t1, v2, t2))[np.ix_(OI, OI, VI, VI)]
    vt['ijab'] += (np.einsum('em,jfbe,afim->ijab', t1, v2, t2))[np.ix_(OI, OI, VI, VI)]
    vt['ijab'] += (np.einsum('mafe,ei,bfjm->ijab', v2, t1, t2))[np.ix_(OI, OI, VI, VI)]
    vt['ijab'] += (np.einsum('mbfe,ej,afim->ijab', v2, t1, t2))[np.ix_(OI, OI, VI, VI)]
    vt['ijab'] += (-np.einsum('mbfe,em,afij->ijab', v2, t1, t2))[np.ix_(OI, OI, VI, VI)]
    vt['ijab'] += (-np.einsum('mafe,ej,bfim->ijab', v2, t1, t2))[np.ix_(OI, OI, VI, VI)]
    vt['ijab'] += (np.einsum('mnij,am,bn->ijab', v2, t1, t1))[np.ix_(OI, OI, VI, VI)]  # [T_int=0]
    vt['ijab'] += (np.einsum('abef,ei,fj->ijab', v2, t1, t1))[np.ix_(OI, OI, VI, VI)]
    vt['ijab'] += ((-1/2) * np.einsum('em,ijae,bm->ijab', t1, v2, t1))[np.ix_(OI, OI, VI, VI)]  # [T_int=0]
    vt['ijab'] += ((1/2) * np.einsum('em,ijbe,am->ijab', t1, v2, t1))[np.ix_(OI, OI, VI, VI)]  # [T_int=0]
    vt['ijab'] += ((1/2) * np.einsum('em,jmab,ei->ijab', t1, v2, t1))[np.ix_(OI, OI, VI, VI)]
    vt['ijab'] += ((-1/2) * np.einsum('em,imab,ej->ijab', t1, v2, t1))[np.ix_(OI, OI, VI, VI)]
    vt['ijab'] += (np.einsum('maie,ej,bm->ijab', v2, t1, t1))[np.ix_(OI, OI, VI, VI)]  # [T_int=0]
    vt['ijab'] += (np.einsum('mbje,ei,am->ijab', v2, t1, t1))[np.ix_(OI, OI, VI, VI)]  # [T_int=0]
    vt['ijab'] += (-np.einsum('mbie,am,ej->ijab', v2, t1, t1))[np.ix_(OI, OI, VI, VI)]  # [T_int=0]
    vt['ijab'] += (-np.einsum('maje,ei,bm->ijab', v2, t1, t1))[np.ix_(OI, OI, VI, VI)]  # [T_int=0]

    # --- PHHP ---
    vt['aijb'] += (-np.einsum('ebmi,nemf,afjn->aijb', t2, v2, t2))[np.ix_(VI, OI, OI, VI)]
    vt['aijb'] += (np.einsum('ebmi,nejf,afmn->aijb', t2, v2, t2))[np.ix_(VI, OI, OI, VI)]
    vt['aijb'] += (np.einsum('ebmn,iemf,afjn->aijb', t2, v2, t2))[np.ix_(VI, OI, OI, VI)]
    vt['aijb'] += (np.einsum('efmi,nemb,afjn->aijb', t2, v2, t2))[np.ix_(VI, OI, OI, VI)]
    vt['aijb'] += ((-1/2) * np.einsum('ebmn,iejf,afmn->aijb', t2, v2, t2))[np.ix_(VI, OI, OI, VI)]
    vt['aijb'] += (np.einsum('ebmi,namf,efjn->aijb', t2, v2, t2))[np.ix_(VI, OI, OI, VI)]
    vt['aijb'] += ((-1/2) * np.einsum('efmi,namb,efjn->aijb', t2, v2, t2))[np.ix_(VI, OI, OI, VI)]
    vt['aijb'] += (np.einsum('ebmn,oijm,aeno->aijb', t2, v2, t2))[np.ix_(VI, OI, OI, VI)]
    vt['aijb'] += ((1/2) * np.einsum('ebmn,oimn,aejo->aijb', t2, v2, t2))[np.ix_(VI, OI, OI, VI)]
    vt['aijb'] += ((1/2) * np.einsum('ebmi,nojm,aeno->aijb', t2, v2, t2))[np.ix_(VI, OI, OI, VI)]
    vt['aijb'] += (np.einsum('efmi,aegb,fgjm->aijb', t2, v2, t2))[np.ix_(VI, OI, OI, VI)]
    vt['aijb'] += ((1/2) * np.einsum('efmi,efgb,agjm->aijb', t2, v2, t2))[np.ix_(VI, OI, OI, VI)]
    vt['aijb'] += (-np.einsum('efmi,nejb,afmn->aijb', t2, v2, t2))[np.ix_(VI, OI, OI, VI)]
    vt['aijb'] += ((-1/2) * np.einsum('efmn,iemb,afjn->aijb', t2, v2, t2))[np.ix_(VI, OI, OI, VI)]
    vt['aijb'] += ((-1/2) * np.einsum('ebmi,najf,efmn->aijb', t2, v2, t2))[np.ix_(VI, OI, OI, VI)]
    vt['aijb'] += (-np.einsum('ebmn,iamf,efjn->aijb', t2, v2, t2))[np.ix_(VI, OI, OI, VI)]
    vt['aijb'] += ((1/2) * np.einsum('ebmi,aefg,fgjm->aijb', t2, v2, t2))[np.ix_(VI, OI, OI, VI)]
    vt['aijb'] += ((1/4) * np.einsum('efmn,iejb,afmn->aijb', t2, v2, t2))[np.ix_(VI, OI, OI, VI)]
    vt['aijb'] += ((1/4) * np.einsum('efmi,najb,efmn->aijb', t2, v2, t2))[np.ix_(VI, OI, OI, VI)]
    vt['aijb'] += ((1/4) * np.einsum('ebmn,iajf,efmn->aijb', t2, v2, t2))[np.ix_(VI, OI, OI, VI)]
    vt['aijb'] += ((1/4) * np.einsum('efmn,iamb,efjn->aijb', t2, v2, t2))[np.ix_(VI, OI, OI, VI)]
    vt['aijb'] += ((1/2) * np.einsum('ei,mabf,efjm->aijb', t1, v2, t2))[np.ix_(VI, OI, OI, VI)]
    vt['aijb'] += (-np.einsum('ei,mebf,afjm->aijb', t1, v2, t2))[np.ix_(VI, OI, OI, VI)]
    vt['aijb'] += ((-1/4) * np.einsum('efmi,jbef,am->aijb', t2, v2, t1))[np.ix_(VI, OI, OI, VI)]  # [T_int=0]
    vt['aijb'] += (np.einsum('bm,nime,aejn->aijb', t1, v2, t2))[np.ix_(VI, OI, OI, VI)]  # [T_int=0]
    vt['aijb'] += ((-1/2) * np.einsum('bm,nije,aemn->aijb', t1, v2, t2))[np.ix_(VI, OI, OI, VI)]  # [T_int=0]
    vt['aijb'] += (-np.einsum('ebmi,mjna,en->aijb', t2, v2, t1))[np.ix_(VI, OI, OI, VI)]
    vt['aijb'] += ((-1/2) * np.einsum('efmi,jbae,fm->aijb', t2, v2, t1))[np.ix_(VI, OI, OI, VI)]
    vt['aijb'] += ((1/2) * np.einsum('ebmn,mjia,en->aijb', t2, v2, t1))[np.ix_(VI, OI, OI, VI)]
    vt['aijb'] += ((1/2) * np.einsum('em,nijb,aemn->aijb', t1, v2, t2))[np.ix_(VI, OI, OI, VI)]
    vt['aijb'] += ((1/4) * np.einsum('ebmn,nmia,ej->aijb', t2, v2, t1))[np.ix_(VI, OI, OI, VI)]
    vt['aijb'] += (-np.einsum('em,nimb,aejn->aijb', t1, v2, t2))[np.ix_(VI, OI, OI, VI)]
    vt['aijb'] += (np.einsum('em,iebf,afjm->aijb', t1, v2, t2))[np.ix_(VI, OI, OI, VI)]
    vt['aijb'] += (-np.einsum('ebmi,mfae,fj->aijb', t2, v2, t1))[np.ix_(VI, OI, OI, VI)]
    vt['aijb'] += ((-1/2) * np.einsum('ebmn,mjie,an->aijb', t2, v2, t1))[np.ix_(VI, OI, OI, VI)]  # [T_int=0]
    vt['aijb'] += (np.einsum('ebmi,jfae,fm->aijb', t2, v2, t1))[np.ix_(VI, OI, OI, VI)]
    vt['aijb'] += ((-1/2) * np.einsum('em,iabf,efjm->aijb', t1, v2, t2))[np.ix_(VI, OI, OI, VI)]
    vt['aijb'] += (np.einsum('ebmi,mjne,an->aijb', t2, v2, t1))[np.ix_(VI, OI, OI, VI)]  # [T_int=0]
    vt['aijb'] += ((1/2) * np.einsum('efmi,mbae,fj->aijb', t2, v2, t1))[np.ix_(VI, OI, OI, VI)]
    vt['aijb'] += ((1/4) * np.einsum('ei,mnjb,aemn->aijb', t1, v2, t2))[np.ix_(VI, OI, OI, VI)]
    vt['aijb'] += ((-1/4) * np.einsum('bm,iafe,efjm->aijb', t1, v2, t2))[np.ix_(VI, OI, OI, VI)]  # [T_int=0]
    vt['aijb'] += (-np.einsum('bm,ei,jmae->aijb', t1, t1, v2))[np.ix_(VI, OI, OI, VI)]  # [T_int=0]
    vt['aijb'] += (np.einsum('bm,nijm,an->aijb', t1, v2, t1))[np.ix_(VI, OI, OI, VI)]  # [T_int=0]
    vt['aijb'] += (np.einsum('ei,aefb,fj->aijb', t1, v2, t1))[np.ix_(VI, OI, OI, VI)]
    vt['aijb'] += (-np.einsum('mieb,ej,am->aijb', v2, t1, t1))[np.ix_(VI, OI, OI, VI)]  # [T_int=0]
    vt['aijb'] += ((1/2) * np.einsum('em,iejb,am->aijb', t1, v2, t1))[np.ix_(VI, OI, OI, VI)]  # [T_int=0]
    vt['aijb'] += (-np.einsum('ei,mejb,am->aijb', t1, v2, t1))[np.ix_(VI, OI, OI, VI)]  # [T_int=0]
    vt['aijb'] += ((1/2) * np.einsum('ei,majb,em->aijb', t1, v2, t1))[np.ix_(VI, OI, OI, VI)]
    vt['aijb'] += ((1/2) * np.einsum('em,iamb,ej->aijb', t1, v2, t1))[np.ix_(VI, OI, OI, VI)]
    vt['aijb'] += ((1/2) * np.einsum('bm,iaje,em->aijb', t1, v2, t1))[np.ix_(VI, OI, OI, VI)]  # [T_int=0]
    vt['aijb'] += (-np.einsum('bm,iame,ej->aijb', t1, v2, t1))[np.ix_(VI, OI, OI, VI)]  # [T_int=0]

    return scalar

def f_3(
    ft, vt, idx,
    f1, t1, t2,
):
    """BCH commutator F_3: Fock operator f1.

    Auto-generated from ExaChem's ducc-t_ccsd.hpp.
    All tensors use full spin-orbital arrays in physicist notation.
    Output is accumulated into active-subspace views (ft, vt).

    Args:
        ft: dict with keys 'ij','ia','ab' -> views into active-space 1e blocks.
        vt: dict with keys 'ijkl','ijka','aijb','ijab','iabc','abcd' -> active 2e blocks.
        idx: dict with index slice arrays: 'O','V','OI','VI'.
        f1: full spin-orbital Fock matrix, shape (nso, nso).
        t1: T1 amplitudes (nso, nso), T_int zeroed.
        t2: T2 amplitudes (nso, nso, nso, nso), T_int zeroed.
    """
    OI, VI = idx['OI'], idx['VI']
    scalar = 0.0

    # --- FC ---
    # PARSE ERROR: (-2.0/3.0) * f1(m, e) * t1(f, n) * t1(f, m) * t1(e, n)
    # PARSE ERROR: (-2.0/3.0) * t1(e, m) * t1(f, n) * f1(m, f) * t1(e, n)
    # PARSE ERROR: (-1.0/2.0) * t1(e, m) * f1(n, m) * t1(f, o) * t2(e, f, n, o)
    # PARSE ERROR: (-1.0/2.0) * t2(e, f, m, n) * f1(e, g) * t1(f, m) * t1(g, n)
    # PARSE ERROR: (-1.0/3.0) * f1(m, e) * t2(f, g, n, o) * t1(f, m) * t2(e, g, n, o)
    # PARSE ERROR: (-1.0/3.0) * f1(m, e) * t2(f, g, n, o) * t1(e, n) * t2(f, g, m, o)
    # PARSE ERROR: (-1.0/3.0) * t1(e, m) * t2(f, g, n, o) * f1(m, f) * t2(e, g, n, o)
    # PARSE ERROR: (-1.0/3.0) * t1(e, m) * t2(f, g, n, o) * f1(n, e) * t2(f, g, m, o)
    scalar += (1/2) * np.einsum('em,ef,gn,fgmn', t1, f1, t1, t2)
    scalar += (1/2) * np.einsum('efmn,om,en,fo', t2, f1, t1, t1)
    scalar += (1/6) * np.einsum('me,fgno,fn,egmo', f1, t2, t1, t2)
    scalar += (1/6) * np.einsum('efmn,me,go,fgno', t2, f1, t1, t2)

    # --- HH ---
    ft['ij'] += ((1/6) * np.einsum('efmj,ng,em,fgin->ij', t2, f1, t1, t2))[np.ix_(OI, OI)]
    ft['ij'] += ((1/3) * np.einsum('efmj,ng,gm,efin->ij', t2, f1, t1, t2))[np.ix_(OI, OI)]
    ft['ij'] += ((-1/6) * np.einsum('em,fgnj,fn,egim->ij', t1, t2, f1, t2))[np.ix_(OI, OI)]
    ft['ij'] += ((1/3) * np.einsum('em,fgnj,en,fgim->ij', t1, t2, f1, t2))[np.ix_(OI, OI)]
    ft['ij'] += ((-1/6) * np.einsum('em,fgnj,fi,egmn->ij', t1, t2, f1, t2))[np.ix_(OI, OI)]
    ft['ij'] += ((2/3) * np.einsum('em,fgnj,fm,egin->ij', t1, t2, f1, t2))[np.ix_(OI, OI)]
    ft['ij'] += ((1/6) * np.einsum('ej,fgmn,fm,egin->ij', t1, t2, f1, t2))[np.ix_(OI, OI)]
    ft['ij'] += ((-1/6) * np.einsum('efmn,jg,em,fgin->ij', t2, f1, t1, t2))[np.ix_(OI, OI)]
    ft['ij'] += ((-1/6) * np.einsum('efmj,ng,ei,fgmn->ij', t2, f1, t1, t2))[np.ix_(OI, OI)]
    ft['ij'] += ((-2/3) * np.einsum('efmj,ng,en,fgim->ij', t2, f1, t1, t2))[np.ix_(OI, OI)]
    ft['ij'] += ((-1/12) * np.einsum('ej,fgmn,fi,egmn->ij', t1, t2, f1, t2))[np.ix_(OI, OI)]
    ft['ij'] += ((1/12) * np.einsum('em,fgnj,ei,fgmn->ij', t1, t2, f1, t2))[np.ix_(OI, OI)]
    ft['ij'] += ((-1/4) * np.einsum('ej,fgmn,em,fgin->ij', t1, t2, f1, t2))[np.ix_(OI, OI)]
    ft['ij'] += ((-1/12) * np.einsum('efmn,jg,gm,efin->ij', t2, f1, t1, t2))[np.ix_(OI, OI)]
    ft['ij'] += ((1/12) * np.einsum('efmn,jg,ei,fgmn->ij', t2, f1, t1, t2))[np.ix_(OI, OI)]
    ft['ij'] += ((-1/4) * np.einsum('efmj,ng,gi,efmn->ij', t2, f1, t1, t2))[np.ix_(OI, OI)]
    ft['ij'] += ((1/2) * np.einsum('efmj,nm,ei,fn->ij', t2, f1, t1, t1))[np.ix_(OI, OI)]
    ft['ij'] += ((1/2) * np.einsum('em,fj,eg,fgim->ij', t1, t1, f1, t2))[np.ix_(OI, OI)]
    ft['ij'] += ((1/2) * np.einsum('efmj,eg,fi,gm->ij', t2, f1, t1, t1))[np.ix_(OI, OI)]
    ft['ij'] += ((-1/2) * np.einsum('efmj,eg,gi,fm->ij', t2, f1, t1, t1))[np.ix_(OI, OI)]
    ft['ij'] += ((1/2) * np.einsum('em,fj,nm,efin->ij', t1, t1, f1, t2))[np.ix_(OI, OI)]
    ft['ij'] += ((-1/6) * np.einsum('efmj,ni,em,fn->ij', t2, f1, t1, t1))[np.ix_(OI, OI)]
    ft['ij'] += ((-1/2) * np.einsum('em,fj,fg,egim->ij', t1, t1, f1, t2))[np.ix_(OI, OI)]
    ft['ij'] += ((-1/3) * np.einsum('efmn,jm,ei,fn->ij', t2, f1, t1, t1))[np.ix_(OI, OI)]
    ft['ij'] += ((-1/6) * np.einsum('em,fn,jm,efin->ij', t1, t1, f1, t2))[np.ix_(OI, OI)]
    ft['ij'] += ((-1/3) * np.einsum('em,fj,ni,efmn->ij', t1, t1, f1, t2))[np.ix_(OI, OI)]
    ft['ij'] += ((-1/6) * np.einsum('em,fj,ei,fm->ij', t1, t1, f1, t1))[np.ix_(OI, OI)]
    ft['ij'] += ((-1/2) * np.einsum('em,fj,fm,ei->ij', t1, t1, f1, t1))[np.ix_(OI, OI)]
    ft['ij'] += ((-1/6) * np.einsum('em,jf,ei,fm->ij', t1, f1, t1, t1))[np.ix_(OI, OI)]
    ft['ij'] += ((-1/2) * np.einsum('ej,mf,fi,em->ij', t1, f1, t1, t1))[np.ix_(OI, OI)]

    # --- PP ---
    ft['ab'] += ((1/6) * np.einsum('em,fbno,fn,aemo->ab', t1, t2, f1, t2))[np.ix_(VI, VI)]
    ft['ab'] += ((-1/6) * np.einsum('ebmn,of,em,afno->ab', t2, f1, t1, t2))[np.ix_(VI, VI)]
    ft['ab'] += ((-1/3) * np.einsum('em,fbno,fm,aeno->ab', t1, t2, f1, t2))[np.ix_(VI, VI)]
    ft['ab'] += ((-1/3) * np.einsum('ebmn,of,eo,afmn->ab', t2, f1, t1, t2))[np.ix_(VI, VI)]
    ft['ab'] += ((1/6) * np.einsum('efmn,ob,em,afno->ab', t2, f1, t1, t2))[np.ix_(VI, VI)]
    ft['ab'] += ((1/6) * np.einsum('ebmn,of,am,efno->ab', t2, f1, t1, t2))[np.ix_(VI, VI)]  # [T_int=0]
    ft['ab'] += ((2/3) * np.einsum('ebmn,of,fm,aeno->ab', t2, f1, t1, t2))[np.ix_(VI, VI)]
    ft['ab'] += ((-2/3) * np.einsum('em,fbno,en,afmo->ab', t1, t2, f1, t2))[np.ix_(VI, VI)]
    ft['ab'] += ((-1/6) * np.einsum('bm,efno,en,afmo->ab', t1, t2, f1, t2))[np.ix_(VI, VI)]  # [T_int=0]
    ft['ab'] += ((1/6) * np.einsum('em,fbno,an,efmo->ab', t1, t2, f1, t2))[np.ix_(VI, VI)]
    ft['ab'] += ((1/12) * np.einsum('efmn,ob,eo,afmn->ab', t2, f1, t1, t2))[np.ix_(VI, VI)]
    ft['ab'] += ((-1/12) * np.einsum('efmn,ob,am,efno->ab', t2, f1, t1, t2))[np.ix_(VI, VI)]  # [T_int=0]
    ft['ab'] += ((1/4) * np.einsum('ebmn,of,ao,efmn->ab', t2, f1, t1, t2))[np.ix_(VI, VI)]  # [T_int=0]
    ft['ab'] += ((1/4) * np.einsum('bm,efno,em,afno->ab', t1, t2, f1, t2))[np.ix_(VI, VI)]  # [T_int=0]
    ft['ab'] += ((-1/12) * np.einsum('em,fbno,am,efno->ab', t1, t2, f1, t2))[np.ix_(VI, VI)]
    ft['ab'] += ((1/12) * np.einsum('bm,efno,an,efmo->ab', t1, t2, f1, t2))[np.ix_(VI, VI)]  # [T_int=0]
    ft['ab'] += ((-1/6) * np.einsum('em,fn,eb,afmn->ab', t1, t1, f1, t2))[np.ix_(VI, VI)]
    ft['ab'] += ((1/2) * np.einsum('em,bn,ef,afmn->ab', t1, t1, f1, t2))[np.ix_(VI, VI)]  # [T_int=0]
    ft['ab'] += ((-1/3) * np.einsum('em,bn,af,efmn->ab', t1, t1, f1, t2))[np.ix_(VI, VI)]  # [T_int=0]
    ft['ab'] += ((1/2) * np.einsum('ebmn,ef,am,fn->ab', t2, f1, t1, t1))[np.ix_(VI, VI)]  # [T_int=0]
    ft['ab'] += ((-1/3) * np.einsum('efmn,eb,am,fn->ab', t2, f1, t1, t1))[np.ix_(VI, VI)]  # [T_int=0]
    ft['ab'] += ((-1/6) * np.einsum('ebmn,af,em,fn->ab', t2, f1, t1, t1))[np.ix_(VI, VI)]
    ft['ab'] += ((1/2) * np.einsum('ebmn,om,an,eo->ab', t2, f1, t1, t1))[np.ix_(VI, VI)]  # [T_int=0]
    ft['ab'] += ((1/2) * np.einsum('em,bn,om,aeno->ab', t1, t1, f1, t2))[np.ix_(VI, VI)]  # [T_int=0]
    ft['ab'] += ((-1/2) * np.einsum('ebmn,om,ao,en->ab', t2, f1, t1, t1))[np.ix_(VI, VI)]  # [T_int=0]
    ft['ab'] += ((-1/2) * np.einsum('bm,en,om,aeno->ab', t1, t1, f1, t2))[np.ix_(VI, VI)]  # [T_int=0]
    ft['ab'] += ((1/6) * np.einsum('em,nb,am,en->ab', t1, f1, t1, t1))[np.ix_(VI, VI)]  # [T_int=0]
    ft['ab'] += ((1/2) * np.einsum('bm,ne,an,em->ab', t1, f1, t1, t1))[np.ix_(VI, VI)]  # [T_int=0]
    ft['ab'] += ((1/6) * np.einsum('em,bn,am,en->ab', t1, t1, f1, t1))[np.ix_(VI, VI)]  # [T_int=0]
    ft['ab'] += ((1/2) * np.einsum('bm,en,em,an->ab', t1, t1, f1, t1))[np.ix_(VI, VI)]  # [T_int=0]

    # --- HP/PH ---
    ft['ia'] += ((-2/3) * np.einsum('efmn,og,egim,afno->ia', t2, f1, t2, t2))[np.ix_(OI, VI)]
    ft['ia'] += ((1/6) * np.einsum('efmn,og,aeim,fgno->ia', t2, f1, t2, t2))[np.ix_(OI, VI)]
    ft['ia'] += ((-1/12) * np.einsum('efmn,og,egio,afmn->ia', t2, f1, t2, t2))[np.ix_(OI, VI)]
    ft['ia'] += ((1/3) * np.einsum('efmn,og,agim,efno->ia', t2, f1, t2, t2))[np.ix_(OI, VI)]
    ft['ia'] += ((1/3) * np.einsum('efmn,og,aeio,fgmn->ia', t2, f1, t2, t2))[np.ix_(OI, VI)]
    ft['ia'] += ((1/12) * np.einsum('efmn,og,efim,agno->ia', t2, f1, t2, t2))[np.ix_(OI, VI)]
    ft['ia'] += ((1/6) * np.einsum('efmn,og,efio,agmn->ia', t2, f1, t2, t2))[np.ix_(OI, VI)]
    ft['ia'] += ((1/2) * np.einsum('efmn,om,en,afio->ia', t2, f1, t1, t2))[np.ix_(OI, VI)]
    ft['ia'] += ((-1/2) * np.einsum('efmn,eg,fm,agin->ia', t2, f1, t1, t2))[np.ix_(OI, VI)]
    ft['ia'] += ((-1/4) * np.einsum('efmn,om,an,efio->ia', t2, f1, t1, t2))[np.ix_(OI, VI)]  # [T_int=0]
    ft['ia'] += ((1/4) * np.einsum('efmn,eg,fi,agmn->ia', t2, f1, t1, t2))[np.ix_(OI, VI)]
    ft['ia'] += ((1/6) * np.einsum('efmn,oi,em,afno->ia', t2, f1, t1, t2))[np.ix_(OI, VI)]
    ft['ia'] += ((-1/2) * np.einsum('efmn,om,eo,afin->ia', t2, f1, t1, t2))[np.ix_(OI, VI)]
    ft['ia'] += ((-1/2) * np.einsum('efmn,om,ei,afno->ia', t2, f1, t1, t2))[np.ix_(OI, VI)]
    ft['ia'] += ((-1/6) * np.einsum('efmn,ag,em,fgin->ia', t2, f1, t1, t2))[np.ix_(OI, VI)]
    ft['ia'] += ((1/2) * np.einsum('efmn,eg,am,fgin->ia', t2, f1, t1, t2))[np.ix_(OI, VI)]  # [T_int=0]
    ft['ia'] += ((1/2) * np.einsum('efmn,eg,gm,afin->ia', t2, f1, t1, t2))[np.ix_(OI, VI)]
    ft['ia'] += ((1/12) * np.einsum('efmn,oi,eo,afmn->ia', t2, f1, t1, t2))[np.ix_(OI, VI)]
    ft['ia'] += ((-1/12) * np.einsum('efmn,oi,am,efno->ia', t2, f1, t1, t2))[np.ix_(OI, VI)]  # [T_int=0]
    ft['ia'] += ((1/4) * np.einsum('efmn,om,ao,efin->ia', t2, f1, t1, t2))[np.ix_(OI, VI)]  # [T_int=0]
    ft['ia'] += ((1/12) * np.einsum('efmn,ag,ei,fgmn->ia', t2, f1, t1, t2))[np.ix_(OI, VI)]
    ft['ia'] += ((-1/12) * np.einsum('efmn,ag,gm,efin->ia', t2, f1, t1, t2))[np.ix_(OI, VI)]
    ft['ia'] += ((-1/4) * np.einsum('efmn,eg,gi,afmn->ia', t2, f1, t1, t2))[np.ix_(OI, VI)]
    ft['ia'] += ((-2/3) * np.einsum('em,nf,fm,aein->ia', t1, f1, t1, t2))[np.ix_(OI, VI)]
    ft['ia'] += ((-1/6) * np.einsum('em,nf,ei,afmn->ia', t1, f1, t1, t2))[np.ix_(OI, VI)]
    ft['ia'] += ((-1/6) * np.einsum('em,fn,am,efin->ia', t1, t1, f1, t2))[np.ix_(OI, VI)]
    ft['ia'] += ((-2/3) * np.einsum('em,nf,en,afim->ia', t1, f1, t1, t2))[np.ix_(OI, VI)]
    ft['ia'] += ((-1/6) * np.einsum('em,fn,ei,afmn->ia', t1, t1, f1, t2))[np.ix_(OI, VI)]
    ft['ia'] += ((-2/3) * np.einsum('em,fn,fm,aein->ia', t1, t1, f1, t2))[np.ix_(OI, VI)]
    ft['ia'] += ((-1/6) * np.einsum('em,nf,am,efin->ia', t1, f1, t1, t2))[np.ix_(OI, VI)]  # [T_int=0]
    ft['ia'] += ((1/2) * np.einsum('em,nf,fi,aemn->ia', t1, f1, t1, t2))[np.ix_(OI, VI)]
    ft['ia'] += ((-1/3) * np.einsum('efmn,em,fi,an->ia', t2, f1, t1, t1))[np.ix_(OI, VI)]  # [T_int=0]
    ft['ia'] += ((1/2) * np.einsum('em,nf,an,efim->ia', t1, f1, t1, t2))[np.ix_(OI, VI)]  # [T_int=0]
    ft['ia'] += ((-1/3) * np.einsum('efmn,am,ei,fn->ia', t2, f1, t1, t1))[np.ix_(OI, VI)]
    ft['ia'] += ((-1/3) * np.einsum('efmn,ei,am,fn->ia', t2, f1, t1, t1))[np.ix_(OI, VI)]  # [T_int=0]
    ft['ia'] += ((1/2) * np.einsum('em,nm,ei,an->ia', t1, f1, t1, t1))[np.ix_(OI, VI)]  # [T_int=0]
    ft['ia'] += ((1/6) * np.einsum('em,ni,am,en->ia', t1, f1, t1, t1))[np.ix_(OI, VI)]  # [T_int=0]
    ft['ia'] += ((-1/6) * np.einsum('em,af,ei,fm->ia', t1, f1, t1, t1))[np.ix_(OI, VI)]
    ft['ia'] += ((-1/2) * np.einsum('em,ef,fi,am->ia', t1, f1, t1, t1))[np.ix_(OI, VI)]  # [T_int=0]

    # --- HHHH ---
    vt['ijkl'] += ((-1/4) * np.einsum('eflk,mg,gi,efjm->ijkl', t2, f1, t1, t2))[np.ix_(OI, OI, OI, OI)]
    vt['ijkl'] += ((-1/12) * np.einsum('em,fglk,ei,fgjm->ijkl', t1, t2, f1, t2))[np.ix_(OI, OI, OI, OI)]
    vt['ijkl'] += ((-1/6) * np.einsum('eflk,mg,ei,fgjm->ijkl', t2, f1, t1, t2))[np.ix_(OI, OI, OI, OI)]
    vt['ijkl'] += ((1/6) * np.einsum('ek,fgml,fm,egij->ijkl', t1, t2, f1, t2))[np.ix_(OI, OI, OI, OI)]
    vt['ijkl'] += ((1/6) * np.einsum('em,fglk,fi,egjm->ijkl', t1, t2, f1, t2))[np.ix_(OI, OI, OI, OI)]
    vt['ijkl'] += ((-1/6) * np.einsum('el,fgmk,fm,egij->ijkl', t1, t2, f1, t2))[np.ix_(OI, OI, OI, OI)]
    vt['ijkl'] += ((2/3) * np.einsum('em,fglk,fm,egij->ijkl', t1, t2, f1, t2))[np.ix_(OI, OI, OI, OI)]
    vt['ijkl'] += ((-2/3) * np.einsum('eflk,mg,em,fgij->ijkl', t2, f1, t1, t2))[np.ix_(OI, OI, OI, OI)]
    vt['ijkl'] += ((-1/6) * np.einsum('em,fglk,fj,egim->ijkl', t1, t2, f1, t2))[np.ix_(OI, OI, OI, OI)]
    vt['ijkl'] += ((1/6) * np.einsum('eflk,mg,ej,fgim->ijkl', t2, f1, t1, t2))[np.ix_(OI, OI, OI, OI)]
    vt['ijkl'] += ((1/12) * np.einsum('em,fglk,ej,fgim->ijkl', t1, t2, f1, t2))[np.ix_(OI, OI, OI, OI)]
    vt['ijkl'] += ((1/4) * np.einsum('eflk,mg,gj,efim->ijkl', t2, f1, t1, t2))[np.ix_(OI, OI, OI, OI)]
    vt['ijkl'] += ((-1/4) * np.einsum('ek,fgml,em,fgij->ijkl', t1, t2, f1, t2))[np.ix_(OI, OI, OI, OI)]
    vt['ijkl'] += ((1/4) * np.einsum('el,fgmk,em,fgij->ijkl', t1, t2, f1, t2))[np.ix_(OI, OI, OI, OI)]
    vt['ijkl'] += ((1/6) * np.einsum('efmk,lg,em,fgij->ijkl', t2, f1, t1, t2))[np.ix_(OI, OI, OI, OI)]
    vt['ijkl'] += ((-1/6) * np.einsum('efml,kg,em,fgij->ijkl', t2, f1, t1, t2))[np.ix_(OI, OI, OI, OI)]
    vt['ijkl'] += ((-1/12) * np.einsum('efml,kg,gm,efij->ijkl', t2, f1, t1, t2))[np.ix_(OI, OI, OI, OI)]
    vt['ijkl'] += ((1/12) * np.einsum('efmk,lg,gm,efij->ijkl', t2, f1, t1, t2))[np.ix_(OI, OI, OI, OI)]
    vt['ijkl'] += ((-1/6) * np.einsum('ek,fgml,fj,egim->ijkl', t1, t2, f1, t2))[np.ix_(OI, OI, OI, OI)]
    vt['ijkl'] += ((1/6) * np.einsum('ek,fgml,fi,egjm->ijkl', t1, t2, f1, t2))[np.ix_(OI, OI, OI, OI)]
    vt['ijkl'] += ((1/6) * np.einsum('el,fgmk,fj,egim->ijkl', t1, t2, f1, t2))[np.ix_(OI, OI, OI, OI)]
    vt['ijkl'] += ((-1/6) * np.einsum('el,fgmk,fi,egjm->ijkl', t1, t2, f1, t2))[np.ix_(OI, OI, OI, OI)]
    vt['ijkl'] += ((-1/6) * np.einsum('efml,kg,ei,fgjm->ijkl', t2, f1, t1, t2))[np.ix_(OI, OI, OI, OI)]
    vt['ijkl'] += ((1/6) * np.einsum('efmk,lg,ei,fgjm->ijkl', t2, f1, t1, t2))[np.ix_(OI, OI, OI, OI)]
    vt['ijkl'] += ((1/6) * np.einsum('efml,kg,ej,fgim->ijkl', t2, f1, t1, t2))[np.ix_(OI, OI, OI, OI)]
    vt['ijkl'] += ((-1/6) * np.einsum('efmk,lg,ej,fgim->ijkl', t2, f1, t1, t2))[np.ix_(OI, OI, OI, OI)]
    vt['ijkl'] += ((-1/3) * np.einsum('el,fk,mi,efjm->ijkl', t1, t1, f1, t2))[np.ix_(OI, OI, OI, OI)]
    vt['ijkl'] += ((-1/2) * np.einsum('eflk,eg,gi,fj->ijkl', t2, f1, t1, t1))[np.ix_(OI, OI, OI, OI)]
    vt['ijkl'] += ((-1/6) * np.einsum('eflk,mi,ej,fm->ijkl', t2, f1, t1, t1))[np.ix_(OI, OI, OI, OI)]
    vt['ijkl'] += ((1/3) * np.einsum('el,fk,mj,efim->ijkl', t1, t1, f1, t2))[np.ix_(OI, OI, OI, OI)]
    vt['ijkl'] += ((-1/6) * np.einsum('em,fl,km,efij->ijkl', t1, t1, f1, t2))[np.ix_(OI, OI, OI, OI)]
    vt['ijkl'] += ((1/6) * np.einsum('eflk,mj,ei,fm->ijkl', t2, f1, t1, t1))[np.ix_(OI, OI, OI, OI)]
    vt['ijkl'] += ((1/3) * np.einsum('efmk,lm,ei,fj->ijkl', t2, f1, t1, t1))[np.ix_(OI, OI, OI, OI)]
    vt['ijkl'] += ((-1/3) * np.einsum('efml,km,ei,fj->ijkl', t2, f1, t1, t1))[np.ix_(OI, OI, OI, OI)]
    vt['ijkl'] += ((-1/2) * np.einsum('el,fk,fg,egij->ijkl', t1, t1, f1, t2))[np.ix_(OI, OI, OI, OI)]
    vt['ijkl'] += ((1/2) * np.einsum('eflk,eg,fi,gj->ijkl', t2, f1, t1, t1))[np.ix_(OI, OI, OI, OI)]
    vt['ijkl'] += ((1/6) * np.einsum('em,fk,lm,efij->ijkl', t1, t1, f1, t2))[np.ix_(OI, OI, OI, OI)]
    vt['ijkl'] += ((1/2) * np.einsum('el,fk,eg,fgij->ijkl', t1, t1, f1, t2))[np.ix_(OI, OI, OI, OI)]

    # --- PPPP ---
    vt['abcd'] += ((-1/6) * np.einsum('em,dcno,bn,aemo->abcd', t1, t2, f1, t2))[np.ix_(VI, VI, VI, VI)]
    vt['abcd'] += ((1/6) * np.einsum('cm,edno,en,abmo->abcd', t1, t2, f1, t2))[np.ix_(VI, VI, VI, VI)]  # [T_int=0]
    vt['abcd'] += ((2/3) * np.einsum('em,dcno,en,abmo->abcd', t1, t2, f1, t2))[np.ix_(VI, VI, VI, VI)]
    vt['abcd'] += ((1/6) * np.einsum('em,dcno,an,bemo->abcd', t1, t2, f1, t2))[np.ix_(VI, VI, VI, VI)]
    vt['abcd'] += ((-1/6) * np.einsum('edmn,oc,em,abno->abcd', t2, f1, t1, t2))[np.ix_(VI, VI, VI, VI)]
    vt['abcd'] += ((-1/6) * np.einsum('dm,ecno,en,abmo->abcd', t1, t2, f1, t2))[np.ix_(VI, VI, VI, VI)]  # [T_int=0]
    vt['abcd'] += ((1/12) * np.einsum('ecmn,od,eo,abmn->abcd', t2, f1, t1, t2))[np.ix_(VI, VI, VI, VI)]  # [T_int=0]
    vt['abcd'] += ((-1/4) * np.einsum('cm,edno,em,abno->abcd', t1, t2, f1, t2))[np.ix_(VI, VI, VI, VI)]  # [T_int=0]
    vt['abcd'] += ((1/6) * np.einsum('ecmn,od,em,abno->abcd', t2, f1, t1, t2))[np.ix_(VI, VI, VI, VI)]
    vt['abcd'] += ((-2/3) * np.einsum('dcmn,oe,em,abno->abcd', t2, f1, t1, t2))[np.ix_(VI, VI, VI, VI)]  # [T_int=0]
    vt['abcd'] += ((1/4) * np.einsum('dm,ecno,em,abno->abcd', t1, t2, f1, t2))[np.ix_(VI, VI, VI, VI)]  # [T_int=0]
    vt['abcd'] += ((1/12) * np.einsum('em,dcno,bm,aeno->abcd', t1, t2, f1, t2))[np.ix_(VI, VI, VI, VI)]
    vt['abcd'] += ((-1/12) * np.einsum('edmn,oc,eo,abmn->abcd', t2, f1, t1, t2))[np.ix_(VI, VI, VI, VI)]  # [T_int=0]
    vt['abcd'] += ((-1/6) * np.einsum('dcmn,oe,am,beno->abcd', t2, f1, t1, t2))[np.ix_(VI, VI, VI, VI)]  # [T_int=0]
    vt['abcd'] += ((-1/12) * np.einsum('em,dcno,am,beno->abcd', t1, t2, f1, t2))[np.ix_(VI, VI, VI, VI)]
    vt['abcd'] += ((1/6) * np.einsum('dcmn,oe,bm,aeno->abcd', t2, f1, t1, t2))[np.ix_(VI, VI, VI, VI)]  # [T_int=0]
    vt['abcd'] += ((-1/4) * np.einsum('dcmn,oe,ao,bemn->abcd', t2, f1, t1, t2))[np.ix_(VI, VI, VI, VI)]  # [T_int=0]
    vt['abcd'] += ((1/4) * np.einsum('dcmn,oe,bo,aemn->abcd', t2, f1, t1, t2))[np.ix_(VI, VI, VI, VI)]  # [T_int=0]
    vt['abcd'] += ((-1/6) * np.einsum('cm,edno,bn,aemo->abcd', t1, t2, f1, t2))[np.ix_(VI, VI, VI, VI)]  # [T_int=0]
    vt['abcd'] += ((1/6) * np.einsum('cm,edno,an,bemo->abcd', t1, t2, f1, t2))[np.ix_(VI, VI, VI, VI)]  # [T_int=0]
    vt['abcd'] += ((-1/6) * np.einsum('dm,ecno,an,bemo->abcd', t1, t2, f1, t2))[np.ix_(VI, VI, VI, VI)]  # [T_int=0]
    vt['abcd'] += ((1/6) * np.einsum('dm,ecno,bn,aemo->abcd', t1, t2, f1, t2))[np.ix_(VI, VI, VI, VI)]  # [T_int=0]
    vt['abcd'] += ((-1/6) * np.einsum('ecmn,od,bm,aeno->abcd', t2, f1, t1, t2))[np.ix_(VI, VI, VI, VI)]  # [T_int=0]
    vt['abcd'] += ((1/6) * np.einsum('ecmn,od,am,beno->abcd', t2, f1, t1, t2))[np.ix_(VI, VI, VI, VI)]  # [T_int=0]
    vt['abcd'] += ((1/6) * np.einsum('edmn,oc,bm,aeno->abcd', t2, f1, t1, t2))[np.ix_(VI, VI, VI, VI)]  # [T_int=0]
    vt['abcd'] += ((-1/6) * np.einsum('edmn,oc,am,beno->abcd', t2, f1, t1, t2))[np.ix_(VI, VI, VI, VI)]  # [T_int=0]
    vt['abcd'] += ((1/6) * np.einsum('em,dn,ec,abmn->abcd', t1, t1, f1, t2))[np.ix_(VI, VI, VI, VI)]  # [T_int=0]
    vt['abcd'] += ((1/3) * np.einsum('edmn,ec,am,bn->abcd', t2, f1, t1, t1))[np.ix_(VI, VI, VI, VI)]  # [T_int=0]
    vt['abcd'] += ((-1/2) * np.einsum('dm,cn,om,abno->abcd', t1, t1, f1, t2))[np.ix_(VI, VI, VI, VI)]  # [T_int=0]
    vt['abcd'] += ((1/2) * np.einsum('cm,dn,om,abno->abcd', t1, t1, f1, t2))[np.ix_(VI, VI, VI, VI)]  # [T_int=0]
    vt['abcd'] += ((-1/3) * np.einsum('ecmn,ed,am,bn->abcd', t2, f1, t1, t1))[np.ix_(VI, VI, VI, VI)]  # [T_int=0]
    vt['abcd'] += ((-1/6) * np.einsum('em,cn,ed,abmn->abcd', t1, t1, f1, t2))[np.ix_(VI, VI, VI, VI)]  # [T_int=0]
    vt['abcd'] += ((1/6) * np.einsum('dcmn,ae,bm,en->abcd', t2, f1, t1, t1))[np.ix_(VI, VI, VI, VI)]  # [T_int=0]
    vt['abcd'] += ((-1/6) * np.einsum('dcmn,be,am,en->abcd', t2, f1, t1, t1))[np.ix_(VI, VI, VI, VI)]  # [T_int=0]
    vt['abcd'] += ((-1/2) * np.einsum('dcmn,om,an,bo->abcd', t2, f1, t1, t1))[np.ix_(VI, VI, VI, VI)]  # [T_int=0]
    vt['abcd'] += ((1/2) * np.einsum('dcmn,om,ao,bn->abcd', t2, f1, t1, t1))[np.ix_(VI, VI, VI, VI)]  # [T_int=0]
    vt['abcd'] += ((1/3) * np.einsum('dm,cn,ae,bemn->abcd', t1, t1, f1, t2))[np.ix_(VI, VI, VI, VI)]  # [T_int=0]
    vt['abcd'] += ((-1/3) * np.einsum('dm,cn,be,aemn->abcd', t1, t1, f1, t2))[np.ix_(VI, VI, VI, VI)]  # [T_int=0]

    # --- HHHP/HPHH ---
    vt['ijka'] += ((1/6) * np.einsum('efmk,ng,aeij,fgmn->ijka', t2, f1, t2, t2))[np.ix_(OI, OI, OI, VI)]
    vt['ijka'] += ((1/4) * np.einsum('efmk,ng,agij,efmn->ijka', t2, f1, t2, t2))[np.ix_(OI, OI, OI, VI)]
    vt['ijka'] += ((-1/12) * np.einsum('efmn,kg,aeij,fgmn->ijka', t2, f1, t2, t2))[np.ix_(OI, OI, OI, VI)]
    vt['ijka'] += ((-2/3) * np.einsum('efmk,ng,egij,afmn->ijka', t2, f1, t2, t2))[np.ix_(OI, OI, OI, VI)]
    vt['ijka'] += ((-1/6) * np.einsum('efmn,kg,egim,afjn->ijka', t2, f1, t2, t2))[np.ix_(OI, OI, OI, VI)]
    vt['ijka'] += ((1/12) * np.einsum('efmk,ng,efij,agmn->ijka', t2, f1, t2, t2))[np.ix_(OI, OI, OI, VI)]
    vt['ijka'] += ((1/12) * np.einsum('efmn,kg,efim,agjn->ijka', t2, f1, t2, t2))[np.ix_(OI, OI, OI, VI)]
    vt['ijka'] += ((2/3) * np.einsum('efmk,ng,aein,fgjm->ijka', t2, f1, t2, t2))[np.ix_(OI, OI, OI, VI)]
    vt['ijka'] += ((1/12) * np.einsum('efmn,kg,egij,afmn->ijka', t2, f1, t2, t2))[np.ix_(OI, OI, OI, VI)]
    vt['ijka'] += ((-1/6) * np.einsum('efmk,ng,egin,afjm->ijka', t2, f1, t2, t2))[np.ix_(OI, OI, OI, VI)]
    vt['ijka'] += ((2/3) * np.einsum('efmk,ng,egim,afjn->ijka', t2, f1, t2, t2))[np.ix_(OI, OI, OI, VI)]
    vt['ijka'] += ((1/6) * np.einsum('efmn,kg,aeim,fgjn->ijka', t2, f1, t2, t2))[np.ix_(OI, OI, OI, VI)]
    vt['ijka'] += ((1/12) * np.einsum('efmn,kg,agim,efjn->ijka', t2, f1, t2, t2))[np.ix_(OI, OI, OI, VI)]
    vt['ijka'] += ((-1/6) * np.einsum('efmk,ng,aeim,fgjn->ijka', t2, f1, t2, t2))[np.ix_(OI, OI, OI, VI)]
    vt['ijka'] += ((-1/24) * np.einsum('efmn,kg,efij,agmn->ijka', t2, f1, t2, t2))[np.ix_(OI, OI, OI, VI)]
    vt['ijka'] += ((1/3) * np.einsum('efmk,ng,efin,agjm->ijka', t2, f1, t2, t2))[np.ix_(OI, OI, OI, VI)]
    vt['ijka'] += ((-1/3) * np.einsum('efmk,ng,agim,efjn->ijka', t2, f1, t2, t2))[np.ix_(OI, OI, OI, VI)]
    vt['ijka'] += ((1/12) * np.einsum('efmk,ni,am,efjn->ijka', t2, f1, t1, t2))[np.ix_(OI, OI, OI, VI)]  # [T_int=0]
    vt['ijka'] += ((-1/6) * np.einsum('efmk,ni,em,afjn->ijka', t2, f1, t1, t2))[np.ix_(OI, OI, OI, VI)]
    vt['ijka'] += ((1/2) * np.einsum('efmk,eg,fm,agij->ijka', t2, f1, t1, t2))[np.ix_(OI, OI, OI, VI)]
    vt['ijka'] += ((1/6) * np.einsum('efmk,ag,em,fgij->ijka', t2, f1, t1, t2))[np.ix_(OI, OI, OI, VI)]
    vt['ijka'] += ((1/12) * np.einsum('efmk,ag,gm,efij->ijka', t2, f1, t1, t2))[np.ix_(OI, OI, OI, VI)]
    vt['ijka'] += ((-1/12) * np.einsum('efmk,nj,am,efin->ijka', t2, f1, t1, t2))[np.ix_(OI, OI, OI, VI)]  # [T_int=0]
    vt['ijka'] += ((1/6) * np.einsum('efmk,nj,em,afin->ijka', t2, f1, t1, t2))[np.ix_(OI, OI, OI, VI)]
    vt['ijka'] += ((-1/2) * np.einsum('efmk,eg,am,fgij->ijka', t2, f1, t1, t2))[np.ix_(OI, OI, OI, VI)]  # [T_int=0]
    vt['ijka'] += ((-1/2) * np.einsum('efmk,eg,gm,afij->ijka', t2, f1, t1, t2))[np.ix_(OI, OI, OI, VI)]
    vt['ijka'] += ((1/2) * np.einsum('efmk,nm,en,afij->ijka', t2, f1, t1, t2))[np.ix_(OI, OI, OI, VI)]
    vt['ijka'] += ((-1/4) * np.einsum('efmk,nm,an,efij->ijka', t2, f1, t1, t2))[np.ix_(OI, OI, OI, VI)]  # [T_int=0]
    vt['ijka'] += ((-1/3) * np.einsum('efmn,km,en,afij->ijka', t2, f1, t1, t2))[np.ix_(OI, OI, OI, VI)]
    vt['ijka'] += ((1/6) * np.einsum('efmn,km,an,efij->ijka', t2, f1, t1, t2))[np.ix_(OI, OI, OI, VI)]  # [T_int=0]
    vt['ijka'] += ((1/2) * np.einsum('efmk,nm,ei,afjn->ijka', t2, f1, t1, t2))[np.ix_(OI, OI, OI, VI)]
    vt['ijka'] += ((-1/2) * np.einsum('efmk,nm,ej,afin->ijka', t2, f1, t1, t2))[np.ix_(OI, OI, OI, VI)]
    vt['ijka'] += ((1/2) * np.einsum('efmk,eg,fi,agjm->ijka', t2, f1, t1, t2))[np.ix_(OI, OI, OI, VI)]
    vt['ijka'] += ((-1/2) * np.einsum('efmk,eg,fj,agim->ijka', t2, f1, t1, t2))[np.ix_(OI, OI, OI, VI)]
    vt['ijka'] += ((-1/3) * np.einsum('efmn,km,ei,afjn->ijka', t2, f1, t1, t2))[np.ix_(OI, OI, OI, VI)]
    vt['ijka'] += ((-1/6) * np.einsum('efmk,nj,ei,afmn->ijka', t2, f1, t1, t2))[np.ix_(OI, OI, OI, VI)]
    vt['ijka'] += ((1/6) * np.einsum('efmk,ni,en,afjm->ijka', t2, f1, t1, t2))[np.ix_(OI, OI, OI, VI)]
    vt['ijka'] += ((1/3) * np.einsum('efmn,km,ej,afin->ijka', t2, f1, t1, t2))[np.ix_(OI, OI, OI, VI)]
    vt['ijka'] += ((1/6) * np.einsum('efmk,ni,ej,afmn->ijka', t2, f1, t1, t2))[np.ix_(OI, OI, OI, VI)]
    vt['ijka'] += ((-1/6) * np.einsum('efmk,nj,en,afim->ijka', t2, f1, t1, t2))[np.ix_(OI, OI, OI, VI)]
    vt['ijka'] += ((-1/2) * np.einsum('efmk,eg,gi,afjm->ijka', t2, f1, t1, t2))[np.ix_(OI, OI, OI, VI)]
    vt['ijka'] += ((1/6) * np.einsum('efmk,ag,ei,fgjm->ijka', t2, f1, t1, t2))[np.ix_(OI, OI, OI, VI)]
    vt['ijka'] += ((1/2) * np.einsum('efmk,eg,gj,afim->ijka', t2, f1, t1, t2))[np.ix_(OI, OI, OI, VI)]
    vt['ijka'] += ((-1/6) * np.einsum('efmk,ag,ej,fgim->ijka', t2, f1, t1, t2))[np.ix_(OI, OI, OI, VI)]
    vt['ijka'] += ((-1/6) * np.einsum('em,kf,ei,afjm->ijka', t1, f1, t1, t2))[np.ix_(OI, OI, OI, VI)]
    vt['ijka'] += ((1/6) * np.einsum('em,kf,fm,aeij->ijka', t1, f1, t1, t2))[np.ix_(OI, OI, OI, VI)]
    vt['ijka'] += ((-1/6) * np.einsum('em,fk,ei,afjm->ijka', t1, t1, f1, t2))[np.ix_(OI, OI, OI, VI)]
    vt['ijka'] += ((1/6) * np.einsum('em,kf,ej,afim->ijka', t1, f1, t1, t2))[np.ix_(OI, OI, OI, VI)]
    vt['ijka'] += ((1/6) * np.einsum('em,kf,am,efij->ijka', t1, f1, t1, t2))[np.ix_(OI, OI, OI, VI)]  # [T_int=0]
    vt['ijka'] += ((-1/2) * np.einsum('ek,mf,am,efij->ijka', t1, f1, t1, t2))[np.ix_(OI, OI, OI, VI)]  # [T_int=0]
    vt['ijka'] += ((1/6) * np.einsum('em,fk,ej,afim->ijka', t1, t1, f1, t2))[np.ix_(OI, OI, OI, VI)]
    vt['ijka'] += ((1/2) * np.einsum('em,fk,fm,aeij->ijka', t1, t1, f1, t2))[np.ix_(OI, OI, OI, VI)]
    vt['ijka'] += ((1/3) * np.einsum('efmk,am,ei,fj->ijka', t2, f1, t1, t1))[np.ix_(OI, OI, OI, VI)]
    vt['ijka'] += ((-1/2) * np.einsum('ek,mf,fi,aejm->ijka', t1, f1, t1, t2))[np.ix_(OI, OI, OI, VI)]
    vt['ijka'] += ((-1/3) * np.einsum('efmk,ej,fi,am->ijka', t2, f1, t1, t1))[np.ix_(OI, OI, OI, VI)]  # [T_int=0]
    vt['ijka'] += ((1/3) * np.einsum('efmk,ei,am,fj->ijka', t2, f1, t1, t1))[np.ix_(OI, OI, OI, VI)]  # [T_int=0]
    vt['ijka'] += ((1/6) * np.einsum('em,fk,am,efij->ijka', t1, t1, f1, t2))[np.ix_(OI, OI, OI, VI)]
    vt['ijka'] += ((1/2) * np.einsum('ek,mf,em,afij->ijka', t1, f1, t1, t2))[np.ix_(OI, OI, OI, VI)]
    vt['ijka'] += ((1/2) * np.einsum('ek,mf,fj,aeim->ijka', t1, f1, t1, t2))[np.ix_(OI, OI, OI, VI)]

    # --- PPPH/PHPP ---
    vt['iabc'] += ((1/6) * np.einsum('eamn,of,ceim,bfno->iabc', t2, f1, t2, t2))[np.ix_(OI, VI, VI, VI)]
    vt['iabc'] += ((-2/3) * np.einsum('eamn,of,cemo,bfin->iabc', t2, f1, t2, t2))[np.ix_(OI, VI, VI, VI)]
    vt['iabc'] += ((1/3) * np.einsum('eamn,of,ceio,bfmn->iabc', t2, f1, t2, t2))[np.ix_(OI, VI, VI, VI)]
    vt['iabc'] += ((1/6) * np.einsum('eamn,of,cfmo,bein->iabc', t2, f1, t2, t2))[np.ix_(OI, VI, VI, VI)]
    vt['iabc'] += ((-2/3) * np.einsum('eamn,of,cfim,beno->iabc', t2, f1, t2, t2))[np.ix_(OI, VI, VI, VI)]
    vt['iabc'] += ((-1/6) * np.einsum('eamn,of,cbim,efno->iabc', t2, f1, t2, t2))[np.ix_(OI, VI, VI, VI)]  # [T_int=0]
    vt['iabc'] += ((2/3) * np.einsum('eamn,of,cbmo,efin->iabc', t2, f1, t2, t2))[np.ix_(OI, VI, VI, VI)]
    vt['iabc'] += ((-1/6) * np.einsum('efmn,oa,ceim,bfno->iabc', t2, f1, t2, t2))[np.ix_(OI, VI, VI, VI)]
    vt['iabc'] += ((1/6) * np.einsum('efmn,oa,cemo,bfin->iabc', t2, f1, t2, t2))[np.ix_(OI, VI, VI, VI)]
    vt['iabc'] += ((-1/4) * np.einsum('eamn,of,cbio,efmn->iabc', t2, f1, t2, t2))[np.ix_(OI, VI, VI, VI)]
    vt['iabc'] += ((-1/3) * np.einsum('eamn,of,cfmn,beio->iabc', t2, f1, t2, t2))[np.ix_(OI, VI, VI, VI)]
    vt['iabc'] += ((-1/12) * np.einsum('eamn,of,cbmn,efio->iabc', t2, f1, t2, t2))[np.ix_(OI, VI, VI, VI)]  # [T_int=0]
    vt['iabc'] += ((-1/12) * np.einsum('efmn,oa,ceio,bfmn->iabc', t2, f1, t2, t2))[np.ix_(OI, VI, VI, VI)]
    vt['iabc'] += ((-1/12) * np.einsum('efmn,oa,cemn,bfio->iabc', t2, f1, t2, t2))[np.ix_(OI, VI, VI, VI)]
    vt['iabc'] += ((1/12) * np.einsum('efmn,oa,cbim,efno->iabc', t2, f1, t2, t2))[np.ix_(OI, VI, VI, VI)]  # [T_int=0]
    vt['iabc'] += ((-1/12) * np.einsum('efmn,oa,cbmo,efin->iabc', t2, f1, t2, t2))[np.ix_(OI, VI, VI, VI)]
    vt['iabc'] += ((1/24) * np.einsum('efmn,oa,cbmn,efio->iabc', t2, f1, t2, t2))[np.ix_(OI, VI, VI, VI)]  # [T_int=0]
    vt['iabc'] += ((-1/2) * np.einsum('eamn,ef,bm,cfin->iabc', t2, f1, t1, t2))[np.ix_(OI, VI, VI, VI)]  # [T_int=0]
    vt['iabc'] += ((1/6) * np.einsum('eamn,oi,em,cbno->iabc', t2, f1, t1, t2))[np.ix_(OI, VI, VI, VI)]
    vt['iabc'] += ((1/2) * np.einsum('eamn,ef,fm,cbin->iabc', t2, f1, t1, t2))[np.ix_(OI, VI, VI, VI)]
    vt['iabc'] += ((1/6) * np.einsum('eamn,bf,em,cfin->iabc', t2, f1, t1, t2))[np.ix_(OI, VI, VI, VI)]
    vt['iabc'] += ((-1/2) * np.einsum('eamn,om,eo,cbin->iabc', t2, f1, t1, t2))[np.ix_(OI, VI, VI, VI)]
    vt['iabc'] += ((1/2) * np.einsum('eamn,om,en,cbio->iabc', t2, f1, t1, t2))[np.ix_(OI, VI, VI, VI)]
    vt['iabc'] += ((-1/4) * np.einsum('eamn,ef,fi,cbmn->iabc', t2, f1, t1, t2))[np.ix_(OI, VI, VI, VI)]  # [T_int=0]
    vt['iabc'] += ((1/12) * np.einsum('eamn,oi,eo,cbmn->iabc', t2, f1, t1, t2))[np.ix_(OI, VI, VI, VI)]  # [T_int=0]
    vt['iabc'] += ((-1/2) * np.einsum('eamn,om,ei,cbno->iabc', t2, f1, t1, t2))[np.ix_(OI, VI, VI, VI)]
    vt['iabc'] += ((-1/3) * np.einsum('efmn,ea,fm,cbin->iabc', t2, f1, t1, t2))[np.ix_(OI, VI, VI, VI)]
    vt['iabc'] += ((-1/6) * np.einsum('eamn,cf,em,bfin->iabc', t2, f1, t1, t2))[np.ix_(OI, VI, VI, VI)]
    vt['iabc'] += ((1/6) * np.einsum('efmn,ea,fi,cbmn->iabc', t2, f1, t1, t2))[np.ix_(OI, VI, VI, VI)]  # [T_int=0]
    vt['iabc'] += ((-1/12) * np.einsum('eamn,bf,ei,cfmn->iabc', t2, f1, t1, t2))[np.ix_(OI, VI, VI, VI)]
    vt['iabc'] += ((1/2) * np.einsum('eamn,ef,cm,bfin->iabc', t2, f1, t1, t2))[np.ix_(OI, VI, VI, VI)]  # [T_int=0]
    vt['iabc'] += ((1/12) * np.einsum('eamn,cf,ei,bfmn->iabc', t2, f1, t1, t2))[np.ix_(OI, VI, VI, VI)]
    vt['iabc'] += ((-1/2) * np.einsum('eamn,om,bn,ceio->iabc', t2, f1, t1, t2))[np.ix_(OI, VI, VI, VI)]  # [T_int=0]
    vt['iabc'] += ((1/2) * np.einsum('eamn,om,cn,beio->iabc', t2, f1, t1, t2))[np.ix_(OI, VI, VI, VI)]  # [T_int=0]
    vt['iabc'] += ((-1/6) * np.einsum('eamn,oi,bm,ceno->iabc', t2, f1, t1, t2))[np.ix_(OI, VI, VI, VI)]  # [T_int=0]
    vt['iabc'] += ((1/2) * np.einsum('eamn,om,bo,cein->iabc', t2, f1, t1, t2))[np.ix_(OI, VI, VI, VI)]  # [T_int=0]
    vt['iabc'] += ((1/6) * np.einsum('eamn,oi,cm,beno->iabc', t2, f1, t1, t2))[np.ix_(OI, VI, VI, VI)]  # [T_int=0]
    vt['iabc'] += ((-1/2) * np.einsum('eamn,om,co,bein->iabc', t2, f1, t1, t2))[np.ix_(OI, VI, VI, VI)]  # [T_int=0]
    vt['iabc'] += ((1/3) * np.einsum('efmn,ea,bm,cfin->iabc', t2, f1, t1, t2))[np.ix_(OI, VI, VI, VI)]  # [T_int=0]
    vt['iabc'] += ((1/6) * np.einsum('eamn,cf,bm,efin->iabc', t2, f1, t1, t2))[np.ix_(OI, VI, VI, VI)]  # [T_int=0]
    vt['iabc'] += ((-1/6) * np.einsum('eamn,bf,fm,cein->iabc', t2, f1, t1, t2))[np.ix_(OI, VI, VI, VI)]
    vt['iabc'] += ((-1/6) * np.einsum('eamn,bf,cm,efin->iabc', t2, f1, t1, t2))[np.ix_(OI, VI, VI, VI)]  # [T_int=0]
    vt['iabc'] += ((1/6) * np.einsum('eamn,cf,fm,bein->iabc', t2, f1, t1, t2))[np.ix_(OI, VI, VI, VI)]
    vt['iabc'] += ((-1/3) * np.einsum('efmn,ea,cm,bfin->iabc', t2, f1, t1, t2))[np.ix_(OI, VI, VI, VI)]  # [T_int=0]
    vt['iabc'] += ((-1/2) * np.einsum('am,ne,bn,ceim->iabc', t1, f1, t1, t2))[np.ix_(OI, VI, VI, VI)]  # [T_int=0]
    vt['iabc'] += ((-1/6) * np.einsum('em,an,ei,cbmn->iabc', t1, t1, f1, t2))[np.ix_(OI, VI, VI, VI)]  # [T_int=0]
    vt['iabc'] += ((1/2) * np.einsum('am,ne,cn,beim->iabc', t1, f1, t1, t2))[np.ix_(OI, VI, VI, VI)]  # [T_int=0]
    vt['iabc'] += ((-1/6) * np.einsum('em,na,ei,cbmn->iabc', t1, f1, t1, t2))[np.ix_(OI, VI, VI, VI)]  # [T_int=0]
    vt['iabc'] += ((-1/3) * np.einsum('eamn,ei,cm,bn->iabc', t2, f1, t1, t1))[np.ix_(OI, VI, VI, VI)]  # [T_int=0]
    vt['iabc'] += ((-1/6) * np.einsum('em,an,bm,cein->iabc', t1, t1, f1, t2))[np.ix_(OI, VI, VI, VI)]  # [T_int=0]
    vt['iabc'] += ((-1/3) * np.einsum('eamn,cm,ei,bn->iabc', t2, f1, t1, t1))[np.ix_(OI, VI, VI, VI)]  # [T_int=0]
    vt['iabc'] += ((1/6) * np.einsum('em,na,cm,bein->iabc', t1, f1, t1, t2))[np.ix_(OI, VI, VI, VI)]  # [T_int=0]
    vt['iabc'] += ((1/2) * np.einsum('am,ne,ei,cbmn->iabc', t1, f1, t1, t2))[np.ix_(OI, VI, VI, VI)]  # [T_int=0]
    vt['iabc'] += ((1/3) * np.einsum('eamn,bm,cn,ei->iabc', t2, f1, t1, t1))[np.ix_(OI, VI, VI, VI)]  # [T_int=0]
    vt['iabc'] += ((-1/6) * np.einsum('em,na,bm,cein->iabc', t1, f1, t1, t2))[np.ix_(OI, VI, VI, VI)]  # [T_int=0]
    vt['iabc'] += ((1/6) * np.einsum('em,an,cm,bein->iabc', t1, t1, f1, t2))[np.ix_(OI, VI, VI, VI)]  # [T_int=0]
    vt['iabc'] += ((-1/2) * np.einsum('am,ne,em,cbin->iabc', t1, f1, t1, t2))[np.ix_(OI, VI, VI, VI)]  # [T_int=0]
    vt['iabc'] += ((-1/6) * np.einsum('em,na,en,cbim->iabc', t1, f1, t1, t2))[np.ix_(OI, VI, VI, VI)]  # [T_int=0]
    vt['iabc'] += ((-1/2) * np.einsum('am,en,em,cbin->iabc', t1, t1, f1, t2))[np.ix_(OI, VI, VI, VI)]  # [T_int=0]

    # --- HHPP/PPHH ---
    vt['ijab'] += ((-1/2) * np.einsum('efmn,om,aeij,bfno->ijab', t2, f1, t2, t2))[np.ix_(OI, OI, VI, VI)]
    vt['ijab'] += ((1/2) * np.einsum('efmn,eg,abim,fgjn->ijab', t2, f1, t2, t2))[np.ix_(OI, OI, VI, VI)]  # [T_int=0]
    vt['ijab'] += ((-1/2) * np.einsum('efmn,eg,afim,bgjn->ijab', t2, f1, t2, t2))[np.ix_(OI, OI, VI, VI)]
    vt['ijab'] += ((1/4) * np.einsum('efmn,om,efij,abno->ijab', t2, f1, t2, t2))[np.ix_(OI, OI, VI, VI)]
    vt['ijab'] += ((1/2) * np.einsum('efmn,om,beij,afno->ijab', t2, f1, t2, t2))[np.ix_(OI, OI, VI, VI)]
    vt['ijab'] += ((1/4) * np.einsum('efmn,eg,bgij,afmn->ijab', t2, f1, t2, t2))[np.ix_(OI, OI, VI, VI)]
    vt['ijab'] += ((1/2) * np.einsum('efmn,eg,agim,bfjn->ijab', t2, f1, t2, t2))[np.ix_(OI, OI, VI, VI)]
    vt['ijab'] += ((1/2) * np.einsum('efmn,eg,fgim,abjn->ijab', t2, f1, t2, t2))[np.ix_(OI, OI, VI, VI)]
    vt['ijab'] += ((-1/2) * np.einsum('efmn,eg,bgim,afjn->ijab', t2, f1, t2, t2))[np.ix_(OI, OI, VI, VI)]
    vt['ijab'] += ((-1/6) * np.einsum('efmn,oj,beim,afno->ijab', t2, f1, t2, t2))[np.ix_(OI, OI, VI, VI)]
    vt['ijab'] += ((-1/4) * np.einsum('efmn,eg,fgij,abmn->ijab', t2, f1, t2, t2))[np.ix_(OI, OI, VI, VI)]  # [T_int=0]
    vt['ijab'] += ((-1/6) * np.einsum('efmn,bg,aeim,fgjn->ijab', t2, f1, t2, t2))[np.ix_(OI, OI, VI, VI)]
    vt['ijab'] += ((1/6) * np.einsum('efmn,bg,egim,afjn->ijab', t2, f1, t2, t2))[np.ix_(OI, OI, VI, VI)]
    vt['ijab'] += ((1/2) * np.einsum('efmn,eg,bfim,agjn->ijab', t2, f1, t2, t2))[np.ix_(OI, OI, VI, VI)]
    vt['ijab'] += ((-1/2) * np.einsum('efmn,om,bein,afjo->ijab', t2, f1, t2, t2))[np.ix_(OI, OI, VI, VI)]
    vt['ijab'] += ((1/6) * np.einsum('efmn,oi,aemo,bfjn->ijab', t2, f1, t2, t2))[np.ix_(OI, OI, VI, VI)]
    vt['ijab'] += ((1/4) * np.einsum('efmn,eg,afij,bgmn->ijab', t2, f1, t2, t2))[np.ix_(OI, OI, VI, VI)]
    vt['ijab'] += ((-1/12) * np.einsum('efmn,bg,agim,efjn->ijab', t2, f1, t2, t2))[np.ix_(OI, OI, VI, VI)]
    vt['ijab'] += ((1/12) * np.einsum('efmn,bg,aeij,fgmn->ijab', t2, f1, t2, t2))[np.ix_(OI, OI, VI, VI)]
    vt['ijab'] += ((-1/4) * np.einsum('efmn,eg,agij,bfmn->ijab', t2, f1, t2, t2))[np.ix_(OI, OI, VI, VI)]
    vt['ijab'] += ((1/12) * np.einsum('efmn,oj,aeio,bfmn->ijab', t2, f1, t2, t2))[np.ix_(OI, OI, VI, VI)]
    vt['ijab'] += ((-1/6) * np.einsum('efmn,oi,aejm,bfno->ijab', t2, f1, t2, t2))[np.ix_(OI, OI, VI, VI)]
    vt['ijab'] += ((-1/12) * np.einsum('efmn,oi,aejo,bfmn->ijab', t2, f1, t2, t2))[np.ix_(OI, OI, VI, VI)]
    vt['ijab'] += ((-1/4) * np.einsum('efmn,eg,bfij,agmn->ijab', t2, f1, t2, t2))[np.ix_(OI, OI, VI, VI)]
    vt['ijab'] += ((-1/6) * np.einsum('efmn,ag,egim,bfjn->ijab', t2, f1, t2, t2))[np.ix_(OI, OI, VI, VI)]
    vt['ijab'] += ((1/6) * np.einsum('efmn,oj,aeim,bfno->ijab', t2, f1, t2, t2))[np.ix_(OI, OI, VI, VI)]
    vt['ijab'] += ((-1/2) * np.einsum('efmn,om,aeio,bfjn->ijab', t2, f1, t2, t2))[np.ix_(OI, OI, VI, VI)]
    vt['ijab'] += ((1/2) * np.einsum('efmn,om,beio,afjn->ijab', t2, f1, t2, t2))[np.ix_(OI, OI, VI, VI)]
    vt['ijab'] += ((1/2) * np.einsum('efmn,om,aein,bfjo->ijab', t2, f1, t2, t2))[np.ix_(OI, OI, VI, VI)]
    vt['ijab'] += ((1/6) * np.einsum('efmn,ag,beim,fgjn->ijab', t2, f1, t2, t2))[np.ix_(OI, OI, VI, VI)]
    vt['ijab'] += ((-1/12) * np.einsum('efmn,bg,egij,afmn->ijab', t2, f1, t2, t2))[np.ix_(OI, OI, VI, VI)]
    vt['ijab'] += ((-1/12) * np.einsum('efmn,oi,abmo,efjn->ijab', t2, f1, t2, t2))[np.ix_(OI, OI, VI, VI)]
    vt['ijab'] += ((-1/12) * np.einsum('efmn,oj,abim,efno->ijab', t2, f1, t2, t2))[np.ix_(OI, OI, VI, VI)]  # [T_int=0]
    vt['ijab'] += ((-1/12) * np.einsum('efmn,bg,efim,agjn->ijab', t2, f1, t2, t2))[np.ix_(OI, OI, VI, VI)]
    vt['ijab'] += ((-1/4) * np.einsum('efmn,om,abin,efjo->ijab', t2, f1, t2, t2))[np.ix_(OI, OI, VI, VI)]
    vt['ijab'] += ((1/4) * np.einsum('efmn,om,abio,efjn->ijab', t2, f1, t2, t2))[np.ix_(OI, OI, VI, VI)]
    vt['ijab'] += ((-1/12) * np.einsum('efmn,oj,efim,abno->ijab', t2, f1, t2, t2))[np.ix_(OI, OI, VI, VI)]
    vt['ijab'] += ((1/12) * np.einsum('efmn,oi,abjm,efno->ijab', t2, f1, t2, t2))[np.ix_(OI, OI, VI, VI)]  # [T_int=0]
    vt['ijab'] += ((1/24) * np.einsum('efmn,bg,efij,agmn->ijab', t2, f1, t2, t2))[np.ix_(OI, OI, VI, VI)]
    vt['ijab'] += ((1/24) * np.einsum('efmn,oi,abmn,efjo->ijab', t2, f1, t2, t2))[np.ix_(OI, OI, VI, VI)]  # [T_int=0]
    vt['ijab'] += ((-1/12) * np.einsum('efmn,oj,beio,afmn->ijab', t2, f1, t2, t2))[np.ix_(OI, OI, VI, VI)]
    vt['ijab'] += ((-1/12) * np.einsum('efmn,ag,beij,fgmn->ijab', t2, f1, t2, t2))[np.ix_(OI, OI, VI, VI)]
    vt['ijab'] += ((1/12) * np.einsum('efmn,ag,efim,bgjn->ijab', t2, f1, t2, t2))[np.ix_(OI, OI, VI, VI)]
    vt['ijab'] += ((-1/12) * np.einsum('efmn,oi,aemn,bfjo->ijab', t2, f1, t2, t2))[np.ix_(OI, OI, VI, VI)]
    vt['ijab'] += ((-1/4) * np.einsum('efmn,om,efin,abjo->ijab', t2, f1, t2, t2))[np.ix_(OI, OI, VI, VI)]
    vt['ijab'] += ((1/12) * np.einsum('efmn,ag,bgim,efjn->ijab', t2, f1, t2, t2))[np.ix_(OI, OI, VI, VI)]
    vt['ijab'] += ((1/12) * np.einsum('efmn,ag,egij,bfmn->ijab', t2, f1, t2, t2))[np.ix_(OI, OI, VI, VI)]
    vt['ijab'] += ((1/4) * np.einsum('efmn,om,efio,abjn->ijab', t2, f1, t2, t2))[np.ix_(OI, OI, VI, VI)]
    vt['ijab'] += ((-1/24) * np.einsum('efmn,oj,efio,abmn->ijab', t2, f1, t2, t2))[np.ix_(OI, OI, VI, VI)]  # [T_int=0]
    vt['ijab'] += ((-1/24) * np.einsum('efmn,ag,efij,bgmn->ijab', t2, f1, t2, t2))[np.ix_(OI, OI, VI, VI)]
    vt['ijab'] += ((-1/3) * np.einsum('efmn,em,bn,afij->ijab', t2, f1, t1, t2))[np.ix_(OI, OI, VI, VI)]  # [T_int=0]
    vt['ijab'] += ((2/3) * np.einsum('em,nf,efij,abmn->ijab', t1, f1, t2, t2))[np.ix_(OI, OI, VI, VI)]  # [T_int=0]
    vt['ijab'] += ((1/3) * np.einsum('efmn,em,an,bfij->ijab', t2, f1, t1, t2))[np.ix_(OI, OI, VI, VI)]  # [T_int=0]
    vt['ijab'] += ((2/3) * np.einsum('em,nf,bfim,aejn->ijab', t1, f1, t2, t2))[np.ix_(OI, OI, VI, VI)]
    vt['ijab'] += ((-1/2) * np.einsum('em,nf,efim,abjn->ijab', t1, f1, t2, t2))[np.ix_(OI, OI, VI, VI)]
    vt['ijab'] += ((1/6) * np.einsum('em,nf,efin,abjm->ijab', t1, f1, t2, t2))[np.ix_(OI, OI, VI, VI)]  # [T_int=0]
    vt['ijab'] += ((-1/2) * np.einsum('em,nf,bfij,aemn->ijab', t1, f1, t2, t2))[np.ix_(OI, OI, VI, VI)]
    vt['ijab'] += ((-1/3) * np.einsum('efmn,em,fj,abin->ijab', t2, f1, t1, t2))[np.ix_(OI, OI, VI, VI)]
    vt['ijab'] += ((1/2) * np.einsum('em,nf,abin,efjm->ijab', t1, f1, t2, t2))[np.ix_(OI, OI, VI, VI)]
    vt['ijab'] += ((-1/6) * np.einsum('em,nf,abim,efjn->ijab', t1, f1, t2, t2))[np.ix_(OI, OI, VI, VI)]  # [T_int=0]
    vt['ijab'] += ((-1/3) * np.einsum('efmn,am,en,bfij->ijab', t2, f1, t1, t2))[np.ix_(OI, OI, VI, VI)]
    vt['ijab'] += ((1/3) * np.einsum('efmn,ej,fm,abin->ijab', t2, f1, t1, t2))[np.ix_(OI, OI, VI, VI)]
    vt['ijab'] += ((1/2) * np.einsum('em,nf,afij,bemn->ijab', t1, f1, t2, t2))[np.ix_(OI, OI, VI, VI)]
    vt['ijab'] += ((1/3) * np.einsum('efmn,bm,en,afij->ijab', t2, f1, t1, t2))[np.ix_(OI, OI, VI, VI)]
    vt['ijab'] += ((-2/3) * np.einsum('em,nf,afim,bejn->ijab', t1, f1, t2, t2))[np.ix_(OI, OI, VI, VI)]
    vt['ijab'] += ((1/6) * np.einsum('em,nf,beij,afmn->ijab', t1, f1, t2, t2))[np.ix_(OI, OI, VI, VI)]
    vt['ijab'] += ((-1/6) * np.einsum('em,nf,aeij,bfmn->ijab', t1, f1, t2, t2))[np.ix_(OI, OI, VI, VI)]
    vt['ijab'] += ((-2/3) * np.einsum('em,nf,aein,bfjm->ijab', t1, f1, t2, t2))[np.ix_(OI, OI, VI, VI)]
    vt['ijab'] += ((1/3) * np.einsum('efmn,em,fi,abjn->ijab', t2, f1, t1, t2))[np.ix_(OI, OI, VI, VI)]
    vt['ijab'] += ((-1/3) * np.einsum('efmn,ei,fm,abjn->ijab', t2, f1, t1, t2))[np.ix_(OI, OI, VI, VI)]
    vt['ijab'] += ((-1/6) * np.einsum('efmn,ej,fi,abmn->ijab', t2, f1, t1, t2))[np.ix_(OI, OI, VI, VI)]  # [T_int=0]
    vt['ijab'] += ((1/6) * np.einsum('efmn,ei,fj,abmn->ijab', t2, f1, t1, t2))[np.ix_(OI, OI, VI, VI)]  # [T_int=0]
    vt['ijab'] += ((2/3) * np.einsum('em,nf,bein,afjm->ijab', t1, f1, t2, t2))[np.ix_(OI, OI, VI, VI)]
    vt['ijab'] += ((1/6) * np.einsum('efmn,am,bn,efij->ijab', t2, f1, t1, t2))[np.ix_(OI, OI, VI, VI)]  # [T_int=0]
    vt['ijab'] += ((-1/6) * np.einsum('efmn,bm,an,efij->ijab', t2, f1, t1, t2))[np.ix_(OI, OI, VI, VI)]  # [T_int=0]
    vt['ijab'] += ((-1/3) * np.einsum('efmn,ei,am,bfjn->ijab', t2, f1, t1, t2))[np.ix_(OI, OI, VI, VI)]  # [T_int=0]
    vt['ijab'] += ((1/3) * np.einsum('efmn,ej,am,bfin->ijab', t2, f1, t1, t2))[np.ix_(OI, OI, VI, VI)]  # [T_int=0]
    vt['ijab'] += ((-1/3) * np.einsum('efmn,ej,bm,afin->ijab', t2, f1, t1, t2))[np.ix_(OI, OI, VI, VI)]  # [T_int=0]
    vt['ijab'] += ((1/3) * np.einsum('efmn,ei,bm,afjn->ijab', t2, f1, t1, t2))[np.ix_(OI, OI, VI, VI)]  # [T_int=0]
    vt['ijab'] += ((-1/3) * np.einsum('efmn,bm,ej,afin->ijab', t2, f1, t1, t2))[np.ix_(OI, OI, VI, VI)]
    vt['ijab'] += ((1/3) * np.einsum('efmn,bm,ei,afjn->ijab', t2, f1, t1, t2))[np.ix_(OI, OI, VI, VI)]
    vt['ijab'] += ((1/3) * np.einsum('efmn,am,ej,bfin->ijab', t2, f1, t1, t2))[np.ix_(OI, OI, VI, VI)]
    vt['ijab'] += ((-1/3) * np.einsum('efmn,am,ei,bfjn->ijab', t2, f1, t1, t2))[np.ix_(OI, OI, VI, VI)]
    vt['ijab'] += ((1/6) * np.einsum('em,nj,bm,aein->ijab', t1, f1, t1, t2))[np.ix_(OI, OI, VI, VI)]  # [T_int=0]
    vt['ijab'] += ((-1/6) * np.einsum('em,bf,am,efij->ijab', t1, f1, t1, t2))[np.ix_(OI, OI, VI, VI)]  # [T_int=0]
    vt['ijab'] += ((1/6) * np.einsum('em,af,ej,bfim->ijab', t1, f1, t1, t2))[np.ix_(OI, OI, VI, VI)]
    vt['ijab'] += ((1/6) * np.einsum('em,bf,ei,afjm->ijab', t1, f1, t1, t2))[np.ix_(OI, OI, VI, VI)]
    vt['ijab'] += ((-1/6) * np.einsum('em,bf,ej,afim->ijab', t1, f1, t1, t2))[np.ix_(OI, OI, VI, VI)]
    vt['ijab'] += ((1/6) * np.einsum('em,nj,ei,abmn->ijab', t1, f1, t1, t2))[np.ix_(OI, OI, VI, VI)]  # [T_int=0]
    vt['ijab'] += ((-1/6) * np.einsum('em,ni,ej,abmn->ijab', t1, f1, t1, t2))[np.ix_(OI, OI, VI, VI)]  # [T_int=0]
    vt['ijab'] += ((-1/6) * np.einsum('em,ni,bm,aejn->ijab', t1, f1, t1, t2))[np.ix_(OI, OI, VI, VI)]  # [T_int=0]
    vt['ijab'] += ((-1/6) * np.einsum('em,ni,en,abjm->ijab', t1, f1, t1, t2))[np.ix_(OI, OI, VI, VI)]  # [T_int=0]
    vt['ijab'] += ((1/6) * np.einsum('em,af,bm,efij->ijab', t1, f1, t1, t2))[np.ix_(OI, OI, VI, VI)]  # [T_int=0]
    vt['ijab'] += ((1/6) * np.einsum('em,nj,en,abim->ijab', t1, f1, t1, t2))[np.ix_(OI, OI, VI, VI)]  # [T_int=0]
    vt['ijab'] += ((-1/6) * np.einsum('em,bf,fm,aeij->ijab', t1, f1, t1, t2))[np.ix_(OI, OI, VI, VI)]
    vt['ijab'] += ((-1/6) * np.einsum('em,nj,am,bein->ijab', t1, f1, t1, t2))[np.ix_(OI, OI, VI, VI)]  # [T_int=0]
    vt['ijab'] += ((-1/2) * np.einsum('em,ef,bm,afij->ijab', t1, f1, t1, t2))[np.ix_(OI, OI, VI, VI)]  # [T_int=0]
    vt['ijab'] += ((1/2) * np.einsum('em,nm,bn,aeij->ijab', t1, f1, t1, t2))[np.ix_(OI, OI, VI, VI)]  # [T_int=0]
    vt['ijab'] += ((-1/2) * np.einsum('em,nm,an,beij->ijab', t1, f1, t1, t2))[np.ix_(OI, OI, VI, VI)]  # [T_int=0]
    vt['ijab'] += ((1/2) * np.einsum('em,ef,am,bfij->ijab', t1, f1, t1, t2))[np.ix_(OI, OI, VI, VI)]  # [T_int=0]
    vt['ijab'] += ((1/2) * np.einsum('em,nm,ej,abin->ijab', t1, f1, t1, t2))[np.ix_(OI, OI, VI, VI)]
    vt['ijab'] += ((-1/2) * np.einsum('em,ef,fj,abim->ijab', t1, f1, t1, t2))[np.ix_(OI, OI, VI, VI)]  # [T_int=0]
    vt['ijab'] += ((1/6) * np.einsum('em,af,fm,beij->ijab', t1, f1, t1, t2))[np.ix_(OI, OI, VI, VI)]
    vt['ijab'] += ((-1/6) * np.einsum('em,af,ei,bfjm->ijab', t1, f1, t1, t2))[np.ix_(OI, OI, VI, VI)]
    vt['ijab'] += ((1/6) * np.einsum('em,ni,am,bejn->ijab', t1, f1, t1, t2))[np.ix_(OI, OI, VI, VI)]  # [T_int=0]
    vt['ijab'] += ((-1/2) * np.einsum('em,nm,ei,abjn->ijab', t1, f1, t1, t2))[np.ix_(OI, OI, VI, VI)]
    vt['ijab'] += ((1/2) * np.einsum('em,ef,fi,abjm->ijab', t1, f1, t1, t2))[np.ix_(OI, OI, VI, VI)]  # [T_int=0]

    # --- PHHP ---
    vt['aijb'] += ((-1/6) * np.einsum('ebmi,nf,am,efjn->aijb', t2, f1, t1, t2))[np.ix_(VI, OI, OI, VI)]  # [T_int=0]
    vt['aijb'] += ((-2/3) * np.einsum('ebmi,nf,fm,aejn->aijb', t2, f1, t1, t2))[np.ix_(VI, OI, OI, VI)]
    vt['aijb'] += ((-1/6) * np.einsum('efmi,nb,em,afjn->aijb', t2, f1, t1, t2))[np.ix_(VI, OI, OI, VI)]
    vt['aijb'] += ((1/6) * np.einsum('em,fbni,an,efjm->aijb', t1, t2, f1, t2))[np.ix_(VI, OI, OI, VI)]
    vt['aijb'] += ((-2/3) * np.einsum('ebmi,nf,en,afjm->aijb', t2, f1, t1, t2))[np.ix_(VI, OI, OI, VI)]
    vt['aijb'] += ((1/12) * np.einsum('efmi,nb,am,efjn->aijb', t2, f1, t1, t2))[np.ix_(VI, OI, OI, VI)]  # [T_int=0]
    vt['aijb'] += ((-1/6) * np.einsum('ebmn,if,em,afjn->aijb', t2, f1, t1, t2))[np.ix_(VI, OI, OI, VI)]
    vt['aijb'] += ((-2/3) * np.einsum('em,fbni,en,afjm->aijb', t1, t2, f1, t2))[np.ix_(VI, OI, OI, VI)]
    vt['aijb'] += ((1/6) * np.einsum('em,fbni,fj,aemn->aijb', t1, t2, f1, t2))[np.ix_(VI, OI, OI, VI)]
    vt['aijb'] += ((-1/6) * np.einsum('ebmi,nf,ej,afmn->aijb', t2, f1, t1, t2))[np.ix_(VI, OI, OI, VI)]
    vt['aijb'] += ((1/12) * np.einsum('ebmn,if,ej,afmn->aijb', t2, f1, t1, t2))[np.ix_(VI, OI, OI, VI)]
    vt['aijb'] += ((-1/6) * np.einsum('bm,efni,en,afjm->aijb', t1, t2, f1, t2))[np.ix_(VI, OI, OI, VI)]  # [T_int=0]
    vt['aijb'] += ((-1/6) * np.einsum('ei,fbmn,fm,aejn->aijb', t1, t2, f1, t2))[np.ix_(VI, OI, OI, VI)]
    vt['aijb'] += ((-2/3) * np.einsum('em,fbni,fm,aejn->aijb', t1, t2, f1, t2))[np.ix_(VI, OI, OI, VI)]
    vt['aijb'] += ((1/12) * np.einsum('bm,efni,an,efjm->aijb', t1, t2, f1, t2))[np.ix_(VI, OI, OI, VI)]  # [T_int=0]
    vt['aijb'] += ((1/12) * np.einsum('ei,fbmn,fj,aemn->aijb', t1, t2, f1, t2))[np.ix_(VI, OI, OI, VI)]
    vt['aijb'] += ((-1/6) * np.einsum('em,fbni,ej,afmn->aijb', t1, t2, f1, t2))[np.ix_(VI, OI, OI, VI)]
    vt['aijb'] += ((1/2) * np.einsum('bm,efni,em,afjn->aijb', t1, t2, f1, t2))[np.ix_(VI, OI, OI, VI)]  # [T_int=0]
    vt['aijb'] += ((-1/6) * np.einsum('bm,efni,ej,afmn->aijb', t1, t2, f1, t2))[np.ix_(VI, OI, OI, VI)]  # [T_int=0]
    vt['aijb'] += ((-1/6) * np.einsum('em,fbni,am,efjn->aijb', t1, t2, f1, t2))[np.ix_(VI, OI, OI, VI)]
    vt['aijb'] += ((1/2) * np.einsum('ei,fbmn,em,afjn->aijb', t1, t2, f1, t2))[np.ix_(VI, OI, OI, VI)]
    vt['aijb'] += ((-1/6) * np.einsum('ei,fbmn,am,efjn->aijb', t1, t2, f1, t2))[np.ix_(VI, OI, OI, VI)]
    vt['aijb'] += ((1/6) * np.einsum('ebmn,if,fm,aejn->aijb', t2, f1, t1, t2))[np.ix_(VI, OI, OI, VI)]
    vt['aijb'] += ((1/6) * np.einsum('ebmn,if,am,efjn->aijb', t2, f1, t1, t2))[np.ix_(VI, OI, OI, VI)]  # [T_int=0]
    vt['aijb'] += ((1/6) * np.einsum('efmi,nb,en,afjm->aijb', t2, f1, t1, t2))[np.ix_(VI, OI, OI, VI)]
    vt['aijb'] += ((1/6) * np.einsum('efmi,nb,ej,afmn->aijb', t2, f1, t1, t2))[np.ix_(VI, OI, OI, VI)]
    vt['aijb'] += ((1/2) * np.einsum('ebmi,nf,an,efjm->aijb', t2, f1, t1, t2))[np.ix_(VI, OI, OI, VI)]  # [T_int=0]
    vt['aijb'] += ((1/2) * np.einsum('ebmi,nf,fj,aemn->aijb', t2, f1, t1, t2))[np.ix_(VI, OI, OI, VI)]
    vt['aijb'] += ((1/6) * np.einsum('em,bn,im,aejn->aijb', t1, t1, f1, t2))[np.ix_(VI, OI, OI, VI)]  # [T_int=0]
    vt['aijb'] += ((-1/2) * np.einsum('bm,ei,ef,afjm->aijb', t1, t1, f1, t2))[np.ix_(VI, OI, OI, VI)]  # [T_int=0]
    vt['aijb'] += ((1/3) * np.einsum('bm,ei,af,efjm->aijb', t1, t1, f1, t2))[np.ix_(VI, OI, OI, VI)]  # [T_int=0]
    vt['aijb'] += ((-1/3) * np.einsum('bm,ei,nj,aemn->aijb', t1, t1, f1, t2))[np.ix_(VI, OI, OI, VI)]  # [T_int=0]
    vt['aijb'] += ((1/2) * np.einsum('bm,ei,nm,aejn->aijb', t1, t1, f1, t2))[np.ix_(VI, OI, OI, VI)]  # [T_int=0]
    vt['aijb'] += ((-1/6) * np.einsum('em,fi,eb,afjm->aijb', t1, t1, f1, t2))[np.ix_(VI, OI, OI, VI)]
    vt['aijb'] += ((-1/3) * np.einsum('ebmn,im,ej,an->aijb', t2, f1, t1, t1))[np.ix_(VI, OI, OI, VI)]  # [T_int=0]
    vt['aijb'] += ((1/3) * np.einsum('efmi,eb,fj,am->aijb', t2, f1, t1, t1))[np.ix_(VI, OI, OI, VI)]  # [T_int=0]
    vt['aijb'] += ((-1/6) * np.einsum('ebmi,af,ej,fm->aijb', t2, f1, t1, t1))[np.ix_(VI, OI, OI, VI)]
    vt['aijb'] += ((-1/2) * np.einsum('ebmi,ef,fj,am->aijb', t2, f1, t1, t1))[np.ix_(VI, OI, OI, VI)]  # [T_int=0]
    vt['aijb'] += ((1/6) * np.einsum('ebmi,nj,am,en->aijb', t2, f1, t1, t1))[np.ix_(VI, OI, OI, VI)]  # [T_int=0]
    vt['aijb'] += ((1/2) * np.einsum('ebmi,nm,ej,an->aijb', t2, f1, t1, t1))[np.ix_(VI, OI, OI, VI)]  # [T_int=0]

    return scalar

