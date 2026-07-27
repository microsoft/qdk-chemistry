# 5-Minute Talk Track — QDK/Chemistry: Why Multi-Configurational?

**Companion notebook:** [`multiconfig_demo.ipynb`](multiconfig_demo.ipynb)
**System:** Carbon monoxide (CO), triple bond pulled from equilibrium to full dissociation
**Format:** Continuous narration to speak *over* the notebook as you scroll and run cells.

---

## How to use this document

This is a **word-for-word talk track**, not a list of cues. It runs a little rich on
purpose: spoken at a brisk pace — and remember you'll be *talking over* cell
execution, especially the ~30-second scan cell — it lands right around five minutes.
Read every word slowly and it's closer to six, so paraphrase or trim freely. Each
section header tells you **which cell is on screen** and **roughly when** you should
be there. Stage directions — what to click, where to point — are in *[italic
brackets]*. Beat marks **[BEAT]** are natural places to pause, run a cell, or let a
result land.

> **Tip:** Pre-run the whole notebook once to warm up the backends, then restart the
> kernel and run live. The full scan takes well under a minute on a laptop.

---

## [0:00–0:35] — The premise

*[On screen: the title cell. Don't run anything yet — let the title and outline sit
there while you set up the story.]*

"Here's a question that sits underneath almost all of modern chemistry: **how do you
know when you can trust your calculation?**

Nearly every method a chemist reaches for by default — Hartree–Fock, MP2, coupled
cluster — quietly makes the *same* assumption: that the molecule is described by
**one** dominant arrangement of electrons, a single configuration. Near a stable
geometry, that's fantastic — it's why these methods work so well.

**[BEAT]**

But the moment you break bonds — in a reaction, at a transition state, in a diradical
or a metal active site — that picture falls apart. Those are the *strongly correlated*
problems, and they demand **multi-configurational** methods. So rather than tell you,
I'll show you: we'll pull apart the triple bond of carbon monoxide and watch the
standard methods break — all inside one toolkit, QDK/Chemistry."

---

## [0:35–1:05] — Setup: one interface for everything

*[Run the imports cell. It's quick and undramatic — that's the point.]*

"First, the setup — and there's only one thing to notice. Every method we're about to
use, from mean-field Hartree–Fock up to a multi-configurational solver, is built from
the **same factory function**: `create`. You don't learn five APIs for five methods
— you ask the toolkit to `create` what you want and hand it the same inputs. That's
what makes the comparison we're about to do *fair*: every method sees exactly the
same molecule, basis set, and active space."

---

## [1:05–1:50] — Where one configuration works

*[Run the equilibrium cell. Point at the printed leading-configuration weight when
it appears.]*

"Let's be fair to the single-configuration methods and start on their home turf: CO
at its equilibrium bond length, about 1.13 ångström.

The number to watch is the **leading-configuration weight**: of the entire
wavefunction, how much is carried by its single most important configuration? If
that's close to one, one configuration dominates and a single-reference method is
fully justified.

**[BEAT — let the result print.]**

There it is: **ninety-five percent**. So Hartree–Fock, MP2, and coupled cluster
should all be excellent here — and they are. Hold onto that number. The rest of this
demo is the story of watching it collapse, and what that does to our methods."

---

## [1:50–2:30] — One evaluator, four levels of theory

*[Run the evaluator-definition cell. Nothing prints — it just defines the function.
Scroll slowly through it so the audience sees the four method calls.]*

"Here's the engine of the experiment, in one function. Give it a bond length, and it
runs four calculations: **Hartree–Fock**, the single-determinant baseline; **MP2** and
**CCSD**, the standard ways of adding correlation *on top of* that determinant; and
**CASCI**, a genuinely **multi-configurational** solver that's free to mix many
configurations together. Each is just another `create` call on the same inputs. And
notice I've wrapped the coupled-cluster call so that if it *fails*, we record it and
keep going — that turns out to matter."

---

## [2:30–3:00] — Walk the bond apart

*[Run the scan cell. While it runs — maybe 30–40 seconds — keep talking. When the
table appears, point at the CCSD line.]*

"Now let's run it across the whole dissociation — sweeping the carbon–oxygen distance
from the bonded minimum out to two atoms drifting apart. Nine geometries, a full
four-method calculation at each: the molecule coming apart, frame by frame.

**[BEAT — wait for the table.]**

And before we even plot anything, there's a headline hiding in this table:
**coupled cluster failed to converge** at the longest bond length. CCSD is the gold
standard of quantum chemistry — and once we stretch this bond far enough, it doesn't
give a wrong answer, it gives *no answer at all*."

---

## [3:00–3:50] — The payoff: watch them break

*[Run the energy-curve plot cell. This is the centerpiece. Use your cursor to trace
the curves left-to-right as you narrate.]*

"Now the picture that ties it together: every method's energy as the bond breaks,
each relative to its own equilibrium so we compare *shapes*.

Start on the **left**, near equilibrium. *[Point.]* Every curve is stacked on top of
every other — the methods completely agree, because one configuration dominates here.

**[BEAT]**

Now follow them **right**, as the bond breaks. *[Trace outward.]* Watch them peel
apart. Hartree–Fock — red — climbs wildly *too high*, because a closed-shell
determinant can't describe two separating atoms. MP2 and CCSD sit on that same broken
foundation and overshoot — and the coupled-cluster curve has **gaps**, the geometries
where it gave up. Only the green curve — **CASCI**, the multi-configurational method
— stays smooth the whole way and levels off where it physically should. One method
survives the bond breaking. The others don't."

---

## [3:50–4:25] — Why it happens

*[Run the weight-collapse plot cell. Point first at the left edge near 0.95, then
trace the curve down past the dashed line.]*

"And here's *why*. This is that leading-configuration weight again — the ninety-five
percent from earlier — now plotted across the whole dissociation. *[Point top-left.]*
It starts up near one, above this dashed line that marks the safe single-reference
regime. Then as the bond stretches, *[trace down]* it **collapses** — past one half,
down toward ten percent.

**[BEAT]**

That falling curve *is* the explanation for everything we just saw. Below that line
there's no dominant configuration left for a single-reference method to stand on. The
bond breaking and the wavefunction going multi-configurational are the same physical
event — and this one plot captures it."

---

## [4:25–4:50] — Look inside the wavefunction

*[Run the determinant-listing cell. Let the list of configurations print and point
at the near-equal weights.]*

"We can even open up the wavefunction at a stretched geometry. *[Gesture at the
list.]* Instead of one configuration carrying everything, we get a handful at nearly
*equal* weight — four of them at around eleven percent each. There's no single 'main'
configuration anymore. That's what **multi-configurational** means, made concrete —
and exactly what a single-determinant method can't represent."

---

## [4:50–5:05] — Wrap up

*[Run the final summary-table cell and leave it on screen through your closing lines.]*

"So, to pull it together: near equilibrium, one configuration rules and every method
agrees. Break the bond, and those methods break with it — Hartree–Fock to the wrong
limit, MP2 and CCSD overshooting, coupled cluster failing outright. Only the
multi-configurational method stays correct the whole way.

**[BEAT]**

And this isn't an edge case. Bond-breaking, diradicals, transition-metal and
*f*-element chemistry, catalytic transition states — these strongly correlated
problems are everywhere in real chemistry, and they're where multi-configurational
methods stop being optional. They're also the natural launch point for quantum
algorithms — but that's another demo. Everything you just saw came out of one
toolkit, behind one interface. That's QDK/Chemistry — `pip install qdk-chemistry`,
and this notebook is in the repo. Thanks for watching."

---

## Backup answers (for live Q&A)

- **"Why does CCSD fail rather than just be inaccurate?"** Coupled cluster solves a
  set of nonlinear amplitude equations built on the Hartree–Fock reference. When that
  reference becomes near-degenerate with other configurations, those equations stop
  having a stable solution and the iterations diverge — so you get non-convergence,
  not a wrong number.
- **"Isn't this just because the basis set is small?"** No — it's a *qualitative*
  failure, not an accuracy one. A bigger basis makes the numbers more precise but
  doesn't fix a single-determinant reference that's fundamentally wrong for
  bond-breaking. That's static correlation, and only a multi-configurational
  treatment removes it.
- **"What is the active space here?"** CO's full valence space — ten electrons in
  eight orbitals, CAS(10,8) — which captures the σ and π bonding/antibonding orbitals
  involved in the triple bond. The toolkit selected it automatically.
- **"Is there a kink near 2 Å?"** Yes, a small one in the Hartree–Fock and
  coupled-cluster curves — that's a genuine reference instability as the mean-field
  solution rearranges, not a numerical bug. Only bring it up if asked.
- **"How long does this take to run?"** The full nine-geometry, four-method scan is
  well under a minute on a laptop — which is itself a nice point about the
  performance of the underlying solvers.
