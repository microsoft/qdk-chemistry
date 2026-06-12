---
name: feature-branch-rubric
description: "Use when: working on a feature branch — both while implementing changes and before merging them. Applies an opinionated rubric that prioritizes readability, convention adherence, and minimum sufficient code. Flags AI-typical overengineering, invented semantics, narrative comments, and over-narrowed contracts."
---

# Feature Branch Rubric

A rubric for the code on a feature branch. The goal is **boring, easy-to-read code that obeys existing conventions and does no more than the task requires**.

The rubric applies in **both directions**:

- **Implementation pass** — *before* writing changes, and continuously *while* writing them. Ground yourself in intent and conventions first; let the principles below govern what you produce.
- **Review pass** — before merging. Walk the diff and flag deviations.

The principles in "Core Ethos" are identical in both passes. The procedure has two entry points — one for each pass — that share most steps. Implementation and review are not separated in this workflow; the same agent applying this skill is responsible for both.

Regardless of pass: **never `git commit`** on the developer's behalf, and **consult the developer** on public-API changes, new dependencies, algorithm semantics, data-model changes, or architectural decisions, per repo policy.

## Core Ethos (Non-Negotiable)

### 1. Overengineering is a defect

Overengineering is just as much a bug as a logic error. It costs the reader time forever. Flag any of the following:

- **Single-use functions.** If a helper is called exactly once and its body is short, **inline it**. Naming a four-line block does not always clarify — often it just forces the reader to jump.
- **Single-use constants** pulled out to module/global scope. A magic number with a comment at the call site is often more readable than `_DEFAULT_TIMEOUT_SECONDS = 30` defined 400 lines away. Constants earn their place by being used in ≥2 locations or by encoding a non-obvious invariant.
- **Premature abstraction.** New base classes, interfaces, protocols, registries, factories, strategies, or visitors with **one implementation**. Use the concrete type. Abstractions are added when a second use case appears, not in anticipation.
- **Speculative configuration.** Settings/parameters/kwargs that nothing in the diff actually varies. If every caller passes the same value, it isn't configuration — it's a constant.
- **Wrapper classes that just forward calls.** A class that holds one field and re-exports its methods is a tax on the reader.
- **Backward-compat shims for unreleased code.** Internal APIs that have never shipped have no compatibility to preserve. Change them directly.
- **New dependencies, new modules, new packages** introduced to solve a problem that 20 lines of in-place code would solve. Dependencies and module boundaries are expensive — they require justification.
- **Design-pattern reflexes.** Strategy/Visitor/Observer/Builder used where a function call would suffice. The pattern name is not a reason.

**Heuristic:** If a reviewer has to ask "why is this here?" and the honest answer is "in case we need it later," delete it.

### 2. Conventions are the law of the codebase

New code must look like it belongs. AI-generated code routinely invents semantics — especially for serialization/deserialization, error types, configuration, and naming. **Do not let it.**

- **Read the surrounding code before judging.** Always consult the **`acquire-codebase-knowledge`** and **`domain-reference`** skills (and the existing files in the area of change) before deciding whether a new addition is well-formed. Conventions are not always documented; they are demonstrated by sibling code.
- **Serialization/deserialization in particular.** Check existing `to_json`/`from_json`/`to_hdf5`/`from_hdf5`/`__getstate__`/`__setstate__` patterns *for similar data structures* before accepting a new one. Do not accept newly invented field names, wrapper envelopes, type tags, version fields, or container layouts unless they match the existing pattern for similar types.
- **Error handling.** Do not accept new exception classes when an existing one applies. Do not accept new `try`/`except` patterns that differ from how sibling code reports failure.
- **Naming.** Match the prefix/suffix/case conventions of the file and module being edited (e.g., C++ private `_member`, Python snake_case, namespace `qdk::chemistry`).
- **Configuration.** In this codebase, algorithm configuration belongs in `Settings`. Reject ad-hoc instance variables or new `**kwargs` on `run()`.
- **Data classes are immutable; algorithms are stateless.** Any change that mutates a data class, or stashes state on an algorithm outside `Settings`, is a design violation.

**Heuristic:** If three nearby files do X one way and the new code does X a second way for no stated reason, the new code is wrong.

### 3. Code additions should be the minimum needed to serve the purpose

Related to (1) but called out explicitly because it applies *per-change*, not just *per-design*.

- Every added line should map to something the task actually requires. If a line can be deleted and the task still ships, delete it.
- This applies to tests too — overengineered tests (elaborate mocks, parameterized scenarios that don't test anything new, deep fixtures, helper hierarchies) are also defects. Tests should be obvious and direct.
- This applies to docstrings and comments. Don't add a 30-line module docstring to a 40-line module. Don't comment what the code already says.
- This applies to documentation. Don't add new conceptual `.rst` pages for a feature that doesn't change user-visible behavior.

**Heuristic:** Ask "if I deleted this, what would break?" If the answer is "nothing," delete it.

### 4. Comments and docstrings convey *scope*, not detail

A reader should be able to understand **what a function/class is for and what its contract is** by reading only its docblock — and nothing more. The body explains itself. Comments that paraphrase the body are noise, and noise rots faster than code.

Applies equally to Python docstrings and to C++ Doxygen blocks. Flag any of:

- **Restating the code in prose.** `// Increment counter` above `counter += 1`. `@param x The x value.` for a parameter named `x`. A docstring whose body is a verbal walkthrough of the function.
- **Step-by-step narration of an algorithm** inside a docblock. If a non-trivial algorithm needs explanation, a *short* `@note` or in-body comment pointing at the key insight (or a reference to a paper/section) is sufficient. Do not transcribe the implementation.
- **Doxygen `@details` paragraphs that paraphrase the brief.** Brief + `@param` + `@return` + `@throws` + `@tparam` (where applicable) is the standard shape. Anything more needs to earn its place by clarifying a non-obvious contract — units, ownership, lifetime, complexity, thread-safety, invariants.
- **Module/file headers that exhaustively list contents.** The file is the source of truth for what it contains.
- **Inline comments restating the next line.** Comment only when a reader would otherwise reasonably ask "why is this here?" or "why this way and not the obvious way?"
- **TODO/FIXME/NOTE blocks that ship without an issue link** and without changing behavior — they age into lies.

**What good comments do:** name a non-obvious *why*, declare a contract the type system can't express (units, invariants, ranges), point at the source of an algorithm (paper, section number), or mark a real hazard.

**Heuristic:** If a reader who understands the surrounding code would not learn anything from the comment that the code itself doesn't already say, delete it.

### 5. Generality over specificity — don't codify the current state

A repeated AI failure mode: encoding the *current* set of callers, consumers, values, or use cases as a *hard constraint* in code that was designed to be general. This narrows a flexible component into something that has to be widened later, with breaking changes, the moment a second consumer appears.

This is the code-level analog of the documentation principle "**Be generic, not specific**" in `.github/copilot-instructions.md` (lines 41–46) and `.github/skills/docs-build/SKILL.md` (the Documentation Writing Principles section). The doc rule and the code rule are the same rule: **describe and accept things by their general contract; mention specific cases as examples, not as definitions.**

Flag any of:

- **Whitelist validation of values the function was designed to accept generically.** `assert backend in {"pyscf", "qiskit"}` inside a function that just needs *a* backend object. Validate against the protocol/interface, not the enumeration of today's plugins.
- **Over-narrowed type signatures.** `Literal["foo", "bar"]` where `str` (or a base class / protocol) is the actual contract. Same for `Union[FooImpl, BarImpl]` where a base/abstract type would do.
- **Names that bind a general helper to its first consumer.** `compute_pyscf_hamiltonian` for a function that takes a `Hamiltonian` and is plugin-agnostic. `format_for_logging` for a serializer that has nothing to do with logging.
- **Docstrings/Doxygen framing a general component as if it serves one purpose.** "This class is used by the SCF solver to ..." — when the class is the generic data structure and the SCF solver is one of several callers. Describe the contract; mention SCF as *an example* if helpful.
- **Tests that pin behavior the design did not intend to guarantee.** Asserting the exact set of supported backends, the exact field order in a serialized blob, the exact list of registered algorithm names — when none of those are part of the contract. Over-specification via tests is just as bad as over-specification in code.
- **Configuration defaults or wiring that hard-code the current consumer.** A factory that defaults to a specific plugin when the plugin is one of several peers.
- **`if`/`switch` ladders enumerating known cases** where dispatch through the existing registry/factory/protocol would work. Adding a new case should not require editing a central enum.

The right reflex on every addition: ask "is this constraint **part of the contract**, or just **a property of today's callers**?" Only the former belongs in the code.

When in doubt, **consult `acquire-codebase-knowledge` and `domain-reference`** to see how the surrounding system describes the touched component's purpose. If the existing description is general and the diff narrows it, the diff is wrong.

**Heuristic:** If a second consumer appearing next week would force this code to change, the code has over-fit its current consumer.

## Additional Anti-Patterns to Flag

These extend the core ethos and address things AI specifically tends to do.

- **Dead and aspirational code.** Unused imports, unused parameters, unreachable branches, `if False:` blocks, TODO placeholders that ship.
- **Tests that mirror the implementation instead of the behavior.** Tests that assert internal state, monkey-patch the unit under test, or re-run the production code path inside the test rather than asserting on observables. Tests should fail when behavior changes, not when implementation changes.
- **Snapshot/golden tests for things that aren't outputs.** Don't snapshot data structures whose representation isn't a contract.
- **Renames and moves bundled with logic changes.** Hard to review. Flag and ask the developer to split.
- **Cross-language drift.** Python change that desyncs from C++ binding (or `.pyi` stub), or vice versa. Check `python/src/pybind11/` and stubs whenever a C++ public type or method changes.
- **Public API changes done quietly.** Anything that adds, renames, or changes the signature of a public symbol — flag and call out for developer consultation per repo policy.
- **New logging / metrics / tracing infrastructure** introduced for a single feature when the project does not already have it. Use existing patterns or none.
- **Generated files committed without their generator.** Stubs, manifests, lock files — make sure the source of truth is updated, not just the artifact.
- **Workarounds disguised as features.** A change that adds a special case to make a failing test pass — when the real bug is elsewhere — is worse than no change.

## Procedure

The procedure has two entry points. Steps marked **[shared]** apply identically to both passes; steps marked **[impl]** or **[review]** apply only to that pass.

### Implementation pass (before / during writing)

Run these *before* the first line of code is added, and re-check **[shared]** steps any time the change grows beyond its original scope.

1. **[shared] Read the stated intent.** See "Read the stated intent" below.
2. **[shared] Acquire context.** See "Acquire context" below. **This step is the directive "look elsewhere in the code before implementing new logic."** Skipping it is the most common path to invented semantics, over-narrowed contracts, and noise comments — the exact defects the principles above forbid. Treat it as load-bearing, not optional.
3. **[impl] Plan against the rubric.** For the change you are about to make, mentally answer the seven questions in "Walk the diff" below — applied to the *intended* additions, not an existing diff. If any answer is wrong, change the plan before writing.
4. **[impl] Write the change.** Apply the principles in Core Ethos as constraints on what you produce: minimum sufficient code, match sibling conventions, comments-as-scope, generality over specificity.
5. **[shared] Verify behavior, surgically.** See "Verify behavior, surgically" below.

### Review pass (before merging)

Run these once the change is on disk, against the diff.

1. **[review] Establish scope.** See "Establish scope" below.
2. **[shared] Read the stated intent.**
3. **[shared] Acquire context.** Re-load if not already in head from the implementation pass; sibling code must still be consulted for the review judgment.
4. **[review] Walk the diff.** See "Walk the diff" below.
5. **[shared] Verify behavior, surgically.**
6. **[review] Report findings.** See "Report findings" below.

If a finding from "Walk the diff" indicates that step 3 (acquire context) was skipped during implementation — for instance, a new addition adopts a pattern that obviously conflicts with two or more sibling files — the **omission itself is the Blocker**, not just the resulting code. Fix the code by adopting the sibling pattern, and treat it as a lesson to do step 3 properly next time.

### Step details

#### Establish scope **[review]**

Identify the base and the head:

```bash
# Determine the merge base against the trunk
git --no-pager merge-base HEAD origin/main
# List changed files
git --no-pager diff --name-status origin/main...HEAD
# Aggregate stats
git --no-pager diff --shortstat origin/main...HEAD
```

Substitute `origin/main` with the actual base branch if different. For uncommitted changes, use `git diff` and `git diff --cached`.

#### Read the stated intent **[shared]**

Understand what the change is trying to do *before* judging or writing code:

- PR description / commit messages (`git --no-pager log origin/main..HEAD`), if any exist yet.
- Linked issue, if any (`gh issue view <n>` or `gh pr view <n>`).
- The first commit's message is often the cleanest statement of intent.
- For the implementation pass: the developer's prompt or the task ticket is the equivalent source.

Write down (mentally, or in a session note) the **one sentence** that summarizes the goal. Every line — written or reviewed — will be judged against this sentence.

#### Acquire context **[shared]**

For non-trivial changes, **do not work in isolation**. Load:

- The **`domain-reference`** skill (specifically the files relevant to the touched area — data structures, plugins, registry, conventions, design rationale).
- The **`acquire-codebase-knowledge`** skill if the change spans an area unfamiliar to you.
- The sibling files of every file you are about to touch (or have touched). New code must match the conventions of its neighbors — conventions are not always documented; they are demonstrated by sibling code.
- For serialization/deserialization changes specifically, read **at least three existing examples** of similar serialization in the codebase before writing or accepting a new one.
- For any new public symbol, search the codebase for existing symbols that play a similar role and match their shape (naming, signature style, docblock format, error-handling pattern).

This step is the operationalization of the directive **"look elsewhere in the code before implementing new logic."** It applies before writing and before reviewing.

#### Walk the diff **[review]** (or "walk the plan" **[impl]**)

For each meaningful hunk (or planned addition), ask:

1. **Does this line need to exist for the stated goal?** (Minimum sufficient code.)
2. **Does it match the conventions of nearby code?** (No invented semantics.)
3. **Is the abstraction level justified by current callers, not future ones?** (No overengineering.)
4. **Does any comment/docstring say more than the contract requires?** (Comments minimal.)
5. **Does any added constraint encode today's callers rather than the actual contract?** (Generality over specificity.)
6. **Is the change reviewable, or is it tangled with renames/moves/refactors?** (Split if not.)
7. **Does it preserve the design invariants of this codebase?** (Immutable data, stateless algorithms, `Settings` for configuration, fixed run signatures, etc.)

Use `view` and `grep` directly. For very large branches, delegate the mechanical walk to the built-in `code-review` agent, but **apply this rubric yourself** to its findings before reporting.

#### Verify behavior, surgically **[shared]**

Run **only** the tests and lints relevant to the diff. Do not run the full suite — CI handles that. See the `cpp-build`, `python-build`, and `lint` skills for commands. Confirm the change actually works before signing off on design.

#### Report findings **[review]**

Group findings by severity. Cite file paths and line numbers. Keep noise low — every finding should be something the developer would actually want to change.

| Severity | Examples |
|----------|----------|
| **Blocker** | Convention violation (esp. serialization), broken invariant, public API change without consultation, behavior regression, broken tests, new dependency without justification, hard-coded enumeration that narrows a generic contract, evidence that sibling code was not consulted before adding a new pattern. |
| **Should-fix** | Overengineering, premature abstraction, single-use helpers, unnecessary constants, dead code, narrative comments/docstrings, Doxygen blocks that paraphrase the body, names that bind a generic helper to one consumer, over-narrowed types, overengineered tests. |
| **Nit** | Naming taste, comment phrasing, ordering. Use sparingly — most "nits" are noise. |

Do **not**:

- Comment on style/formatting that linters already handle.
- Suggest changes outside the diff's scope unless they are directly caused by the change.
- Auto-fix during the review pass — surface, do not rewrite. (The implementation pass *is* where writing happens; the review pass is for judgment, not rework.)
- Run `git commit`.

## What Good Looks Like

A reviewed feature branch should leave the codebase in a state where:

- Six months from now, a new contributor reading any changed file cannot tell which lines are new.
- Every added line has a clear job.
- The diff matches the one-sentence intent and no more.
- The patterns used are the patterns already in the codebase.
- Each new function/class can be understood from its docblock alone; the body is not narrated.
- No new code narrows a generic component to its current consumers. A new caller could be added next week without touching the code added by this diff.

If the branch does not meet those bars, the review's job is to explain — clearly and specifically — why, and to point at the existing code that demonstrates the right pattern.
