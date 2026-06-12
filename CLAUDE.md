# CLAUDE.md

> **Compatibility shim** — This file exists for tools that auto-read `CLAUDE.md`.
> The primary instructions live in `.github/copilot-instructions.md`.

## Quick Reference

- **Design rule**: Data classes are immutable. Algorithms are stateless — all
  config lives in Settings. Run signatures are fixed per algorithm type.
- **Never commit** on behalf of the developer.
- **Be surgical with tests** — run only what's relevant, not the full suite.
- **Use skills** — see `.github/skills/` for build, test, lint, docs, and
  domain reference workflows.

## Where to Find Things

| What | Where |
|------|-------|
| Full instructions | `.github/copilot-instructions.md` |
| Domain knowledge (conventions, data structures, plugins) | `.github/skills/domain-reference/` |
| Build / test / lint / docs workflows | `.github/skills/{cpp-build,python-build,lint,docs-build}/` |
| Adding algorithms or bindings | `.github/skills/{add-algorithm,add-binding}/` |
| Codebase structural scan | `.github/skills/acquire-codebase-knowledge/` |
