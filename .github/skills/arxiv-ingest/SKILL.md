---
name: arxiv-ingest
description: "Use when: the user asks to read, ingest, summarize, or discuss an arxiv paper by ID. Downloads the LaTeX source from arxiv, extracts it, and reads all .tex files into context."
---

# Arxiv Paper Ingest

Downloads and extracts the LaTeX source of an arxiv paper so you can read and
discuss its content.

## Inputs

The user provides an **arxiv paper ID** (e.g., `2602.19411`, `2301.08322`,
`quant-ph/0512170`). Extract the ID from whatever form the user gives — a bare
ID, a full URL (`https://arxiv.org/abs/2602.19411`), or inline mention.

## Procedure

### 1. Download and extract (LaTeX source — preferred)

```bash
id="<ARXIV_ID>"
d="/tmp/arxiv_${id//\//_}"

mkdir -p "$d" &&
  wget -q -O "$d/src.tar.gz" "https://arxiv.org/src/$id" &&
  tar xzf "$d/src.tar.gz" -C "$d"
```

- Replace `<ARXIV_ID>` with the actual paper ID.
- IDs containing `/` (e.g., `quant-ph/0512170`) are converted to `_` for the
  directory name to avoid nested paths.
- Use `wget -q` to suppress download noise.

If `tar` fails, the download may be a single `.tex` file rather than an
archive. Rename and continue:

```bash
mv "$d/src.tar.gz" "$d/paper.tex"
```

### 2. Find `.tex` files

```bash
find "$d" -name '*.tex'
```

If **no `.tex` files are found** (source unavailable or extraction failed),
fall back to the PDF path in step 3.

### 3. PDF fallback

Use this path when LaTeX source is unavailable (step 1 failed, or no `.tex`
files were found in step 2).

```bash
wget -q -O "$d/paper.pdf" "https://arxiv.org/pdf/$id"
```

Then invoke the **pdf** skill to read `$d/paper.pdf`. The pdf skill handles
text extraction and pagination for large documents.

### 4. Read the paper

**From LaTeX (preferred):**

- Use the **view** tool to read each `.tex` file found in step 2.
- Start with the main file (usually the one that contains `\documentclass` or
  `\begin{document}`). If unclear, check the shortest-named `.tex` file or
  look for a file matching the arxiv ID.
- Read supporting files (sections, appendices) as needed for the user's question.

**From PDF (fallback):**

- Use the **pdf** skill on `$d/paper.pdf`.

### 5. Clean up

After you have finished reading and discussing the paper, remove the temp
directory:

```bash
rm -rf "$d"
```

## Notes

- Arxiv may rate-limit downloads. If `wget` fails, wait a moment and retry once.
- Some papers ship as a single `.tex` file; others split across many. Adapt
  accordingly.
- Very large `.tex` files (>50 KB) should be read with `view_range` to avoid
  truncation.
- **Always prefer LaTeX source over PDF** — it preserves equations, references,
  and structure more faithfully. Only fall back to PDF when source is
  unavailable.
