---
description: Generates and maintains an AI-written pull request description summary
on:
  pull_request:
    types: [opened, synchronize, reopened]

engine: copilot

permissions:
  contents: read
  pull-requests: read
  copilot-requests: write

safe-outputs:
  jobs:
    update-pr-summary:
      description: >
        Replace the AI-generated section of the PR description with fresh
        content, preserving everything the user wrote above it.
      runs-on: ubuntu-latest
      output: "PR description updated."
      permissions:
        pull-requests: write
      inputs:
        summary:
          description: "The markdown summary body (do not include the horizontal rule or heading, those are added automatically)"
          required: true
          type: string
      steps:
      - name: Splice AI section into PR description
        uses: actions/github-script@v9
        env:
          PR_NUMBER: ${{ github.event.pull_request.number }}
        with:
          script: |
            const fs = require('fs');
            // Marker delimiting the previously-appended AI section; everything
            // from this marker onward is replaced on every run, everything
            // above it (the user's own description) is left untouched.
            const marker = '\n\n---\n\n### AI description\n\n';

            const outputFile = process.env.GH_AW_AGENT_OUTPUT;
            const agentOutput = JSON.parse(fs.readFileSync(outputFile, 'utf8'));
            const item = agentOutput.items.find((i) => i.type === 'update_pr_summary');
            if (!item) {
              core.setFailed('No update_pr_summary item found in agent output.');
              return;
            }

            const prNumber = Number(process.env.PR_NUMBER);
            const { data: pr } = await github.rest.pulls.get({
              owner: context.repo.owner,
              repo: context.repo.repo,
              pull_number: prNumber,
            });

            const markerIndex = pr.body ? pr.body.indexOf(marker) : -1;
            const preserved = markerIndex === -1 ? (pr.body || '') : pr.body.slice(0, markerIndex);

            await github.rest.pulls.update({
              owner: context.repo.owner,
              repo: context.repo.repo,
              pull_number: prNumber,
              body: `${preserved}${marker}${item.summary}`,
            });
---

# PR Description Generator

Analyze the changes introduced by this pull request and write a concise, factual
summary suitable for the bottom of the PR description.

## Output format

Call the `update-pr-summary` tool once, passing the summary content as the
`summary` argument. Do not include a horizontal rule or heading in `summary` —
those are added automatically, and everything before them in the current PR
description is preserved as-is.

## What to include

- A short (2-4 sentence) overview of what changed and why, in plain language based
  on the diff and commit messages.
- A bullet list of the key changes, grouped by area when the diff touches multiple
  parts of the repository (for example `cpp/`, `python/`, `docs/`,
  `.github/workflows/`).
- Explicitly call out any changes to public APIs, build/CI configuration, or
  dependencies (`vcpkg.json`, `pyproject.toml`, manifest files).

## Style

- Be factual and concise. Do not speculate about intent beyond what the diff and
  commit messages show.
- Do not repeat the PR title.
- Use GitHub-flavored Markdown.
